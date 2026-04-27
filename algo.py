import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import trainer_base
import utils
import math
import models
from torch.func import functional_call
from models.dit import modulate_fused
import functools
from entmax import entmax_bisect

class AR(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer,
                         vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == 'none'

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return - output.gather(
            -1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (torch.distributions.Gumbel(0, 1)
                 .sample((num_samples, num_pred_tokens, self.vocab_size))
                 .to(self.device))
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(num_pred_tokens):
            output = self.backbone(x[:, :i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == 'ancestral_cache'

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(
            model_output)
        unmasked_score = torch.scatter(
            unmasked_score,
            -1,
            x[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = - (
            log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(
            model_output.dtype)[:, :, None]
        model_output = (
            masked_score * masked_indices
            + unmasked_score * (1 - masked_indices))
        return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.noise.type == 'log-linear'
        assert self.parameterization == 'mean'

    def _process_model_output(self, model_output, xt, sigma):
        del xt
        del sigma
        if self.subs_masking:
            model_output[:, :, self.mask_index] += self.neg_infinity
        return model_output.log_softmax(dim=-1)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        del dalpha_t
        assert not low_var
        dt = 1 / self.T
        t = 1 - alpha_t  # Only valid for log-linear schedule.
        t = t.clamp(0., 1.0 - 1e-4)
        alpha_t = alpha_t + torch.zeros_like(xt)
        alpha_s = t - dt + torch.zeros_like(xt)
        assert alpha_s.shape == xt.shape
        assert alpha_t.shape == xt.shape
        log_x_theta_at_x0 = torch.gather(
            log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(
            alpha_s * x_theta_at_m / (t - dt) + 1)
        L_vb_masked = (
            term_1_coef * (term_1_log_nr - term_1_log_dr)
            + term_2_coef * (term_2_log_nr - term_2_log_dr))

        diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
        return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.predictor == 'analytic'

    def _get_score(self, x, sigma):
        return self.forward(x, sigma).exp()

    def _process_model_output(self, model_output, xt, sigma):
        esigm1_log = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            sigma.exp() - 1).log().to(model_output.dtype)
        # logits shape
        # (batch_size, context_length, vocab_size)
        model_output = (model_output
                        - esigm1_log[:, None, None]
                        - np.log(model_output.shape[-1] - 1))
        # The below scatter operation sets the log score
        # for the input word to 0.
        model_output = torch.scatter(
            model_output, -1, xt[..., None],
            torch.zeros_like(model_output[..., :1]))
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        """Computes the SEDD loss for the Absorbing State Diffusion.

        Args:
          log_x_theta: float torch.Tensor with shape (batch_size,
              context_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              context_length), input.
          x0: int torch.Tensor with shape (batch_size,
              context_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
              time derivative of signal level.
          low_var: bool, low variance loss during training.

        Returns:
          loss with shape (batch_size, context_length).
        """
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = self._sigma_from_alphat(alpha_t)
        dsigma = - dalpha_t / alpha_t

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices],
            -1,
            words_that_were_masked[..., None]).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(
                dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy


def stopgrad(x):
    """Stop gradient for x."""
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)  # (B,)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


def mse_loss(error):
    per_sample = (error ** 2).mean(dim=(1, 2))  # [B]
    return per_sample.mean()


class DUO_BASE(trainer_base.UniformState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        # Handle _orig_mod prefix from torch.compile and filter teacher keys
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            # Filter out teacher keys
            if k.startswith('teacher'):
                continue
            # Strip _orig_mod prefix from torch.compile
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v
        checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
          x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
          xt: The noisy latent (as indices) of shape (B, L).
          alpha_s: Noise level at s of shape (B, [L | 1], 1).
          alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
          Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(
            self.dtype).to(self.device)
        return (
            (alpha_t * self.vocab_size * x * xt_one_hot + (
                alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
                1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
                alpha_t * self.vocab_size * torch.gather(
                    x, -1, xt[..., None]) + (1 - alpha_t)))

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):  # Computes Eq 5.
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        if torch.is_tensor(dalpha_t) and dalpha_t.ndim == 1:
            dalpha_t = dalpha_t.unsqueeze(-1)
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()  # convert logits to probabilities
        x_bar_theta = self.vocab_size * alpha_t[
            :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(
            x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(
            x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt
                                   - 1 / xbar_theta_xt)  # Eq 5. term 1

        const = (1 - alpha_t) / (self.vocab_size * alpha_t
                                 + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                        - (1 / const) * x_neq_xt) * const.log()
        term2_theta = - term2_coefs * (
            x_bar_theta.log().sum(-1)
            - self.vocab_size * xbar_theta_xt.log())
        term2_theta = (
            term2_theta
            - self.vocab_size * alpha_t / (1 - alpha_t) * (
                xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _ancestral_update(self, x, t, dt, p_x0=None,
                          noise_removal_step=False, step_index=None):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)

        assert alpha_t.ndim == 2

        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(),
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t)
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(
                q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs, self.config.sampling.temperature)


class Integral(torch.autograd.Function):
    """
    torch module calculating UDLM's p_t 
    """

    @staticmethod
    def forward(ctx, gamma_t, data):
        gamma_max = data['gamma_max']
        gamma_min = data['gamma_min']
        if (gamma_t.max() > gamma_max) or (
                gamma_t.min() < gamma_min):
            # print('max:{} {}'.format(gamma_t.max(), gamma_max))
            # print('min:{} {}'.format(gamma_t.min(), gamma_min))
            gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
        indices = torch.round(
            (data['num_points'] - 1) * (gamma_t - gamma_min) / (
                gamma_max - gamma_min)).long()
        grad_pt = data['grad_pt']
        ctx.grad_pt = grad_pt[indices]

        pt = data['pt'][indices]
        assert pt.shape == gamma_t.shape
        return pt

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        with fsspec.open(self.config.algo.integral_cache_path,
                                         'rb') as f:
            self.integral_cache = pickle.load(f)
        self.integral_cache['pt'] = torch.from_numpy(
            self.integral_cache['pt'])
        self.integral_cache['grad_pt'] = torch.from_numpy(
            self.integral_cache['grad_pt'])
        self.gamma_min = self.config.algo.gamma_min
        self.gamma_max = self.config.algo.gamma_max
        self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
        self.curriculum_start = self.config.algo.curriculum_start
        self.curriculum_end = self.config.algo.curriculum_end
        self.loss_type = self.config.algo.loss_type
        self._validate_configuration()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.integral_cache['pt'] = self.integral_cache[
            'pt'].to(*args, **kwargs)
        self.integral_cache['grad_pt'] = self.integral_cache[
            'grad_pt'].to(*args, **kwargs)
        return self

    def _compute_gumbel_tau_inverse(self):
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = -10
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(name='gumbel_tau_log10',
                         value=1 / self._compute_gumbel_tau_inverse(),
                         on_step=True,
                         on_epoch=False,
                         sync_dist=True)
        return super().training_step(batch, batch_idx)

    def _gamma_to_alphat(self, gamma_t): # eq 10.
        integral = Integral.apply(gamma_t, self.integral_cache)
        return (self.vocab_size * integral - 1) / (
            self.vocab_size - 1)

    def _prior_loss(self):
        alpha_1 = self._gamma_to_alphat(
            torch.tensor(self.gamma_max))
        loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
            (self.vocab_size - 1) * alpha_1 + 1) + (
                1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
        return loss.item()

    def _q_xt_gaussian(self, x, gamma_t):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert x.ndim == 3
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()
        epsilon = torch.randn(x.shape, dtype=torch.float32,
                                                    device=self.device)
        return alpha_t * x + sigma_t * epsilon

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, **kwargs):
        # TODO: use xT
        use_true_nll = (self.global_step > self.curriculum_end
                                        or not train_mode)
        if use_true_nll:
            return super().nll(x0, output_tokens,
                                                 current_accumulation_step)
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)    
        gamma_t_prime = self.gamma_max - self.gamma_min
        alpha_t = self._gamma_to_alphat(gamma_t)
        T = 1000
        dalpha_t = gamma_t_prime * T * (
            self._gamma_to_alphat(gamma_t + 1 / T) - alpha_t)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        x0_one_hot = F.one_hot(x0, self.vocab_size)
        xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse()
        xt_usdm = xt.argmax(-1)
        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                                            xt=xt_usdm,
                                                            x0=x0,
                                                            alpha_t=alpha_t,
                                                            dalpha_t=dalpha_t,
                                                            low_var=False)




class Distillation(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.update_teacher_every = config.algo.update_teacher_every
        self.save_hyperparameters()
        self.teacher = None
        self.teacher_ema = config.algo.teacher_ema
        self.linear_growth_dt = config.algo.linear_growth_dt
        self.linear_growth_min = config.algo.linear_growth_min
        self.linear_growth_max = config.algo.linear_growth_max

    def _validate_configuration(self):
        assert os.path.exists(
            self.config.algo.integral_cache_path), (
            'The integral cache (Eq. 10 in the paper) for '
            f'the {self.config.data.tokenizer_name_or_path} '
            ' tokenizer doesnt exist at '
            f'{self.config.algo.integral_cache_path}. '
            'Please generate it by running the utils.py script, '
            'and ensure the correct path is specified using the '
            'algo.integral_cache_path flag.')
        assert self.loss_type in {
            'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

    def _maybe_update_teacher_weights(self):
        if self.global_step % self.update_teacher_every != 0:
            return
        if self.teacher_ema:
            self.ema.copy_to(self.teacher.parameters())
        else:
            for better_param, current_param in zip(
                    self.backbone.parameters(), self.teacher.parameters()):
                if current_param.requires_grad:
                    current_param.data.copy_(better_param.data)

    @torch.no_grad()
    def _teacher_logits(self, xt, sigma):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.backbone)
        self._maybe_update_teacher_weights()

        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.teacher(xt, sigma)
        logits = self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)
        return logits.detach()

    def _sample_trajectory(self, x0, gamma_t, gamma_s):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert gamma_s.ndim == 1
        assert x0.ndim == 2
        x0 = F.one_hot(x0, self.vocab_size).to(
            self.dtype).to(self.device)
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
        alpha_s = torch.sigmoid(-gamma_s).sqrt()
        sigma_s = torch.sigmoid(gamma_s).sqrt()

        epsilon = torch.randn(x0.shape, dtype=torch.float32,
                              device=self.device)
        xt = alpha_t * x0 + sigma_t * epsilon
        xs = alpha_s * x0 + sigma_s * epsilon
        return xt, xs

    def _compute_dt(self):
        if self.linear_growth_dt:
            scale = self.global_step / self.trainer.max_steps
            return self.linear_growth_min + scale * (
                self.linear_growth_max - self.linear_growth_min)
        n = self.global_step // self.update_teacher_every
        return 2 ** n / self.T

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=None, xT=None):
        # TODO: use xT
        del output_tokens, train_mode
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        gamma_t = self.gamma_min + t * (self.gamma_max
                                        - self.gamma_min)
        gamma_s = self.gamma_min + (
            t - dt) * (self.gamma_max - self.gamma_min)

        alpha_t = self._gamma_to_alphat(gamma_t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        usdm_alpha_s = self._gamma_to_alphat(gamma_s)
        usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
        assert usdm_alpha_s.ndim == 2

        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)
        log_x_theta_student = self.forward(
            xt_discrete, sigma=self._sigma_from_alphat(alpha_t))
        log_x_theta_teacher = self._teacher_logits(
            xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
        if self.config.training.loss_precision == 'float64':
            log_x_theta_student = log_x_theta_student.to(torch.float64)
            log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
        if self.loss_type == 'kl-fwd':
            return (log_x_theta_teacher.exp() * (
                log_x_theta_teacher - log_x_theta_student)).sum(-1)
        elif self.loss_type == 'kl-bwd':
            return (log_x_theta_student.exp() * (
                log_x_theta_student - log_x_theta_teacher)).sum(-1)

    def training_step(self, batch, batch_idx):
        self.log(name='dt',
                 value=self._compute_dt(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return super().training_step(batch, batch_idx)


class Rectification(DUO):  # Training as duo, without curriculum
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.save_hyperparameters()
        self.use_linear_schedule = config.algo.use_linear_schedule
        self.use_simple_loss = config.algo.use_simple_loss
        self.onestep_mode = config.algo.onestep_mode
        self.debug = getattr(config.algo, 'debug', False)

    def _compute_gumbel_tau_inverse(self):
        return 1e-10

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False, simple_loss=False):
        if simple_loss:
            loss = F.cross_entropy(
                log_x_theta.view(-1, self.vocab_size),
                x0.view(-1),
                reduction='none')
            loss = loss.view(xt.shape)
            return loss
        else:
            return super().nll_per_token(
                log_x_theta=log_x_theta,
                xt=xt,
                x0=x0,
                alpha_t=alpha_t,
                dalpha_t=dalpha_t,
                low_var=low_var
            )

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del output_tokens
        if given_t is not None:
            if not_sampling_t:
                assert torch.is_tensor(given_t)
                t = 1-given_t
            else:
                t = self._sample_t(
                    x0.shape[0], current_accumulation_step, given_t=1-given_t)
        else:
            t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            assert 0

        dalpha_t, alpha_t = self.noise(t)

        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        if given_t is not None and xT is not None:
            # x0 with alpha_t, xT with (1-alpha_t)
            random = torch.rand_like(x0, dtype=torch.float32)
            given_t = given_t.unsqueeze(1)
            random = given_t + random * (1 - given_t)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)
        elif xT is None or self.debug:
            if not self.debug:
                assert not self.training, 'xT should be provided during training'
            xT = self.prior_sample(x0.shape[0], x0.shape[1])
            random = torch.rand_like(x0, dtype=torch.float32)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)
        else:
            # x0 with alpha_t, xT with (1-alpha_t)
            random = torch.rand_like(x0, dtype=torch.float32)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)

        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                  xt=xt,
                                  x0=x0,
                                  alpha_t=alpha_t,
                                  dalpha_t=dalpha_t,
                                  low_var=train_mode and self.loss_type == 'low_var',
                                  simple_loss=self.use_simple_loss,
                                  )

class FLMBase(trainer_base.TrainerBase):
    """Base class for FLM/FMLM.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    def _validate_configuration(self):
        pass

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False):
        """Override to always dispatch to self.loss() for all FLM classes."""
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        loss = self.loss(input_tokens, output_tokens,
                         current_accumulation_step, train_mode,
                         xT=xT, given_t=given_t,
                         not_sampling_t=not_sampling_t)
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens
        return trainer_base.Loss(loss=token_nll,
                                 nlls=nlls,
                                 prior_loss=0.0,
                                 num_tokens=num_tokens)

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]
        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t = t[:batch_dim]
        return t

    def _tau_to_t(self, tau):
        """Convert t to reparameterized time tau."""
        return utils.alpha_to_gamma(tau, self.lut_a2g)

    def _t_to_tau(self, t):
        """Convert t to reparameterized time tau."""
        return utils.gamma_to_alpha(t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        """Corrupt data x0 at time t using linear interpolation with Gaussian noise."""
        t = t.unsqueeze(-1).unsqueeze(-1)
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        print("Resuming training from checkpoint...")
        self._is_resuming = True
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = self._filter_checkpoint_state_dict(
                checkpoint['state_dict'])
        if self.config.mode == 'sample_eval':
            if getattr(self.backbone, 'learnable_loss_weighting', None) is not None:
                if not any(k.startswith('backbone.learnable_loss_weighting')
                           for k in checkpoint['state_dict'].keys()):
                    print("Learnable_loss_weighting not found in checkpoint. "
                          "Initializing from scratch for eval mode.")
                    for name, param in self.backbone.learnable_loss_weighting.named_parameters():
                        param_key = f'backbone.learnable_loss_weighting.{name}'
                        checkpoint['state_dict'][param_key] = param.data.clone()
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def _filter_checkpoint_state_dict(self, state_dict):
        """Filter teacher keys and strip _orig_mod from checkpoint state_dict."""
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('teacher'):
                continue
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v
        return new_state_dict

    def forward_no_softmax(self, xt, tau, tau_prime=None, self_cond=None,
                           **kwargs):
        tau = self._process_sigma(tau)
        if tau_prime is not None:
            tau_prime = self._process_sigma(tau_prime)
        backbone_kwargs = dict(kwargs)
        if self_cond is not None:
            backbone_kwargs['self_cond'] = self_cond
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(
                xt, tau, tau_prime, **backbone_kwargs)
        return model_output

    def _extract_ema_state_dict(self, model, checkpoint):
        """Extract EMA parameters from checkpoint into a state_dict for model."""
        ema_state = checkpoint.get('ema', None)
        if not ema_state:
            print("Warning: No EMA found, using regular state_dict")
            return {k.replace('backbone.', '').replace('._orig_mod.', ''): v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith('backbone.')}

        new_sd = collections.OrderedDict()
        shadow_params = ema_state['shadow_params']
        param_names = [n for n, p in model.named_parameters()
                       if p.requires_grad]
        print(f"EMA shadow_params: {len(shadow_params)}, "
              f"Model param_names: {len(param_names)}")
        min_len = min(len(shadow_params), len(param_names))
        for name, val in zip(param_names[:min_len],
                             shadow_params[:min_len]):
            new_sd[name] = val
        for k, v in checkpoint['state_dict'].items():
            clean_k = k.replace('backbone.', '').replace('._orig_mod.', '')
            if (clean_k not in new_sd
                    and clean_k in [n for n, _ in model.named_parameters()]):
                new_sd[clean_k] = v
                print(f"Loaded missing param from state_dict: {clean_k}")
        if len(shadow_params) != len(param_names):
            print(f"Warning: EMA param count mismatch. "
                  f"Loaded {min_len}/{len(param_names)} from EMA, "
                  f"rest from state_dict")
        return new_sd

    def _load_teacher_model(self, path, use_plain_config=True):
        """Load a frozen teacher model from checkpoint.

        Args:
            path: Path to checkpoint file.
            use_plain_config: If True, temporarily disable double_temb and
                learnable_loss_weighting when building the teacher
                (to match EMA parameter shapes from a base model).
        """
        print(f"Loading teacher model from: {path}")
        if use_plain_config:
            saved = (self.config.algo.double_temb,
                     self.config.algo.learnable_loss_weighting)
            self.config.algo.double_temb = False
            self.config.algo.learnable_loss_weighting = False

        assert self.config.algo.backbone == 'dit', \
            "Only DIT backbone supported for teacher model"
        model = models.dit.DIT(self.config, vocab_size=self.vocab_size)

        if use_plain_config:
            (self.config.algo.double_temb,
             self.config.algo.learnable_loss_weighting) = saved

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _copy_teacher_weights_to_student(self, teacher_dict):
        """Copy teacher weights to student backbone and zero-init sigma_map_prime."""
        with torch.no_grad():
            student_dict = self.backbone.state_dict()
            for name, param in teacher_dict.items():
                print(f"Copying parameter: {name}")
                if name in student_dict:
                    student_dict[name].copy_(param)
            if (hasattr(self.backbone, 'sigma_map_prime')
                    and self.backbone.sigma_map_prime is not None):
                for name, param in self.backbone.sigma_map_prime.named_parameters():
                    if 'mlp.2' in name:
                        param.zero_()
                        print(f"Zero initialized student sigma_map_prime: {name}")

    @staticmethod
    def _zero_init_module(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def _random_init_module(module, std=0.02):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std=std)
                if m.bias is not None:
                    m.bias.data.zero_()


class FLM(FLMBase):
    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens
        B = x0.shape[0]
        tau_t = self._sample_t_interval(B, current_accumulation_step,
                                    t_min=self.t_min, t_max=self.t_max)
        t = self._tau_to_t(tau_t)
        x_t, target_data = self.corrupt_continuous(x0, t)
        f = self.forward(x_t, tau_t) #condition on tau_t
        loss = -(target_data * f).sum(dim=-1)
        self.log('loss', loss.mean(), prog_bar=True)
        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(tau_t)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            self.log('loss_weighted', loss.mean(), prog_bar=True)
        # dt_dtau = utils.d_alpha_to_gamma(tau_t, self.lut_a2g)
        # loss_weight = (2 * t * dt_dtau) / (1 - t)**3
        # return loss * loss_weight[:, None]
        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples using Euler ODE solver."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)

        for i in range(num_steps):
            tau_t_curr = tau_vals[i]
            tau_t_next = tau_vals[i + 1]
            tau_t_in = tau_t_curr.expand(B)
            t_in = self._tau_to_t(tau_t_in)
            dt = self._tau_to_t(tau_t_next.expand(B)) - t_in
            x_1_pred = self.forward(z, tau_t_in)
            x_1_pred_probs = x_1_pred.exp()

            if i == num_steps - 1:
                z = x_1_pred_probs
                break

            v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
            z = z + dt.view(-1, 1, 1) * v

        return z.argmax(dim=-1)
    
class FLMBlock(FLM):
    """Blockwise FLM with duplicated clean/noisy streams.

    The model input has length 2L. The first half is clean block context, the
    second half is the noisy target stream. Losses and samples are taken only
    from the noisy half.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.block_size = int(config.algo.block_size)
        self._validate_block_configuration()
        self._block_attention_mask_cache = {}
        self._block_position_ids_cache = {}

    def _validate_block_configuration(self):
        assert self.block_size > 0, "block_size must be positive"
        assert self.num_tokens % self.block_size == 0, \
            "model.length must be divisible by block_size"
        assert not self.config.algo.causal_attention, \
            "FLMBlock requires non-causal DIT attention with a block mask"

    def _num_blocks(self, block_size=None):
        if block_size is None:
            block_size = self.block_size
        return self.num_tokens // block_size

    def _block_attention_mask(self, block_size=None, device=None):
        if block_size is None:
            block_size = self.block_size
        if device is None:
            device = self.device
        key = (block_size, device)
        cached = self._block_attention_mask_cache.get(key)
        if cached is not None:
            return cached

        L = self.num_tokens
        S = 2 * L
        token_ids = torch.arange(S, device=device)
        query_is_noisy = token_ids >= L
        key_is_noisy = query_is_noisy
        query_block = (token_ids % L) // block_size
        key_block = query_block

        q_noisy = query_is_noisy[:, None]
        k_noisy = key_is_noisy[None, :]
        q_block = query_block[:, None]
        k_block = key_block[None, :]

        clean_query = (~q_noisy) & (~k_noisy) & (k_block <= q_block)
        noisy_query_clean_context = q_noisy & (~k_noisy) & (k_block < q_block)
        noisy_query_same_block = q_noisy & k_noisy & (k_block == q_block)
        mask = clean_query | noisy_query_clean_context | noisy_query_same_block
        self._block_attention_mask_cache[key] = mask
        return mask

    def _block_position_ids(self, device=None):
        if device is None:
            device = self.device
        cached = self._block_position_ids_cache.get(device)
        if cached is not None:
            return cached
        pos = torch.arange(self.num_tokens, device=device)
        pos = torch.cat([pos, pos], dim=0)
        self._block_position_ids_cache[device] = pos
        return pos

    def _sample_block_tau(self, n, accum_step, block_size=None,
                          t_min=None, t_max=None):
        if block_size is None:
            block_size = self.block_size
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        num_blocks = self._num_blocks(block_size)
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        total = n * num_blocks
        eps_t = torch.rand(total, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(total, device=self.device) / total
            eps_t = (eps_t / total + offset) % 1
            perm = torch.randperm(total, device=self.device)
            eps_t = eps_t[perm]
        tau = ((t_max - t_min) * eps_t + t_min).view(n, num_blocks)
        if accum_step is not None:
            tau = tau.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            tau = tau.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            tau = tau.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            tau = tau[:batch_dim]
        return tau

    def _sample_two_block_tau(self, n, accum_step, block_size=None):
        tau_a = self._sample_block_tau(n, accum_step, block_size)
        tau_b = self._sample_block_tau(n, accum_step, block_size)
        tau_s = torch.minimum(tau_a, tau_b)
        tau_t = torch.maximum(tau_a, tau_b)
        return tau_s, tau_t

    def _tau_blocks_to_tokens(self, tau, block_size=None):
        if block_size is None:
            block_size = self.block_size
        if tau.ndim == 1:
            tau = tau[:, None].expand(-1, self._num_blocks(block_size))
        return tau.repeat_interleave(block_size, dim=1)

    def _block_t_to_tokens(self, t, block_size=None):
        if block_size is None:
            block_size = self.block_size
        if t.ndim == 1:
            t = t[:, None].expand(-1, self._num_blocks(block_size))
        return t.repeat_interleave(block_size, dim=1)

    def corrupt_continuous_blocks(self, x0, tau, block_size=None):
        if block_size is None:
            block_size = self.block_size
        t = self._tau_to_t(tau.reshape(-1)).view_as(tau)
        t_tokens = self._block_t_to_tokens(t, block_size)
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = ((1 - t_tokens[:, :, None]) * noise
               + t_tokens[:, :, None] * target_data)
        return x_t, target_data

    def _block_forward_model(self, model, noisy_tokens, clean_tokens,
                             tau, tau_prime=None, use_jvp_attn=False,
                             block_size=None):
        if block_size is None:
            block_size = self.block_size
        clean = F.one_hot(clean_tokens, self.vocab_size).to(noisy_tokens.dtype)
        x_full = torch.cat([clean, noisy_tokens], dim=1)
        tau_tokens = self._tau_blocks_to_tokens(tau, block_size)
        tau_full = torch.cat([torch.zeros_like(tau_tokens), tau_tokens], dim=1)
        tau_prime_full = None
        if tau_prime is not None:
            tau_prime_tokens = self._tau_blocks_to_tokens(tau_prime, block_size)
            tau_prime_full = torch.cat(
                [torch.zeros_like(tau_prime_tokens), tau_prime_tokens], dim=1)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = model(
                x_full,
                tau_full,
                tau_prime_full,
                use_jvp_attn=use_jvp_attn,
                attention_mask=self._block_attention_mask(block_size, x_full.device),
                position_ids=self._block_position_ids(x_full.device))
        model_output = self._process_model_output(
            model_output=model_output, xt=x_full, sigma=tau_full)
        return model_output[:, self.num_tokens:]

    def _block_forward(self, noisy_tokens, clean_tokens, tau, tau_prime=None,
                       use_jvp_attn=False, block_size=None):
        return self._block_forward_model(
            self.backbone, noisy_tokens, clean_tokens, tau, tau_prime,
            use_jvp_attn=use_jvp_attn, block_size=block_size)

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens, train_mode, xT
        B = x0.shape[0]
        block_size = self.block_size
        tau_t = self._sample_block_tau(B, current_accumulation_step, block_size)
        x_t, target_data = self.corrupt_continuous_blocks(x0, tau_t, block_size)
        f = self._block_forward(x_t, x0, tau_t, block_size=block_size)
        loss = -(target_data * f).sum(dim=-1)
        self.log('loss', loss.mean(), prog_bar=True)
        if self.config.algo.learnable_loss_weighting is True:
            tau_for_weight = tau_t.mean(dim=1)
            loss_weight = self.backbone.learnable_loss_weighting(tau_for_weight)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            self.log('loss_weighted', loss.mean(), prog_bar=True)
        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples block-by-block using the FLM Euler solver."""
        del eps
        if num_steps is None:
            num_steps = self.config.sampling.steps
        B = num_samples
        L = self.num_tokens
        V = self.vocab_size
        block_size = self.block_size
        num_blocks = self._num_blocks(block_size)
        device = self.device
        tokens = torch.zeros((B, L), dtype=torch.long, device=device)
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size
            z_block = torch.randn((B, block_size, V), device=device,
                                  dtype=self.dtype)
            for i in range(num_steps):
                tau_curr = tau_vals[i]
                tau_next = tau_vals[i + 1]
                tau_blocks = torch.zeros((B, num_blocks), device=device)
                tau_blocks[:, block_idx] = tau_curr
                tau_next_blocks = torch.zeros((B, num_blocks), device=device)
                tau_next_blocks[:, block_idx] = tau_next
                t_in = self._tau_to_t(tau_curr.expand(B))
                dt = self._tau_to_t(tau_next.expand(B)) - t_in

                noisy = torch.zeros((B, L, V), device=device, dtype=self.dtype)
                noisy[:, start:end] = z_block
                log_pred = self._block_forward(
                    noisy, tokens, tau_blocks, block_size=block_size)
                pred = log_pred[:, start:end].exp()
                if i == num_steps - 1:
                    z_block = pred
                    break
                v = (pred - z_block) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
                z_block = z_block + dt.view(-1, 1, 1) * v

            tokens[:, start:end] = z_block.argmax(dim=-1)
        return tokens

class FMLM(FLMBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.teacher_model = None

    def setup(self, stage: str):
        if self.teacher_model is None:
            self._initialize_teacher()
        if stage == 'fit' and not self._is_resuming:
            print(">>> Initializing student from teacher...")
            self._initialize_student_from_teacher()
        elif self._is_resuming:
            print(">>> Skipping student initialization (resuming from checkpoint).")

    def _initialize_teacher(self):
        path = self.config.algo.teacher_path
        if path is None or path == "":
            print("No teacher model specified, skipping teacher initialization")
            return
        self.teacher_model = self._load_teacher_model(path, use_plain_config=True)

    def _initialize_student_from_teacher(self):
        if self.teacher_model is None or not self.config.algo.initialize_student_from_teacher:
            return
        self._copy_teacher_weights_to_student(self.teacher_model.state_dict())

    def _validate_configuration(self):
        assert self.config.algo.double_temb == True, \
            "FMLM denoiser requires double time-emb conditioning to be True"

        assert type(self.config.algo.diagonal_fraction) == float, \
            "diagonal_fraction must be a float"
        assert 0 <= self.config.algo.diagonal_fraction <= 1, \
            "diagonal_fraction must be between 0 and 1"
        assert self.config.algo.distillation_method in ["PSD", "LSD", "ESD"], \
            "FMLM denoiser must distill using PSD, LSD, or ESD"
        if self.config.algo.distillation_method == "LSD":
            assert 2 >= self.config.algo.entmax_temp_lsd >= 1, \
                "entmax_temp_lsd must be in [1,2] for LSD distillation"
            assert not self.config.algo.backprop_entmax_temp_lsd, \
                "backprop_entmax_temp_lsd is not coded for LSD distillation"

    def forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False, **kwargs):
        return super().forward(xt, sigma, sigma_prime, use_jvp_attn=use_jvp_attn, **kwargs)

    def forward_with_ema(self, *args, **kwargs):
        ema_to_use = self.ema
        assert ema_to_use is not None, "EMA must be available"
        ema_to_use.store(self._get_parameters())
        ema_to_use.copy_to(self._get_parameters())
        try:
            with torch.no_grad():
                self.backbone.eval()
                out = self.forward(*args, **kwargs)
            return out
        finally:
            ema_to_use.restore(self._get_parameters())
            self.backbone.train()

    def teacher_forward(self, xt, tau=None, d=None, use_jvp_attn=False):
        del d, use_jvp_attn
        sigma = tau.unsqueeze(-1) if tau.ndim == 1 else tau
        sigma = self._process_sigma(sigma)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model(xt, sigma)
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def _d_tau_by_d_t(self, t):
        return utils.d_alpha_by_d_gamma(t, self.lut_g2a)

    def _sample_multi_t_interval(self, n, accum_step, num_per_sample, t_min=None, t_max=None):
        """Sample num_per_sample times from [t_min, t_max], returned sorted."""
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max

        if accum_step is not None:
            batch_dim = n
            total_n = self.config.loader.global_batch_size * num_per_sample
        else:
            batch_dim = n
            total_n = n * num_per_sample

        _eps_t = torch.rand(total_n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(total_n, device=self.device) / total_n
            _eps_t = (_eps_t / total_n + offset) % 1
            perm = torch.randperm(total_n, device=self.device)
            _eps_t = _eps_t[perm]

        t_all = (t_max - t_min) * _eps_t + t_min

        if accum_step is not None:
            t_all = t_all.view(-1, num_per_sample)
            t_all = t_all.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t_all = t_all.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t_all = t_all.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t_all = t_all[:batch_dim]
        else:
            t_all = t_all.view(batch_dim, num_per_sample)

        t_sorted, _ = torch.sort(t_all, dim=-1)
        return [t_sorted[:, i] for i in range(num_per_sample)]

    def _get_split_indices(self, n, accum_step, ratios):
        """Split batch indices into category sets"""
        assert sum(ratios) < 0.999, "Sum of ratios must be less than 1.0, leave one ratio out"

        batch_dim = n
        if accum_step is not None:
            n_total = self.config.loader.global_batch_size
            num_nodes = self.trainer.num_nodes
            num_devices = self.trainer.num_devices
            accum_batches = self.trainer.accumulate_grad_batches
            node_rank = self.trainer.node_rank
            local_rank = self.trainer.local_rank
            current_accum = accum_step
        else:
            n_total = n
            num_nodes = num_devices = accum_batches = 1
            node_rank = local_rank = current_accum = 0

        total_chunks = num_nodes * num_devices * accum_batches
        chunk_idx = (node_rank * num_devices * accum_batches
                     + local_rank * accum_batches
                     + current_accum)

        num_categories = len(ratios) + 1
        global_counts = [0] * num_categories
        net_ratio, prev_num = 0, 0
        for i, ratio in enumerate(ratios):
            net_ratio += ratio
            num_a = int(n_total * net_ratio)
            global_counts[i] = num_a - prev_num
            prev_num = num_a
        global_counts[-1] = n_total - prev_num

        local_counts = []
        for cnt in global_counts:
            base, rem = divmod(cnt, total_chunks)
            local_counts.append(base + (1 if chunk_idx < rem else 0))

        local_size = sum(local_counts)
        local_assignments = torch.empty(local_size, device=self.device, dtype=torch.long)
        offset = 0
        for i, cnt in enumerate(local_counts):
            local_assignments[offset:offset + cnt] = i
            offset += cnt

        seed = self.global_step * total_chunks + chunk_idx
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        perm = torch.randperm(local_size, device=self.device, generator=g)
        local_assignments = local_assignments[perm][:batch_dim]

        return [(local_assignments == i).nonzero(as_tuple=True)[0]
                for i in range(num_categories)]

    def loss(self, x1, output_tokens,
             current_accumulation_step=None, train_mode=False, xT=None,
             given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens, train_mode, xT
        B, L = x1.shape[0], x1.shape[1]

        tau_diag = self._sample_t_interval(B, current_accumulation_step,
                                           t_min=self.t_min, t_max=self.t_max)
        set_midpoint = getattr(self.config.algo, 'set_midpoint', 'midpoint')
        if self.config.algo.offdiagonal_sampling == "uniform_st":
            tau_s_offdiag, tau_t_offdiag = self._sample_multi_t_interval(
                B, current_accumulation_step, 2,
                t_min=self.t_min, t_max=self.t_max)
        else:  # uniform_diff
            tau_d_offdiag = self._sample_t_interval(
                B, current_accumulation_step, t_min=self.t_min, t_max=self.t_max)
            tau_s_offdiag = self._sample_t_interval(
                B, current_accumulation_step, t_min=self.t_min, t_max=self.t_max)
            tau_s_offdiag = tau_s_offdiag * (1 - tau_d_offdiag)
            tau_t_offdiag = tau_s_offdiag + tau_d_offdiag

        idx_diag, idx_offdiag_bndry, idx_offdiag = self._get_split_indices(
            B, current_accumulation_step,
            ratios=(self.config.algo.diagonal_fraction,
                    (1.0 - self.config.algo.diagonal_fraction)
                    * (1.0 / self.config.algo.boundary_prob)))
        tau_s = torch.zeros(B, device=self.device)
        tau_t = torch.zeros(B, device=self.device)
        tau_s[idx_diag] = tau_diag[idx_diag]
        tau_s[idx_offdiag_bndry] = 0.0
        tau_s[idx_offdiag] = tau_s_offdiag[idx_offdiag]
        tau_t[idx_diag] = tau_diag[idx_diag]
        tau_t[idx_offdiag_bndry] = 1.0
        tau_t[idx_offdiag] = tau_t_offdiag[idx_offdiag]

        tau_s = torch.clamp(tau_s, 0.0, 1.0)
        tau_t = torch.clamp(tau_t, 0.0, 1.0)

        idx_offdiag = torch.cat([idx_offdiag, idx_offdiag_bndry])
        has_diag = idx_diag.numel() > 0
        has_offdiag = idx_offdiag.numel() > 0

        if set_midpoint == 'midpoint':
            tau_u = 0.5 * (tau_s + tau_t)
        else:  # random
            tau_u = tau_s + torch.rand_like(tau_s) * (tau_t - tau_s)
        s = self._tau_to_t(tau_s)
        u = self._tau_to_t(tau_u)
        t = self._tau_to_t(tau_t)

        x_s, target_data = self.corrupt_continuous(x1, s)

        if self.teacher_model is None or not has_diag:
            on_diagonal_target = target_data[idx_diag]
        else:
            on_diagonal_target = self.teacher_forward(x_s[idx_diag], tau_s[idx_diag]).exp()
        on_diagonal_target = stopgrad(on_diagonal_target)

        if self.config.algo.distillation_method == "PSD": # Progressive Distillation. (semigroup)
            log_D_st = self.forward(x_s, tau_s, tau_t)
            _fwd = (self.forward_with_ema
                    if getattr(self.config.algo, 'use_ema_for_psd_target', False)
                    else self.forward)
            if has_offdiag:
                with torch.no_grad():
                    x_s_od = x_s[idx_offdiag]
                    s_od = s[idx_offdiag].view(-1, 1, 1)
                    u_od = u[idx_offdiag].view(-1, 1, 1)
                    t_od = t[idx_offdiag].view(-1, 1, 1)
                    tau_s_od = tau_s[idx_offdiag]
                    tau_u_od = tau_u[idx_offdiag]
                    tau_t_od = tau_t[idx_offdiag]
                    D_su_offdiag = _fwd(x_s_od, tau_s_od, tau_u_od).exp()
                    X_su = ((1 - u_od) / (1 - s_od + 1e-8)) * x_s_od \
                           + ((u_od - s_od) / (1 - s_od + 1e-8)) * D_su_offdiag
                    D_ut_offdiag = _fwd(X_su, tau_u_od, tau_t_od).exp()
                    lambda_sut = ((1 - t_od) * (u_od - s_od)
                                  / ((1 - u_od) * (t_od - s_od) + 1e-8))
                    offdiag_target = stopgrad(
                        lambda_sut * D_su_offdiag + (1 - lambda_sut) * D_ut_offdiag)

                if not self.config.algo.use_mse_loss_psd:
                    offdiag_loss = -(offdiag_target * log_D_st[idx_offdiag]).sum(dim=-1)
                else:
                    offdiag_loss = F.mse_loss(
                        log_D_st[idx_offdiag].exp(), offdiag_target,
                        reduction='none').sum(dim=-1)
            else:
                offdiag_loss = x_s.new_empty((0, L))

            if has_diag:
                if not self.config.algo.use_mse_loss_psd:
                    diag_loss = -(on_diagonal_target * log_D_st[idx_diag]).sum(dim=-1)
                else:
                    diag_loss = F.mse_loss(
                        log_D_st[idx_diag].exp(), on_diagonal_target,
                        reduction='none').sum(dim=-1)
            else:
                diag_loss = x_s.new_empty((0, L))

            if self.config.algo.rescale_offdiag_loss_psd is True and has_offdiag:
                offdiag_loss = offdiag_loss * (
                    (t_od - s_od) / (1 - s_od + 1e-8)).view(-1, 1).pow(2)

        elif self.config.algo.distillation_method == "ESD": # Eulerian Distillation
            use_jvp_attn = True
            
            if has_diag:
                log_D_st_diag = self.forward(
                    x_s[idx_diag], s[idx_diag], t[idx_diag], use_jvp_attn=False
                )
                diag_loss = -(on_diagonal_target * log_D_st_diag).sum(dim=-1)
            else:
                log_D_st_diag = x_s.new_empty((0, L, self.vocab_size))
                diag_loss = x_s.new_empty((0, L))
            
            if has_offdiag:
                x_s_od = x_s[idx_offdiag]
                s_od = s[idx_offdiag]
                t_od = t[idx_offdiag]
                tau_s_od = tau_s[idx_offdiag]
                tau_t_od = tau_t[idx_offdiag]
                
                with torch.no_grad():
                    use_teacher = (
                        self.teacher_model is not None
                        and getattr(self.config.algo, 'use_teacher_for_D_s_esd', True)
                    )
                    if use_teacher:
                        D_s = self.teacher_forward(x_s_od, tau_s_od).exp()
                    else:
                        D_s = self.forward(
                            x_s_od, tau_s_od, tau_t_od, use_jvp_attn=False
                        ).exp()
                
                d_tau_s_by_d_s = self._d_tau_by_d_t(
                    s_od.view(-1, 1, 1)
                ).squeeze()
                
                with torch.enable_grad():
                    def forward_s_x(tau_s_val, x_s_val):
                        return self.forward(
                            x_s_val, tau_s_val, tau_t_od,
                            use_jvp_attn=use_jvp_attn
                        )
                    
                    tangent_tau_s = d_tau_s_by_d_s * torch.ones_like(tau_s_od)
                    tangent_x_s = (D_s - x_s_od) / (1 - s_od.view(-1, 1, 1) + 1e-8)
                    
                    log_D_st_offdiag, d_ds_log_D_st = torch.func.jvp(
                        forward_s_x,
                        (tau_s_od, x_s_od),
                        (tangent_tau_s, tangent_x_s),
                    )
                    d_ds_log_D_st = stopgrad(d_ds_log_D_st)
                
                with torch.no_grad():
                    s_g = s_od.view(-1, 1, 1)
                    t_g = t_od.view(-1, 1, 1)
                    
                    D_st = log_D_st_offdiag.exp()
                    d_ds_D_st = D_st * d_ds_log_D_st
                    
                    coeff = (1 - s_g) * (t_g - s_g) / (1 - t_g + 1e-8)
                    offdiag_target = stopgrad(D_s + coeff * d_ds_D_st)
                
                offdiag_loss = F.mse_loss(log_D_st_offdiag.exp(), offdiag_target, reduction='none').sum(dim=-1)
            else:
                log_D_st_offdiag = x_s.new_empty((0, L, self.vocab_size))
                offdiag_loss = x_s.new_empty((0, L))
            
            log_D_st = torch.zeros(B, L, self.vocab_size, device=self.device)
            log_D_st[idx_offdiag] = log_D_st_offdiag
            log_D_st[idx_diag] = log_D_st_diag
            
        else: # Langrangian distillation
            use_jvp_attn = True

            if has_diag:
                log_D_st_diag = self.forward(
                    x_s[idx_diag], tau_s[idx_diag], tau_t[idx_diag], use_jvp_attn=False)
                diag_loss = -(on_diagonal_target * log_D_st_diag).sum(dim=-1)
            else:
                log_D_st_diag = x_s.new_empty((0, L, self.vocab_size))
                diag_loss = x_s.new_empty((0, L))

            if has_offdiag:
                with torch.no_grad():
                    x_s_od = x_s[idx_offdiag]
                    s_od = s[idx_offdiag].view(-1, 1, 1)
                    t_od = t[idx_offdiag].view(-1, 1, 1)
                    tau_s_od = tau_s[idx_offdiag]
                    tau_t_od = tau_t[idx_offdiag]

                with torch.enable_grad():
                    def forward_t(tau_t_val):
                        return self.forward(
                            x_s_od, tau_s_od, tau_t_val,
                            use_jvp_attn=use_jvp_attn)
                    tangent_t = torch.ones_like(tau_t_od)
                    log_D_st_offdiag, grad_tau_t_log_D_st_offdiag = torch.func.jvp(
                        forward_t, (tau_t_od,), (tangent_t,))
                    grad_tau_t_log_D_st_offdiag = stopgrad(grad_tau_t_log_D_st_offdiag)

                d_tau_by_d_t = self._d_tau_by_d_t(t_od)
                grad_t_log_D_st_offdiag = grad_tau_t_log_D_st_offdiag * d_tau_by_d_t

                with torch.no_grad():
                    D_st = log_D_st_offdiag.exp()
                    partial_t_D_st = D_st * grad_t_log_D_st_offdiag
                    X_st = ((1 - t_od) / (1 - s_od + 1e-8)) * x_s_od \
                           + ((t_od - s_od) / (1 - s_od + 1e-8)) * D_st
                    use_teacher_for_D_t = (
                        self.teacher_model is not None
                        and self.config.algo.use_teacher_for_D_t_lsd)
                    if use_teacher_for_D_t:
                        D_t__X_st = self.teacher_forward(X_st, tau_t_od).exp()
                    else:
                        D_t__X_st = self.forward(
                            X_st, tau_t_od, tau_t_od,
                            use_jvp_attn=False).exp()
                    offdiag_target = stopgrad(
                        D_t__X_st
                        - (t_od - s_od) * ((1 - t_od) / (1 - s_od + 1e-8))
                        * partial_t_D_st)
                    offdiag_target = entmax_bisect(
                        offdiag_target,
                        torch.tensor(
                            self.config.algo.entmax_temp_lsd,
                            dtype=torch.float32,
                            requires_grad=self.config.algo.backprop_entmax_temp_lsd,
                        ).to(self.device),
                        dim=-1,
                    )

                offdiag_loss = -(offdiag_target * log_D_st_offdiag).sum(dim=-1)
            else:
                log_D_st_offdiag = x_s.new_empty((0, L, self.vocab_size))
                offdiag_loss = x_s.new_empty((0, L))

            log_D_st = torch.zeros(B, L, self.vocab_size, device=self.device)
            log_D_st[idx_offdiag] = log_D_st_offdiag
            log_D_st[idx_diag] = log_D_st_diag

        loss = torch.zeros(B, L, device=self.device)
        
        if has_diag:
            loss[idx_diag] = diag_loss
            diag_loss_to_log = diag_loss.mean()
        else:
            diag_loss_to_log = loss.new_tensor(0.0)
        self.log('diag_loss', diag_loss_to_log, prog_bar=True, sync_dist=True)
        
        if has_offdiag:
            loss[idx_offdiag] = offdiag_loss
            offdiag_loss_to_log = offdiag_loss.mean()
        else:
            offdiag_loss_to_log = loss.new_tensor(0.0)
        self.log('offdiag_loss', offdiag_loss_to_log, prog_bar=True, sync_dist=True)
        self.log('loss', loss.mean(), prog_bar=True, sync_dist=True)

        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(tau_s, tau_t)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            self.log('loss_weighted', loss.mean(), prog_bar=True, sync_dist=True)

        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        gamma = getattr(self.config.sampling, 'gamma', 0.0)
        print(f"Sampling with {num_steps} steps")
        
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)


        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)

        for i in range(num_steps):
            tau_curr = tau_vals[i]
            tau_next = tau_vals[i + 1]

            t_curr = self._tau_to_t(tau_curr.expand(B))

            t_next = self._tau_to_t(tau_next.expand(B))
            sigma_target = 1.0 - t_next

            sigma_tilde = sigma_target * torch.sqrt(torch.tensor(1.0 - gamma**2))
            t_tilde = 1.0 - sigma_tilde
            tau_tilde = self._t_to_tau(t_tilde)

            log_D_st_pred = self.forward(z, tau_curr.expand(B), tau_tilde)
            D_st_pred = log_D_st_pred.exp()

            if i == num_steps - 1:
                z = D_st_pred
                break

            weight_z = (1.0 - t_tilde.view(-1, 1, 1)) / (1.0 - t_curr.view(-1, 1, 1))
            weight_D = ((t_tilde.view(-1, 1, 1) - t_curr.view(-1, 1, 1))
                        / (1.0 - t_curr.view(-1, 1, 1)))
            z_tilde = weight_z * z + weight_D * D_st_pred

            if gamma > 0:
                noise_std = gamma * sigma_target.view(-1, 1, 1)
                mean_adjustment = sigma_tilde.view(-1, 1, 1) - sigma_target.view(-1, 1, 1)
                z = z_tilde + mean_adjustment * D_st_pred + noise_std * torch.randn_like(z)
            else:
                z = z_tilde
                
        return z.argmax(dim=-1)


class FMLMBlock(FMLM, FLMBlock):
    """Blockwise FMLM with PSD flow-map distillation."""

    def __init__(self, config, tokenizer):
        FMLM.__init__(self, config, tokenizer)
        self.block_size = int(config.algo.block_size)
        self._validate_block_configuration()
        self._block_attention_mask_cache = {}
        self._block_position_ids_cache = {}

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.algo.distillation_method == "PSD", \
            "FMLMBlock currently supports PSD distillation"

    def teacher_forward(self, xt, tau=None, d=None, use_jvp_attn=False,
                        clean_tokens=None, block_size=None):
        del d
        if clean_tokens is None:
            return super().teacher_forward(
                xt, tau=tau, use_jvp_attn=use_jvp_attn)
        with torch.no_grad():
            return self._block_forward_model(
                self.teacher_model, xt, clean_tokens, tau,
                use_jvp_attn=use_jvp_attn, block_size=block_size)

    def _block_forward_with_ema(self, noisy_tokens, clean_tokens,
                                tau, tau_prime=None, block_size=None):
        ema_to_use = self.ema
        assert ema_to_use is not None, "EMA must be available"
        ema_to_use.store(self._get_parameters())
        ema_to_use.copy_to(self._get_parameters())
        try:
            with torch.no_grad():
                self.backbone.eval()
                return self._block_forward(
                    noisy_tokens, clean_tokens, tau, tau_prime,
                    block_size=block_size)
        finally:
            ema_to_use.restore(self._get_parameters())
            self.backbone.train()

    def _sample_fmlm_block_times(self, n, accum_step, block_size):
        tau_diag = self._sample_block_tau(
            n, accum_step, block_size, t_min=self.t_min, t_max=self.t_max)
        set_midpoint = getattr(self.config.algo, 'set_midpoint', 'midpoint')
        if self.config.algo.offdiagonal_sampling == "uniform_st":
            tau_s_offdiag, tau_t_offdiag = self._sample_two_block_tau(
                n, accum_step, block_size)
        else:
            tau_d_offdiag = self._sample_block_tau(
                n, accum_step, block_size, t_min=self.t_min, t_max=self.t_max)
            tau_s_offdiag = self._sample_block_tau(
                n, accum_step, block_size, t_min=self.t_min, t_max=self.t_max)
            tau_s_offdiag = tau_s_offdiag * (1 - tau_d_offdiag)
            tau_t_offdiag = tau_s_offdiag + tau_d_offdiag

        idx_diag, idx_offdiag_bndry, idx_offdiag = self._get_split_indices(
            n, accum_step,
            ratios=(self.config.algo.diagonal_fraction,
                    (1.0 - self.config.algo.diagonal_fraction)
                    * (1.0 / self.config.algo.boundary_prob)))

        num_blocks = self._num_blocks(block_size)
        tau_s = torch.zeros((n, num_blocks), device=self.device)
        tau_t = torch.zeros((n, num_blocks), device=self.device)
        tau_s[idx_diag] = tau_diag[idx_diag]
        tau_s[idx_offdiag_bndry] = 0.0
        tau_s[idx_offdiag] = tau_s_offdiag[idx_offdiag]
        tau_t[idx_diag] = tau_diag[idx_diag]
        tau_t[idx_offdiag_bndry] = 1.0
        tau_t[idx_offdiag] = tau_t_offdiag[idx_offdiag]
        tau_s = torch.clamp(tau_s, 0.0, 1.0)
        tau_t = torch.clamp(tau_t, 0.0, 1.0)

        if set_midpoint == 'midpoint':
            tau_u = 0.5 * (tau_s + tau_t)
        else:
            tau_u = tau_s + torch.rand_like(tau_s) * (tau_t - tau_s)

        idx_offdiag = torch.cat([idx_offdiag, idx_offdiag_bndry])
        return tau_s, tau_u, tau_t, idx_diag, idx_offdiag

    def loss(self, x1, output_tokens,
             current_accumulation_step=None, train_mode=False, xT=None,
             given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens, train_mode, xT
        B, L = x1.shape[0], x1.shape[1]
        block_size = self.block_size
        tau_s, tau_u, tau_t, idx_diag, idx_offdiag = (
            self._sample_fmlm_block_times(
                B, current_accumulation_step, block_size))

        has_diag = idx_diag.numel() > 0
        has_offdiag = idx_offdiag.numel() > 0

        s = self._tau_to_t(tau_s.reshape(-1)).view_as(tau_s)
        u = self._tau_to_t(tau_u.reshape(-1)).view_as(tau_u)
        t = self._tau_to_t(tau_t.reshape(-1)).view_as(tau_t)
        x_s, target_data = self.corrupt_continuous_blocks(
            x1, tau_s, block_size)

        if self.teacher_model is None or not has_diag:
            on_diagonal_target = target_data[idx_diag]
        else:
            on_diagonal_target = self.teacher_forward(
                x_s[idx_diag], tau=tau_s[idx_diag],
                clean_tokens=x1[idx_diag], block_size=block_size).exp()
        on_diagonal_target = stopgrad(on_diagonal_target)

        log_D_st = self._block_forward(
            x_s, x1, tau_s, tau_t, block_size=block_size)
        _fwd = (self._block_forward_with_ema
                if getattr(self.config.algo, 'use_ema_for_psd_target', False)
                else self._block_forward)

        if has_offdiag:
            with torch.no_grad():
                x_s_od = x_s[idx_offdiag]
                x1_od = x1[idx_offdiag]
                s_od = self._block_t_to_tokens(
                    s[idx_offdiag], block_size)[:, :, None]
                u_od = self._block_t_to_tokens(
                    u[idx_offdiag], block_size)[:, :, None]
                t_od = self._block_t_to_tokens(
                    t[idx_offdiag], block_size)[:, :, None]
                tau_s_od = tau_s[idx_offdiag]
                tau_u_od = tau_u[idx_offdiag]
                tau_t_od = tau_t[idx_offdiag]

                D_su_offdiag = _fwd(
                    x_s_od, x1_od, tau_s_od, tau_u_od,
                    block_size=block_size).exp()
                X_su = ((1 - u_od) / (1 - s_od + 1e-8)) * x_s_od \
                       + ((u_od - s_od) / (1 - s_od + 1e-8)) * D_su_offdiag
                D_ut_offdiag = _fwd(
                    X_su, x1_od, tau_u_od, tau_t_od,
                    block_size=block_size).exp()
                lambda_sut = ((1 - t_od) * (u_od - s_od)
                              / ((1 - u_od) * (t_od - s_od) + 1e-8))
                offdiag_target = stopgrad(
                    lambda_sut * D_su_offdiag
                    + (1 - lambda_sut) * D_ut_offdiag)

            if not self.config.algo.use_mse_loss_psd:
                offdiag_loss = -(
                    offdiag_target * log_D_st[idx_offdiag]).sum(dim=-1)
            else:
                offdiag_loss = F.mse_loss(
                    log_D_st[idx_offdiag].exp(), offdiag_target,
                    reduction='none').sum(dim=-1)
            if self.config.algo.rescale_offdiag_loss_psd is True:
                offdiag_loss = offdiag_loss * (
                    (t_od - s_od) / (1 - s_od + 1e-8)).squeeze(-1).pow(2)
        else:
            offdiag_loss = x_s.new_empty((0, L))

        if has_diag:
            if not self.config.algo.use_mse_loss_psd:
                diag_loss = -(
                    on_diagonal_target * log_D_st[idx_diag]).sum(dim=-1)
            else:
                diag_loss = F.mse_loss(
                    log_D_st[idx_diag].exp(), on_diagonal_target,
                    reduction='none').sum(dim=-1)
        else:
            diag_loss = x_s.new_empty((0, L))

        loss = torch.zeros(B, L, device=self.device)
        if has_diag:
            loss[idx_diag] = diag_loss
            diag_loss_to_log = diag_loss.mean()
        else:
            diag_loss_to_log = loss.new_tensor(0.0)
        self.log('diag_loss', diag_loss_to_log, prog_bar=True, sync_dist=True)

        if has_offdiag:
            loss[idx_offdiag] = offdiag_loss
            offdiag_loss_to_log = offdiag_loss.mean()
        else:
            offdiag_loss_to_log = loss.new_tensor(0.0)
        self.log('offdiag_loss', offdiag_loss_to_log, prog_bar=True,
                 sync_dist=True)
        self.log('loss', loss.mean(), prog_bar=True, sync_dist=True)

        if self.config.algo.learnable_loss_weighting is True:
            tau_s_for_weight = tau_s.mean(dim=1)
            tau_t_for_weight = tau_t.mean(dim=1)
            loss_weight = self.backbone.learnable_loss_weighting(
                tau_s_for_weight, tau_t_for_weight)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            self.log('loss_weighted', loss.mean(), prog_bar=True,
                     sync_dist=True)
        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        del eps
        if num_steps is None:
            num_steps = self.config.sampling.steps
        gamma = getattr(self.config.sampling, 'gamma', 0.0)
        print(f"Sampling with {num_steps} steps")

        B = num_samples
        L = self.num_tokens
        V = self.vocab_size
        block_size = self.block_size
        num_blocks = self._num_blocks(block_size)
        device = self.device
        tokens = torch.zeros((B, L), dtype=torch.long, device=device)
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size
            z_block = torch.randn((B, block_size, V), device=device,
                                  dtype=self.dtype)
            for i in range(num_steps):
                tau_curr = tau_vals[i]
                tau_next = tau_vals[i + 1]
                t_curr = self._tau_to_t(tau_curr.expand(B))
                t_next = self._tau_to_t(tau_next.expand(B))
                sigma_target = 1.0 - t_next
                sigma_tilde = sigma_target * torch.sqrt(
                    torch.tensor(1.0 - gamma**2, device=device))
                t_tilde = 1.0 - sigma_tilde
                tau_tilde = self._t_to_tau(t_tilde)

                tau_blocks = torch.zeros((B, num_blocks), device=device)
                tau_blocks[:, block_idx] = tau_curr
                tau_tilde_blocks = torch.zeros((B, num_blocks), device=device)
                tau_tilde_blocks[:, block_idx] = tau_tilde

                noisy = torch.zeros((B, L, V), device=device, dtype=self.dtype)
                noisy[:, start:end] = z_block
                log_pred = self._block_forward(
                    noisy, tokens, tau_blocks, tau_tilde_blocks,
                    block_size=block_size)
                pred = log_pred[:, start:end].exp()
                if i == num_steps - 1:
                    z_block = pred
                    break

                weight_z = ((1.0 - t_tilde.view(-1, 1, 1))
                            / (1.0 - t_curr.view(-1, 1, 1)))
                weight_D = ((t_tilde.view(-1, 1, 1)
                             - t_curr.view(-1, 1, 1))
                            / (1.0 - t_curr.view(-1, 1, 1)))
                z_tilde = weight_z * z_block + weight_D * pred
                if gamma > 0:
                    noise_std = gamma * sigma_target.view(-1, 1, 1)
                    mean_adjustment = (
                        sigma_tilde.view(-1, 1, 1)
                        - sigma_target.view(-1, 1, 1))
                    z_block = (z_tilde + mean_adjustment * pred
                               + noise_std * torch.randn_like(z_block))
                else:
                    z_block = z_tilde

            tokens[:, start:end] = z_block.argmax(dim=-1)
        return tokens

class FLMVDM(FLM):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.train_t_min = self.t_min
        self.train_t_max = self.t_max
        self.val_t_min = getattr(config.algo, 'val_t_min', self.train_t_min)
        self.val_t_max = getattr(config.algo, 'val_t_max', self.train_t_max)
        self.interpolant_type = getattr(config.algo, 'interpolant_type', 'vp_vdm')
        self.cond_t = getattr(config.algo, 'cond_t', 'tau')
        self.gamma_min = getattr(config.algo, 'gamma_min', -5.)
        self.gamma_max = getattr(config.algo, 'gamma_max', 5.)
        self.noise_proposal = getattr(config.algo, 'noise_proposal', 'tau')
        self.train_objective = getattr(config.algo, 'train_objective', 'vlb')
        self.gamma_proposal_loc = float(
            getattr(config.algo, 'gamma_proposal_loc', 0.0))
        self.gamma_proposal_scale = float(
            getattr(config.algo, 'gamma_proposal_scale', 1.0))
        self.vlb_proposal_floor = float(
            getattr(config.algo, 'vlb_proposal_floor', 0.05))
        self.train_loss = getattr(config.algo, 'train_loss', 'ce')
        self.train_on_weighted_loss = getattr(config.algo, 'train_on_weighted_loss', True)
        self.diagnostic_num_bins = getattr(config.algo, 'diagnostic_num_bins', 8)
        self.self_conditioning_cfg = getattr(config.algo, 'self_conditioning', None)
        self.self_conditioning_enabled = bool(
            getattr(self.self_conditioning_cfg, 'enabled', False))
        self.self_conditioning_train_prob = float(
            getattr(self.self_conditioning_cfg, 'train_prob', 0.25))
        if self.interpolant_type != 'vp_vdm':
            raise NotImplementedError(
                "FLMVDM currently supports only interpolant_type='vp_vdm', "
                f"got {self.interpolant_type}")
        self._validate_proposal_config()
        if not 0.0 <= self.self_conditioning_train_prob <= 1.0:
            raise ValueError(
                "self_conditioning.train_prob must be in [0, 1], "
                f"got {self.self_conditioning_train_prob}")
        if (self.self_conditioning_enabled
                and self.config.algo.backbone != 'dit'):
            raise NotImplementedError(
                "FLM-VDM self-conditioning is only implemented for the DiT "
                f"backbone, got {self.config.algo.backbone}")
        self.lut_tau2gamma, self.lut_gamma2tau = utils.build_vp_luts(
            K=self.vocab_size,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
        )

    def _tau_to_gamma(self, tau):
        return utils.tau_to_gamma(tau, self.lut_tau2gamma)

    def _gamma_to_tau(self, gamma):
        return utils.gamma_to_tau(gamma, self.lut_gamma2tau)

    def _d_gamma_by_d_tau(self, tau):
        return utils.d_tau_to_gamma(tau, self.lut_tau2gamma)

    def _validate_proposal_config(self):
        if self.noise_proposal not in {'tau', 'gamma_gumbel'}:
            raise ValueError(
                "algo.noise_proposal must be one of "
                "{'tau', 'gamma_gumbel'}, "
                f"got {self.noise_proposal}")
        if self.train_objective not in {'vlb', 'bounded_vlb', 'proposal_ce'}:
            raise ValueError(
                "algo.train_objective must be one of "
                "{'vlb', 'bounded_vlb', 'proposal_ce'}, "
                f"got {self.train_objective}")
        if self.noise_proposal == 'tau' and self.train_objective != 'vlb':
            raise ValueError(
                "algo.train_objective must be 'vlb' when "
                "algo.noise_proposal='tau'")
        if self.noise_proposal == 'gamma_gumbel' and self.train_objective == 'vlb':
            raise ValueError(
                "algo.train_objective='vlb' is only defined for "
                "algo.noise_proposal='tau'; use 'bounded_vlb' for "
                "importance-corrected gamma proposals")
        if self.gamma_proposal_scale <= 0:
            raise ValueError(
                "algo.gamma_proposal_scale must be positive, "
                f"got {self.gamma_proposal_scale}")
        if not 0.0 < self.vlb_proposal_floor <= 1.0:
            raise ValueError(
                "algo.vlb_proposal_floor must be in (0, 1], "
                f"got {self.vlb_proposal_floor}")

    def _get_noise_conditioning(self, tau_t, gamma):
        if self.cond_t == 'tau':
            return tau_t
        if self.cond_t == 'snr':
            return torch.exp(-gamma)
        if self.cond_t == 'log_snr':
            return -gamma
        if self.cond_t == 'nsr':
            return torch.exp(gamma)
        if self.cond_t == 'log_nsr':
            return gamma
        raise ValueError(f"Unknown cond_t: {self.cond_t}")

    def _log_stage_metric(self, stage, name, value, group=None):
        if torch.is_tensor(value):
            value = value.detach()
        if group is None:
            metric_name = f'{stage}/{name}'
        else:
            metric_name = f'{stage}_{group}/{name}'
        self.log(metric_name,
                 value,
                 on_step=self.training,
                 on_epoch=not self.training,
                 sync_dist=True)

    def _log_distribution_stats(self, stage, name, values, group=None):
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return

        self._log_stage_metric(stage, f'{name}_mean', values.mean(),
                               group=group)
        self._log_stage_metric(stage, f'{name}_median', values.median(),
                               group=group)
        quantiles = torch.quantile(
            values, torch.tensor([0.9, 0.99], device=values.device))
        self._log_stage_metric(stage, f'{name}_p90', quantiles[0],
                               group=group)
        self._log_stage_metric(stage, f'{name}_p99', quantiles[1],
                               group=group)
        self._log_stage_metric(stage, f'{name}_max', values.max(),
                               group=group)

    def _log_weight_diagnostics(self, stage, tau, gamma, snr, snr_width,
                                snr_prime_tau, dgamma_dtau, loss_weight,
                                train_weight, vlb_weight,
                                normalized_vlb_weight,
                                proposal_density=None,
                                vlb_gamma_density=None):
        self._log_distribution_stats(stage, 'tau', tau, group='stats')
        self._log_distribution_stats(stage, 'gamma', gamma, group='stats')
        self._log_distribution_stats(stage, 'snr', snr, group='stats')
        self._log_distribution_stats(stage, 'snr_width', snr_width,
                                     group='stats')
        self._log_distribution_stats(stage, 'snr_prime_tau', snr_prime_tau,
                                     group='stats')
        self._log_distribution_stats(stage, 'dgamma_dtau', dgamma_dtau,
                                     group='stats')
        self._log_distribution_stats(stage, 'loss_weight', loss_weight,
                                     group='stats')
        self._log_distribution_stats(stage, 'train_weight', train_weight,
                                     group='stats')
        self._log_distribution_stats(stage, 'vlb_weight', vlb_weight,
                                     group='stats')
        self._log_distribution_stats(stage, 'normalized_vlb_weight',
                                     normalized_vlb_weight,
                                     group='stats')
        if proposal_density is not None:
            self._log_distribution_stats(stage, 'proposal_density',
                                         proposal_density,
                                         group='stats')
        if vlb_gamma_density is not None:
            self._log_distribution_stats(stage, 'vlb_gamma_density',
                                         vlb_gamma_density,
                                         group='stats')

        weight_sum = loss_weight.sum().clamp_min(1e-12)
        ess = weight_sum.square() / loss_weight.square().sum().clamp_min(1e-12)
        self._log_stage_metric(stage, 'loss_weight_ess', ess, group='stats')
        self._log_stage_metric(stage, 'loss_weight_ess_frac',
                               ess / max(loss_weight.numel(), 1),
                               group='stats')

        vlb_weight_sum = vlb_weight.sum().clamp_min(1e-12)
        vlb_ess = (
            vlb_weight_sum.square()
            / vlb_weight.square().sum().clamp_min(1e-12))
        self._log_stage_metric(stage, 'vlb_weight_ess', vlb_ess,
                               group='stats')
        self._log_stage_metric(stage, 'vlb_weight_ess_frac',
                               vlb_ess / max(vlb_weight.numel(), 1),
                               group='stats')

    def _log_objective_diagnostics(self, stage, tau, target_data,
                                   log_softmax_pred, vlb_weight,
                                   train_weight):
        probs = log_softmax_pred.exp()
        ce_loss = -(target_data * log_softmax_pred).sum(dim=-1)
        l2_loss = ((target_data - probs) ** 2).sum(dim=-1)
        true_token_prob = (target_data * probs).sum(dim=-1)
        pred_tokens = log_softmax_pred.argmax(dim=-1)
        target_tokens = target_data.argmax(dim=-1)
        token_accuracy = (pred_tokens == target_tokens).float()

        broadcast_weight = vlb_weight.expand_as(ce_loss)
        weight_sum = broadcast_weight.sum().clamp_min(1e-12)
        broadcast_train_weight = train_weight.expand_as(ce_loss)
        train_weight_sum = broadcast_train_weight.sum().clamp_min(1e-12)

        self._log_stage_metric(
            stage, 'ce_weighted_normalized',
            (ce_loss * broadcast_weight).sum() / weight_sum,
            group='objective')
        self._log_stage_metric(
            stage, 'l2_weighted_normalized',
            (0.5 * l2_loss * broadcast_weight).sum() / weight_sum,
            group='objective')
        self._log_stage_metric(
            stage, 'ce_train_weighted_normalized',
            (ce_loss * broadcast_train_weight).sum() / train_weight_sum,
            group='objective')
        self._log_stage_metric(
            stage, 'l2_train_weighted_normalized',
            (0.5 * l2_loss * broadcast_train_weight).sum()
            / train_weight_sum,
            group='objective')
        self._log_stage_metric(stage, 'token_accuracy',
                               token_accuracy.mean(),
                               group='objective')
        self._log_stage_metric(
            stage, 'token_accuracy_weighted_normalized',
            (token_accuracy * broadcast_weight).sum() / weight_sum,
            group='objective')
        self._log_stage_metric(
            stage, 'token_accuracy_train_weighted_normalized',
            (token_accuracy * broadcast_train_weight).sum()
            / train_weight_sum,
            group='objective')
        self._log_stage_metric(stage, 'true_token_prob_mean',
                               true_token_prob.mean(),
                               group='objective')
        self._log_distribution_stats(stage, 'true_token_prob',
                                     true_token_prob,
                                     group='objective')

        if self.training:
            return

        ce_per_sample = ce_loss.mean(dim=-1)
        weighted_ce_per_sample = ce_per_sample * vlb_weight.squeeze(-1)
        token_accuracy_per_sample = token_accuracy.mean(dim=-1)
        num_bins = max(int(self.diagnostic_num_bins), 1)
        edges = torch.linspace(0.0, 1.0, num_bins + 1, device=tau.device)
        bin_ids = torch.bucketize(tau.detach(), edges[1:-1])

        zero = tau.new_tensor(0.0)
        for bin_idx in range(num_bins):
            mask = bin_ids == bin_idx
            if mask.any():
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_frac',
                    mask.float().mean(),
                    group='objective')
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_mean',
                    tau[mask].mean(),
                    group='objective')
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_ce',
                    ce_per_sample[mask].mean(),
                    group='objective')
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_weight',
                    vlb_weight.squeeze(-1)[mask].mean(),
                    group='objective')
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_weighted_ce',
                    weighted_ce_per_sample[mask].mean(),
                    group='objective')
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_accuracy',
                    token_accuracy_per_sample[mask].mean(),
                    group='objective')
            else:
                self._log_stage_metric(stage, f'tau_bin_{bin_idx}_frac', zero,
                                       group='objective')
                self._log_stage_metric(stage, f'tau_bin_{bin_idx}_mean', zero,
                                       group='objective')
                self._log_stage_metric(stage, f'tau_bin_{bin_idx}_ce', zero,
                                       group='objective')
                self._log_stage_metric(stage, f'tau_bin_{bin_idx}_weight', zero,
                                       group='objective')
                self._log_stage_metric(
                    stage, f'tau_bin_{bin_idx}_weighted_ce', zero,
                    group='objective')
                self._log_stage_metric(stage, f'tau_bin_{bin_idx}_accuracy', zero,
                                       group='objective')
    
    def corrupt_continuous(self, x0, tau_t, gamma, dgamma_dtau, stage,
                           proposal_density=None, vlb_gamma_density=None):
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        train_weight, vlb_weight, diagnostics = self._compute_loss_weights(
            tau_t,
            gamma,
            dgamma_dtau,
            proposal_density=proposal_density,
            vlb_gamma_density=vlb_gamma_density,
        )

        alpha = torch.sigmoid(-gamma).sqrt().unsqueeze(-1).unsqueeze(-1)
        sigma = torch.sigmoid(gamma).sqrt().unsqueeze(-1).unsqueeze(-1)
        self._log_stage_metric(stage, 'alpha_mean', alpha.mean(),
                               group='stats')
        self._log_stage_metric(stage, 'sigma_mean', sigma.mean(),
                               group='stats')

        x_t = alpha * target_data + sigma * noise
        cond_t = self._get_noise_conditioning(tau_t, gamma)
        return (
            x_t,
            cond_t,
            target_data,
            train_weight.unsqueeze(-1),
            vlb_weight.unsqueeze(-1),
            diagnostics,
        )
    
    def _ce_loss(self, target_data, log_softmax_pred, vlb_weight,
                 train_weight, stage):
        loss = -(target_data * log_softmax_pred).sum(dim=-1)
        self._log_stage_metric(stage, 'ce_unweighted',
                               loss.mean().detach(),
                               group='objective')
        self._log_stage_metric(stage, 'ce_weighted',
                               (loss * vlb_weight).mean().detach(),
                               group='objective')
        self._log_stage_metric(stage, 'ce_train_weighted',
                               (loss * train_weight).mean().detach(),
                               group='objective')
        return loss, train_weight

    def _l2_loss(self, target_data, log_softmax_pred, vlb_weight,
                 train_weight, stage):
        loss = ((target_data - log_softmax_pred.exp()) ** 2).sum(dim=-1)
        self._log_stage_metric(stage, 'l2_unweighted',
                               loss.mean().detach(),
                               group='objective')
        self._log_stage_metric(stage, 'l2_weighted',
                               (0.5 * loss * vlb_weight).mean().detach(),
                               group='objective')
        self._log_stage_metric(stage, 'l2_train_weighted',
                               (0.5 * loss * train_weight).mean().detach(),
                               group='objective')
        return loss, 0.5 * train_weight

    def _gamma_support(self, device, dtype):
        gamma_min = torch.tensor(self.gamma_min, device=device, dtype=dtype)
        gamma_max = torch.tensor(self.gamma_max, device=device, dtype=dtype)
        return gamma_min, gamma_max

    def _snr_width(self, device, dtype):
        gamma_min, gamma_max = self._gamma_support(device, dtype)
        return (torch.exp(-gamma_min) - torch.exp(-gamma_max)).clamp_min(1e-12)

    def _truncated_gumbel_cdf(self, gamma, loc, scale):
        z = (gamma - loc) / scale
        return torch.exp(-torch.exp(-z))

    def _truncated_gumbel_pdf(self, gamma, loc, scale,
                              gamma_min, gamma_max):
        z = (gamma - loc) / scale
        pdf = torch.exp(-(z + torch.exp(-z))) / scale
        cdf_min = self._truncated_gumbel_cdf(gamma_min, loc, scale)
        cdf_max = self._truncated_gumbel_cdf(gamma_max, loc, scale)
        return pdf / (cdf_max - cdf_min).clamp_min(1e-12)

    def _sample_truncated_gumbel_gamma(self, unit):
        dtype = unit.dtype
        device = unit.device
        work_unit = unit.double()
        gamma_min, gamma_max = self._gamma_support(device, torch.float64)
        loc = torch.tensor(self.gamma_proposal_loc, device=device,
                           dtype=torch.float64)
        scale = torch.tensor(self.gamma_proposal_scale, device=device,
                             dtype=torch.float64)
        cdf_min = self._truncated_gumbel_cdf(gamma_min, loc, scale)
        cdf_max = self._truncated_gumbel_cdf(gamma_max, loc, scale)
        u = cdf_min + work_unit * (cdf_max - cdf_min)
        u = u.clamp(1e-12, 1.0 - 1e-7)
        gamma = loc - scale * torch.log(-torch.log(u))
        return gamma.to(dtype=dtype)

    def _sample_vlb_gamma(self, unit):
        dtype = unit.dtype
        device = unit.device
        gamma_min, gamma_max = self._gamma_support(device, dtype)
        snr_max = torch.exp(-gamma_min)
        snr_min = torch.exp(-gamma_max)
        snr = snr_max - unit * (snr_max - snr_min)
        return -torch.log(snr.clamp_min(1e-12))

    def _vlb_gamma_density(self, gamma):
        snr_width = self._snr_width(gamma.device, gamma.dtype)
        return torch.exp(-gamma) / snr_width

    def _proposal_gamma_density(self, gamma):
        gamma_min, gamma_max = self._gamma_support(gamma.device, gamma.dtype)
        loc = torch.tensor(self.gamma_proposal_loc, device=gamma.device,
                           dtype=gamma.dtype)
        scale = torch.tensor(self.gamma_proposal_scale, device=gamma.device,
                             dtype=gamma.dtype)
        gumbel_density = self._truncated_gumbel_pdf(
            gamma, loc, scale, gamma_min, gamma_max)
        vlb_density = self._vlb_gamma_density(gamma)
        floor = self.vlb_proposal_floor
        proposal_density = floor * vlb_density + (1.0 - floor) * gumbel_density
        return proposal_density, vlb_density

    def _sample_gamma_proposal(self, B, accum_step):
        mix_unit = self._sample_t_interval(
            B, accum_step, t_min=0.0, t_max=1.0)
        sample_unit = self._sample_t_interval(
            B, accum_step, t_min=0.0, t_max=1.0)
        gamma_gumbel = self._sample_truncated_gumbel_gamma(sample_unit)
        gamma_vlb = self._sample_vlb_gamma(sample_unit)
        use_vlb_component = mix_unit < self.vlb_proposal_floor
        gamma = torch.where(use_vlb_component, gamma_vlb, gamma_gumbel)
        proposal_density, vlb_density = self._proposal_gamma_density(gamma)
        return gamma, proposal_density, vlb_density

    def _compute_loss_weights(self, tau_t, gamma, dgamma_dtau,
                              proposal_density=None,
                              vlb_gamma_density=None):
        snr = torch.exp(-gamma)
        snr_width = self._snr_width(gamma.device, gamma.dtype)
        # Positive derivative dSNR / dtau under the VP parameterization.
        snr_prime_tau = snr * (-dgamma_dtau)

        if proposal_density is None:
            vlb_weight = snr_prime_tau
            normalized_vlb_weight = vlb_weight / snr_width
        else:
            normalized_vlb_weight = (
                vlb_gamma_density
                / proposal_density.clamp_min(1e-12))
            vlb_weight = snr_width * normalized_vlb_weight

        if not self.training or self.train_objective == 'vlb':
            train_weight = vlb_weight
        elif self.train_objective == 'bounded_vlb':
            train_weight = normalized_vlb_weight
        elif self.train_objective == 'proposal_ce':
            train_weight = torch.ones_like(vlb_weight)
        else:
            raise ValueError(f"Unknown train_objective: {self.train_objective}")

        diagnostics = {
            'tau': tau_t,
            'gamma': gamma,
            'snr': snr,
            'snr_width': snr_width,
            'snr_prime_tau': snr_prime_tau,
            'dgamma_dtau': dgamma_dtau,
            'loss_weight': train_weight,
            'train_weight': train_weight,
            'vlb_weight': vlb_weight,
            'normalized_vlb_weight': normalized_vlb_weight,
            'proposal_density': proposal_density,
            'vlb_gamma_density': vlb_gamma_density,
        }
        return train_weight, vlb_weight, diagnostics

    def _sample_tau_coordinates(self, B, accum_step):
        if self.training and self.noise_proposal == 'gamma_gumbel':
            gamma, proposal_density, vlb_gamma_density = (
                self._sample_gamma_proposal(B, accum_step))
            tau_t = self._gamma_to_tau(gamma)
        else:
            t_min, t_max = (
                (self.train_t_min, self.train_t_max)
                if self.training else (self.val_t_min, self.val_t_max))
            tau_t = self._sample_t_interval(
                B,
                accum_step,
                t_min=t_min,
                t_max=t_max,
            )
            gamma = self._tau_to_gamma(tau_t)
            proposal_density = None
            vlb_gamma_density = None
        dgamma_dtau = self._d_gamma_by_d_tau(tau_t)
        return tau_t, gamma, dgamma_dtau, proposal_density, vlb_gamma_density

    def _build_self_conditioning(self, x_t, cond_t, self_cond_mask=None):
        if not self.self_conditioning_enabled:
            return None

        if self_cond_mask is None:
            with torch.no_grad():
                return self.forward(x_t, cond_t).detach().exp()

        self_cond = torch.zeros_like(x_t)
        if not self_cond_mask.any():
            return self_cond

        with torch.no_grad():
            self_cond[self_cond_mask] = self.forward(
                x_t[self_cond_mask],
                cond_t[self_cond_mask],
            ).detach().exp()
        return self_cond

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens
        stage = 'train' if self.training else 'val'
        B = x0.shape[0]
        (tau_t, gamma, dgamma_dtau, proposal_density,
         vlb_gamma_density) = self._sample_tau_coordinates(
             B, current_accumulation_step)

        (x_t, cond_t, target_data, train_weight, vlb_weight,
         diagnostics) = self.corrupt_continuous(
             x0,
             tau_t,
             gamma,
             dgamma_dtau,
             stage,
             proposal_density=proposal_density,
             vlb_gamma_density=vlb_gamma_density,
         )
        self_cond = None
        if self.self_conditioning_enabled:
            if self.training:
                self_cond_mask = (
                    torch.rand(B, device=self.device)
                    < self.self_conditioning_train_prob)
                self_cond = self._build_self_conditioning(
                    x_t, cond_t, self_cond_mask=self_cond_mask)
                self._log_stage_metric(
                    stage, 'selfcond_frac', self_cond_mask.float().mean(),
                    group='stats')
            else:
                self_cond = self._build_self_conditioning(x_t, cond_t)
                self._log_stage_metric(
                    stage, 'selfcond_frac',
                    x_t.new_tensor(1.0),
                    group='stats')
        else:
            self._log_stage_metric(
                stage, 'selfcond_frac',
                x_t.new_tensor(0.0),
                group='stats')

        f = self.forward(x_t, cond_t, self_cond=self_cond)
        self._log_weight_diagnostics(stage, **diagnostics)
        self._log_objective_diagnostics(stage, diagnostics['tau'],
                                        target_data, f, vlb_weight,
                                        train_weight)
        if self.train_loss == 'ce':
            with torch.no_grad():
                _, _ = self._l2_loss(target_data, f, vlb_weight,
                                     train_weight, stage)
            loss, loss_weight = self._ce_loss(target_data, f, vlb_weight,
                                             train_weight, stage)
        elif self.train_loss == 'l2':
            with torch.no_grad():
                _, _ = self._ce_loss(target_data, f, vlb_weight,
                                     train_weight, stage)
            loss, loss_weight = self._l2_loss(target_data, f, vlb_weight,
                                             train_weight, stage)
        else:
            raise NotImplementedError(f"Train loss {self.train_loss} not implemented!")
        self.log('loss', loss.mean(), prog_bar=True)
        if self.train_on_weighted_loss is True:
            loss = loss * loss_weight
            self.log('loss_weighted', loss.mean(), prog_bar=True)
        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples with a deterministic Gaussian-path update."""
        del eps
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device
        sigma_floor = 1e-5

        t_min, t_max = self.val_t_min, self.val_t_max
        tau_vals = torch.linspace(
            t_min, t_max, num_steps + 1, device=device)
        gamma_vals = self._tau_to_gamma(tau_vals)

        gamma_start = gamma_vals[0]
        sigma_start = torch.sigmoid(gamma_start).sqrt()
        z = sigma_start * torch.randn(
            (num_samples, L, V), device=device, dtype=self.dtype)
        self_cond_probs = None

        for i in range(num_steps):
            tau_curr = tau_vals[i].expand(B)
            tau_next = tau_vals[i + 1].expand(B)
            gamma_curr = gamma_vals[i].expand(B)
            alpha_curr = torch.sigmoid(-gamma_curr).sqrt().view(-1, 1, 1)
            sigma_curr = torch.sigmoid(gamma_curr).sqrt().view(-1, 1, 1)
            cond_curr = self._get_noise_conditioning(tau_curr, gamma_curr)

            x_hat0 = self.forward(
                z, cond_curr, self_cond=self_cond_probs).exp()
            if self.self_conditioning_enabled:
                self_cond_probs = x_hat0.detach()
            eps_hat = (
                z - alpha_curr * x_hat0
            ) / sigma_curr.clamp_min(sigma_floor)

            gamma_next = gamma_vals[i + 1].expand(B)
            alpha_next = torch.sigmoid(-gamma_next).sqrt().view(-1, 1, 1)
            sigma_next = torch.sigmoid(gamma_next).sqrt().view(-1, 1, 1)
            z = alpha_next * x_hat0 + sigma_next * eps_hat

        tau_end = tau_vals[-1].expand(B)
        gamma_end = gamma_vals[-1].expand(B)
        cond_end = self._get_noise_conditioning(tau_end, gamma_end)
        return self.forward(
            z, cond_end, self_cond=self_cond_probs).argmax(dim=-1)

class FMLM_TwoModel(FLMBase):
    """FMLM two-model parameterization (appendix: semigroup loss, first stage of two-stage MSE distillation)."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.log_flag = False
        self.teacher_model = None
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
        )

    def setup(self, stage: str):
        if self.teacher_model is None:
            self._initialize_teacher()
        if stage == 'fit' and not self._is_resuming:
            print(">>> Initializing student from teacher...")
            self._initialize_student_from_teacher()
        elif self._is_resuming:
            print(">>> Skipping student initialization (resuming from checkpoint).")

    def _initialize_teacher(self):
        self.teacher_model = self._load_teacher_model(
            self.config.algo.teacher_path, use_plain_config=True)

    def _initialize_student_from_teacher(self):
        self._copy_teacher_weights_to_student(self.teacher_model.state_dict())
        if hasattr(self.backbone, 'output_layer'):
            self._zero_init_module(self.backbone.output_layer)
            print("Zero initialized student output_layer")

    def _on_load_checkpoint_extra(self, checkpoint):
        if self.config.mode == 'sample_eval':
            self._initialize_teacher()

    def on_train_start(self):
        super().on_train_start()
        if self.teacher_model is None:
            print("Initializing teacher model...")
            self._initialize_teacher()
            print("Initializing student from teacher")
            self._initialize_student_from_teacher()

    def teacher_forward(self, xt, tau):
        tau = self._process_sigma(tau)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model(xt, tau)
        return self._process_model_output(model_output, xt, tau)

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens
        B = x0.shape[0]

        d_tau = self._sample_t_interval(
            B, current_accumulation_step, t_min=0.0, t_max=1.0).clamp(min=1e-6)
        tau_s = torch.rand(B, device=self.device) * (1.0 - d_tau)
        if self.config.algo.add_boundary:
            p_boundary = 1.0 / self.config.algo.boundary_prob
            is_boundary = torch.rand(B, device=self.device) < p_boundary
            tau_s = torch.where(is_boundary, torch.tensor(0.0, device=self.device), tau_s)
            d_tau = torch.where(is_boundary, torch.tensor(1.0, device=self.device), d_tau)

        t_s = self._tau_to_t(tau_s)
        x_t_s, _ = self.corrupt_continuous(x0, t_s)
        f_final_sc = self.forward_no_softmax(x_t_s, tau_s, tau_s + d_tau)

        with torch.no_grad():
            d_tau_half = d_tau / 2.0
            t_s = t_s.view(-1, 1, 1)

            t_mid = self._tau_to_t(tau_s + d_tau_half).view(-1, 1, 1)
            t_e = self._tau_to_t(tau_s + d_tau).view(-1, 1, 1)
            dt_half_1 = t_mid - t_s
            dt_half_2 = t_e - t_mid
            dt = t_e - t_s

            f_theta_s = self.teacher_forward(x_t_s, tau_s).exp()
            v_s_hat = (f_theta_s - x_t_s) / (1.0 - t_s + 1e-5)
            g_theta_s_u = self.forward_no_softmax(x_t_s, tau_s, tau_s + d_tau_half)

            # v_su_hat = v_s(x_s) + 1/2(u-s)g_theta(x_s,u)
            v_s_u_hat = ((f_theta_s - x_t_s) / (1.0 - t_s + 1e-5)
                         + dt_half_1 / 2.0 * g_theta_s_u)

            # F_s,u(x_s) = x_s + (u-s)v_s(x_s) + 1/2(u-s)^2 g_theta(x_s,u)
            large_f_s_u = (x_t_s + dt_half_1 * v_s_hat
                           + 0.5 * dt_half_1 ** 2 * g_theta_s_u)

            f_theta_u = self.teacher_forward(large_f_s_u, tau_s + d_tau_half).exp()
            v_u_hat = (f_theta_u - large_f_s_u) / (1.0 - t_mid + 1e-5)
            g_theta_u_t = self.forward_no_softmax(
                large_f_s_u, tau_s + d_tau_half, tau_s + d_tau)

            # v_u,t_hat(x_u') = v_u(x_u') + 1/2*(t-u)*g_theta(x_u',u, t)
            v_u_t_hat = v_u_hat + dt_half_2 / 2.0 * g_theta_u_t

            v_hat = (dt_half_1 * v_s_u_hat + dt_half_2 * v_u_t_hat) / dt

            # x_1_hat = stopgrad(x_s + (1-s)*v_hat)
            x_boot = x_t_s + (1.0 - t_s) * v_hat
            x_boot = x_boot.detach()

        f_final_fm = f_theta_s
        weight = 0.5 * dt * (1.0 - t_s)
        f_final = f_final_fm + weight * f_final_sc
        error = x_boot - f_final  # (B, L, V)
        loss = (error ** 2).mean(dim=-1) * self.vocab_size  # (B, L)

        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(tau_s, tau_s + d_tau)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight

        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples using flow map."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for i in range(num_steps):
            tau_curr = tau_vals[i]
            tau_next = tau_vals[i + 1]
            tau_in = tau_curr.expand(B)
            t_in = self._tau_to_t(tau_in)
            dt_in = self._tau_to_t(tau_next.expand(B)) - t_in

            x_1_pred_fm = self.teacher_forward(z, tau_in).exp()
            x_1_pred_sc = self.forward_no_softmax(z, tau_in, tau_next.expand(B))
            v_pred = (x_1_pred_fm - z) / (1.0 - t_in.view(-1, 1, 1) + eps)
            z = (z + v_pred * dt_in.view(-1, 1, 1)
                 + 0.5 * (dt_in.view(-1, 1, 1) ** 2) * x_1_pred_sc)

        return z.argmax(dim=-1)


class FMLM_TwoStage(FLMBase):
    """FMLM two-stage distillation (appendix: second stage compresses two-model teacher into single model)."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.teacher_model_f = None
        self.teacher_model_g = None
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    def setup(self, stage: str):
        if self.teacher_model_f is None or self.teacher_model_g is None:
            self._initialize_teacher_f()
            self._initialize_teacher_g()
        if stage == 'fit' and not self._is_resuming:
            print(">>> Initializing student from teacher...")
            self._initialize_student_from_teacher()
        elif self._is_resuming:
            print(">>> Skipping student initialization (resuming from checkpoint).")

    def _initialize_teacher_f(self):
        self.teacher_model_f = self._load_teacher_model(
            self.config.algo.teacher_f_path, use_plain_config=True)

    def _initialize_teacher_g(self):
        """Load the residual teacher model (uses current config, no plain override)."""
        self.teacher_model_g = self._load_teacher_model(
            self.config.algo.teacher_g_path, use_plain_config=False)

    def _initialize_student_from_teacher(self):
        self._copy_teacher_weights_to_student(self.teacher_model_f.state_dict())


    def on_train_start(self):
        super().on_train_start()
        if self.teacher_model_f is None or self.teacher_model_g is None:
            print("Initializing teacher models...")
            self._initialize_teacher_f()
            self._initialize_teacher_g()
            print("Initializing student from teacher")
            self._initialize_student_from_teacher()


    def teacher_f_forward(self, xt, tau=None, d=None, use_jvp_attn=False):
        del d, use_jvp_attn
        sigma = tau.unsqueeze(-1) if tau.ndim == 1 else tau
        sigma = self._process_sigma(sigma)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model_f(xt, sigma)
        return self._process_model_output(model_output, xt, sigma)

    def teacher_g_forward(self, xt, tau, tau_prime=None, **kwargs):
        sigma = self._process_sigma(tau)
        if tau_prime is not None:
            sigma_prime = self._process_sigma(tau_prime)
        else:
            sigma_prime = None
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.teacher_model_g(
                xt, sigma, sigma_prime, **kwargs)
        return model_output

    def loss(self, x1, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t, output_tokens
        B = x1.shape[0]

        d_tau = self._sample_t_interval(
            B, current_accumulation_step,
            t_min=0.0, t_max=1.0).clamp(min=1e-5, max=1.0)
        tau_s = torch.rand(B, device=self.device) * (1.0 - d_tau)

        if self.config.algo.add_boundary:
            p_boundary = 1.0 / self.config.algo.boundary_prob
            is_boundary = torch.rand(B, device=self.device) < p_boundary
            tau_s = torch.where(is_boundary, torch.tensor(0.0, device=self.device), tau_s)
            d_tau = torch.where(is_boundary, torch.tensor(1.0, device=self.device), d_tau)

        t_s = self._tau_to_t(tau_s)
        x_t_s, _ = self.corrupt_continuous(x1, t_s)
        dt = self._tau_to_t(tau_s + d_tau) - t_s

        f_final_f = self.teacher_f_forward(x_t_s, tau_s).exp()
        v_f = (f_final_f - x_t_s) / (1.0 - t_s.view(-1, 1, 1) + 1e-5)
        f_final_g = self.teacher_g_forward(x_t_s, tau_s, tau_s + d_tau)
        F_s_t = (x_t_s + v_f * dt.view(-1, 1, 1)
                 + 0.5 * (dt.view(-1, 1, 1) ** 2) * f_final_g)

        student_pred = self.forward(x_t_s, tau_s, tau_s + d_tau).exp()
        student_v = (student_pred - x_t_s) / (1.0 - t_s.view(-1, 1, 1) + 1e-5)
        F_s_t_distilled = x_t_s + dt.view(-1, 1, 1) * student_v

        error = F_s_t - F_s_t_distilled
        loss = (error ** 2).mean(dim=-1) * self.vocab_size

        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                        eps=1e-5):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        gamma = getattr(self.config.sampling, 'gamma', 0.0)
        print(f"Sampling with {num_steps} steps")
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
        
        for i in range(num_steps):
            tau_curr = tau_vals[i]
            tau_next = tau_vals[i + 1]

            t_curr = self._tau_to_t(tau_curr.expand(B))

            t_next = self._tau_to_t(tau_next.expand(B))
            sigma_target = 1.0 - t_next

            sigma_tilde = sigma_target * torch.sqrt(torch.tensor(1.0 - gamma**2))
            t_tilde = 1.0 - sigma_tilde
            tau_tilde = self._t_to_tau(t_tilde)

            log_D_st_pred = self.forward(z, tau_curr.expand(B), tau_tilde)
            D_st_pred = log_D_st_pred.exp()

            if i == num_steps - 1:
                z = D_st_pred
                break

            weight_z = (1.0 - t_tilde.view(-1, 1, 1)) / (1.0 - t_curr.view(-1, 1, 1))
            weight_D = ((t_tilde.view(-1, 1, 1) - t_curr.view(-1, 1, 1))
                        / (1.0 - t_curr.view(-1, 1, 1)))
            z_tilde = weight_z * z + weight_D * D_st_pred

            if gamma > 0:
                noise_std = gamma * sigma_target.view(-1, 1, 1)
                mean_adjustment = sigma_tilde.view(-1, 1, 1) - sigma_target.view(-1, 1, 1)
                z = z_tilde + mean_adjustment * D_st_pred + noise_std * torch.randn_like(z)
            else:
                z = z_tilde
                
        return z.argmax(dim=-1)
