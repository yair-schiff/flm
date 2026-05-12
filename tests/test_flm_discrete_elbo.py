import types
import unittest

import torch
import torch.nn.functional as F

import algo


class LinearFieldFLM(algo.FLM):
    def __init__(self, scale=0.0, num_tokens=2, vocab_size=3,
                 algo_name='flm'):
        torch.nn.Module.__init__(self)
        self._device = torch.device('cpu')
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.config = types.SimpleNamespace(
            algo=types.SimpleNamespace(name=algo_name))
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def _continuous_likelihood_vector_field(self, x, t, endpoint_eps):
        del t, endpoint_eps
        return self.scale.to(dtype=x.dtype) * x


class FLMDiscreteELBOTest(unittest.TestCase):
    @staticmethod
    def _manual_discrete_elbo_terms(model, x0, mc_samples, endpoint_eps, seed):
        endpoint_t = 1.0 - endpoint_eps
        target = F.one_hot(x0, model.vocab_size).to(torch.float32)
        mean = endpoint_t * target
        continuous_terms = []
        decoder_terms = []
        encoder_terms = []
        sample_log_weights = []
        for sample_idx in range(mc_samples):
            sample_seed = seed + sample_idx
            y_endpoint = mean + endpoint_eps * model._sample_endpoint_noise(
                mean.shape, sample_seed, mean.device, mean.dtype)
            continuous = model.continuous_endpoint_log_density(
                y_endpoint,
                num_steps=16,
                endpoint_eps=endpoint_eps,
                trace_method='exact',
                solver='rk4')
            decoder = model._endpoint_decoder_log_prob(
                y_endpoint, x0, endpoint_t, endpoint_eps)
            encoder = model._diagonal_gaussian_log_prob(
                y_endpoint, mean, endpoint_eps)
            continuous_terms.append(continuous)
            decoder_terms.append(decoder)
            encoder_terms.append(encoder)
            sample_log_weights.append(continuous + decoder - encoder)
        return {
            'continuous_mean': torch.stack(continuous_terms, dim=0).mean(dim=0),
            'decoder_mean': torch.stack(decoder_terms, dim=0).mean(dim=0),
            'encoder_mean': torch.stack(encoder_terms, dim=0).mean(dim=0),
            'sample_elbo': torch.stack(sample_log_weights, dim=0).mean(dim=0),
            'iw_elbo': model._logmeanexp(
                torch.stack(sample_log_weights, dim=0), dim=0),
            'encoder_entropy': model._diagonal_gaussian_entropy(
                batch_size=x0.shape[0],
                num_dims=mean[0].numel(),
                std=endpoint_eps,
                device=mean.device,
                dtype=mean.dtype),
        }

    def test_endpoint_density_matches_one_hot_wrapper(self):
        model = LinearFieldFLM(scale=0.2, num_tokens=2, vocab_size=4)
        x0 = torch.tensor([[0, 1], [3, 2]])
        one_hot = F.one_hot(x0, model.vocab_size).to(torch.float32)

        wrapped = model.continuous_log_likelihood(
            x0,
            num_steps=16,
            endpoint_eps=0.2,
            trace_method='exact',
            solver='rk4')
        direct = model.continuous_endpoint_log_density(
            one_hot,
            num_steps=16,
            endpoint_eps=0.2,
            trace_method='exact',
            solver='rk4')

        torch.testing.assert_close(direct, wrapped, rtol=1e-6, atol=1e-6)

    def test_discrete_elbo_matches_component_sum(self):
        model = LinearFieldFLM(scale=0.0, num_tokens=2, vocab_size=3)
        x0 = torch.tensor([[0, 1], [2, 0]])
        endpoint_eps = 0.2
        seed = 7

        details = model.discrete_elbo(
            x0,
            mc_samples=1,
            num_steps=16,
            endpoint_eps=endpoint_eps,
            trace_method='exact',
            solver='rk4',
            seed=seed,
            return_details=True)

        manual = self._manual_discrete_elbo_terms(
            model, x0, mc_samples=1, endpoint_eps=endpoint_eps, seed=seed)
        expected_elbo = (
            manual['continuous_mean']
            + manual['decoder_mean']
            + manual['encoder_entropy'])

        torch.testing.assert_close(details['continuous_log_prob'],
                                   manual['continuous_mean'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['decoder_log_prob'],
                                   manual['decoder_mean'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['encoder_log_prob'],
                                   manual['encoder_mean'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['encoder_entropy'],
                                   manual['encoder_entropy'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['elbo'], expected_elbo,
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['sample_elbo'],
                                   manual['sample_elbo'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['iw_elbo'],
                                   manual['iw_elbo'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['nll_upper_bound'], -expected_elbo,
                                   rtol=1e-6, atol=1e-6)

    def test_discrete_elbo_iw_matches_logmeanexp_and_tightens(self):
        model = LinearFieldFLM(scale=0.0, num_tokens=2, vocab_size=3)
        x0 = torch.tensor([[0, 1], [2, 0]])
        endpoint_eps = 0.2
        seed = 13

        details = model.discrete_elbo(
            x0,
            mc_samples=3,
            num_steps=16,
            endpoint_eps=endpoint_eps,
            trace_method='exact',
            solver='rk4',
            seed=seed,
            return_details=True)

        manual = self._manual_discrete_elbo_terms(
            model, x0, mc_samples=3, endpoint_eps=endpoint_eps, seed=seed)

        torch.testing.assert_close(details['sample_elbo'],
                                   manual['sample_elbo'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['iw_elbo'], manual['iw_elbo'],
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(details['encoder_entropy'],
                                   manual['encoder_entropy'],
                                   rtol=1e-6, atol=1e-6)
        self.assertTrue(
            torch.all(details['iw_elbo'] >= details['sample_elbo']).item())
        self.assertTrue(
            torch.all(
                details['iw_nll_upper_bound']
                <= details['sample_nll_upper_bound']).item())

    def test_discrete_elbo_restores_training_mode_and_parameter_grads(self):
        model = LinearFieldFLM(scale=0.1, num_tokens=2, vocab_size=3)
        model.train()
        model.scale.grad = torch.tensor(5.0)
        grad_before = model.scale.grad.detach().clone()

        details = model.discrete_elbo(
            torch.tensor([[0, 1]]),
            mc_samples=2,
            num_steps=8,
            endpoint_eps=0.25,
            trace_method='hutchinson',
            trace_samples=2,
            seed=11,
            return_details=True)

        self.assertTrue(model.training)
        self.assertEqual(details['mc_samples'], 2)
        self.assertIn('iw_elbo', details)
        self.assertIn('encoder_entropy', details)
        torch.testing.assert_close(model.scale.grad, grad_before)

    def test_discrete_elbo_raises_for_non_base_flm(self):
        model = LinearFieldFLM(algo_name='flm_vdm')

        with self.assertRaisesRegex(NotImplementedError, 'base FLM'):
            model.discrete_elbo(torch.tensor([[0, 1]]))


if __name__ == '__main__':
    unittest.main()
