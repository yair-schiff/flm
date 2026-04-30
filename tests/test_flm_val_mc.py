import unittest

import torch

import algo


class FakeFLM(algo.FLMBase):
    def __init__(self, val_mc_samples=1, recon=False):
        torch.nn.Module.__init__(self)
        self._device = torch.device('cpu')
        self.val_mc_samples = val_mc_samples
        self.ignore_bos = False
        self.recon = recon
        self.t_min = 0.0
        self.t_max = 1.0
        self.antithetic_sampling = False
        self.loss_inputs = []
        self.logged = []

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del output_tokens, current_accumulation_step, train_mode
        del xT, given_t, not_sampling_t
        self.loss_inputs.append(x0.detach().clone())
        loss = torch.arange(
            x0.numel(), device=x0.device, dtype=torch.float32).view_as(x0)
        if self.recon:
            self._nll_with_recon_loss = loss + 100.0
        return loss

    def log(self, name, value, **kwargs):
        self.logged.append((name, value.detach().clone(), kwargs))


class SamplingFakeFLM(FakeFLM):
    def __init__(self, val_mc_samples=1):
        super().__init__(val_mc_samples=val_mc_samples)
        self.antithetic_sampling = True
        self.sampled_tau = None

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del output_tokens, train_mode, xT, given_t, not_sampling_t
        self.sampled_tau = self._sample_t_interval(
            x0.shape[0], current_accumulation_step)
        return torch.zeros_like(x0, dtype=torch.float32)


class FLMValidationMCTest(unittest.TestCase):
    def test_single_mc_sample_preserves_current_aggregation(self):
        model = FakeFLM(val_mc_samples=1)
        model.eval()
        x0 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        valid_tokens = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        losses = model._loss(x0, valid_tokens)

        self.assertEqual(model.loss_inputs[0].shape, x0.shape)
        self.assertEqual(losses.num_tokens.item(), 4.0)
        self.assertEqual(losses.nlls.item(), 9.0)
        self.assertEqual(losses.loss.item(), 2.25)

    def test_validation_mc_samples_average_per_item_before_token_counting(self):
        model = FakeFLM(val_mc_samples=3)
        model.eval()
        x0 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        valid_tokens = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        losses = model._loss(x0, valid_tokens)

        self.assertEqual(model.loss_inputs[0].shape, (6, 3))
        torch.testing.assert_close(
            model.loss_inputs[0],
            x0.unsqueeze(0).expand(3, *x0.shape).reshape(6, 3))
        self.assertEqual(losses.num_tokens.item(), 4.0)
        self.assertEqual(losses.nlls.item(), 33.0)
        self.assertEqual(losses.loss.item(), 8.25)

    def test_recon_side_loss_is_averaged_over_validation_mc_samples(self):
        model = FakeFLM(val_mc_samples=3, recon=True)
        model.eval()
        x0 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        valid_tokens = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        model._loss(x0, valid_tokens)

        self.assertEqual(len(model.logged), 1)
        name, value, kwargs = model.logged[0]
        self.assertEqual(name, 'val/nll_with_recon')
        self.assertEqual(value.item(), 108.25)
        self.assertFalse(kwargs['on_step'])
        self.assertTrue(kwargs['on_epoch'])
        self.assertTrue(kwargs['sync_dist'])
        self.assertIsNone(model._nll_with_recon_loss)

    def test_antithetic_validation_mc_stratifies_times_per_item(self):
        torch.manual_seed(123)
        model = SamplingFakeFLM(val_mc_samples=4)
        model.eval()
        x0 = torch.zeros((5, 2), dtype=torch.long)
        valid_tokens = torch.ones_like(x0, dtype=torch.float32)

        model._loss(x0, valid_tokens)

        tau = model.sampled_tau.reshape(4, 5)
        strata = torch.floor(tau * 4).long().sort(dim=0).values
        expected = torch.arange(4).unsqueeze(1).expand(4, 5)
        torch.testing.assert_close(strata, expected)


if __name__ == '__main__':
    unittest.main()
