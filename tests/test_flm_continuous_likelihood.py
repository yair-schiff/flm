import math
import types
import unittest

import torch

import algo


class LinearFieldFLM(algo.FLM):
    def __init__(self, scale=0.2, num_tokens=2, vocab_size=3,
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


class BaseOnlyFLM(algo.FLMBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


def expected_linear_log_prob(model, x0, endpoint_eps):
    t1 = 1.0 - endpoint_eps
    scale = float(model.scale.detach())
    dim = model.num_tokens * model.vocab_size
    one_hot_norm_sq = model.num_tokens * math.exp(-2.0 * scale * t1)
    base_log_prob = -0.5 * (
        one_hot_norm_sq + dim * math.log(2.0 * math.pi))
    divergence_integral = dim * scale * t1
    value = base_log_prob - divergence_integral
    return torch.full((x0.shape[0],), value, dtype=torch.float32)


class FLMContinuousLikelihoodTest(unittest.TestCase):
    def test_exact_trace_matches_linear_field_closed_form(self):
        model = LinearFieldFLM(scale=0.3, num_tokens=2, vocab_size=3)
        model.eval()
        x0 = torch.tensor([[0, 1], [2, 0]])
        endpoint_eps = 0.1

        details = model.continuous_log_likelihood(
            x0,
            num_steps=64,
            endpoint_eps=endpoint_eps,
            trace_method='exact',
            solver='rk4',
            return_details=True)

        expected = expected_linear_log_prob(model, x0, endpoint_eps)
        expected_div = torch.full(
            (x0.shape[0],),
            model.num_tokens * model.vocab_size
            * float(model.scale.detach()) * (1.0 - endpoint_eps))
        torch.testing.assert_close(details['log_prob'], expected,
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(details['divergence_integral'],
                                   expected_div, rtol=1e-6, atol=1e-6)
        self.assertTrue(details['is_trace_exact'])
        self.assertEqual(details['trace_method'], 'exact')

    def test_hutchinson_matches_exact_for_diagonal_linear_field(self):
        model = LinearFieldFLM(scale=-0.15, num_tokens=2, vocab_size=4)
        x0 = torch.tensor([[0, 3], [2, 1]])

        exact = model.continuous_log_likelihood(
            x0,
            num_steps=32,
            endpoint_eps=0.2,
            trace_method='exact',
            solver='rk4')
        hutchinson = model.continuous_log_likelihood(
            x0,
            num_steps=32,
            endpoint_eps=0.2,
            trace_method='hutchinson',
            trace_samples=3,
            noise='rademacher',
            seed=123,
            solver='rk4')

        torch.testing.assert_close(hutchinson, exact,
                                   rtol=1e-6, atol=1e-6)

    def test_non_base_flm_config_raises(self):
        model = LinearFieldFLM(algo_name='flm_vdm')
        x0 = torch.tensor([[0, 1]])

        with self.assertRaisesRegex(NotImplementedError, 'base FLM'):
            model.continuous_log_likelihood(x0)

    def test_flmbase_stub_raises_for_other_flm_like_models(self):
        model = BaseOnlyFLM()

        with self.assertRaisesRegex(NotImplementedError, 'base FLM'):
            model.continuous_log_likelihood(torch.tensor([[0, 1]]))

    def test_restores_training_mode_and_preserves_parameter_grads(self):
        model = LinearFieldFLM(scale=0.1)
        model.train()
        model.scale.grad = torch.tensor(7.0)
        grad_before = model.scale.grad.detach().clone()
        x0 = torch.tensor([[0, 1]])

        _ = model.continuous_log_likelihood(
            x0,
            num_steps=8,
            endpoint_eps=0.25,
            trace_method='hutchinson',
            seed=5)

        self.assertTrue(model.training)
        torch.testing.assert_close(model.scale.grad, grad_before)


if __name__ == '__main__':
    unittest.main()
