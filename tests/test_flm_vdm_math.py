import unittest

import torch

import utils


class FLMVDMMathTest(unittest.TestCase):
    def test_vp_alpha_sigma_constraint(self):
        gamma = torch.linspace(-5.0, 5.0, 101)
        alpha, sigma = utils.vdm_alpha_sigma_from_gamma(gamma, 'vp')
        self.assertTrue(torch.allclose(
            alpha.square() + sigma.square(),
            torch.ones_like(gamma),
            atol=1e-6))
        self.assertTrue(torch.allclose(
            alpha.square() / sigma.square(),
            torch.exp(-gamma),
            atol=1e-5))

    def test_linear_alpha_sigma_constraint(self):
        gamma = torch.linspace(-5.0, 5.0, 101)
        alpha, sigma = utils.vdm_alpha_sigma_from_gamma(gamma, 'linear')
        self.assertTrue(torch.allclose(
            alpha + sigma,
            torch.ones_like(gamma),
            atol=1e-6))
        self.assertTrue(torch.allclose(
            alpha.square() / sigma.square(),
            torch.exp(-gamma),
            atol=1e-5))

    def test_linear_half_gamma_relation(self):
        gamma = torch.linspace(-5.0, 5.0, 101)
        alpha, sigma = utils.vdm_alpha_sigma_from_gamma(gamma, 'linear')
        reconstructed_gamma = 2.0 * (sigma.log() - alpha.log())
        self.assertTrue(torch.allclose(
            reconstructed_gamma,
            gamma,
            atol=1e-6))

    def test_tau_progress_lut_endpoints(self):
        lut_tau2gamma, lut_gamma2tau = utils.build_vp_luts(
            K=64,
            gamma_min=-4.0,
            gamma_max=4.0,
            n_points=400,
        )
        tau = torch.tensor([0.0, 1.0])
        gamma = utils.tau_to_gamma(tau, lut_tau2gamma)
        self.assertTrue(torch.allclose(
            gamma,
            torch.tensor([4.0, -4.0]),
            atol=1e-4))
        interior_tau = torch.tensor([0.25, 0.5, 0.75])
        interior_gamma = utils.tau_to_gamma(interior_tau, lut_tau2gamma)
        roundtrip_tau = utils.gamma_to_tau(interior_gamma, lut_gamma2tau)
        self.assertTrue(torch.allclose(interior_tau, roundtrip_tau,
                                       atol=5e-3))

    def test_uniform_gamma_elbo_weight_integral(self):
        gamma_min = -3.0
        gamma_max = 5.0
        n = 20000
        unit_midpoints = (torch.arange(n, dtype=torch.float32) + 0.5) / n
        gamma = gamma_min + unit_midpoints * (gamma_max - gamma_min)
        width = gamma_max - gamma_min
        q_gamma = 1.0 / width
        weights = torch.exp(-gamma) / q_gamma
        estimate = weights.mean()
        target = torch.exp(torch.tensor(-gamma_min)) - torch.exp(
            torch.tensor(-gamma_max))
        self.assertLess((estimate - target).abs().item(), 2e-3)


if __name__ == '__main__':
    unittest.main()
