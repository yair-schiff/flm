import math
import unittest

import torch

import noise_schedules


class NoiseScheduleTest(unittest.TestCase):
    def test_linear_schedule_and_weight(self):
        schedule = noise_schedules.LinearGammaSchedule(
            gamma_min=-5.0, gamma_max=5.0)
        tau = torch.tensor([0.0, 0.5, 1.0])
        gamma, weight = schedule(tau)
        torch.testing.assert_close(
            gamma, torch.tensor([5.0, 0.0, -5.0]))
        torch.testing.assert_close(weight, torch.exp(-gamma) * 10.0)
        self.assertTrue(torch.all(weight > 0))

    def test_linear_alpha_schedule_makes_vp_alpha_linear(self):
        schedule = noise_schedules.LinearAlphaSchedule(
            latent_type='vp', t_min=0.2, t_max=0.6)
        tau = torch.linspace(0.2, 0.6, 5)
        gamma, weight = schedule(tau)
        alpha = torch.sigmoid(-gamma).sqrt()
        eps = torch.finfo(tau.dtype).eps
        expected_alpha = tau.clamp(eps, 1.0 - eps)

        torch.testing.assert_close(alpha, expected_alpha)
        torch.testing.assert_close(
            schedule.tau_from_snr(torch.exp(-gamma)), tau, atol=1e-6,
            rtol=1e-6)
        self.assertTrue(torch.all(weight > 0))

    def test_linear_alpha_schedule_makes_linear_interp_alpha_linear(self):
        schedule = noise_schedules.build_noise_schedule(
            'alpha_linear', vocab_size=16, t_min=0.2, t_max=0.6,
            latent_type='linear_interp')
        tau = torch.linspace(0.2, 0.6, 5)
        gamma, weight = schedule(tau)
        alpha = torch.sigmoid(-0.5 * gamma)
        eps = torch.finfo(tau.dtype).eps
        expected_alpha = tau.clamp(eps, 1.0 - eps)

        self.assertIsInstance(schedule, noise_schedules.LinearAlphaSchedule)
        torch.testing.assert_close(alpha, expected_alpha)
        self.assertTrue(torch.all(weight > 0))

    def test_linear_alpha_schedule_uses_tau_in_train_and_eval(self):
        schedule = noise_schedules.LinearAlphaSchedule(
            latent_type='linear_interp', t_min=0.2, t_max=0.6,
            val_t_min=0.0, val_t_max=0.9)
        train_tau = torch.tensor([0.2, 0.4, 0.6])
        train_gamma, _ = schedule(train_tau)
        train_alpha = torch.sigmoid(-0.5 * train_gamma)
        train_expected = train_tau.clamp(
            torch.finfo(train_tau.dtype).eps, 1.0 - torch.finfo(
                train_tau.dtype).eps)

        schedule.eval()
        val_tau = torch.tensor([0.0, 0.45, 0.9])
        val_gamma, _ = schedule(val_tau)
        val_alpha = torch.sigmoid(-0.5 * val_gamma)
        val_expected = val_tau.clamp(
            torch.finfo(val_tau.dtype).eps, 1.0 - torch.finfo(
                val_tau.dtype).eps)

        torch.testing.assert_close(train_alpha, train_expected)
        torch.testing.assert_close(val_alpha, val_expected)

    def test_linear_alpha_schedule_supports_alpha_range(self):
        schedule = noise_schedules.LinearAlphaSchedule(
            latent_type='vp', alpha_min=0.2, alpha_max=0.8)
        tau = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(
            schedule._alpha(tau), torch.tensor([0.2, 0.5, 0.8]))

        probe_tau = torch.tensor([0.25, 0.75])
        gamma, _ = schedule(probe_tau)
        torch.testing.assert_close(
            schedule.tau_from_snr(torch.exp(-gamma)), probe_tau,
            atol=1e-6, rtol=1e-6)

    def test_cosine_alpha_schedule_makes_vp_alpha_cosine(self):
        schedule = noise_schedules.CosineAlphaSchedule(
            latent_type='vp', t_min=0.2, t_max=0.6)
        tau = torch.linspace(0.2, 0.55, 5)
        gamma, weight = schedule(tau)
        alpha = torch.sigmoid(-gamma).sqrt()
        eps = torch.finfo(tau.dtype).eps
        unit_tau = tau.clamp(0.0, 1.0)
        expected_alpha = torch.sin(0.5 * math.pi * unit_tau).clamp(
            eps, 1.0 - eps)

        torch.testing.assert_close(alpha, expected_alpha)
        torch.testing.assert_close(
            schedule.tau_from_snr(torch.exp(-gamma)), tau, atol=1e-6,
            rtol=1e-6)
        self.assertTrue(torch.all(weight > 0))
        _, endpoint_weight = schedule(torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.isfinite(endpoint_weight).all())
        self.assertTrue(torch.all(endpoint_weight >= 0))

    def test_cosine_alpha_schedule_makes_linear_interp_alpha_cosine(self):
        schedule = noise_schedules.build_noise_schedule(
            'alpha_cosine', vocab_size=16, t_min=0.2, t_max=0.6,
            latent_type='linear_interp')
        tau = torch.linspace(0.2, 0.55, 5)
        gamma, weight = schedule(tau)
        alpha = torch.sigmoid(-0.5 * gamma)
        eps = torch.finfo(tau.dtype).eps
        unit_tau = tau.clamp(0.0, 1.0)
        expected_alpha = torch.sin(0.5 * math.pi * unit_tau).clamp(
            eps, 1.0 - eps)

        self.assertIsInstance(schedule, noise_schedules.CosineAlphaSchedule)
        torch.testing.assert_close(alpha, expected_alpha)
        torch.testing.assert_close(
            schedule.tau_from_snr(torch.exp(-gamma)), tau, atol=1e-6,
            rtol=1e-6)
        self.assertTrue(torch.all(weight > 0))

    def test_ode_alpha_schedule_endpoints_and_monotonicity(self):
        schedule = noise_schedules.ODEAlphaSchedule(
            alpha_min=1e-5, alpha_max=0.9, latent_type='linear_interp',
            eps=1e-12)
        tau = torch.tensor([0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(
            schedule._alpha(tau),
            torch.tensor([1e-5, 0.9], dtype=torch.float64),
            atol=1e-12, rtol=1e-9)

        grid = torch.linspace(0.0, 1.0, 128, dtype=torch.float64)
        alpha = schedule._alpha(grid)
        self.assertTrue(torch.all(alpha[1:] > alpha[:-1]))

    def test_ode_alpha_schedule_linear_interp_snr_prime_identity(self):
        schedule = noise_schedules.ODEAlphaSchedule(
            alpha_min=1e-5, alpha_max=0.9, latent_type='linear_interp',
            eps=1e-12)
        tau = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
        alpha = schedule._alpha(tau)
        _, weight = schedule(tau)
        expected = 2.0 * schedule._lambda(tau) / (1.0 - alpha)

        torch.testing.assert_close(weight, expected, atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.all(weight > 0))

    def test_ode_alpha_schedule_vp_snr_prime_identity(self):
        schedule = noise_schedules.ODEAlphaSchedule(
            alpha_min=1e-5, alpha_max=0.9, latent_type='vp', eps=1e-12)
        tau = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
        alpha = schedule._alpha(tau)
        _, weight = schedule(tau)
        expected = 2.0 * schedule._lambda(tau) / (1.0 + alpha).square()

        torch.testing.assert_close(weight, expected, atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.all(weight > 0))

    def test_ode_alpha_schedule_inverse(self):
        schedule = noise_schedules.ODEAlphaSchedule(
            alpha_min=1e-5, alpha_max=0.9, latent_type='linear_interp',
            eps=1e-12)
        tau = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
        gamma, _ = schedule(tau)

        torch.testing.assert_close(
            schedule.tau_from_snr(torch.exp(-gamma)), tau,
            atol=1e-9, rtol=1e-9)

    def test_ode_alpha_schedule_validates_bounds(self):
        invalid_kwargs = [
            {'alpha_min': 0.0, 'alpha_max': 0.9},
            {'alpha_min': -1e-5, 'alpha_max': 0.9},
            {'alpha_min': 1e-5, 'alpha_max': 1.0},
            {'alpha_min': 0.9, 'alpha_max': 0.5},
            {'alpha_min': 1e-5, 'alpha_max': 0.9,
             'val_alpha_min': 0.0, 'val_alpha_max': 0.8},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    noise_schedules.ODEAlphaSchedule(**kwargs)

    def test_ode_alpha_schedule_factory(self):
        schedule = noise_schedules.build_noise_schedule(
            'ode_alpha', vocab_size=16, alpha_min=1e-5, alpha_max=0.9,
            latent_type='linear_interp', eps=1e-12, bisect_iters=32)

        self.assertIsInstance(schedule, noise_schedules.ODEAlphaSchedule)
        self.assertEqual(schedule.bisect_iters, 32)

    def test_piecewise_alpha_schedule_maps_knots_and_slopes(self):
        schedule = noise_schedules.PiecewiseLinearAlphaSchedule(
            latent_type='vp',
            alpha_knots=[0.0, 0.2, 0.6, 0.8, 1.0],
            tau_knots=[0.0, 0.1, 0.5, 0.9, 1.0])
        tau = torch.tensor([0.0, 0.05, 0.1, 0.7, 0.9, 1.0])
        expected_alpha = torch.tensor([0.0, 0.1, 0.2, 0.7, 0.8, 1.0])
        torch.testing.assert_close(schedule._alpha(tau), expected_alpha)

        slopes = schedule._dalpha_dtau(torch.tensor([0.05, 0.7]))
        torch.testing.assert_close(slopes, torch.tensor([2.0, 0.5]))

        probe_tau = torch.tensor([0.05, 0.7])
        gamma, weight = schedule(probe_tau)
        torch.testing.assert_close(
            schedule.tau_from_snr(torch.exp(-gamma)), probe_tau,
            atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(weight > 0))

    def test_piecewise_alpha_schedule_accepts_tau_density(self):
        schedule = noise_schedules.build_noise_schedule(
            'piecewise_alpha', vocab_size=16, latent_type='linear_interp',
            alpha_knots=[0.0, 0.2, 0.6, 0.8, 1.0],
            tau_density=[0.5, 1.0, 2.0, 1.0])
        slopes = schedule._dalpha_dtau(torch.tensor([0.04, 0.6]))

        self.assertIsInstance(
            schedule, noise_schedules.PiecewiseLinearAlphaSchedule)
        self.assertGreater(slopes[0].item(), slopes[1].item())

    def test_snr_power_is_finite_near_boundaries(self):
        C = 17.276323318481445
        p = 0.43169814348220825
        schedule = noise_schedules.SNRPowerSchedule(C=C, p=p, eps=1e-6)
        tau = torch.tensor([0.0, 1e-8, 0.5, 1.0 - 1e-8, 1.0])
        gamma, weight = schedule(tau)
        self.assertTrue(torch.isfinite(gamma).all())
        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue(torch.all(weight > 0))

        L_mid = -math.log1p(-0.5)
        expected_mid = -math.log(C) - p * math.log(L_mid)
        self.assertAlmostEqual(gamma[2].item(), expected_mid, places=6)

    def test_importance_sampling_uses_constant_snr_span_weight(self):
        schedule = noise_schedules.SNRPowerSchedule(C=2.0, p=1.0, eps=1e-6)
        unit = torch.tensor([0.0, 0.25, 1.0])
        tau, gamma, weight = schedule.importance_sample(unit, t_max=0.5)
        snr_min = schedule.snr_at(0.0, unit)
        snr_max = schedule.snr_at(0.5, unit)
        expected_snr_span = snr_max - snr_min
        torch.testing.assert_close(
            weight, expected_snr_span.expand_as(unit))
        self.assertTrue(torch.isfinite(tau).all())
        self.assertTrue(torch.isfinite(gamma).all())

    def test_importance_sampling_respects_interval(self):
        schedule = noise_schedules.LinearGammaSchedule(
            gamma_min=-5.0, gamma_max=5.0)
        unit = torch.tensor([0.0, 0.5, 1.0])
        tau, gamma, weight = schedule.importance_sample(
            unit, t_min=0.2, t_max=0.6)
        snr_min = schedule.snr_at(0.2, unit)
        snr_max = schedule.snr_at(0.6, unit)
        expected_snr = snr_min + unit * (snr_max - snr_min)
        torch.testing.assert_close(tau, schedule.tau_from_snr(expected_snr))
        torch.testing.assert_close(gamma, -torch.log(expected_snr))
        torch.testing.assert_close(weight, (snr_max - snr_min).expand_as(unit))
        self.assertTrue(torch.all(tau >= 0.2))
        self.assertTrue(torch.all(tau <= 0.6))

    def test_learned_vdm_schedule_is_monotone_at_init(self):
        schedule = noise_schedules.LearnedVDMSchedule(
            gamma_min=-5.0, gamma_max=5.0, hidden_size=4)
        tau = torch.linspace(0.0, 1.0, 11)
        gamma, weight = schedule(tau)
        self.assertGreater(gamma[0].item(), gamma[-1].item())
        self.assertTrue(torch.all(gamma[:-1] >= gamma[1:]))
        self.assertTrue(torch.all(weight > 0))

    def test_argmax_uncertainty_lut_endpoints(self):
        schedule = noise_schedules.ArgmaxUncertaintySchedule(
            vocab_size=16, gamma_min=-5.0, gamma_max=5.0,
            n_points=256, n_gh=16)
        tau = torch.tensor([0.0, 1.0])
        gamma, weight = schedule(tau)
        torch.testing.assert_close(
            gamma, torch.tensor([5.0, -5.0]), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.all(weight > 0))

    def test_argmax_uncertainty_pchip_stays_within_gamma_bounds(self):
        schedule = noise_schedules.ArgmaxUncertaintySchedule(
            vocab_size=256, gamma_min=-3.0, gamma_max=20.0,
            n_points=512, n_gh=16, interpolation='pchip')
        tau = torch.linspace(0.0, 1.0, 2048)
        gamma, weight = schedule(tau)
        self.assertGreaterEqual(gamma.min().item(), -3.0)
        self.assertLessEqual(gamma.max().item(), 20.0)
        self.assertTrue(torch.all(gamma[:-1] >= gamma[1:]))
        self.assertTrue(torch.all(weight > 0))

    def test_argmax_uncertainty_accepts_legacy_cubic_interpolation(self):
        schedule = noise_schedules.ArgmaxUncertaintySchedule(
            vocab_size=16, gamma_min=-5.0, gamma_max=5.0,
            n_points=256, n_gh=16, interpolation='cubic')
        tau = torch.tensor([0.0, 0.5, 1.0])
        gamma, weight = schedule(tau)
        self.assertTrue(torch.isfinite(gamma).all())
        self.assertTrue(torch.all(gamma >= -5.0))
        self.assertTrue(torch.all(gamma <= 5.0))
        self.assertTrue(torch.all(weight > 0))


if __name__ == '__main__':
    unittest.main()
