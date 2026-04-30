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


if __name__ == '__main__':
    unittest.main()
