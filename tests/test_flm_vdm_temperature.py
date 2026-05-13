import unittest

import torch

import algo


class FakeFLMVDM(algo.FLMVDM):
    def __init__(self, softmax_temperature=1.0):
        torch.nn.Module.__init__(self)
        self.softmax_temperature = softmax_temperature


class FLMVDMTemperatureTest(unittest.TestCase):
    def test_temperature_scales_logits_before_log_softmax(self):
        logits = torch.tensor([[[-1.0, 0.0, 2.0]]])
        temperature = 0.5
        model = FakeFLMVDM(softmax_temperature=temperature)

        actual = model._process_model_output(logits, None, None)
        capped = 30.0 * torch.tanh(logits / 30.0)
        expected = torch.log_softmax(capped / temperature, dim=-1)

        torch.testing.assert_close(actual, expected)

    def test_lower_temperature_makes_distribution_sharper(self):
        logits = torch.tensor([[[-1.0, 0.0, 2.0]]])

        cold = FakeFLMVDM(softmax_temperature=0.5)._process_model_output(
            logits, None, None).exp()
        base = FakeFLMVDM(softmax_temperature=1.0)._process_model_output(
            logits, None, None).exp()
        hot = FakeFLMVDM(softmax_temperature=2.0)._process_model_output(
            logits, None, None).exp()

        self.assertGreater(cold.max().item(), base.max().item())
        self.assertGreater(base.max().item(), hot.max().item())


if __name__ == '__main__':
    unittest.main()
