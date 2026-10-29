import unittest
import torch
from models.temporal_encoders import build_temporal_encoder


class NoConvEncoderTest(unittest.TestCase):
    def test_ablation_and_padding(self):
        torch.manual_seed(42)
        model = build_temporal_encoder('td_conv_gru_no_conv', 8, hidden_dim=4, dropout=0).eval()
        self.assertFalse(any(isinstance(m, torch.nn.Conv1d) for m in model.modules()))
        x = torch.randn(2, 4, 8)
        lengths = torch.tensor([1, 3])
        delta = torch.zeros(2, 4)
        out = model(x, lengths, delta=delta)
        self.assertEqual(out.shape, (2, 8))
        altered = x.clone()
        altered[0, 1:] = 1000
        altered[1, 3:] = -1000
        torch.testing.assert_close(out, model(altered, lengths, delta=delta))
        self.assertFalse(torch.allclose(out, model(x, lengths, delta=delta + 2)))
        out.sum().backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_original_still_has_convolutions(self):
        model = build_temporal_encoder('td_conv_gru', 8, hidden_dim=4)
        self.assertTrue(any(isinstance(m, torch.nn.Conv1d) for m in model.modules()))


if __name__ == '__main__':
    unittest.main()
