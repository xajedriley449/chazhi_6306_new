"""Run: python -m unittest discover -s risk_6306 -p test_core.py -v"""
import unittest
import importlib.util
import numpy as np
import pandas as pd
from common import alarm_weights, fit_scales, choose_threshold, threshold_metrics
from prepare_6306_spatial_risk import render


class CoreTests(unittest.TestCase):
    def test_energy_count_conservation_and_hard_labels(self):
        e = pd.DataFrame({'x': [0., 200., 99.], 'y': [0., 200., 100.],
                          'energy_J': [100., 100000., 200000.]})
        x, n = render(e, np.array([0., 100., 200.]), np.array([0., 100., 200.]), 50., 100000.)
        np.testing.assert_allclose(x.sum((1, 2)), [3, 100, 300000], atol=1e-6)
        self.assertEqual(n.sum(), 2)
        self.assertEqual(n[1, 1], 1)
        self.assertEqual(n[1, 0], 1)

    def test_ties_are_fractional_and_zero_map_does_not_alarm(self):
        w = alarm_weights(np.ones((10, 10)), .1)
        np.testing.assert_allclose(w, .1)
        self.assertEqual(alarm_weights(np.zeros((10, 10)), .1).sum(), 0.)
        score = np.arange(100.).reshape(10, 10)
        self.assertEqual(alarm_weights(score, .1).sum(), 10.)

    def test_normalization_excludes_target_and_future(self):
        x = np.ones((8, 3, 2, 2))
        x[4:] = 1e9
        s = fit_scales(x, np.array([3, 4]), 2)
        np.testing.assert_allclose(s, np.log(2), rtol=1e-6)

    def test_validation_threshold_allows_zero_alarm(self):
        p = np.full((3, 2, 2), .01, dtype=np.float32)
        y = np.zeros_like(p); y[0, 0, 0] = 1
        t = choose_threshold(p, y, .1)
        self.assertGreater(t, float(p.max()))
        self.assertEqual(threshold_metrics(p, y, y, t)['mean_alarm_area'], 0.)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_model_forward_backward(self):
        import torch
        from model import ConvLSTM
        model = ConvLSTM(3, 4, .01)
        x = torch.randn(2, 3, 3, 5, 7)
        z = model(x)
        self.assertEqual(tuple(z.shape), (2, 5, 7))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
        loss.backward()
        self.assertTrue(torch.isfinite(model.gates.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
