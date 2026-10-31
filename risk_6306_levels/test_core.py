"""Run: python risk_6306_levels/test_core.py"""
import importlib.util
import unittest
import numpy as np
import pandas as pd
from common import alarm_weights, fit_scales, metrics
from prepare_6306_levels import render
from run_6306_levels import select_period, validate_arrays


class DataTests(unittest.TestCase):
    def test_labels_boundaries_mass(self):
        e = pd.DataFrame({'x': [0., 100., 50., 25.], 'y': [0., 100., 50., 25.],
                          'energy_J': [9999., 10000., 100000., 30000.]})
        x, n = render(e, np.array([0., 50., 100.]), np.array([0., 50., 100.]), 20., 100000.)
        np.testing.assert_allclose(x.sum((1, 2)), [4, 49999, 100000], rtol=1e-8)
        self.assertEqual(n[0].sum(), 3); self.assertEqual(n[1].sum(), 1)
        self.assertEqual(n[0, 1, 1], 2)
        self.assertTrue((n[1] <= n[0]).all())
        validate_arrays({'target': (n[None] > 0).astype('uint8'), 'event_counts': n[None]})

    def test_exact_area_including_zero_ties(self):
        for score in [np.zeros((10, 10)), np.ones((10, 10)), np.arange(100).reshape(10, 10)]:
            w = alarm_weights(score, .1)
            self.assertAlmostEqual(w.sum(), 10.)
            self.assertTrue(((w >= 0) & (w <= 1)).all())
        self.assertAlmostEqual(alarm_weights(np.zeros((10, 10)), .1)[0, 0], .1)

    def test_causal_scales(self):
        x = np.ones((10, 3, 2, 2))
        expected = fit_scales(x, np.array([3, 4]), 3)
        x[4:] = 1e10  # targets and future are not input history for these samples
        np.testing.assert_array_equal(fit_scales(x, np.array([3, 4]), 3), expected)

    def test_periods_and_exclusions(self):
        d = pd.DataFrame({'usable': [True, True, True, False, True],
                          'target_date': ['2021-12-31','2022-12-31','2023-06-30','2023-07-01','2023-07-02'],
                          'target_index': [0, 1, 2, 3, 4], 'split': ['train','val','test','test','test']})
        ta, ea = select_period(d, 'A'); tb, eb = select_period(d, 'B')
        np.testing.assert_array_equal(ta, [0, 1]); np.testing.assert_array_equal(ea, [2])
        np.testing.assert_array_equal(tb, [0, 1, 2]); np.testing.assert_array_equal(eb, [4])
        self.assertFalse(set(ea) & set(eb))

    def test_no_positive_evaluation_is_reported(self):
        z = np.zeros((2, 10, 10))
        m = metrics(z, z, z, True)
        self.assertIsNone(m['cell_average_precision'])
        self.assertIsNone(m['top10']['event_recall'])
        self.assertAlmostEqual(m['top10']['mean_alarm_area'], .1)

    def test_reject_non_nested_counts(self):
        n = np.zeros((1, 2, 2, 2), dtype=int); n[0, 1, 0, 0] = 1
        with self.assertRaises(ValueError):
            validate_arrays({'target': n > 0, 'event_counts': n})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class ModelTests(unittest.TestCase):
    def test_nested_probabilities_and_gradients(self):
        import torch
        from model import nested_logits
        a = torch.tensor([[-100., 100.], [100., -100.], [0., 0.], [5., 2.]], requires_grad=True)
        z = nested_logits(a[:, :, None, None])
        p = torch.sigmoid(z)[:, :, 0, 0]
        torch.testing.assert_close(p[:, 1], torch.sigmoid(a[:, 0])*torch.sigmoid(a[:, 1]), atol=1e-6, rtol=1e-5)
        self.assertTrue(bool((p[:, 1] <= p[:, 0]+1e-7).all()))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
        loss.backward(); self.assertTrue(bool(torch.isfinite(a.grad).all()))

    def test_all_tasks_training_step_and_roundtrip(self):
        import torch
        from model import ConvLSTM
        torch.set_num_threads(1)
        for task, outputs in [('dual', 2), ('low', 1), ('high', 1)]:
            model = ConvLSTM(3, 4, (.1, .02), task)
            x = torch.rand(2, 3, 3, 4, 5)
            z = model(x); self.assertEqual(tuple(z.shape), (2, outputs, 4, 5))
            y = torch.zeros_like(z); y[0, :, 0, 0] = 1
            opt = torch.optim.AdamW(model.parameters())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward(); opt.step()
            other = ConvLSTM(3, 4, (.1, .02), task); other.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(x), other(x))


if __name__ == '__main__':
    unittest.main(verbosity=2)
