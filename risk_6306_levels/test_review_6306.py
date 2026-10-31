"""Numerical and synthetic end-to-end checks; no real experiment is changed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from common import alarm_weights, digest, save_json
import run_6306_levels as runner
import review_6306_thesis as review


class ReviewTests(unittest.TestCase):
    def test_every_area_matches_reference_including_ties(self):
        rng = np.random.default_rng(17)
        scores = rng.integers(0, 4, (5, 3, 7)).astype(float)
        scores[0] = 0
        events = rng.integers(0, 4, scores.shape)
        curves = review.coverage_curves(scores, events)
        for day in range(len(scores)):
            for k in range(22):
                # nextafter avoids a floating-point floor falling just below integer k
                fraction = np.nextafter(k/21, np.inf) if k < 21 else 1.
                wanted = (alarm_weights(scores[day], fraction)*events[day]).sum()
                self.assertAlmostEqual(curves[day, k], wanted)
        np.testing.assert_allclose(curves[:, -1], events.sum((1, 2)))
        self.assertTrue((np.diff(curves, axis=1) >= -1e-12).all())

    def test_calendar_blocks_do_not_compress_missing_dates(self):
        dates = ["2023-01-01", "2023-01-02", "2023-01-29"]
        values = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(review.block_totals(dates, values, 14), [[4, 6], [5, 6]])

    def test_paired_bootstrap_strata_and_zero_event_replicates(self):
        parts = [(["2023-01-01", "2023-01-15"], np.array([1., 0]), np.array([1, 0])),
                 (["2023-07-01", "2023-07-15"], np.array([-1., 0]), np.array([2, 0]))]
        result = review.paired_bootstrap(parts, 14, 1000, 42)
        self.assertEqual(result["events"], 3)
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["valid_replicates"]+result["zero_event_replicates"], 1000)
        self.assertGreater(result["zero_event_replicates"], 0)
        self.assertLessEqual(result["ci_low"], result["delta"])
        self.assertGreaterEqual(result["ci_high"], result["delta"])
        self.assertEqual(result, review.paired_bootstrap(parts, 14, 1000, 42))

    def test_metric_mismatch_is_rejected(self):
        p = np.ones((2, 3, 7))
        n = np.zeros_like(p); n[0, 0, 0] = 1
        a = runner.metrics(p, n > 0, n, True)
        b = json.loads(json.dumps(a)); b["events"] += 1
        with self.assertRaises(ValueError):
            review.assert_metric_match(a, b, "synthetic")

    def test_full_24_fit_postprocessing_on_synthetic_files(self):
        # Fixture checks alignment, report generation, figures and provenance together.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, runs, out = root/"data", root/"runs", root/"review"
            data.mkdir(); runs.mkdir()
            dates = np.array(["2022-12-28", "2022-12-29", "2022-12-30", "2022-12-31",
                              "2023-01-01", "2023-01-15", "2023-02-01", "2023-06-30",
                              "2023-07-01", "2023-07-15", "2023-08-01", "2023-08-15"])
            rng = np.random.default_rng(6306)
            n = np.zeros((12, 2, 3, 7), dtype=int)
            for d in range(12):
                n[d, 0, d % 3, d % 7] = 1+(d % 2)
                if d % 2 == 0:
                    n[d, 1, d % 3, d % 7] = 1
            arr = dict(inputs=rng.random((12, 3, 3, 7)), event_counts=n,
                       target=(n > 0).astype("uint8"), dates=dates,
                       x_edges=np.arange(8)*100., y_edges=np.arange(4)*100.)
            np.savez_compressed(data/"dataset.npz", **arr)
            samples = pd.DataFrame(dict(target_date=dates, target_index=np.arange(12),
                                        usable=np.arange(12) >= 2,
                                        split=["train"]*4+["test"]*8))
            samples.to_csv(data/"sample_manifest.csv", index=False)
            pd.DataFrame(dict(date=dates)).to_csv(data/"daily_audit.csv", index=False)
            cfg = dict(seq_len=2, thresholds_J=[10000., 100000.],
                       dataset_sha256=digest(data/"dataset.npz"),
                       samples_sha256=digest(data/"sample_manifest.csv"),
                       daily_audit_sha256=digest(data/"daily_audit.csv"))
            save_json(data/"config.json", cfg)
            data_id, core_id = runner.identity(data), runner.code_identity()
            for period in ("A", "B"):
                ti, ei = runner.select_period(samples, period)
                folder = runs/f"baselines_{period}"; folder.mkdir()
                save_json(folder/"run_config.json", dict(identity=data_id, code_identity=core_id))
                save_json(folder/"completed.json", dict(action="baseline", period=period))
                runner.report(folder/"evaluation", arr, ei, ti, 2)
                for order, task in (("ordered", "low"), ("ordered", "high"),
                                    ("ordered", "dual"), ("shuffled", "dual")):
                    for seed in review.SEEDS:
                        folder = runs/f"{'train' if order == 'ordered' else 'shuffled'}_{period}_{task}_all_seed{seed}"
                        folder.mkdir()
                        code_id = dict(core_id)
                        if order == "shuffled":
                            code_id["run_6306_shuffled.py"] = digest(Path(runner.__file__).parent/"run_6306_shuffled.py")
                        save_json(folder/"run_config.json", dict(identity=data_id, code_identity=code_id,
                                  period=period, task=task, seed=seed, epochs=15, input_mode="all"))
                        save_json(folder/"training_summary.json", dict(train_ids=ti.tolist(), eval_ids=ei.tolist()))
                        save_json(folder/"completed.json", dict(action="train", period=period))
                        low = rng.random((len(ei), 3, 7)).astype("float32")
                        high = low*rng.random(low.shape).astype("float32")
                        levels = runner.level_indices(task)
                        p = np.stack([low, high], axis=1)[:, levels]
                        runner.report(folder/"evaluation", arr, ei, ti, 2, p, levels)
            with patch.object(sys, "argv", ["review", "--data", str(data), "--runs", str(runs),
                                           "--output", str(out), "--bootstrap", "100"]):
                review.main()
            self.assertEqual(review.read_json(out/"completed.json")["fits"], 24)
            self.assertEqual(len(pd.read_csv(out/"verification.csv")), 56)
            self.assertEqual(len(list((out/"figures").glob("coverage_*.png"))), 6)
            self.assertGreater(len(list((out/"cases").glob("*.png"))), 0)
            combined = pd.read_csv(out/"metrics_with_combined.csv")
            self.assertTrue(combined.loc[combined.period.eq("A+B"), "AP"].isna().all())
            for b in review.BUDGETS:
                for (level, method, seed), group in combined.groupby(["level", "method", "seed"]):
                    both = group[group.period.eq("A+B")].iloc[0]
                    separate = group[~group.period.eq("A+B")]
                    self.assertAlmostEqual(both[f"recall_{b}"],
                                           separate[f"hits_{b}"].sum()/separate.events.sum())
            self.assertIn("按阶段计算的结果解读", (out/"REPORT.md").read_text(encoding="utf-8"))
            self.assertIn("source_sha256", review.read_json(out/"provenance.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
