#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate existing scalar-energy checkpoints on the untouched 6306 test split.

Place this file beside pred_6306_xy_v4.py. This script does not train models.
It evaluates the linear and Softplus best checkpoints once on the test dates and
writes per-day metrics plus baseline summaries.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pred_6306_xy_v4 as core


def build_model(hidden, mode):
    t, nn = core.torch, core.nn

    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = core.Predictor(2, hidden, 'random')
            del self.encoder.head
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(hidden * 16, 32), nn.SiLU(), nn.Linear(32, 1)
            )
            nn.init.constant_(self.head[-1].bias, -2.0)

        def forward(self, x):
            b, _, _, h, w = x.shape
            hx = x.new_zeros((b, hidden, h, w))
            cx = t.zeros_like(hx)
            for j in range(x.shape[1]):
                i, f, o, g = self.encoder.gates(t.cat([x[:, j], hx], 1)).chunk(4, 1)
                cx = t.sigmoid(f) * cx + t.sigmoid(i) * t.tanh(g)
                hx = t.sigmoid(o) * t.tanh(cx)
            return self.head(self.pool(hx)).squeeze(1)

    return ScalarModel()


def summarize(frame):
    result = {}
    for label, mask in [
        ('all', np.ones(len(frame), dtype=bool)),
        ('high_event_days', frame['n_high'].to_numpy() > 0),
        ('no_high_event_days', frame['n_high'].to_numpy() == 0),
    ]:
        part = frame.loc[mask]
        result[label] = {
            'days': int(len(part)),
            'MAE_J': float(part['absolute_error_J'].mean()) if len(part) else None,
            'true_mean_J': float(part['true_total_J'].mean()) if len(part) else None,
            'pred_mean_J': float(part['pred_total_J'].mean()) if len(part) else None,
            'log_MAE': float(part['log_error'].mean()) if len(part) else None,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--linear-checkpoint', type=Path, required=True,
                   help='linear/best_model.pth from the completed diagnostic run')
    p.add_argument('--softplus-checkpoint', type=Path, required=True,
                   help='softplus/best_model.pth from the completed diagnostic run')
    p.add_argument('--data-dir', type=Path, default=core.PROJECT / 'input_6306_daily_xy')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()

    t = core.torch
    if t is None:
        raise RuntimeError('当前环境未安装 PyTorch')
    for path in [a.linear_checkpoint, a.softplus_checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(f'检查点不存在：{path}')
    device = t.device('cuda' if a.device == 'auto' and t.cuda.is_available()
                      else ('cpu' if a.device == 'auto' else a.device))
    if device.type == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('指定了 CUDA，但当前环境不可用')

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(
        data_dir=a.data_dir, seq_len=10, train_end='2022-01-01',
        test_start='2023-01-01', coverage='exploratory', input_mode='both'
    )
    cfg, manifest, events, data, sums, counts, edges, audit = core.load_catalog(settings, out)
    ids = core.samples(manifest, settings, out)
    test_ids = ids.get('test')
    if test_ids is None or len(test_ids) == 0:
        raise ValueError('测试集为空；请检查数据目录和时间划分')
    energy_ref = float(cfg['energy_ref'])
    targets = np.log10(1.0 + sums[:, 1] / energy_ref).astype(np.float32)

    all_summaries = {}
    for mode, ckpt_path in [
        ('linear', a.linear_checkpoint), ('softplus', a.softplus_checkpoint)
    ]:
        payload = t.load(ckpt_path, map_location='cpu', weights_only=False)
        saved_mode = payload.get('output_mode')
        if saved_mode != mode:
            raise ValueError(f'{ckpt_path} 标记为 {saved_mode!r}，期望 {mode!r}')
        if not np.isclose(float(payload['energy_ref']), energy_ref):
            raise ValueError(f'{mode} 检查点 energy_ref 与当前数据不一致')
        scales = payload['scales']
        hidden = int(payload['hidden'])

        class TestDataset(core.Dataset):
            def __init__(self):
                self.indices = test_ids
                self.maps = core.Sequences(data, self.indices, settings, scales)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, j):
                x, _ = self.maps[j]
                idx = self.indices[j]
                return x, t.tensor(targets[idx], dtype=t.float32)

        model = build_model(hidden, mode)
        model.load_state_dict(payload['model'], strict=True)
        model.to(device).eval()
        predictions = []
        with t.no_grad():
            loader = core.DataLoader(TestDataset(), batch_size=8, shuffle=False)
            for x, _ in loader:
                predictions.extend(model(x.to(device)).cpu().numpy().tolist())

        raw_log = np.asarray(predictions, dtype=np.float64)
        clipped_log = np.maximum(raw_log, 0.0)
        predicted_j = core.to_j(clipped_log, energy_ref)
        true_j = sums[test_ids, 1].astype(np.float64)
        truth_log = targets[test_ids].astype(np.float64)
        dates = manifest.iloc[test_ids]['date'].astype(str).to_numpy()
        n_high = counts[test_ids, 1].astype(int)
        baselines = {
            'scalar_convlstm': predicted_j,
            'zero': np.zeros(len(test_ids), dtype=np.float64),
            'training_mean': np.full(len(test_ids), float(sums[ids['train'], 1].mean())),
            'history_10day_mean': np.array(
                [sums[i - 10:i, 1].mean() for i in test_ids], dtype=np.float64
            ),
        }
        frames = []
        mode_summary = {}
        for name, pred_j in baselines.items():
            pred_log = np.log10(1.0 + pred_j / energy_ref)
            frame = pd.DataFrame({
                'date': dates, 'method': name, 'n_high': n_high,
                'positive_day': n_high > 0,
                'true_total_J': true_j, 'pred_total_J': pred_j,
                'total_absolute_error_J': np.abs(pred_j - true_j),
                'total_bias_J': pred_j - true_j,
                'log_error': np.abs(pred_log - truth_log),
            })
            if name == 'scalar_convlstm':
                frame['raw_log_prediction'] = raw_log
                frame['negative_raw_prediction'] = raw_log < 0
            frames.append(frame)
            mode_summary[name] = summarize(frame.rename(columns={
                'total_absolute_error_J': 'absolute_error_J'
            }))

        pd.concat(frames, ignore_index=True).to_csv(
            out / f'test_daily_metrics_{mode}.csv', index=False, encoding='utf-8-sig'
        )
        all_summaries[mode] = {
            'checkpoint': str(ckpt_path.resolve()),
            'output_mode': mode,
            'test_days': int(len(test_ids)),
            'positive_test_days': int((n_high > 0).sum()),
            'negative_raw_log_predictions': int((raw_log < 0).sum()),
            'raw_log_prediction_min': float(raw_log.min()),
            'raw_log_prediction_max': float(raw_log.max()),
            'metrics': mode_summary,
        }
        print(f'Finished {mode}: {len(test_ids)} test days', flush=True)

    report = {
        'test_split': 'target dates from 2023-01-01 onward, using the existing exploratory coverage policy',
        'data_audit': audit,
        'energy_ref_J': energy_ref,
        'negative_linear_log_policy': 'preserve raw log predictions in CSV; clip at zero only for physical energy conversion',
        'models': all_summaries,
        'selection_note': 'Report both test results; do not select/tune a model using these test metrics.',
    }
    core.save_json(out / 'test_comparison_summary.json', report)
    print(f'Results saved to: {out}', flush=True)


if __name__ == '__main__':
    main()
