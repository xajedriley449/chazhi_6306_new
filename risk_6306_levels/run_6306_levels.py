"""Fixed-epoch, chronological two-threshold spatial exceedance experiment."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from common import (PROJECT, FRACTIONS, load_data, fresh, save_json, digest, channels,
                    fit_scales, transform, alarm_weights, metrics, baselines, truth)

LEVELS = ['ge1e4', 'ge1e5']


def level_indices(task):
    return {'dual': [0, 1], 'low': [0], 'high': [1]}[task]


def select_period(samples, period):
    """A/B are expanding-history retrospective tests; no model selection on eval."""
    usable = samples[samples.usable]
    if period == 'development':
        train = usable[usable.split.eq('train')]
        evaluation = usable[usable.split.eq('val')]
    elif period == 'A':
        train = usable[usable.target_date < '2023-01-01']
        evaluation = usable[(usable.target_date >= '2023-01-01') & (usable.target_date < '2023-07-01')]
    else:
        train = usable[usable.target_date < '2023-07-01']
        evaluation = usable[usable.target_date >= '2023-07-01']
    ti, ei = train.target_index.to_numpy(int), evaluation.target_index.to_numpy(int)
    if not len(ti) or not len(ei) or ti.max() >= ei.min():
        raise ValueError('Empty or nonchronological train/evaluation period')
    return ti, ei


def validate_arrays(arr):
    y, n = arr['target'], arr['event_counts']
    if y.ndim != 4 or y.shape[1] != 2 or n.shape != y.shape:
        raise ValueError('Use prepare_6306_levels.py; expected [day,2,y,x] labels')
    if not np.array_equal(y, n > 0) or (n < 0).any() or (n[:, 1] > n[:, 0]).any():
        raise ValueError('Invalid cumulative event counts/labels')
    if not np.isfinite(n).all() or not np.equal(n, np.floor(n)).all():
        raise ValueError('Event counts must be finite integers')
    if not (y[:, 1] <= y[:, 0]).all():
        raise ValueError('High label must imply low label')


def identity(root):
    return {p: digest(Path(root)/p) for p in ('config.json', 'dataset.npz', 'sample_manifest.csv', 'daily_audit.csv')}


def code_identity():
    here = Path(__file__).parent
    return {p: digest(here/p) for p in ('model.py', 'common.py', 'run_6306_levels.py')}


def plot_examples(out, p, y, dates, arr, level):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = sorted(set([0, len(p)-1] + np.flatnonzero(y.sum((1, 2)))[:8].tolist()))
    for j in chosen:
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.pcolormesh(arr['x_edges'], arr['y_edges'], p[j], shading='flat', cmap='magma', vmin=0)
        iy, ix = np.nonzero(y[j])
        xc = (arr['x_edges'][:-1]+arr['x_edges'][1:])/2
        yc = (arr['y_edges'][:-1]+arr['y_edges'][1:])/2
        ax.scatter(xc[ix], yc[iy], marker='x', color='cyan', label='Observed positive cell')
        ax.set(title=f'{dates[j]} {level}', xlabel='X (m)', ylabel='Y (m)', aspect='equal')
        fig.colorbar(im, ax=ax, label='Model score (calibration unverified)')
        fig.tight_layout(); fig.savefig(out/f'{level}_{dates[j]}.png', dpi=140); plt.close(fig)


def report(out, arr, eval_ids, train_ids, length, model_scores=None, levels=(0, 1), figures=False):
    out.mkdir(parents=True, exist_ok=True)
    result, daily, cells, reliability = {}, [], [], []
    dates = arr['dates'][eval_ids]
    for j, level in enumerate(levels):
        label = LEVELS[level]
        single = dict(arr, target=arr['target'][:, level], event_counts=arr['event_counts'][:, level])
        maps = baselines(single, eval_ids, train_ids, length)
        if model_scores is not None:
            maps['convlstm'] = (model_scores[:, j], True)
        result[label] = {}
        y, n = single['target'][eval_ids], single['event_counts'][eval_ids]
        for name, (p, probability) in maps.items():
            result[label][name] = metrics(p, y, n, probability)
            for k, date in enumerate(dates):
                row = {'date': str(date), 'level': label, 'method': name, 'events': int(n[k].sum()),
                       'positive_cells': int(y[k].sum()), 'mean_score': float(p[k].mean())}
                weights = {f: alarm_weights(p[k], f) for f in FRACTIONS}
                for f, w in weights.items():
                    row[f'hit_events_top{int(100*f)}'] = float((w*n[k]).sum())
                    row[f'area_top{int(100*f)}'] = float(w.mean())
                daily.append(row)
                for iy, ix in zip(*np.nonzero(n[k])):
                    cells.append({'date': str(date), 'level': label, 'method': name,
                                  'iy': int(iy), 'ix': int(ix), 'events': int(n[k, iy, ix]),
                                  'score': float(p[k, iy, ix]),
                                  **{f'coverage_top{int(100*f)}': float(w[iy, ix]) for f, w in weights.items()}})
            if probability:
                for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
                    m = (p >= lo) & ((p <= hi) if hi == 1 else (p < hi))
                    reliability.append({'level': label, 'method': name, 'lo': lo, 'hi': hi,
                                        'cells': int(m.sum()), 'mean_score': float(p[m].mean()) if m.any() else None,
                                        'observed_frequency': float(y[m].mean()) if m.any() else None})
        if model_scores is not None and figures:
            plot_examples(out, model_scores[:, j], y, dates, arr, label)
    save_json(out/'summary.json', result)
    pd.DataFrame(daily).to_csv(out/'daily_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cells).to_csv(out/'positive_cell_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(reliability).to_csv(out/'reliability.csv', index=False)
    if model_scores is not None:
        if len(levels) == 2 and np.any(model_scores[:, 1] > model_scores[:, 0] + 1e-7):
            raise ValueError('Nested output violated')
        np.savez_compressed(out/'predictions.npz', scores=model_scores, targets=arr['target'][eval_ids][:, levels],
                            event_counts=arr['event_counts'][eval_ids][:, levels], dates=dates,
                            level_names=np.asarray([LEVELS[i] for i in levels]),
                            x_edges=arr['x_edges'], y_edges=arr['y_edges'])
    return result


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['baseline', 'train', 'evaluate', 'forecast'])
    p.add_argument('--data', type=Path, default=PROJECT/'input_6306_levels')
    p.add_argument('--output', type=Path)
    p.add_argument('--period', choices=['development', 'A', 'B'], default='development')
    p.add_argument('--task', choices=['dual', 'low', 'high'], default='dual')
    p.add_argument('--input-mode', choices=['all', 'energy', 'count'], default='all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='auto')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--forecast-date', help='YYYY-MM-DD, strictly after checkpoint training dates')
    p.add_argument('--figures', action='store_true')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.hidden, a.threads) < 1 or not np.isfinite([a.lr, a.weight_decay]).all() or a.lr <= 0 or a.weight_decay < 0:
        p.error('Invalid optimization arguments')
    if a.action in ('evaluate', 'forecast') and not a.checkpoint:
        p.error('--checkpoint required')
    if a.action in ('baseline', 'train') and a.checkpoint:
        p.error('Use evaluate or forecast for a saved checkpoint')
    if a.action == 'forecast' and not a.forecast_date:
        p.error('--forecast-date required')
    return a


def inference(torch, model, x, ids, length, scales, selected, batch_size, device):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            xx = np.stack([transform(x[i-length:i], scales, selected) for i in ids[start:start+batch_size]])
            result.append(torch.sigmoid(model(torch.from_numpy(xx).to(device))).cpu().numpy())
    return np.concatenate(result)


def main():
    a = arguments()
    cfg, arr, samples, _ = load_data(a.data)
    validate_arrays(arr)
    if cfg.get('thresholds_J') != [10000., 100000.]:
        raise ValueError('Unexpected thresholds')
    length = cfg['seq_len']
    train_ids, eval_ids = select_period(samples, a.period)
    data_identity = identity(a.data)
    checkpoint = None
    if a.action != 'baseline':
        try:
            import torch
            from torch import nn
            from torch.utils.data import TensorDataset, DataLoader
            from model import ConvLSTM
        except ImportError as exc:
            raise RuntimeError('Activate env2 and install requirements.txt') from exc
        torch.set_num_threads(a.threads)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device))
        if a.checkpoint:
            # Only load your own locally trained checkpoints (pickle format).
            try: checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
            except TypeError: checkpoint = torch.load(a.checkpoint, map_location='cpu')
            if checkpoint['identity'] != data_identity or checkpoint['code_identity'] != code_identity():
                raise ValueError('Checkpoint dataset or implementation differs')
            a.task, a.input_mode, a.hidden = checkpoint['task'], checkpoint['input_mode'], checkpoint['hidden']
            train_ids = np.asarray(checkpoint['train_ids'], dtype=int)
            eval_ids = np.asarray(checkpoint['eval_ids'], dtype=int)
            a.period = checkpoint['period']
            scales = np.asarray(checkpoint['scales'], dtype=np.float32)
        else:
            scales = fit_scales(arr['inputs'], train_ids, length)
        selected, levels = channels(a.input_mode), level_indices(a.task)
        prevalence = arr['target'][train_ids].mean((0, 2, 3))
        if np.any(prevalence[levels] == 0):
            raise ValueError('Training contains no positive cells for a requested level')
        model = ConvLSTM(len(selected), a.hidden, prevalence, a.task).to(device)
        if checkpoint: model.load_state_dict(checkpoint['model'])
    out = fresh(a.output or PROJECT/'output_6306_levels'/f'{a.action}_{a.period}_{a.task}_{a.input_mode}_seed{a.seed}')
    save_json(out/'run_config.json', {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                    'identity': data_identity, 'code_identity': code_identity(), 'dataset_config': cfg,
                                    'train_target_start': str(arr['dates'][train_ids[0]]),
                                    'train_target_end': str(arr['dates'][train_ids[-1]]),
                                    'evaluation_start': str(arr['dates'][eval_ids[0]]),
                                    'evaluation_end': str(arr['dates'][eval_ids[-1]]),
                                    'selection': 'fixed final epoch; no evaluation-driven stopping or alarm threshold selection',
                                    'alarm_policy': 'exact floor(fraction*cells), fractional ties including zero; expected coverage',
                                    'loss': 'equal average of unweighted marginal BCE across requested levels',
                                    'calibration': 'unverified; retrospective catalog experiment'})
    if a.action == 'baseline':
        report(out/'evaluation', arr, eval_ids, train_ids, length)
    elif a.action == 'train':
        xx = np.stack([transform(arr['inputs'][i-length:i], scales, selected) for i in train_ids])
        yy = arr['target'][train_ids][:, levels].astype(np.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(xx), torch.from_numpy(yy)),
                            batch_size=a.batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        logs = []
        for epoch in range(1, a.epochs+1):
            model.train(); sums = np.zeros(len(levels)); num = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad(set_to_none=True)
                per_level = nn.functional.binary_cross_entropy_with_logits(model(bx), by, reduction='none').mean((0, 2, 3))
                loss = per_level.mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
                sums += per_level.detach().cpu().numpy()*len(bx); num += len(bx)
            row = {'epoch': epoch, 'train_bce': float(sums.mean()/num),
                   **{f'train_bce_{LEVELS[level]}': float(sums[j]/num) for j, level in enumerate(levels)}}
            logs.append(row); pd.DataFrame(logs).to_csv(out/'loss_log.csv', index=False)
            print(row, flush=True)
        payload = {'model': model.state_dict(), 'task': a.task, 'input_mode': a.input_mode, 'hidden': a.hidden,
                   'scales': scales.tolist(), 'train_ids': train_ids.tolist(), 'eval_ids': eval_ids.tolist(),
                   'identity': data_identity, 'code_identity': code_identity(), 'period': a.period,
                   'epochs': a.epochs, 'seed': a.seed, 'prevalence': prevalence.tolist()}
        torch.save(payload, out/'last_model.pth')
        save_json(out/'training_summary.json', {k: v for k, v in payload.items() if k != 'model'})
        p = inference(torch, model, arr['inputs'], eval_ids, length, scales, selected, a.batch_size, device)
        report(out/'evaluation', arr, eval_ids, train_ids, length, p, levels, a.figures)
    elif a.action == 'evaluate':
        p = inference(torch, model, arr['inputs'], eval_ids, length, scales, selected, a.batch_size, device)
        report(out/'evaluation', arr, eval_ids, train_ids, length, p, levels, a.figures)
    else:
        date = pd.Timestamp(a.forecast_date).normalize()
        if str(date.date()) <= str(arr['dates'][train_ids[-1]]):
            raise ValueError('Forecast must follow all training targets')
        dates = pd.date_range(end=date-pd.Timedelta(days=1), periods=length).strftime('%Y-%m-%d').tolist()
        positions = {str(d): i for i, d in enumerate(arr['dates'])}
        if any(d not in positions for d in dates): raise ValueError('Required history unavailable')
        audit = pd.read_csv(a.data/'daily_audit.csv').set_index('date').loc[dates]
        bad = audit.monitoring_status.eq('missing') | audit.n_outside_grid.gt(0)
        bad |= audit.catalog_edge_day.map(truth) & audit.monitoring_status.ne('normal')
        if cfg['coverage'] == 'strict': bad |= audit.monitoring_status.ne('normal')
        if bad.any(): raise ValueError('Forecast history fails coverage rules')
        x = transform(arr['inputs'][[positions[d] for d in dates]], scales, selected)
        model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(torch.from_numpy(x[None]).to(device))).cpu().numpy()[0]
        np.savez_compressed(out/'forecast.npz', scores=p, level_names=np.asarray([LEVELS[i] for i in levels]),
                            target_date=np.array(str(date.date())), x_edges=arr['x_edges'], y_edges=arr['y_edges'])
    save_json(out/'completed.json', {'action': a.action, 'period': a.period, 'calibration': 'unverified'})
    print(f'Completed: {out}', flush=True)


if __name__ == '__main__':
    main()
