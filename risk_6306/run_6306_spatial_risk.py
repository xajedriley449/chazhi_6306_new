"""Train / evaluate / baseline / forecast a single high-energy spatial occurrence model.
All defaults are relative to this folder's parent (the existing project root).
PyTorch is imported only for train/evaluate/forecast; baseline does not require it.
"""
import argparse
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from common import (PROJECT, fresh, save_json, digest, load_data, channels, fit_scales, transform,
                    baselines, evaluate_maps, choose_threshold, threshold_metrics)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate', 'baseline', 'forecast'])
    p.add_argument('--data', type=Path, default=PROJECT/'input_6306_spatial_risk')
    p.add_argument('--output', type=Path, help='Must be empty. Default output_6306_spatial_risk/<mode>_seed<seed> for train')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--input-mode', choices=['all', 'count', 'energy'], default='all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-alarm-area', type=float, default=0.10, help='Validation global area-time budget for threshold selection')
    p.add_argument('--forecast-date', help='YYYY-MM-DD; uses only the preceding seq-len days')
    a = p.parse_args()
    if min(a.hidden, a.epochs, a.patience, a.batch_size, a.threads) < 1 or not 0 < a.max_alarm_area < 1:
        p.error('positive model/training sizes required; alarm area in (0,1)')
    if not np.isfinite([a.lr, a.weight_decay]).all() or a.lr <= 0 or a.weight_decay < 0:
        p.error('invalid learning rate/weight decay')
    if a.action in ['evaluate', 'forecast'] and not a.checkpoint:
        p.error('evaluate/forecast require --checkpoint')
    if a.action == 'forecast' and not a.forecast_date:
        p.error('forecast requires --forecast-date')
    if a.action != 'train' and a.output is None:
        p.error('baseline/evaluate/forecast require a new --output directory')
    return a


def figure(path, p, y, xe, ye, date):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    extent = [0, xe[-1]-xe[0], 0, ye[-1]-ye[0]]
    # Display range is explicitly the map maximum, not a danger grading standard.
    im = axes[0].imshow(p, origin='lower', extent=extent, aspect='equal', vmin=0,
                        vmax=max(float(p.max()), 1e-7), cmap='YlOrRd')
    axes[0].set_title(f'Predicted occurrence; max={p.max():.3g}')
    fig.colorbar(im, ax=axes[0], label='Sigmoid score (calibration unverified)')
    if y is None:
        axes[1].text(.5, .5, 'Truth unavailable', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Future observation unavailable')
    else:
        axes[1].imshow(y, origin='lower', extent=extent, aspect='equal', vmin=0, vmax=1, cmap='Greys')
        axes[1].set_title('Observed high-energy occupied cells')
    for ax in axes: ax.set(xlabel='Relative X (m)', ylabel='Relative Y (m)')
    fig.suptitle(str(date)); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def inference(torch, model, x, ids, length, scales, selected, batch, device):
    model.eval(); predictions = []
    with torch.no_grad():
        for j in range(0, len(ids), batch):
            block = np.stack([transform(x[i-length:i], scales, selected) for i in ids[j:j+batch]])
            z = model(torch.from_numpy(block).to(device))
            if not torch.isfinite(z).all(): raise FloatingPointError('Non-finite model logits')
            predictions.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(predictions)


def validate_thresholds(maps, y, events, budget):
    result = {}
    for name, (p, _) in maps.items():
        t = choose_threshold(p, y, budget)
        result[name] = {'threshold': t, 'validation': threshold_metrics(p, y, events, t)}
    return result


def report(out, maps, arr, ids, thresholds, split):
    out.mkdir(parents=True, exist_ok=True)
    summaries = evaluate_maps(out, maps, arr, ids, arr['dates'])
    frozen = {}
    for name, (p, _) in maps.items():
        frozen[name] = threshold_metrics(p, arr['target'][ids], arr['event_counts'][ids], thresholds[name]['threshold'])
    save_json(out/'frozen_threshold_metrics.json', frozen)
    save_json(out/'evaluation_context.json', {'split': split,
              'test_status': 'previously inspected historical period; descriptive recheck' if split == 'test' else 'validation used for model/threshold selection',
              'scores': 'sigmoid model and empirical baseline occurrence scores; calibration must be checked',
              'top_area': 'daily fixed area budget; fractional ties are expected coverage, not a deployed binary alarm',
              'global_threshold': 'chosen using validation only; frozen in other periods'})
    for name, stats in summaries.items():
        print(name, 'AP=', stats['cell_average_precision'], 'top10=', stats['top10'])


def main():
    a = arguments()
    cfg, arr, samples, ids = load_data(a.data)
    length = cfg['seq_len']
    if not len(ids['train']) or not arr['target'][ids['train']].any():
        raise ValueError('No usable high-energy training targets. Inspect split_summary.json')
    if a.action != 'forecast' and (not len(ids['val']) or not arr['target'][ids['val']].any()):
        raise ValueError('Validation has no usable high-energy targets; cannot choose thresholds/model reliably')
    out = fresh(a.output or PROJECT/'output_6306_spatial_risk'/f'{a.input_mode}_seed{a.seed}')
    identity = {'dataset': cfg['dataset_sha256'], 'samples': cfg['samples_sha256'], 'config': digest(a.data/'config.json')}
    save_json(out/'run_config.json', {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                     'identity': identity, 'dataset_config': cfg,
                                     'script_sha256': digest(__file__), 'common_sha256': digest(Path(__file__).with_name('common.py')),
                                     'loss': 'unweighted BCE, natural chronological targets, no positive oversampling'})
    if a.action == 'baseline':
        maps = baselines(arr, ids['val'], ids['train'], length)
        thresholds = validate_thresholds(maps, arr['target'][ids['val']], arr['event_counts'][ids['val']], a.max_alarm_area)
        save_json(out/'thresholds.json', thresholds)
        report(out/'val', maps, arr, ids['val'], thresholds, 'val')
        if a.split == 'test':
            if not len(ids['test']): raise ValueError('Empty test set')
            report(out/'test', baselines(arr, ids['test'], ids['train'], length), arr, ids['test'], thresholds, 'test')
        return
    try:
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader
        from model import ConvLSTM
    except ImportError as exc:
        raise RuntimeError('PyTorch is required for train/evaluate/forecast. Use your existing env2 environment.') from exc
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device))
    if device.type == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    checkpoint = None
    if a.checkpoint:
        try:
            checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(a.checkpoint, map_location='cpu')
        if checkpoint['identity'] != identity: raise ValueError('Checkpoint data/config mismatch')
        if checkpoint['model_source_sha256'] != digest(Path(__file__).with_name('model.py')):
            raise ValueError('模型实现已改变，请使用训练时的model.py')
        selected = checkpoint['selected_channels']; hidden = checkpoint['hidden']; scales = checkpoint['scales']
    else:
        selected = channels(a.input_mode); hidden = a.hidden
        scales = fit_scales(arr['inputs'], ids['train'], length)
    prevalence = float(arr['target'][ids['train']].mean())
    model = ConvLSTM(len(selected), hidden, prevalence).to(device)
    if checkpoint: model.load_state_dict(checkpoint['model'])
    print(f'Device={device}; selected={selected}; train prevalence={prevalence:.6g}', flush=True)
    if a.action == 'train':
        # Eager materialization is small at 100m cells and avoids Windows worker import issues.
        loaders = {}
        for split in ['train', 'val']:
            xx = np.stack([transform(arr['inputs'][i-length:i], scales, selected) for i in ids[split]])
            yy = arr['target'][ids[split]].astype(np.float32)
            loaders[split] = DataLoader(TensorDataset(torch.from_numpy(xx), torch.from_numpy(yy)),
                                        batch_size=a.batch_size, shuffle=split == 'train', num_workers=0)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        best, stale, logs = float('inf'), 0, []
        for epoch in range(1, a.epochs+1):
            row = {'epoch': epoch}
            for split in ['train', 'val']:
                model.train(split == 'train'); total, num = 0., 0
                with torch.set_grad_enabled(split == 'train'):
                    for xx, yy in loaders[split]:
                        xx, yy = xx.to(device), yy.to(device)
                        if split == 'train': optimizer.zero_grad(set_to_none=True)
                        loss = loss_fn(model(xx), yy)
                        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
                        if split == 'train':
                            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                        total += float(loss.detach())*len(xx); num += len(xx)
                row[f'{split}_bce'] = total/num
            logs.append(row); pd.DataFrame(logs).to_csv(out/'loss_log.csv', index=False)
            payload = {'model': model.state_dict(), 'selected_channels': selected, 'hidden': hidden,
                       'scales': scales, 'identity': identity, 'epoch': epoch, 'seed': a.seed,
                       'validation_bce': row['val_bce'], 'model_source_sha256': digest(Path(__file__).with_name('model.py'))}
            if row['val_bce'] < best:
                best, stale = row['val_bce'], 0
                torch.save(payload, out/'best_model.pth')
            else: stale += 1
            torch.save(payload, out/'last_model.pth')
            print(f'Epoch {epoch}: train={row["train_bce"]:.6g} val={row["val_bce"]:.6g}', flush=True)
            if stale >= a.patience: break
        try: checkpoint = torch.load(out/'best_model.pth', map_location='cpu', weights_only=False)
        except TypeError: checkpoint = torch.load(out/'best_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        maps = baselines(arr, ids['val'], ids['train'], length)
        maps['convlstm'] = (inference(torch, model, arr['inputs'], ids['val'], length, scales, selected, a.batch_size, device), True)
        thresholds = validate_thresholds(maps, arr['target'][ids['val']], arr['event_counts'][ids['val']], a.max_alarm_area)
        checkpoint['thresholds'] = thresholds
        torch.save(checkpoint, out/'best_model.pth')
        save_json(out/'thresholds.json', thresholds)
        save_json(out/'training_summary.json', {'best_epoch': checkpoint['epoch'], 'best_val_bce': best,
                   'epochs_completed': len(logs), 'natural_sampling': True, 'calibration': 'unverified',
                   'prevalence': prevalence, 'selected_channels': selected,
                   'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__)})
        split = 'val'
    elif a.action == 'forecast':
        date = pd.Timestamp(a.forecast_date)
        expected = pd.date_range(end=date-pd.Timedelta(days=1), periods=length).strftime('%Y-%m-%d').tolist()
        positions = {str(d): i for i, d in enumerate(arr['dates'])}
        if any(d not in positions for d in expected): raise ValueError('Forecast history unavailable')
        ix = [positions[d] for d in expected]
        audit = pd.read_csv(a.data/'daily_audit.csv').set_index('date').loc[expected]
        from common import truth
        bad = audit.monitoring_status.eq('missing') | audit.n_outside_grid.gt(0)
        bad |= audit.catalog_edge_day.map(truth) & audit.monitoring_status.ne('normal')
        if cfg['coverage'] == 'strict': bad |= audit.monitoring_status.ne('normal')
        if bad.any(): raise ValueError(f'Unusable forecast history: {audit.index[bad].tolist()}')
        xx = transform(arr['inputs'][ix], scales, selected)
        model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(torch.from_numpy(xx[None]).to(device)))[0].cpu().numpy()
        np.savez_compressed(out/'forecast.npz', score=p, x_edges=arr['x_edges'], y_edges=arr['y_edges'], target_date=np.array(str(date.date())))
        figure(out/'forecast.png', p, None, arr['x_edges'], arr['y_edges'], f'{date.date()} forecast; truth unavailable')
        save_json(out/'completed.json', {'action': 'forecast', 'target_date': str(date.date()), 'coverage': cfg['coverage']})
        return
    else:
        split = a.split
        if not len(ids[split]): raise ValueError('Empty evaluation split')
        if 'thresholds' not in checkpoint: raise ValueError('Checkpoint lacks frozen validation thresholds; use best_model.pth')
        thresholds = checkpoint['thresholds']
        maps = baselines(arr, ids[split], ids['train'], length)
        maps['convlstm'] = (inference(torch, model, arr['inputs'], ids[split], length, scales, selected, a.batch_size, device), True)
    report(out/split, maps, arr, ids[split], thresholds, split)
    p = maps['convlstm'][0]
    np.savez_compressed(out/split/'predictions.npz', scores=p, targets=arr['target'][ids[split]],
                        event_counts=arr['event_counts'][ids[split]], dates=arr['dates'][ids[split]],
                        x_edges=arr['x_edges'], y_edges=arr['y_edges'])
    # Reliability bins report observed cell frequencies; they do not recalibrate the model.
    flat, target = p.ravel(), arr['target'][ids[split]].ravel()
    rows = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (flat >= lo) & ((flat <= hi) if hi == 1 else (flat < hi))
        rows.append({'bin_lower': lo, 'bin_upper': hi, 'cells': int(mask.sum()),
                     'mean_score': float(flat[mask].mean()) if mask.any() else None,
                     'observed_frequency': float(target[mask].mean()) if mask.any() else None})
    pd.DataFrame(rows).to_csv(out/split/'reliability.csv', index=False)
    show = sorted(set([0, len(p)//2, len(p)-1] + np.flatnonzero(arr['target'][ids[split]].sum((1, 2))).tolist()))
    (out/split/'figures').mkdir(exist_ok=True)
    for j in show:
        figure(out/split/'figures'/f'{arr["dates"][ids[split][j]]}.png', p[j], arr['target'][ids[split][j]],
               arr['x_edges'], arr['y_edges'], arr['dates'][ids[split][j]])
    save_json(out/'completed.json', {'action': a.action, 'split': split, 'coverage': cfg['coverage']})
    print(f'Completed: {out}')


if __name__ == '__main__':
    main()
