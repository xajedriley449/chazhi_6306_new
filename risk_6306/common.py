"""6306 high-energy spatial occurrence experiment. Python >=3.10."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.05, 0.10, 0.20)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fresh(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'输出目录非空，请换一个目录：{path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def truth(v):
    return str(v).strip().lower() in ('true', '1', '1.0')


def load_data(root):
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    if (digest(root/'dataset.npz') != cfg['dataset_sha256'] or
        digest(root/'sample_manifest.csv') != cfg['samples_sha256'] or
        digest(root/'daily_audit.csv') != cfg['daily_audit_sha256']):
        raise ValueError('数据文件已改变，请重新准备到新目录。')
    with np.load(root/'dataset.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    samples = pd.read_csv(root/'sample_manifest.csv', keep_default_na=False)
    samples['usable'] = samples.usable.map(truth)
    if not np.isfinite(arrays['inputs']).all() or (arrays['inputs'] < 0).any():
        raise ValueError('输入有非法值')
    if not np.isin(arrays['target'], [0, 1]).all():
        raise ValueError('标签必须是硬二值占据标签')
    ids = {s: samples.loc[samples.usable & samples.split.eq(s), 'target_index'].to_numpy(int)
           for s in ('train', 'val', 'test')}
    return cfg, arrays, samples, ids


def channels(mode):
    return {'all': [0, 1, 2], 'count': [0], 'energy': [1, 2]}[mode]


def fit_scales(x, ids, length):
    history = np.unique(np.concatenate([np.arange(i-length, i) for i in ids]))
    z = np.log1p(x[history].astype(np.float64))
    # Fit only actual training histories. No per-day normalization.
    return np.maximum(z.max(axis=(0, 2, 3)), 1e-6).astype(np.float32)


def transform(x, scales, selected):
    return (np.log1p(x[:, selected]).astype(np.float32) / scales[selected, None, None]).copy()


def alarm_weights(score, fraction):
    """Fractional ties = expected alarm under uniform tie breaking; never over budget."""
    flat = np.asarray(score, dtype=float).ravel()
    if not np.isfinite(flat).all() or (flat < 0).any():
        raise ValueError('报警分数必须非负且有限')
    k = int(np.floor(len(flat)*fraction))
    w = np.zeros_like(flat)
    if k == 0:
        return w.reshape(score.shape)
    positive = flat > 0
    if positive.sum() <= k:
        w[positive] = 1
    else:
        cutoff = np.partition(flat, -k)[-k]
        above, ties = flat > cutoff, flat == cutoff
        w[above] = 1
        w[ties] = (k-above.sum())/ties.sum()
    return w.reshape(score.shape)


def metrics(score, target, events, probabilities=False):
    from sklearn.metrics import average_precision_score
    y = np.asarray(target).ravel().astype(int)
    p = np.asarray(score).ravel().astype(float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('非法预测')
    result = {'days': len(target), 'positive_days': int(target.reshape(len(target), -1).any(1).sum()),
              'events': int(events.sum()), 'positive_cells': int(y.sum()),
              'cell_average_precision': float(average_precision_score(y, p)) if y.sum() else None,
              'cell_prevalence': float(y.mean())}
    if probabilities:
        if (p > 1).any():
            raise ValueError('概率超过1')
        q = np.clip(p, 1e-7, 1-1e-7)
        result.update(brier=float(np.mean((p-y)**2)),
                      log_loss=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))))
    negative = ~target.reshape(len(target), -1).any(1)
    for f in FRACTIONS:
        w = np.stack([alarm_weights(a, f) for a in score])
        key = f'top{int(f*100)}'
        result[key] = {'event_recall': float((w*events).sum()/events.sum()) if events.sum() else None,
                       'positive_cell_recall': float((w*target).sum()/y.sum()) if y.sum() else None,
                       'mean_alarm_area': float(w.mean()),
                       'negative_day_mean_alarm_area': float(w[negative].mean()) if negative.any() else None}
    return result


def choose_threshold(p, y, max_area):
    # Global time-space threshold selected on validation only. Budget includes zero-event days.
    p = np.asarray(p, dtype=np.float64)
    candidates = np.unique(np.r_[np.quantile(p, np.linspace(0, 1, 501)), np.nextafter(float(p.max()), np.inf)])
    best, best_key = None, None
    for t in candidates:
        a = p >= t
        area = float(a.mean())
        if area > max_area+1e-12:
            continue
        tp = int((a*y).sum())
        recall = tp/y.sum() if y.sum() else 0.
        key = (recall, -area, float(t))
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    return best


def threshold_metrics(p, y, events, t):
    a = np.asarray(p, dtype=np.float64) >= t
    tp, fp, fn = int((a*y).sum()), int((a*(1-y)).sum()), int(((~a)*y).sum())
    neg = ~y.reshape(len(y), -1).any(1)
    return {'threshold': t, 'mean_alarm_area': float(a.mean()),
            'event_recall': float((a*events).sum()/events.sum()) if events.sum() else None,
            'cell_precision': tp/(tp+fp) if tp+fp else None,
            'cell_recall': tp/(tp+fn) if tp+fn else None,
            'negative_day_alarm_rate': float(a[neg].reshape(neg.sum(), -1).any(1).mean()) if neg.any() else None,
            'tp_cells': tp, 'fp_cells': fp, 'fn_cells': fn}


def baselines(arr, ids, train_ids, length):
    x, y = arr['inputs'], arr['target']
    # Beta(0.5,0.5) smoothing; train-only spatial occupancy climatology.
    climate = (y[train_ids].sum(0)+0.5)/(len(train_ids)+1.)
    history = np.stack([(y[i-length:i].sum(0)+climate)/(length+1.) for i in ids])
    count = np.stack([x[i-length:i, 0].mean(0) for i in ids])
    energy = np.stack([x[i-length:i, 1:3].sum(1).mean(0) for i in ids])
    tc = np.stack([x[i-length:i, 0].mean(0) for i in train_ids])
    te = np.stack([x[i-length:i, 1:3].sum(1).mean(0) for i in train_ids])
    cs, es = max(float(np.log1p(tc).max()), 1e-6), max(float(np.log1p(te).max()), 1e-6)
    return {'training_climatology': (np.broadcast_to(climate, history.shape).copy(), True),
            'history_high_occupancy': (history, True),
            'history_count': (count, False), 'history_energy': (energy, False),
            'history_count_energy_equal': (0.5*np.log1p(count)/cs+0.5*np.log1p(energy)/es, False)}


def evaluate_maps(out, maps, arr, ids, dates, prefix=''):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, rows = {}, []
    y, n = arr['target'][ids], arr['event_counts'][ids]
    for name, (p, probability) in maps.items():
        summaries[name] = metrics(p, y, n, probability)
        for j, i in enumerate(ids):
            row = {'date': str(dates[i]), 'method': name, 'high_events': int(n[j].sum()),
                   'positive_cells': int(y[j].sum()), 'mean_score': float(p[j].mean())}
            for f in FRACTIONS:
                w = alarm_weights(p[j], f)
                row[f'hit_events_top{int(100*f)}'] = float((w*n[j]).sum())
                row[f'area_top{int(100*f)}'] = float(w.mean())
            rows.append(row)
    save_json(out/f'{prefix}summary.json', summaries)
    pd.DataFrame(rows).to_csv(out/f'{prefix}daily_metrics.csv', index=False, encoding='utf-8-sig')
    return summaries
