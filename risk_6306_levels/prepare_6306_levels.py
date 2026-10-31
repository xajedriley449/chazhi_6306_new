"""Build three historical channels and two nested next-day exceedance labels.
Run from project root: python risk_6306/prepare_6306_levels.py
Source: existing input_6306_daily_xy events, manifest, config; no PNG/NPZ decoding.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from common import PROJECT, fresh, save_json, digest, truth


def render(events, xe, ye, sigma, threshold):
    """Gaussian probability mass integrated over each cell; normalize within domain.
    Same spatial kernel for counts and energies. Kernel is a representation, not hazard.
    """
    x = np.zeros((3, len(ye)-1, len(xe)-1), dtype=np.float64)
    counts = np.zeros((2, *x.shape[1:]), dtype=np.int32)
    for r in events.itertuples(index=False):
        wx = np.diff(ndtr((xe-float(r.x))/sigma))
        wy = np.diff(ndtr((ye-float(r.y))/sigma))
        k = np.outer(wy, wx)
        total = k.sum()
        if total <= 0:
            raise FloatingPointError('核质量为零，请检查坐标或带宽')
        k /= total
        x[0] += k
        c = 2 if r.energy_J >= threshold else 1
        x[c] += r.energy_J*k
        ix = min(len(xe)-2, np.searchsorted(xe, r.x, side='right')-1)
        iy = min(len(ye)-2, np.searchsorted(ye, r.y, side='right')-1)
        for level, cutoff in enumerate((10000., 100000.)):
            if r.energy_J >= cutoff:
                counts[level, iy, ix] += 1
    sums = [len(events), events.loc[events.energy_J < threshold, 'energy_J'].sum(),
            events.loc[events.energy_J >= threshold, 'energy_J'].sum()]
    np.testing.assert_allclose(x.sum((1, 2)), sums, rtol=1e-9, atol=1e-6)
    return x, counts


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PROJECT/'input_6306_daily_xy')
    p.add_argument('--output', type=Path, default=PROJECT/'input_6306_levels')
    p.add_argument('--cell-m', type=float, default=100., help='Candidate maximum cell width, not verified location accuracy')
    p.add_argument('--sigma-m', type=float, default=50., help='Historical input smoothing only')
    p.set_defaults(threshold=100000.)
    p.add_argument('--seq-len', type=int, default=10)
    p.add_argument('--train-end', default='2022-01-01', help='exclusive target-date cutoff')
    p.add_argument('--test-start', default='2023-01-01')
    p.add_argument('--coverage', choices=['exploratory', 'strict'], default='exploratory')
    p.add_argument('--reuse-samples', type=Path, help='Optional prior sample_manifest.csv; retain its usable dates and splits')
    a = p.parse_args()
    if not np.isfinite([a.cell_m, a.sigma_m, a.threshold]).all() or min(a.cell_m, a.sigma_m, a.threshold) <= 0 or a.seq_len < 1:
        p.error('cell-m/sigma-m/threshold must be finite positive; seq-len >=1')
    for k in ['train_end', 'test_start']:
        setattr(a, k, pd.Timestamp(getattr(a, k)).strftime('%Y-%m-%d'))
    if a.train_end >= a.test_start:
        p.error('test-start must follow train-end')
    return a


def main():
    a = arguments()
    source = a.source.resolve()
    cfg0 = json.loads((source/'dataset_config.json').read_text(encoding='utf-8-sig'))
    if pd.Timestamp(cfg0['train_end']) > pd.Timestamp(a.train_end):
        raise ValueError('源空间范围使用了模型训练截止后的事件，请先重建源数据')
    m = pd.read_csv(source/'daily_manifest.csv', keep_default_na=False).sort_values('date').reset_index(drop=True)
    m['date'] = pd.to_datetime(m.date).dt.strftime('%Y-%m-%d')
    if m.date.duplicated().any() or m.date.tolist() != pd.date_range(m.date.iloc[0], m.date.iloc[-1]).strftime('%Y-%m-%d').tolist():
        raise ValueError('源日清单必须覆盖完整连续日历且无重复')
    if not m.monitoring_status.isin(['normal', 'missing', 'unknown']).all():
        raise ValueError('非法监测状态')
    e = pd.read_csv(source/'events_selected.csv')
    required = ['event_id', 'date', 'x', 'y', 'energy_J']
    if not set(required).issubset(e.columns):
        raise ValueError(f'事件表缺少列：{required}')
    if e.event_id.duplicated().any():
        raise ValueError('清洗事件表仍存在重复event_id')
    e['date'] = pd.to_datetime(e.date).dt.strftime('%Y-%m-%d')
    if not e.date.isin(m.date).all():
        raise ValueError('事件日期不在日清单中')
    if not np.isfinite(e[['x', 'y', 'energy_J']]).all().all() or (e.energy_J <= 0).any():
        raise ValueError('事件表含非法能量/坐标')
    xmin, xmax, ymin, ymax = map(float, cfg0['bounds_xy'])
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmax <= xmin or ymax <= ymin:
        raise ValueError('非法源边界')
    nx, ny = int(np.ceil((xmax-xmin)/a.cell_m)), int(np.ceil((ymax-ymin)/a.cell_m))
    xe, ye = np.linspace(xmin, xmax, nx+1), np.linspace(ymin, ymax, ny+1)
    area = np.outer(np.diff(ye), np.diff(xe))
    inside = e.x.between(xmin, xmax) & e.y.between(ymin, ymax)
    groups = {d: g for d, g in e.groupby('date')}
    inputs, counts, daily = [], [], []
    for r in m.itertuples(index=False):
        g = groups.get(r.date, e.iloc[:0])
        good = g[g.x.between(xmin, xmax) & g.y.between(ymin, ymax)]
        if int(r.n_events) != len(g) or int(r.n_outside_grid) != len(g)-len(good):
            raise ValueError(f'{r.date}: 事件表和日清单数量不一致')
        if int(r.n_high) != int((g.energy_J >= float(cfg0['threshold'])).sum()):
            raise ValueError(f'{r.date}:源高能计数不一致')
        if r.monitoring_status == 'missing' and len(g):
            raise ValueError(f'{r.date}:missing日却存在事件')
        x, n = render(good, xe, ye, a.sigma_m, a.threshold)
        # Units: events/hectare and J/hectare, not probability or physical strain energy.
        inputs.append((x*10000./area).astype(np.float32))
        counts.append(n)
        daily.append({'date': r.date, 'monitoring_status': r.monitoring_status,
                      'n_events': len(g), 'n_outside_grid': len(g)-len(good),
                      'n_high': int((g.energy_J >= a.threshold).sum()),
                      'catalog_edge_day': truth(getattr(r, 'catalog_edge_day', False))})
    d = pd.DataFrame(daily)
    reuse = None
    if a.reuse_samples:
        reuse = pd.read_csv(a.reuse_samples, keep_default_na=False)
        if not {'target_date', 'split', 'usable', 'input_start', 'input_end'}.issubset(reuse.columns):
            raise ValueError('旧样本清单缺少日期、划分、可用状态或输入边界')
        reuse['target_date'] = pd.to_datetime(reuse.target_date).dt.strftime('%Y-%m-%d')
        if reuse.target_date.duplicated().any():
            raise ValueError('旧样本日期重复')
        reuse = reuse.set_index('target_date')
    records = []
    for i, r in d.iterrows():
        reasons = []
        split = 'train' if r.date < a.train_end else ('val' if r.date < a.test_start else 'test')
        if i < a.seq_len:
            reasons.append('insufficient_history')
        else:
            b = d.iloc[i-a.seq_len:i+1]
            if (b.monitoring_status == 'missing').any(): reasons.append('monitoring_missing')
            if a.coverage == 'strict' and (b.monitoring_status != 'normal').any(): reasons.append('unconfirmed_coverage')
            if (b.n_outside_grid > 0).any(): reasons.append('outside_grid')
            if (b.catalog_edge_day & b.monitoring_status.ne('normal')).any(): reasons.append('unconfirmed_catalog_edge')
        start = d.iloc[i-a.seq_len].date if i >= a.seq_len else ''
        end = d.iloc[i-1].date if i else ''
        if reuse is not None:
            if r.date not in reuse.index:
                reasons.append('not_in_old_manifest')
            else:
                old = reuse.loc[r.date]
                if old['split'] != split: raise ValueError(f'{r.date}:旧清单时间划分不一致')
                if not truth(old.usable): reasons.append('old_manifest_excluded')
                elif old.input_start != start or old.input_end != end:
                    raise ValueError(f'{r.date}:旧样本历史长度/边界不一致')
        records.append({'target_index': i, 'target_date': r.date, 'input_start': start, 'input_end': end,
                        'split': split, 'usable': not reasons, 'reason': ';'.join(reasons), 'n_high': r.n_high})
    s = pd.DataFrame(records)
    out = fresh(a.output)
    n = np.stack(counts)
    assert (n[:, 1] <= n[:, 0]).all()
    np.savez_compressed(out/'dataset.npz', inputs=np.stack(inputs), target=(n > 0).astype(np.uint8),
                        event_counts=n, dates=d.date.to_numpy(dtype='U10'), x_edges=xe, y_edges=ye)
    s.to_csv(out/'sample_manifest.csv', index=False, encoding='utf-8-sig')
    d.to_csv(out/'daily_audit.csv', index=False, encoding='utf-8-sig')
    e[~inside].to_csv(out/'out_of_bounds.csv', index=False, encoding='utf-8-sig')
    split = {}
    for name in ['train', 'val', 'test']:
        ids = s.loc[s.usable & s.split.eq(name), 'target_index'].to_numpy(int)
        split[name] = {'samples': len(ids), 'levels': {}}
        for j, label in enumerate(('ge1e4', 'ge1e5')):
            v = n[ids, j]
            split[name]['levels'][label] = {
                'events': int(v.sum()),
                'event_days': int((v.sum((1, 2)) > 0).sum()),
                'positive_cells': int((v > 0).sum()),
                'cell_prevalence': float((v > 0).mean()) if len(ids) else None}
    c = {'version': 2, 'thresholds_J': [10000., 100000.], 'level_order': ['ge1e4', 'ge1e5'], 'source': str(source), 'source_config': cfg0,
         'threshold_J': a.threshold, 'seq_len': a.seq_len, 'sigma_m': a.sigma_m,
         'requested_max_cell_m': a.cell_m, 'cell_size_m': [float(xe[1]-xe[0]), float(ye[1]-ye[0])],
         'grid_shape_yx': [ny, nx], 'bounds_xy': [xmin, xmax, ymin, ymax],
         'channels': ['all_event_count_per_ha', 'low_energy_J_per_ha', 'high_energy_J_per_ha'],
         'train_end': a.train_end, 'test_start': a.test_start, 'coverage': a.coverage,
         'interpretation': 'exploratory catalog occurrence' if a.coverage == 'exploratory' else 'confirmed monitoring occurrence',
         'target': 'two cumulative hard occupancy labels, E>=1e4 and E>=1e5, next calendar day',
         'kernel': 'Gaussian integrated over cells; per-event in-domain mass normalized; input only',
         'spatial_scale_status': 'candidate experiment parameters; not validated engineering resolution',
         'source_hashes': {p: digest(source/p) for p in ['events_selected.csv', 'daily_manifest.csv', 'dataset_config.json']},
         'reused_sample_manifest_sha256': digest(a.reuse_samples) if a.reuse_samples else None,
         'dataset_sha256': digest(out/'dataset.npz'), 'samples_sha256': digest(out/'sample_manifest.csv'),
         'daily_audit_sha256': digest(out/'daily_audit.csv')}
    save_json(out/'config.json', c)
    save_json(out/'split_summary.json', split)
    print(json.dumps(split, ensure_ascii=False, indent=2))
    print(f'Saved: {out}; grid={ny}x{nx}; actual cell={c["cell_size_m"]} m')


if __name__ == '__main__':
    main()

