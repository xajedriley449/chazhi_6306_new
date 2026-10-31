"""Collect existing retrospective results without selecting a best run."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def main():
    rows = []
    for folder in sorted((ROOT / 'output_6306_levels').iterdir()):
        summary = folder / 'evaluation/summary.json'
        config = folder / 'run_config.json'
        if not summary.exists() or not config.exists():
            continue
        cfg = json.loads(config.read_text(encoding='utf-8'))
        results = json.loads(summary.read_text(encoding='utf-8'))
        for level, methods in results.items():
            for method, values in methods.items():
                row = dict(run=folder.name, period=cfg.get('period'), task=cfg.get('task'),
                           seed=cfg.get('seed'), method=method, level=level,
                           events=values['events'], days=values['days'],
                           AP=values.get('cell_average_precision'), brier=values.get('brier'))
                for budget in (5,10,20):
                    item = values['top'+str(budget)]
                    row['recall_'+str(budget)] = item['event_recall']
                    row['area_'+str(budget)] = item['mean_alarm_area']
                rows.append(row)
    if not rows:
        raise RuntimeError('No evaluation results found')
    out = ROOT / 'output_6306_levels/thesis_tables'
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'all_results.csv', index=False, encoding='utf-8-sig')
    net = frame[frame.method.eq('convlstm')]
    if len(net):
        net.assign(order=net.run.str.startswith('shuffled_').map({True:'shuffled',False:'ordered'})).groupby(['period','task','level','order'])[['AP','recall_5','recall_10','recall_20']].agg(
            ['count','mean','min','max']).to_csv(out/'seed_summary.csv', encoding='utf-8-sig')
    print('Saved:', out)
    print('A/B remain separate. Seeds are repeated fits, not independent events.')
if __name__ == '__main__':
    main()
