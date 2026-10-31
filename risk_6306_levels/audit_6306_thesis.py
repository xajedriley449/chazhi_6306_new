"""Audit chronological samples and nested labels using the frozen dataset."""
from common import PROJECT, load_data
from run_6306_levels import validate_arrays, select_period
import pandas as pd
def main():
    cfg, arr, samples, _ = load_data(PROJECT/'input_6306_levels')
    validate_arrays(arr)
    good = samples[samples.usable]
    assert (pd.to_datetime(good.input_end) < pd.to_datetime(good.target_date)).all()
    out = PROJECT/'output_6306_levels/thesis_audit'
    out.mkdir(parents=True, exist_ok=True)
    samples.groupby(['split','usable','reason'], dropna=False).size().to_csv(out/'sample_counts.csv')
    rows = []
    for period in ('development','A','B'):
        train, evaluation = select_period(samples, period)
        for name, ids in [('train',train),('evaluation',evaluation)]:
            for level in range(2):
                n = arr['event_counts'][ids,level]
                rows.append(dict(period=period,partition=name,level=level,days=len(ids),
                                 events=int(n.sum()),event_days=int((n.sum((1,2))>0).sum())))
    pd.DataFrame(rows).to_csv(out/'period_counts.csv',index=False)
    print('Audit passed:',out)
    print('Coverage:',cfg['coverage'],'; monitoring and blasting provenance require external records.')
if __name__ == '__main__':
    main()
