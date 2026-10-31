"""Collect positive-cell diagnostics without selecting favorable examples."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    frames=[]
    for p in sorted((ROOT/'output_6306_levels').glob('*/evaluation/positive_cell_diagnostics.csv')):
        frame=pd.read_csv(p)
        frame.insert(0,'run',p.parents[1].name)
        frames.append(frame)
    if not frames:
        raise RuntimeError('No positive-cell diagnostics')
    out=ROOT/'output_6306_levels/thesis_tables'
    out.mkdir(exist_ok=True)
    pd.concat(frames,ignore_index=True).to_csv(out/'all_positive_cells.csv',index=False,encoding='utf-8-sig')
    print('Saved:',out/'all_positive_cells.csv')
    print('Rows describe occupied cells, not unique event IDs; repeated runs are not independent events.')
if __name__=='__main__':
    main()
