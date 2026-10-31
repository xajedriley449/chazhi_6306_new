#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-only fixed-sample fitting. Requires pred_6306_xy_v3.py alongside.
Runs random-head baseline and the same model with spatial KL.
Default spatial weight 0.01 is an experimental starting value.
All controls reuse samples, gate initialization, seed, scales and step budget.
This measures memorization/optimization, not forecasting generalization.
"""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import numpy as np
import pandas as pd
import pred_6306_xy_v3 as core


def select_samples(data, ids, counts, sums, n, seed):
    pos=ids[counts[ids,1]>0]; neg=ids[counts[ids,1]==0]
    if len(pos)<n or len(neg)<n: raise ValueError('训练期正/负样本不足')
    # Spread positive choices over energy ranks; train data only.
    pos=pos[np.argsort(sums[pos,1],kind='stable')]
    chosen=pos[np.linspace(0,len(pos)-1,n,dtype=int)]
    negatives=np.random.default_rng(seed).choice(neg,n,replace=False)
    return np.concatenate([chosen,negatives])


def grad_norm(module):
    return float(sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**0.5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=core.PROJECT/'input_6306_daily_xy')
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--per-class',type=int,default=4)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--variants',nargs='+',choices=['random','random_spatial'],default=['random','random_spatial'])
    p.add_argument('--spatial-weight',type=float,default=0.01)
    p.add_argument('--dates',nargs='+',help='可选固定目标日期；必须全部属于可用训练期且同时有正负样本')
    args=p.parse_args()
    if not np.isfinite(args.spatial_weight) or args.spatial_weight<0: p.error('spatial-weight必须非负')
    if min(args.steps,args.save_every,args.per_class,args.hidden)<1 or not np.isfinite(args.lr) or args.lr<=0: p.error('参数必须为正')
    if len(set(args.variants))!=len(args.variants): p.error('variants不能重复')
    if core.torch is None: raise RuntimeError('请在安装PyTorch的env2运行')
    t=core.torch
    device=t.device('cuda' if args.device=='auto' and t.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    out=(args.output or core.PROJECT/'output_6306_diagnose_v3'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_seed{args.seed}').resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    a=SimpleNamespace(data_dir=args.data_dir,seq_len=10,train_end='2022-01-01',test_start='2023-01-01',coverage='exploratory',input_mode='both',background_weight=1.,total_weight=.1)
    cfg,m,events,data,sums,counts,edges,audit=core.load_catalog(a,out)
    ids=core.samples(m,a,out)
    if not len(ids['train']): raise ValueError('无训练样本')
    scales=core.fit_scales(data,ids['train'],a.seq_len)
    chosen=select_samples(data,ids['train'],counts,sums,args.per_class,args.seed)
    if args.dates:
        lookup={m.iloc[i].date:i for i in ids['train']}
        if len(set(args.dates))!=len(args.dates) or any(d not in lookup for d in args.dates): raise ValueError('日期重复或不属于可用训练期')
        chosen=np.array([lookup[d] for d in args.dates])
        positive=counts[chosen,1]>0
        if not positive.any() or positive.all(): raise ValueError('需要同时包含正负样本')
    ds=core.Sequences(data,chosen,a,scales)
    x=t.stack([ds[j][0] for j in range(len(ds))]).to(device)
    y=t.stack([ds[j][1] for j in range(len(ds))]).to(device)
    records=[]
    for j,i in enumerate(chosen):
        q=core.to_j(data[i,1],cfg['energy_ref']);iy,ix=np.unravel_index(q.argmax(),q.shape)
        records.append({'date':m.iloc[i].date,'n_high':int(counts[i,1]),'target_total_J':float(sums[i,1]),'peak_x_index':int(ix),'peak_y_index':int(iy),'input_sha256':hashlib.sha256(x[j].cpu().numpy().tobytes()).hexdigest()})
    pd.DataFrame(records).to_csv(out/'selected_samples.csv',index=False,encoding='utf-8-sig')
    core.save_json(out/'config.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'scales':scales,'script_sha256':core.digest(__file__),'core_sha256':core.digest(core.__file__),'interpretation':'train-only memorization; not generalization'})
    if len({r['input_sha256'] for r in records})<len(records): print('WARNING: 存在相同历史输入，查看selected_samples.csv；不同标签可能无法同时拟合。',flush=True)
    for variant in args.variants:
        dest=out/variant;dest.mkdir()
        core.seed_all(args.seed)
        model=core.Predictor(2,args.hidden,'random').to(device)
        a.total_weight=.1
        a.spatial_weight=args.spatial_weight if variant=='random_spatial' else 0.
        opt=t.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        logs=[];metrics=[]
        def snapshot(step):
            model.eval()
            with t.no_grad(): pred=model(x).cpu().numpy()[:,0]*scales['target']
            folder=dest/f'step_{step:05d}';folder.mkdir()
            for j,i in enumerate(chosen):
                date=m.iloc[i].date;q=core.to_j(pred[j],cfg['energy_ref']);truth=core.to_j(data[i,1],cfg['energy_ref'])
                points=core.event_pixels(events,date,edges)
                hit,area=core.alarm_hits(q,points,.1)
                metrics.append({'step':step,'date':date,'true_total_J':float(truth.sum()),'pred_total_J':float(q.sum()),'grid_MAE_J':float(abs(q-truth).mean()),'spatial_std_J':float(q.std()),'covered_events_top10':hit,'n_high':len(points),'area_top10':area, 'normalized_overlap':float(np.minimum(q/q.sum(),truth/truth.sum()).sum()) if truth.sum()>0 and q.sum()>0 else None, 'pred_fraction_in_target_support':float(q[truth>0].sum()/q.sum()) if truth.sum()>0 and q.sum()>0 else None})
                np.savez_compressed(folder/f'{date}.npz',pred_energy_J=q,target_energy_J=truth,pred_energy_log10=pred[j],target_energy_log10=data[i,1],x_edges=edges[0],y_edges=edges[1],target_date=np.array(date))
                core.plot_comparison(folder/f'{date}.png',truth,q,edges,date,250000.)
            pd.DataFrame(metrics).to_csv(dest/'sample_metrics.csv',index=False)
        snapshot(0)
        for step in range(1,args.steps+1):
            model.train();opt.zero_grad(set_to_none=True)
            pred=model(x);loss,stats=core.objective(pred,y,scales['target'],a)
            if not t.isfinite(loss): raise FloatingPointError(f'{variant} step={step}: nonfinite loss')
            loss.backward()
            row={'step':step,**core.aggregate(stats,a),'head_grad_norm':grad_norm(model.head),'gates_grad_norm':grad_norm(model.gates)}
            t.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();logs.append(row)
            if step%10==0 or step==1 or step==args.steps:
                pd.DataFrame(logs).to_csv(dest/'loss_log.csv',index=False)
                print(f'{variant} {step}/{args.steps}: loss={float(loss):.7g}, hot={row["hot"]:.7g}, gate_grad={row["gates_grad_norm"]:.3g}',flush=True)
            if step%args.save_every==0 or step==args.steps: snapshot(step)
        t.save({'model':model.state_dict(),'variant':variant,'scales':scales,'dates':[r['date'] for r in records]},dest/'diagnostic_weights.pth')
        core.save_json(dest/'summary.json',{'initial_training_loss':logs[0]['loss'],'last_training_loss_before_update':logs[-1]['loss'],'steps':args.steps,'interpretation':'inspect sample_metrics and images; no automatic claim of predictive skill'})
    core.save_json(out/'completed.json',{'completed':True,'train_only':True})
    print(f'完成：{out}')


if __name__=='__main__': main()
