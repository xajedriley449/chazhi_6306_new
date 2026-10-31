#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6306 V4 controlled experiment. Standalone; no V1 imports.
Place in project/run_convlstm-early/pred_6306_xy_v3.py.
Adds positive-day spatial KL (default weight 0.01, unvalidated). Changes head weights to N(0,0.05); bias stays -9.
This is an experimental candidate, NOT a verified cure for constant output.
Existing data, split, loss and baselines are retained. Validation only by default.
Use --evaluate-test explicitly for final evaluation; --init-mode original for control.
--prepare-only, --predict-only --checkpoint, --forecast-only are supported.
Training logs distinguish dataset-aggregated loss from batch objective mean.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = nn = F = None
    Dataset = object

PROJECT = Path(__file__).resolve().parent.parent
FRACTIONS = (0.05, 0.10, 0.20)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(v):
    return str(v).strip().lower() in ('true', '1')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sampling',choices=['natural','balanced'],default='balanced')
    p.add_argument('--spatial-weight', type=float, default=0.01, help='positive-day energy-distribution KL weight; experimental')
    p.add_argument('--init-mode', choices=['original','random'], default='random', help='random只改变输出头权重；保留-9偏置控制初始总量')
    p.add_argument('--evaluate-test', action='store_true', help='明确请求时才评价测试集')
    p.add_argument('--data-dir', type=Path, default=PROJECT/'input_6306_daily_xy')
    p.add_argument('--output', type=Path)
    p.add_argument('--seq-len', type=int, default=10)
    p.add_argument('--train-end', default='2022-01-01', help='训练目标截止，不含当天')
    p.add_argument('--test-start', default='2023-01-01')
    p.add_argument('--coverage', choices=['exploratory','strict'], default='exploratory')
    p.add_argument('--input-mode', choices=['both','high'], default='both')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--background-weight', type=float, default=1.)
    p.add_argument('--total-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--plot-every', type=int, default=30, help='每N个目标日绘图，0关闭；有高能真值日总是绘图（非0时）')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--predict-only', action='store_true')
    p.add_argument('--forecast-only', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    a=p.parse_args()
    a.train_end=pd.Timestamp(a.train_end).strftime('%Y-%m-%d')
    a.test_start=pd.Timestamp(a.test_start).strftime('%Y-%m-%d')
    if any(getattr(a,k)<1 for k in ['seq_len','hidden','batch_size','epochs','patience']): p.error('长度、通道和训练整数参数必须>=1')
    if a.num_workers<0 or a.plot_every<0: p.error('num-workers/plot-every必须>=0')
    for k in ['lr','weight_decay','background_weight','total_weight','spatial_weight']:
        v=getattr(a,k)
        if not math.isfinite(v) or v<0 or (k=='lr' and v==0): p.error(f'{k}无效')
    if pd.Timestamp(a.test_start)<=pd.Timestamp(a.train_end): p.error('test-start必须晚于train-end')
    if (a.predict_only or a.forecast_only) and a.checkpoint is None: p.error('预测模式需要--checkpoint')
    return a


def load_catalog(a, out):
    root=a.data_dir.resolve()
    cfg=json.loads((root/'dataset_config.json').read_text(encoding='utf-8-sig'))
    if cfg['axis_order']!=['channel','y','x']: raise ValueError('NPZ轴顺序不匹配')
    if cfg['energy_ref']<=0: raise ValueError('energy_ref必须>0')
    m=pd.read_csv(root/'daily_manifest.csv').sort_values('date').reset_index(drop=True)
    if m.date.duplicated().any(): raise ValueError('日清单有重复日期')
    m['date']=pd.to_datetime(m.date).dt.strftime('%Y-%m-%d')
    events=pd.read_csv(root/'events_selected.csv')
    events['date']=pd.to_datetime(events.date).dt.strftime('%Y-%m-%d')
    if not m.monitoring_status.isin(['normal','missing','unknown']).all(): raise ValueError('监测状态无效')
    if pd.Timestamp(cfg['train_end'])>pd.Timestamp(a.train_end): raise ValueError('预处理范围拟合截止晚于模型训练截止，会引入未来信息')
    frames=[]; sums=[]; counts=[]; hashes=[]; edges=None
    for i,row in m.iterrows():
        path=root/row.matrix_file
        with np.load(path,allow_pickle=False) as z:
            x=z['energy_log10'].astype(np.float32)
            q=z['energy_J'].astype(np.float64)
            raw=z['energy_grid_J'].astype(np.float64)
            cnt=z['count_grid'].astype(np.int64)
            xe,ye=z['x_edges'].copy(),z['y_edges'].copy()
            start=pd.Timestamp(str(z['window_start'].item()))
            end=pd.Timestamp(str(z['window_end'].item()))
        if x.shape!=(2,cfg['grid'],cfg['grid']) or not np.isfinite(x).all() or (x<0).any(): raise ValueError(f'{path}:矩阵非法')
        if not np.isfinite(q).all() or (q<0).any() or (cnt<0).any(): raise ValueError(f'{path}:能量/计数非法')
        if start.strftime('%Y-%m-%d')!=row.date or start.hour or start.minute or start.second or end-start!=pd.Timedelta(days=1): raise ValueError(f'{path}:日窗口不一致')
        np.testing.assert_allclose(x,np.log10(1+q/cfg['energy_ref']),rtol=1e-5,atol=1e-8)
        np.testing.assert_allclose(q.sum((1,2)),raw.sum((1,2)),rtol=1e-6,atol=.01)
        np.testing.assert_allclose(raw.sum((1,2)),[row.raster_low_J,row.raster_high_J],rtol=1e-6,atol=.01)
        if edges is None: edges=(xe,ye)
        else:
            np.testing.assert_allclose(xe,edges[0],rtol=0,atol=1e-6)
            np.testing.assert_allclose(ye,edges[1],rtol=0,atol=1e-6)
        np.testing.assert_allclose([xe[0],xe[-1],ye[0],ye[-1]],cfg['bounds_xy'],rtol=0,atol=1e-5)
        frames.append(x);sums.append(raw.sum((1,2)));counts.append(cnt.sum((1,2)))
        hashes.append(f'{row.date}:{digest(path)}')
    data=np.stack(frames)
    audit={'days':len(m),'shape':list(data.shape),'unknown_days':int(m.monitoring_status.eq('unknown').sum()),
           'coverage_mode':a.coverage,'interpretation':'catalog backtest; monitoring completeness unverified' if a.coverage=='exploratory' else 'confirmed monitoring only',
           'manifest_sha256':digest(root/'daily_manifest.csv'),'events_sha256':digest(root/'events_selected.csv'),
           'npz_digest':hashlib.sha256('\n'.join(hashes).encode()).hexdigest()}
    save_json(out/'data_audit.json',audit)
    return cfg,m,events,data,np.array(sums),np.array(counts),edges,audit


def day_invalid(row,coverage):
    reasons=[]
    if row.monitoring_status=='missing': reasons.append('monitoring_missing')
    if coverage=='strict' and row.monitoring_status!='normal': reasons.append('coverage_not_confirmed')
    if row.n_outside_grid>0: reasons.append('outside_grid')
    if truth(row.get('catalog_edge_day',False)) and row.monitoring_status!='normal': reasons.append('unconfirmed_catalog_edge')
    return reasons


def samples(m,a,out):
    records=[]
    for i,row in m.iterrows():
        split='train' if row.date<a.train_end else ('val' if row.date<a.test_start else 'test')
        reasons=[]
        if i<a.seq_len: reasons.append('insufficient_history')
        else:
            block=m.iloc[i-a.seq_len:i+1]
            expected=pd.date_range(end=row.date,periods=a.seq_len+1).strftime('%Y-%m-%d').tolist()
            if block.date.tolist()!=expected: reasons.append('calendar_gap')
            for _,d in block.iterrows():
                reasons.extend(day_invalid(d,a.coverage))
        records.append({'target_index':i,'target_date':row.date,'input_start':m.iloc[i-a.seq_len].date if i>=a.seq_len else None,
                        'input_end':m.iloc[i-1].date if i else None,'split':split,'usable':not reasons,
                        'reason':';'.join(sorted(set(reasons))),'n_high':int(row.n_high)})
    table=pd.DataFrame(records);table.to_csv(out/'sample_manifest.csv',index=False,encoding='utf-8-sig')
    return {s:table.loc[table.usable & table.split.eq(s),'target_index'].to_numpy(dtype=int) for s in ['train','val','test']}


def to_j(z,ref):
    t=np.asarray(z,dtype=np.float64)*np.log(10.)
    if not np.isfinite(t).all() or (t<0).any() or t.max()>650:
        raise FloatingPointError('预测反变换异常；不静默裁剪能量，请检查训练日志/学习率')
    q=ref*np.expm1(t)
    if not np.isfinite(q).all(): raise FloatingPointError('预测能量溢出')
    return q


def fit_scales(data,idx,n):
    used=np.unique(np.concatenate([np.arange(i-n,i+1) for i in idx]))
    ins=np.maximum(data[used].max(axis=(0,2,3)),1e-6)
    target=max(float(data[idx,1].max()),1e-6)
    return {'input':ins.tolist(),'target':target,'fit_source':'usable training samples only; max, no clipping'}


class Sequences(Dataset):
    def __init__(self,data,indices,a,scales):
        self.data=data;self.ids=indices;self.n=a.seq_len;self.channels=[0,1] if a.input_mode=='both' else [1]
        self.ins=np.array(scales['input'],dtype=np.float32)[self.channels,None,None];self.ts=scales['target']
    def __len__(self): return len(self.ids)
    def __getitem__(self,j):
        i=int(self.ids[j]);x=self.data[i-self.n:i][:,self.channels]/self.ins
        y=self.data[i,1:2]/self.ts
        return torch.from_numpy(x.copy()),torch.from_numpy(y.copy())


class Predictor(nn.Module if nn else object):
    def __init__(self,channels,hidden,init_mode="random"):
        super().__init__();self.hidden=hidden
        self.gates=nn.Conv2d(channels+hidden,4*hidden,3,padding=1)
        self.head=nn.Conv2d(hidden,1,1)
        # 小的初始正值，避免64x64背景上较大的Softplus默认输出。
        nn.init.zeros_(self.head.weight) if init_mode == "original" else nn.init.normal_(self.head.weight, mean=0., std=0.05)
        nn.init.constant_(self.head.bias,-9.)
    def forward(self,x):
        b,_,_,h,w=x.shape
        hx=x.new_zeros((b,self.hidden,h,w));cx=torch.zeros_like(hx)
        for t in range(x.shape[1]):
            i,f,o,g=self.gates(torch.cat([x[:,t],hx],1)).chunk(4,dim=1)
            cx=torch.sigmoid(f)*cx+torch.sigmoid(i)*torch.tanh(g)
            hx=torch.sigmoid(o)*torch.tanh(cx)
        return F.softplus(self.head(hx))


def total_log(z):
    # 稳定计算log10(1+sum(expm1(z*ln10)))，训练时避免直接指数溢出。
    t=z.flatten(1)*math.log(10.)
    logmass=t+torch.log((-torch.expm1(-t)).clamp_min(1e-30))
    return F.softplus(torch.logsumexp(logmass,dim=1))/math.log(10.)


def spatial_kl(pred, target, scale):
    """KL(target energy proportions || predicted energy proportions), positive days only.
    log(expm1(t)) = t + log(1-exp(-t)); energy_ref cancels in normalization.
    Uses natural logs. Zero target cells contribute exactly zero.
    """
    positive = target.flatten(1).sum(1) > 0
    if not positive.any():
        return pred.sum()*0, 0
    def log_mass(z):
        t = z.flatten(1).double() * float(scale) * math.log(10.)
        return t + torch.log((-torch.expm1(-t)).clamp_min(1e-30))
    lp = F.log_softmax(log_mass(pred[positive]), dim=1)
    truth = target[positive].flatten(1)
    lt = log_mass(target[positive]).masked_fill(truth == 0, -torch.inf)
    lt = lt - torch.logsumexp(lt, dim=1, keepdim=True)
    weights = torch.exp(lt)
    safe_lt = torch.where(truth > 0, lt, torch.zeros_like(lt))
    per_day = (weights * (safe_lt-lp)).sum(1).clamp_min(0)
    return per_day.sum().to(pred.dtype), int(positive.sum())


def objective(pred,target,scale,a):
    err=F.smooth_l1_loss(pred,target,reduction='none')
    hot=target>0; bg=~hot
    dims=(1,2,3);hc=hot.sum(dims);bc=bg.sum(dims)
    hp=(err*hot).sum(dims)/hc.clamp_min(1);bp=(err*bg).sum(dims)/bc.clamp_min(1)
    valid=hc>0
    h=hp[valid].mean() if valid.any() else pred.sum()*0
    b=bp.mean()
    totals=F.smooth_l1_loss(total_log(pred*scale),total_log(target*scale),reduction='none')
    spatial_sum, spatial_count = spatial_kl(pred,target,scale)
    spatial = spatial_sum/max(spatial_count,1)
    loss=h+a.background_weight*b+a.total_weight*totals.mean()+a.spatial_weight*spatial
    stats=[float(hp[valid].sum().detach()),int(valid.sum()),float(bp.sum().detach()),float(totals.sum().detach()),len(pred),float(spatial_sum.detach()),spatial_count]
    return loss,stats


def aggregate(stats,a):
    h,n,b,t,k,ss,sn=stats; spatial=ss/max(sn,1);h=h/max(n,1);b=b/max(k,1);t=t/max(k,1)
    return {'loss':h+a.background_weight*b+a.total_weight*t+a.spatial_weight*spatial,'hot':h,'background':b,'total':t,'spatial':spatial}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def checkpoint_payload(model,optimizer,epoch,best_epoch,best_loss,a,scales,cfg,audit):
    return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best_epoch':best_epoch,
            'best_val_loss':best_loss,'settings':{k:v for k,v in vars(a).items() if isinstance(v,(str,int,float,bool,type(None)))},
            'scales':scales,'dataset_config':cfg,'data_audit':audit}


class BalancedBatches:
    """Exactly 2 positive + 6 negative training targets per batch; replacement.
    85 batches for 676 samples, matching V3 optimizer steps per epoch.
    Logs actual draws; this intentionally changes the training distribution.
    """
    def __init__(self, data, indices, seed):
        positive=np.array([bool((data[i,1]>0).any()) for i in indices])
        self.pos=np.flatnonzero(positive);self.neg=np.flatnonzero(~positive)
        if not len(self.pos) or not len(self.neg): raise ValueError('balanced requires both classes')
        self.steps=math.ceil(len(indices)/8);self.seed=seed;self.epoch=0
    def __len__(self): return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);self.epoch+=1
        self.draws=[]
        for _ in range(self.steps):
            batch=np.concatenate([rng.choice(self.pos,2,replace=True),rng.choice(self.neg,6,replace=True)])
            rng.shuffle(batch);self.draws.extend(batch.tolist());yield batch.tolist()


def training(model,data,ids,a,scales,cfg,audit,out,device):
    loaders={s:DataLoader(Sequences(data,ids[s],a,scales),batch_size=a.batch_size,shuffle=s=='train',
                         num_workers=a.num_workers,pin_memory=device.type=='cuda',drop_last=False) for s in ['train','val']}
    sampler=None
    if a.sampling=='balanced':
        if a.batch_size!=8: raise ValueError('balanced固定batch-size=8；显存不足请使用natural对照或调整设备')
        sampler=BalancedBatches(data,ids['train'],a.seed)
        loaders['train']=DataLoader(Sequences(data,ids['train'],a,scales),batch_sampler=sampler,num_workers=a.num_workers,pin_memory=device.type=='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=4,min_lr=1e-6)
    best=float('inf');best_epoch=0;stale=0;logs=[];interrupted=False
    try:
        for epoch in range(1,a.epochs+1):
            row={'epoch':epoch,'lr':optimizer.param_groups[0]['lr']}
            for split in ['train','val']:
                model.train(split=='train');stats=np.zeros(7,dtype=float); batch_loss_sum=0.; batch_samples=0
                with torch.set_grad_enabled(split=='train'):
                    for x,y in loaders[split]:
                        x,y=x.to(device),y.to(device)
                        if split=='train': optimizer.zero_grad(set_to_none=True)
                        p=model(x);loss,terms=objective(p,y,scales['target'],a)
                        if not torch.isfinite(loss): raise FloatingPointError('损失非有限，请检查数据和学习率')
                        if split=='train':
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                        stats+=terms
                        batch_loss_sum += float(loss.detach())*len(x); batch_samples += len(x)
                row.update({f'{split}_{k}':v for k,v in aggregate(stats,a).items()})
                row[f'{split}_batch_objective_mean']=batch_loss_sum/max(batch_samples,1)
            if sampler is not None:
                row.update({'train_positive_draws':2*len(sampler),'train_negative_draws':6*len(sampler),'train_unique_draws':len(set(sampler.draws))})
            scheduler.step(row['val_loss'])
            improved=row['val_loss']<best
            if improved: best=row['val_loss'];best_epoch=epoch;stale=0
            else: stale+=1
            payload=checkpoint_payload(model,optimizer,epoch,best_epoch,best,a,scales,cfg,audit)
            torch.save(payload,out/'last_model.pth')
            if improved: torch.save(payload,out/'best_model.pth')
            logs.append(row);pd.DataFrame(logs).to_csv(out/'loss_log.csv',index=False)
            print(f"Epoch {epoch}: train={row['train_loss']:.6g} val={row['val_loss']:.6g} best={best_epoch}",flush=True)
            if stale>=a.patience: break
    except KeyboardInterrupt:
        interrupted=True
        print('已中断；仅使用已完整完成epoch的最佳检查点。')
    if not (out/'best_model.pth').exists(): raise RuntimeError('尚未完成有效epoch，没有可评价模型')
    ck=torch.load(out/'best_model.pth',map_location=device,weights_only=False)
    model.load_state_dict(ck['model'])
    save_json(out/'training_summary.json',{'best_epoch':best_epoch,'best_val_loss':best,'completed_epochs':len(logs),'interrupted':interrupted})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([r['epoch'] for r in logs],[r['train_loss'] for r in logs],label='Train')
    ax.plot([r['epoch'] for r in logs],[r['val_loss'] for r in logs],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Composite loss');ax.legend();fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)


def alarm_hits(scores,points,fraction):
    # 并列采用面积预算下的分数权重，等价于随机打破并列的期望覆盖率。
    # 全零预测不报警。所有方法使用同样规则。
    weights=np.zeros(scores.size,dtype=float);flat=scores.ravel();positive=flat>0
    k=max(1,int(np.floor(fraction*len(flat))))
    if positive.sum()<=k: weights[positive]=1
    else:
        cutoff=np.partition(flat,-k)[-k];above=flat>cutoff;ties=flat==cutoff
        weights[above]=1;weights[ties]=(k-int(above.sum()))/int(ties.sum())
    hit=float(weights[points].sum()) if len(points) else 0.
    return hit,float(weights.mean())


def event_pixels(events,date,edges):
    xe,ye=edges;e=events[(events.date==date)&(events.energy_class==1)]
    inside=e.x.between(xe[0],xe[-1]) & e.y.between(ye[0],ye[-1])
    if not inside.all(): raise ValueError(f'{date}:可用目标中仍有界外事件')
    ix=np.minimum(np.searchsorted(xe,e.x,side='right')-1,len(xe)-2)
    iy=np.minimum(np.searchsorted(ye,e.y,side='right')-1,len(ye)-2)
    return (iy*(len(xe)-1)+ix).astype(int)


def plot_comparison(path,true,pred,edges,date,vmax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    xe,ye=edges;fig,axs=plt.subplots(1,3,figsize=(16,4.8))
    norm=SymLogNorm(linthresh=1.,vmin=0,vmax=vmax,base=10)
    cmap=matplotlib.colormaps['YlOrRd'].copy();cmap.set_bad('white')
    for ax,a,title in zip(axs,[true,pred,np.abs(pred-true)],['Observed','Predicted','Absolute error']):
        im=ax.pcolormesh(xe-xe[0],ye-ye[0],np.ma.masked_equal(a,0),cmap=cmap,norm=norm,shading='flat')
        ax.set_aspect('equal');ax.set(xlabel='Relative X (m)',ylabel='Relative Y (m)',title=f'{title}; sum={a.sum():.3g} J')
        fig.colorbar(im,ax=ax,label='J/cell; nonlinear',extend='max' if (a>vmax).any() else 'neither')
    fig.suptitle(f'{date} | high-energy spatial energy | fixed display scale')
    fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)


def evaluate(model,data,indices,m,events,edges,train_mean,a,scales,cfg,out,device,split):
    dest=out/split;dest.mkdir();(dest/'predictions').mkdir();(dest/'figures').mkdir()
    rows=[];model.eval()
    ds=Sequences(data,indices,a,scales)
    for j,i in enumerate(indices):
        date=m.iloc[i].date;x,_=ds[j]
        with torch.no_grad(): p=model(x.unsqueeze(0).to(device))[0,0].cpu().numpy()*scales['target']
        q=to_j(p,cfg['energy_ref']);target=to_j(data[i,1],cfg['energy_ref'])
        points=event_pixels(events,date,edges)
        if len(points)!=int(m.iloc[i].n_high): raise ValueError(f'{date}:高能事件表与日清单不匹配')
        predmap={'convlstm':q,'zero':np.zeros_like(q),'training_mean':train_mean,
                 f'history_{a.seq_len}day_mean':to_j(data[i-a.seq_len:i,1],cfg['energy_ref']).mean(0)}
        for method,arr in predmap.items():
            row={'date':date,'method':method,'n_high':len(points),'positive_day':bool(len(points)),
                 'true_total_J':float(target.sum()),'pred_total_J':float(arr.sum()),
                 'total_absolute_error_J':float(abs(arr.sum()-target.sum())),
                 'total_bias_J':float(arr.sum()-target.sum()),'grid_MAE_J':float(np.abs(arr-target).mean()),
                 'log_MAE':float(np.abs(np.log10(1+arr/cfg['energy_ref'])-data[i,1]).mean())}
            for frac in FRACTIONS:
                hit,area=alarm_hits(arr,points,frac);key=str(int(frac*100))
                row[f'covered_events_top{key}']=hit;row[f'actual_area_top{key}']=area
            rows.append(row)
        np.savez_compressed(dest/'predictions'/f'{date}.npz',pred_energy_J=q,pred_energy_log10=p,
                            target_energy_J=target,target_energy_log10=data[i,1],
                            x_edges=edges[0],y_edges=edges[1],target_date=np.array(date))
        if a.plot_every and (j%a.plot_every==0 or len(points)):
            plot_comparison(dest/'figures'/f'{date}.png',target,q,edges,date,250000.)
        if j%100==0: print(f'{split}: {j+1}/{len(indices)}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'daily_metrics.csv',index=False,encoding='utf-8-sig')
    summary={}
    for method,group in frame.groupby('method'):
        stats={}
        for label,g in [('all',group),('high_event_days',group[group.positive_day]),('no_high_event_days',group[~group.positive_day])]:
            d={'days':len(g),'events':int(g.n_high.sum())}
            for col in ['grid_MAE_J','log_MAE','total_absolute_error_J','total_bias_J','pred_total_J']:
                d[col]=float(g[col].mean()) if len(g) else None
            for frac in FRACTIONS:
                key=str(int(frac*100));d[f'event_coverage_top{key}']=float(g[f'covered_events_top{key}'].sum()/g.n_high.sum()) if g.n_high.sum() else None
                d[f'mean_alarm_area_top{key}']=float(g[f'actual_area_top{key}'].mean()) if len(g) else None
            stats[label]=d
        negative=group.loc[~group.positive_day,'pred_total_J']
        stats['negative_day_pred_energy_quantiles_J']={str(v):float(negative.quantile(v)) for v in [.5,.9,.95]} if len(negative) else {}
        summary[method]=stats
    save_json(dest/'summary.json',summary)


def forecast(model,data,m,edges,a,scales,cfg,out,device):
    block=m.iloc[-a.seq_len:]
    if len(block)!=a.seq_len or block.date.tolist()!=pd.date_range(end=block.date.iloc[-1],periods=a.seq_len).strftime('%Y-%m-%d').tolist(): raise ValueError('末尾历史不足或日期不连续')
    bad=[(r.date,day_invalid(r,a.coverage)) for _,r in block.iterrows() if day_invalid(r,a.coverage)]
    if bad: raise ValueError(f'末尾预测历史不可用：{bad}；请补充完整日覆盖，不能使用未完成的最后一天')
    ch=[0,1] if a.input_mode=='both' else [1]
    x=data[-a.seq_len:,ch]/np.array(scales['input'],dtype=np.float32)[ch,None,None]
    model.eval()
    with torch.no_grad(): z=model(torch.from_numpy(x.copy()).unsqueeze(0).to(device))[0,0].cpu().numpy()*scales['target']
    date=(pd.Timestamp(block.date.iloc[-1])+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    np.savez_compressed(out/f'forecast_{date}.npz',pred_energy_J=to_j(z,cfg['energy_ref']),pred_energy_log10=z,
                        target_date=np.array(date),x_edges=edges[0],y_edges=edges[1])
    print(f'已生成未来日{date}预测；无真值评价。')


def main():
    a=parse_args();ck=None
    if a.checkpoint:
        if torch is None: raise RuntimeError('请先安装PyTorch')
        # 仅加载自己生成、可信的检查点；其中含配置而不仅是权重。
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        for k in ['seq_len','train_end','test_start','coverage','input_mode','hidden','background_weight','total_weight','init_mode','spatial_weight']:
            setattr(a,k,ck['settings'].get(k, 0.) if k=='spatial_weight' else (ck['settings'].get(k, 'original') if k=='init_mode' else ck['settings'][k]))
        if not (a.predict_only or a.forecast_only): raise ValueError('--checkpoint仅用于预测模式；本版本不伪装支持断点续训')
    out=a.output or PROJECT/'output_6306_convlstm_v4'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_seed{a.seed}'
    out=out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'输出目录非空：{out}')
    out.mkdir(parents=True,exist_ok=True)
    cfg,m,events,data,sums,counts,edges,audit=load_catalog(a,out)
    ids=samples(m,a,out)
    splitinfo={s:{'samples':len(v),'high_event_days':int((counts[v,1]>0).sum()),'high_events':int(counts[v,1].sum())} for s,v in ids.items()}
    save_json(out/'split_summary.json',splitinfo);print(json.dumps(splitinfo,ensure_ascii=False,indent=2))
    if a.forecast_only:
        scales=ck['scales']
    else:
        if any(len(ids[s])==0 for s in ids): raise ValueError(f'有空集合；查看{out}/sample_manifest.csv。strict模式需要监测覆盖记录。')
        if not (counts[ids['train'],1]>0).any(): raise ValueError('训练集没有高能事件，无法开展此实验')
        scales=ck['scales'] if ck else fit_scales(data,ids['train'],a.seq_len)
    if ck:
        for k in ['bounds_xy','grid','energy_ref','threshold','sigma_m','truncate','axis_order']:
            if cfg[k]!=ck['dataset_config'][k]: raise ValueError(f'检查点与当前预处理参数不一致：{k}')
        if not a.forecast_only:
            for key in ['npz_digest','manifest_sha256','events_sha256']:
                if audit[key]!=ck['data_audit'][key]:
                    raise ValueError(f'回测数据或元数据已改变：{key}。请固定原实验数据。')
    save_json(out/'scales.json',scales)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    settings.update({'data_dir':str(a.data_dir.resolve()),'output':str(out),'alarm_ties':'fractional expected coverage at fixed area; zero map has no alarm',
                     'spatial_domain':'entire fixed rectangular grid','plot_vmax_J_per_cell':250000.,'script_sha256':digest(__file__),'numpy_version':np.__version__,'torch_version':str(torch.__version__) if torch else None})
    save_json(out/'run_config.json',settings)
    if a.prepare_only: print(f'数据准备完成：{out}');return
    if torch is None: raise RuntimeError('数据准备完成，但未安装PyTorch。请在训练环境安装torch后运行。')
    seed_all(a.seed)
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('当前PyTorch未检测到CUDA，可改--device cpu')
    model=Predictor(2 if a.input_mode=='both' else 1,a.hidden,a.init_mode).to(device)
    print(f'Device={device}; parameters={sum(p.numel() for p in model.parameters()):,}')
    if ck: model.load_state_dict(ck['model'])
    if a.forecast_only: forecast(model,data,m,edges,a,scales,cfg,out,device);return
    if not a.predict_only: training(model,data,ids,a,scales,cfg,audit,out,device)
    mean=np.zeros_like(data[0,1],dtype=np.float64)
    for i in ids['train']: mean+=to_j(data[i,1],cfg['energy_ref'])
    mean/=len(ids['train']);np.save(out/'training_mean_energy_J.npy',mean)
    for split in (['train','val','test'] if a.evaluate_test else ['train','val']): evaluate(model,data,ids[split],m,events,edges,mean,a,scales,cfg,out,device,split)
    save_json(out/'completed.json',{'completed':True,'coverage_mode':a.coverage,'seed':a.seed})
    print(f'完成：{out}')


if __name__=='__main__': main()
