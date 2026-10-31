#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Softplus and linear scalar output on the same fixed 8 train dates.
Put beside pred_6306_xy_v4.py. Diagnostic only; no test-set evaluation.
"""
import argparse, math
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pred_6306_xy_v4 as core

DATES=['2020-03-13','2020-04-09','2020-07-12','2020-05-15',
       '2020-12-24','2020-04-29','2021-05-19','2021-08-07']

class ScalarModel:
    def __new__(cls, hidden, output_mode, torch, nn, F):
        class Net(nn.Module):
            def __init__(self):
                super().__init__(); self.hidden=hidden; self.mode=output_mode
                self.gates=nn.Conv2d(2+hidden,4*hidden,3,padding=1)
                self.pool=nn.AdaptiveAvgPool2d((4,4))
                self.head=nn.Sequential(nn.Flatten(),nn.Linear(hidden*16,32),nn.SiLU(),nn.Linear(32,1))
                # Same initialization for both output modes.
                nn.init.normal_(self.head[-1].weight,mean=0.,std=.01)
                nn.init.constant_(self.head[-1].bias,-2.)
            def forward(self,x):
                b,_,_,h,w=x.shape;hx=x.new_zeros((b,self.hidden,h,w));cx=torch.zeros_like(hx)
                for j in range(x.shape[1]):
                    i,f,o,g=self.gates(torch.cat([x[:,j],hx],1)).chunk(4,dim=1)
                    cx=torch.sigmoid(f)*cx+torch.sigmoid(i)*torch.tanh(g)
                    hx=torch.sigmoid(o)*torch.tanh(cx)
                z=self.head(self.pool(hx)).squeeze(1)
                return F.softplus(z) if self.mode=='softplus' else z
        return Net()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=core.PROJECT/'input_6306_daily_xy')
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if min(a.steps,a.save_every,a.hidden)<1 or not np.isfinite(a.lr) or a.lr<=0:p.error('invalid positive parameters')
    if core.torch is None:raise RuntimeError('需要PyTorch')
    t=core.torch;out=(a.output or core.PROJECT/'output_6306_head_diagnose'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_seed{a.seed}').resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    cfgargs=SimpleNamespace(data_dir=a.data_dir,seq_len=10,train_end='2022-01-01',test_start='2023-01-01',coverage='exploratory',input_mode='both')
    cfg,m,events,data,sums,counts,edges,audit=core.load_catalog(cfgargs,out)
    ids=core.samples(m,cfgargs,out);lookup={m.iloc[i].date:i for i in ids['train']}
    if any(d not in lookup for d in DATES):raise ValueError('固定日期不在可用训练期')
    ix=np.array([lookup[d] for d in DATES]);scales=core.fit_scales(data,ids['train'],10)
    ds=core.Sequences(data,ix,cfgargs,scales);x=t.stack([ds[j][0] for j in range(8)])
    y=np.log10(1+sums[ix,1]/cfg['energy_ref']).astype(np.float32)
    x=x.to('cpu');y=t.tensor(y);device=t.device('cuda' if a.device=='auto' and t.cuda.is_available() else ('cpu' if a.device=='auto' else a.device));x=x.to(device);y=y.to(device)
    core.save_json(out/'config.json',{'dates':DATES,'steps':a.steps,'seed':a.seed,'hidden':a.hidden,'lr':a.lr,'energy_ref':cfg['energy_ref'],'target':'log10(1+high_energy_J/energy_ref)','data_audit':audit,'interpretation':'fixed training-set optimization diagnostic only'})
    for mode in ['softplus','linear']:
        dest=out/mode;dest.mkdir();core.seed_all(a.seed)
        model=ScalarModel(a.hidden,mode,t,core.nn,core.F).to(device);opt=t.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);logs=[]
        def snapshot(step):
            model.eval()
            with t.no_grad():raw=model(x).cpu().numpy()
            pred=np.maximum(raw,0);J=core.to_j(pred,cfg['energy_ref']);true=sums[ix,1]
            rows=[]
            for k,date in enumerate(DATES):rows.append({'step':step,'date':date,'n_high':int(counts[ix[k],1]),'true_total_J':float(true[k]),'raw_output_log10':float(raw[k]),'pred_total_J':float(J[k]),'abs_error_J':float(abs(J[k]-true[k]))})
            pd.DataFrame(rows).to_csv(dest/f'step_{step:05d}_metrics.csv',index=False)
            np.savez_compressed(dest/f'step_{step:05d}_outputs.npz',raw_output_log10=raw,pred_total_J=J,true_total_J=true,dates=np.array(DATES))
        for step in range(a.steps+1):
            if step%a.save_every==0 or step==a.steps:snapshot(step)
            if step==a.steps:break
            model.train();opt.zero_grad(set_to_none=True);z=model(x);loss=core.F.smooth_l1_loss(z,y)
            if not t.isfinite(loss):raise FloatingPointError(f'{mode}: nonfinite loss')
            loss.backward();
            with t.no_grad():
                grad_norm=float(sum(float(q.grad.square().sum()) for q in model.parameters() if q.grad is not None)**.5)
                head_grad=float(sum(float(q.grad.square().sum()) for q in model.head.parameters() if q.grad is not None)**.5)
                raw_min=float(z.min());raw_max=float(z.max());raw_mean=float(z.mean())
            t.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            logs.append({'step':step+1,'loss_before_update':float(loss.detach()),'grad_norm':grad_norm,'head_grad_norm':head_grad,'raw_min':raw_min,'raw_max':raw_max,'raw_mean':raw_mean})
            if (step+1)%10==0:pd.DataFrame(logs).to_csv(dest/'loss_log.csv',index=False)
            if (step+1)%100==0:print(mode,logs[-1],flush=True)
        pd.DataFrame(logs).to_csv(dest/'loss_log.csv',index=False)
        t.save({'model':model.state_dict(),'mode':mode,'scales':scales,'energy_ref':cfg['energy_ref'],'dates':DATES},dest/'last_model.pth')
    core.save_json(out/'completed.json',{'completed':True,'train_only':True,'comparison':['softplus','linear']});print(out)
if __name__=='__main__':main()
