#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6306 scalar energy diagnostic. Place alongside pred_6306_xy_v4.py.
Input remains 10-day two-channel maps. Target log10(1+daily high energy/1e5).
Train/validation only. Diagnostic does not replace the spatial forecasting task.
"""
import argparse, math
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pred_6306_xy_v4 as c

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--no-early-stop',action='store_true')
    p.add_argument('--data-dir',type=Path,default=c.PROJECT/'input_6306_daily_xy')
    p.add_argument('--output',type=Path)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    
    a=p.parse_args()
    if min(a.epochs,a.patience,a.hidden)<1: p.error('invalid parameters')
    if c.torch is None: raise RuntimeError('需要PyTorch')
    t=c.torch; nn=c.nn
    class Model(nn.Module):
        mode='softplus'
        def __init__(self):
            super().__init__(); self.encoder=c.Predictor(2,a.hidden,'random')
            del self.encoder.head
            self.pool=nn.AdaptiveAvgPool2d((4,4));self.head=nn.Sequential(nn.Flatten(),nn.Linear(a.hidden*16,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.constant_(self.head[-1].bias,-2.)
        def forward(self,x):
            b,_,_,h,w=x.shape;hx=x.new_zeros((b,a.hidden,h,w));cx=t.zeros_like(hx)
            for j in range(x.shape[1]):
                i,f,o,g=self.encoder.gates(t.cat([x[:,j],hx],1)).chunk(4,1)
                cx=t.sigmoid(f)*cx+t.sigmoid(i)*t.tanh(g);hx=t.sigmoid(o)*t.tanh(cx)
            z=self.head(self.pool(hx)).squeeze(1)
            return c.F.softplus(z) if self.mode=='softplus' else z
    out=(a.output or c.PROJECT/'output_6306_full_head_diagnose'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_seed{a.seed}').resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    settings=SimpleNamespace(data_dir=a.data_dir,seq_len=10,train_end='2022-01-01',test_start='2023-01-01',coverage='exploratory',input_mode='both')
    cfg,m,events,data,sums,counts,edges,audit=c.load_catalog(settings,out);ids=c.samples(m,settings,out)
    if any(len(ids[s])==0 for s in ['train','val']):raise ValueError('训练/验证为空')
    scales=c.fit_scales(data,ids['train'],10)
    targets=np.log10(1+sums[:,1]/cfg['energy_ref']).astype(np.float32)
    class DS(c.Dataset):
        def __init__(self,split):self.indices=ids[split];self.maps=c.Sequences(data,self.indices,settings,scales)
        def __len__(self):return len(self.indices)
        def __getitem__(self,j):return self.maps[j][0],t.tensor(targets[self.indices[j]])
    device=t.device('cuda' if a.device=='auto' and t.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    c.save_json(out/'config.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'scales':scales,'target':'log10(1+sum high energy_J/energy_ref)','energy_ref':cfg['energy_ref'],'script_sha256':c.digest(__file__),'core_sha256':c.digest(c.__file__),'interpretation':'full train/validation scalar output head comparison; not a spatial model','linear_negative_output_policy':'clip raw predicted log10 energy to 0 only for physical-energy conversion/evaluation; preserve raw values in metrics','data_audit':audit})
    def evaluate(model,split,dest):
        model.eval();pred=[]
        with t.no_grad():
            for x,y in c.DataLoader(DS(split),batch_size=8):pred.extend(model(x.to(device)).cpu().numpy().tolist())
        z=np.asarray(pred,dtype=np.float64);z_clipped=np.maximum(z,0.0);q=c.to_j(z_clipped,cfg['energy_ref']);ix=ids[split]
        base=float(sums[ids['train'],1].mean());rows=[];summary={}
        methods={'scalar_convlstm':q,'zero':np.zeros(len(ix)),'training_mean':np.full(len(ix),base),'history_10day_mean':np.array([sums[i-10:i,1].mean() for i in ix])}
        for name,v in methods.items():
            frame=pd.DataFrame({'date':m.iloc[ix].date.to_numpy(),'method':name,'n_high':counts[ix,1],'true_total_J':sums[ix,1],'pred_total_J':v})
            if name=='scalar_convlstm':
                frame['raw_log_prediction']=z
                frame['negative_raw_prediction']=z<0
                frame['raw_prediction_min']=float(z.min());frame['raw_prediction_max']=float(z.max())
            frame['absolute_error_J']=abs(frame.pred_total_J-frame.true_total_J);frame['log_error']=abs(np.log10(1+v/cfg['energy_ref'])-targets[ix]);rows.append(frame)
            d={}
            for label,mask in [('all',np.ones(len(ix),dtype=bool)),('high_event_days',counts[ix,1]>0),('no_high_event_days',counts[ix,1]==0)]:
                f=frame[mask];d[label]={'days':len(f),'MAE_J':float(f.absolute_error_J.mean()) if len(f) else None,'true_mean_J':float(f.true_total_J.mean()) if len(f) else None,'pred_mean_J':float(f.pred_total_J.mean()) if len(f) else None,'log_MAE':float(f.log_error.mean()) if len(f) else None}
            summary[name]=d
        pd.concat(rows).to_csv(dest/f'{split}_daily_metrics.csv',index=False,encoding='utf-8-sig');c.save_json(dest/f'{split}_summary.json',summary)
    for mode in ['linear']:
        dest=out/mode;dest.mkdir();c.seed_all(a.seed);Model.mode=mode;model=Model().to(device)
        train=DS('train');loader=c.DataLoader(train,batch_size=8,shuffle=True)
        val=c.DataLoader(DS('val'),batch_size=8);opt=t.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=math.inf;stale=0;logs=[];best_epoch=0
        for epoch in range(1,a.epochs+1):
            row={'epoch':epoch}
            for split,ld in [('train',loader),('val',val)]:
                model.train(split=='train');total=0.;n=0;pos_sum=0.;neg_sum=0.;pos_n=0;neg_n=0
                with t.set_grad_enabled(split=='train'):
                    for x,y in ld:
                        x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
                        z=model(x); errors=c.F.smooth_l1_loss(z,y,reduction='none');loss=errors.mean()
                        positive=y>0;pos_sum+=float(errors[positive].sum().detach());neg_sum+=float(errors[~positive].sum().detach());pos_n+=int(positive.sum());neg_n+=int((~positive).sum())
                        if not t.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        if split=='train':loss.backward();t.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                        total+=float(loss.detach())*len(x);n+=len(x)
                row[split+'_loss']=total/n;row[split+'_positive_loss']=pos_sum/pos_n if pos_n else None;row[split+'_negative_loss']=neg_sum/neg_n if neg_n else None
            logs.append(row);pd.DataFrame(logs).to_csv(dest/'loss_log.csv',index=False)
            payload={'model':model.state_dict(),'scales':scales,'hidden':a.hidden,'energy_ref':cfg['energy_ref'],'epoch':epoch,'output_mode':mode}
            t.save(payload,dest/'last_model.pth')
            if row['val_loss']<best:best=row['val_loss'];best_epoch=epoch;stale=0;t.save(payload,dest/'best_model.pth')
            print(mode,row,flush=True)
            if not a.no_early_stop and stale>=a.patience:break
        for label in ['best','last']:
            model.load_state_dict(t.load(dest/f'{label}_model.pth',map_location=device,weights_only=False)['model']);Model.mode=mode
            result=dest/label;result.mkdir()
            for split in ['train','val']:evaluate(model,split,result)
        c.save_json(dest/'training_summary.json',{'best_epoch':best_epoch,'completed_epochs':epoch,'best_val_loss':best,'output_mode':mode,'loss':'Huber in log10 total energy','checkpoint':'best_model.pth'})
    c.save_json(out/'completed.json',{'completed':True,'test_evaluated':False});print(out)

if __name__=='__main__':main()
