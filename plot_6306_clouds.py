#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已有daily/*.npz生成论文云图，不修改训练矩阵。
安装：pip install numpy pandas matplotlib
运行：python plot_6306_clouds.py --root output_6306_daily_xy --dates 2020-03-13 2020-05-15 2023-12-19
全部日期：python plot_6306_clouds.py --root output_6306_daily_xy --all
输出PNG(300 dpi)、PDF、审计CSV、绘图参数JSON。无事件图仍保留零值。
色标是固定的J/格显示尺度，不是逐日归一化，也不是事件能量/危险半径。
默认low上限10000 J/格，high上限250000 J/格；超过上限的像素用顶色显示，
标题与审计表记录截色数量，不截断NPZ。可统一修改--vmax-low/--vmax-high。
SymLogNorm在线性阈值1 J/格以上使用对数显示，零值区域白色。
图上圆圈为原始目录事件位置，不能解释为模型预测。界外事件不画进范围内。
仅可视化原始64x64网格，最近邻显示，不通过增大sigma制造大面积热点。
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frame(path, cfg):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in ('energy_J','energy_log10','energy_grid_J','count_grid','x_edges','y_edges')}
    a=result['energy_J']
    if a.ndim!=3 or a.shape[0]!=2 or not np.isfinite(a).all() or (a<0).any():
        raise ValueError(f'{path}:能量矩阵形状或数值无效')
    xe,ye=result['x_edges'],result['y_edges']
    if len(xe)!=a.shape[2]+1 or len(ye)!=a.shape[1]+1 or not (np.diff(xe)>0).all() or not (np.diff(ye)>0).all():
        raise ValueError(f'{path}:坐标边界无效')
    np.testing.assert_allclose([xe[0],xe[-1],ye[0],ye[-1]],cfg['bounds_xy'],rtol=0,atol=1e-5)
    np.testing.assert_allclose(np.log10(1+a.astype(float)/cfg['energy_ref']),result['energy_log10'],rtol=1e-5,atol=1e-8)
    np.testing.assert_allclose(a.astype(float).sum((1,2)),result['energy_grid_J'].sum((1,2)),rtol=1e-6,atol=1e-3)
    return result


def channel_plot(fig,ax,z,date,ch,events,cfg,args):
    name=('Low','High')[ch]; vmax=(args.vmax_low,args.vmax_high)[ch]
    a=z['energy_J'][ch].astype(float)
    xe,ye=z['x_edges'],z['y_edges'];x0,y0=xe[0],ye[0]
    cmap=matplotlib.colormaps['YlOrRd'].copy();cmap.set_bad('white')
    norm=SymLogNorm(linthresh=args.linthresh,linscale=1,base=10,vmin=0,vmax=vmax)
    im=ax.pcolormesh(xe-x0,ye-y0,np.ma.masked_equal(a,0),norm=norm,cmap=cmap,shading='flat',rasterized=True)
    clipped=int((a>vmax).sum())
    bar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.025,extend='max' if clipped else 'neither')
    ticks=[0.0]+[float(10**k) for k in range(-3,10) if args.linthresh<=10**k<vmax]+[vmax]
    ticks=sorted(set(ticks))
    bar.set_ticks(ticks);bar.set_ticklabels([f'{v:g}' for v in ticks]);bar.ax.tick_params(labelsize=8)
    bar.set_label('Allocated energy (J/cell); nonlinear scale',fontsize=9)
    count=int(z['count_grid'][ch].sum()); total=float(z['energy_grid_J'][ch].sum())
    oob=0
    if events is not None:
        selected=events[(events.date==date)&(events.energy_class==ch)]
        inside=selected.x.between(xe[0],xe[-1]) & selected.y.between(ye[0],ye[-1])
        oob=int((~inside).sum()); shown=selected[inside]
        if len(shown)!=count:
            raise ValueError(f'{date}/{name}:事件表与矩阵事件数不一致')
        np.testing.assert_allclose(shown.energy_J.sum(),total,rtol=1e-6,atol=1e-3)
        ax.scatter(shown.x-x0,shown.y-y0,s=24,facecolors='none',edgecolors='#263238',linewidths=0.75,zorder=3)
    relation='<' if ch==0 else '>='
    ax.set_title(f'{date} | {name}: E {relation} {cfg["threshold"]:g} J\n'
                 f'n = {count} in grid; total = {total:.3g} J',fontsize=10,loc='left',pad=7)
    if count==0:
        ax.text(.5,.5,f'No {name.lower()}-energy events in grid\nCatalog map = 0',transform=ax.transAxes,
                ha='center',va='center',color='#63717d',fontsize=11)
    if clipped or oob:
        ax.text(.98,.02,f'Clipped display cells: {clipped}; outside events: {oob}',transform=ax.transAxes,
                ha='right',fontsize=7,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax.set_xlim(0,xe[-1]-x0);ax.set_ylim(0,ye[-1]-y0);ax.set_aspect('equal')
    ax.set_xlabel('Relative X (m)',fontsize=9);ax.set_ylabel('Relative Y (m)',fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#8b969d')
    return {'date':date,'channel':name.lower(),'events_in_grid':count,'events_outside_grid':oob if events is not None else None,
            'total_energy_J':total,'matrix_sum_J':float(a.sum()),'max_J_per_cell':float(a.max()),
            'nonzero_cells':int((a>0).sum()),'display_vmax_J_per_cell':vmax,'clipped_display_cells':clipped}


def footer(fig,cfg,events):
    x0,_,y0,_=cfg['bounds_xy']
    fig.text(.055,.013,f'Origin: X0={x0:.2f} m, Y0={y0:.2f} m | '
             f'Grid: {cfg["grid"]} x {cfg["grid"]} | Gaussian sigma: {cfg["sigma_m"]:g} m\n'
             'White = zero allocated energy; fixed scale per channel across dates. Monitoring completeness is not inferred.',
             fontsize=8,color='#4d5961')
    if events is not None:
        fig.legend(handles=[Line2D([],[],marker='o',markerfacecolor='none',markeredgecolor='#263238',
                   linestyle='None',label='Catalog event location')],loc='upper right',bbox_to_anchor=(.98,.995),frameon=False,fontsize=8)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path('output_6306_daily_xy'))
    p.add_argument('--config',type=Path)
    p.add_argument('--events',type=Path)
    p.add_argument('--dates',nargs='+')
    p.add_argument('--all',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--vmax-low',type=float,default=10000.)
    p.add_argument('--vmax-high',type=float,default=250000.)
    p.add_argument('--linthresh',type=float,default=1.)
    p.add_argument('--dpi',type=int,default=300)
    args=p.parse_args()
    if any(not np.isfinite(v) or v<=0 for v in [args.vmax_low,args.vmax_high,args.linthresh]) or args.linthresh>=min(args.vmax_low,args.vmax_high) or args.dpi<72:
        p.error('色标参数必须为有限正数，linthresh小于vmax，dpi>=72')
    if args.all and args.dates: p.error('--all和--dates不能同时使用')
    configs=sorted(args.root.glob('dataset_config*.json'))
    if args.config is None:
        if len(configs)!=1: p.error('请使用--config指定唯一配置文件')
        args.config=configs[0]
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if cfg.get('axis_order')!=['channel','y','x']: raise ValueError('不支持此数组轴顺序')
    event_path=args.events or args.root/'events_selected.csv'
    events=pd.read_csv(event_path) if event_path.exists() else None
    if args.events and events is None: raise FileNotFoundError(event_path)
    daily=args.root/'daily' if (args.root/'daily').is_dir() else args.root
    paths={p.stem:p for p in sorted(daily.glob('*.npz'))}
    dates=sorted(paths) if args.all else (args.dates or ['2020-03-13','2020-05-15','2023-12-19'])
    dates=list(dict.fromkeys(dates))
    if not dates or any(d not in paths for d in dates): p.error('所选日期NPZ不存在，请检查--root和--dates')
    out=args.output or args.root/'paper_figures'
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'输出目录非空，请指定新的--output：{out}')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'savefig.facecolor':'white'})
    audits=[]; hashes={}
    for date in dates:
        path=paths[date];hashes[date]=sha(path);z=load_frame(path,cfg)
        fig,axs=plt.subplots(1,2,figsize=(12,4.7))
        fig.subplots_adjust(left=.06,right=.94,bottom=.20,top=.85,wspace=.35)
        for ch in range(2): audits.append(channel_plot(fig,axs[ch],z,date,ch,events,cfg,args))
        footer(fig,cfg,events)
        fig.savefig(out/f'{date}_energy_clouds.png',dpi=args.dpi)
        fig.savefig(out/f'{date}_energy_clouds.pdf');plt.close(fig)
        if sha(path)!=hashes[date]: raise AssertionError('源NPZ发生变化')
        print(f'完成 {date}',flush=True)
    # 最多6天合成比较图，避免--all生成数米长的图。
    if 1<len(dates)<=6:
        fig,axs=plt.subplots(len(dates),2,figsize=(12,3.8*len(dates)+.7),squeeze=False)
        fig.subplots_adjust(left=.06,right=.94,bottom=.085,top=.95,hspace=.48,wspace=.35)
        for i,date in enumerate(dates):
            z=load_frame(paths[date],cfg)
            for ch in range(2): channel_plot(fig,axs[i,ch],z,date,ch,events,cfg,args)
        footer(fig,cfg,events)
        fig.savefig(out/'comparison.png',dpi=args.dpi);fig.savefig(out/'comparison.pdf');plt.close(fig)
    pd.DataFrame(audits).to_csv(out/'plot_audit.csv',index=False,encoding='utf-8-sig')
    params={'vmax_low_J_per_cell':args.vmax_low,'vmax_high_J_per_cell':args.vmax_high,
            'linthresh_J_per_cell':args.linthresh,'normalization':'SymLogNorm(base=10,linscale=1)',
            'scale_selection':'fixed display constants; not per-day fit; no test-dependent calibration',
            'dpi':args.dpi,'dates':dates,'source_npz_sha256':hashes,'source_config_sha256':sha(args.config),
            'source_events_sha256':sha(event_path) if events is not None else None,
            'npz_modified':False,'colormap':'YlOrRd; zero masked white',
            'note':'Color is allocated smoothed energy per grid cell, not original single-event energy.'}
    (out/'plot_config.json').write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'图片与审计结果：{out.resolve()}')

if __name__=='__main__': main()
