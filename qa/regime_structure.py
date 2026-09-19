"""Regime structure on the 0918 test split, model-free.

Error here is the constant-position baseline error, which is exactly ||y|| averaged over the twelve
horizons -- the same quantity the published skill numbers are relative to. No trained model needed.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0,'/Users/shuvo/anaconda_projects/ais_liverpool/training_5_60_v1')
from workflow import motion_regime, KNOTS_PER_MS
REL=Path('/Users/shuvo/anaconda_projects/ais_liverpool/outputs/vessel_group_forecasting/data/full_5_60_20260918T072424Z')
SHARDS=8
idx=pd.read_csv(REL/'shard_index.csv')

def load(group):
    fs=idx[(idx.group==group)&(idx.split=='test')].sort_values('file').file.tolist()[:SHARDS]
    X=[];M=[];Y=[]
    for f in fs:
        with np.load(REL/f,allow_pickle=False) as z:
            X.append(z['X']);M.append(z['X_mask']);Y.append(z['y'])
    return dict(X=np.concatenate(X),X_mask=np.concatenate(M)),np.concatenate(Y)

rows=[];dep=[];trans=[];sweep=[]
for g in sorted(idx.group.unique()):
    b,y=load(g)
    reg=motion_regime(b,0.5)
    err=np.linalg.norm(y,axis=-1).mean(1)              # constant-position ADE per window
    d60=np.linalg.norm(y[:,-1,:],axis=-1)              # net 60-min displacement, metres
    hs_kn=d60/3600/KNOTS_PER_MS                        # implied mean speed over the horizon
    n=len(y);E=err.sum()
    for r in ('under_way','not_under_way','sog_unknown'):
        ix=reg==r
        if ix.sum():
            rows.append(dict(group=g,regime=r,share=100*ix.mean(),err_share=100*err[ix].sum()/E,
                             ade_m=err[ix].mean()))
    nu=reg=='not_under_way'
    if nu.sum():
        q=np.percentile(d60[nu],[50,75,90,95,99])
        dep.append(dict(group=g,n=int(nu.sum()),p50=q[0],p75=q[1],p90=q[2],p95=q[3],p99=q[4],
                        pct_moving=100*(hs_kn[nu]>=0.5).mean(),
                        err_share_of_movers=100*err[nu][hs_kn[nu]>=0.5].sum()/max(err[nu].sum(),1e-9)))
    # input regime vs horizon regime
    hz=np.where(hs_kn>=0.5,'moves','stays')
    for r in ('under_way','not_under_way'):
        for h in ('stays','moves'):
            ix=(reg==r)&(hz==h)
            if ix.sum():trans.append(dict(group=g,input=r,horizon=h,share=100*ix.mean(),
                                          err_share=100*err[ix].sum()/E))
    for th in (0.2,0.5,1.0,2.0):
        sweep.append(dict(group=g,threshold_kn=th,under_way_pct=100*(motion_regime(b,th)=='under_way').mean()))

fmt=lambda v:f'{v:8.1f}'
print('=== 1. regime shares and error shares (0.5 kn threshold) ===')
print(pd.DataFrame(rows).pivot(index='group',columns='regime',values=['share','err_share']).to_string(float_format=fmt))
print('\n=== 2. within NOT-under-way: 60-min net displacement (m) ===')
print(pd.DataFrame(dep).set_index('group').to_string(float_format=fmt))
print('\n=== 3. input regime -> horizon behaviour ===')
print(pd.DataFrame(trans).pivot_table(index='group',columns=['input','horizon'],values='err_share').to_string(float_format=fmt))
print('\n=== 4. threshold sensitivity: % labelled under_way ===')
print(pd.DataFrame(sweep).pivot(index='group',columns='threshold_kn',values='under_way_pct').to_string(float_format=fmt))
