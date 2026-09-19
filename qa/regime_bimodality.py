"""Is the trained model's failure on transition windows a bimodality problem?

Models are the 0915 bilstm_attention runs, trained against release 0913 (manifest sha verified),
so they are evaluated on 0913's OWN test split. Using 0918's test split would pull in windows that
were calibration or validation for these models.

Prediction: on transition windows the Gaussian should sit BETWEEN the modes -- predicting motion
where the vessel stays, or stillness where it departs -- rather than simply being wide.
"""
import json,sys,numpy as np,pandas as pd
from pathlib import Path
sys.path.insert(0,'/Users/shuvo/anaconda_projects/ais_liverpool/training_5_60_v1')
import tensorflow as tf, keras
from workflow import motion_regime,KNOTS_PER_MS,iter_batches
from models import make_predictor
REL=Path('/Users/shuvo/anaconda_projects/ais_liverpool/outputs/vessel_group_forecasting/data/full_5_60_20260913T163619Z')
RES=Path('/Users/shuvo/anaconda_projects/ais_liverpool/training_5_60_v1/ais_results')
idx=pd.read_csv(REL/'shard_index.csv')
models={p.parent.name:p.parent for p in RES.rglob('best_model.keras')}
cfg=dict(use_spatial_context=True,probabilistic=True,target_scale_m=1000.)

rows=[]
for g in ('Port_Service','Research_Offshore','Cargo'):
    d=models[g];scaler=json.loads((d/'scaler.json').read_text())
    m=keras.models.load_model(d/'best_model.keras',compile=False)
    predict=make_predictor(m,scaler,cfg)
    files=idx[(idx.group==g)&(idx.split=='test')].sort_values('file').file.tolist()[:6]
    plan=[dict(file=f,source_windows=int(idx[idx.file==f].windows.iloc[0]),rows=None,
               n=int(idx[idx.file==f].windows.iloc[0])) for f in files]
    MU=[];Y=[];SG=[];X=[];XM=[]
    for b in iter_batches(REL,plan,256):
        mu,sg=predict(b);MU.append(mu);SG.append(sg);Y.append(b['y']);X.append(b['X']);XM.append(b['X_mask'])
    mu=np.concatenate(MU);sg=np.concatenate(SG);y=np.concatenate(Y)
    reg=motion_regime(dict(X=np.concatenate(X),X_mask=np.concatenate(XM)),0.5)
    err=np.linalg.norm(mu-y,axis=-1).mean(1)
    true60=np.linalg.norm(y[:,-1,:],axis=-1);pred60=np.linalg.norm(mu[:,-1,:],axis=-1)
    hz=np.where(true60/3600/KNOTS_PER_MS>=0.5,'moves','stays')
    for r in ('not_under_way','under_way'):
        for h in ('stays','moves'):
            ix=(reg==r)&(hz==h)
            if ix.sum()<20:continue
            rows.append(dict(group=g,input=r,horizon=h,n=int(ix.sum()),
                pct=100*ix.mean(),err_share=100*err[ix].sum()/err.sum(),
                ade_m=err[ix].mean(),true60_med=np.median(true60[ix]),
                pred60_med=np.median(pred60[ix]),sigma_med=np.median(sg[ix][:,-1,:].mean(-1)),
                err_over_sigma=err[ix].mean()/np.median(sg[ix][:,-1,:].mean(-1))))
    del m;keras.backend.clear_session()
pd.set_option('display.width',220)
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda v:f'{v:9.1f}'))
