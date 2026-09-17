"""Read-only AIS release consumers. Experiment output paths are configurable."""
from pathlib import Path
import os,json,hashlib,math,copy
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np
import pandas as pd
from settings import read_settings,resolve_path,positive_int
HORIZONS=np.arange(5,61,5)
ROLES=('train','validation','calibration','test')
GROUPS=('Cargo','Tanker','Passenger','Port_Service','Research_Offshore','Fishing','Unknown')
KEYS=('X','X_mask','y','origin_xy_m','origin_time_ns','input_observation_time_ns','MMSI','segment_id')
BASELINES=('constant_position','cv_last_step','cv_mean_last_3','cv_mean_last_5','cv_median_last_5','cv_linear_fit_20','sog_cog_last')
# beta-NLL is applied only where the targets are zero-inflated. There, sigma collapses onto the mass of
# stationary windows and the tail then dominates the gradient, so the plain likelihood selects badly.
# Measured share of 60-minute training targets under one metre: Port_Service 36.9%, Research_Offshore
# 14.4%, and every remaining group at or below 6.7%. Pilot evidence matches: on Port_Service beta=0.5
# cut validation-loss jitter from 0.993 to 0.226, while on Cargo it raised jitter from 0.424 to 1.061.
BETA_NLL_BY_GROUP={'Port_Service':0.5,'Research_Offshore':0.5}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)))

def config(mode='smoke',groups=None,*,device=None,release=None,runs_dir=None,jobs_dir=None,env_file=None,batch_size=None):
 if mode not in ('smoke','pilot','full'):raise ValueError('mode must be smoke, pilot or full')
 settings,loaded_env=read_settings(env_file)
 device=(device or settings.get('AIS_DEVICE') or 'cpu').lower()
 if device not in ('cpu','gpu','auto'):raise ValueError('device must be cpu, gpu or auto')
 groups=[groups] if isinstance(groups,str) else list(groups or ['Cargo'])
 if groups==['all']:groups=list(GROUPS)
 if not set(groups).issubset(GROUPS) or len(set(groups))!=len(groups):raise ValueError('Use unique supported groups, or all on its own')
 dataset=release or settings.get('AIS_DATASET_DIR')
 if dataset:
  dataset=resolve_path(dataset)
 else:
  pointer_path=resolve_path(settings.get('AIS_RELEASE_POINTER') or '../outputs/vessel_group_forecasting/current_release.json')
  pointer=json.loads(pointer_path.read_text());dataset=Path(pointer['output']).expanduser()
  if not dataset.is_absolute():dataset=pointer_path.parent/dataset
  dataset=dataset.resolve()
 outputs=resolve_path(runs_dir or settings.get('AIS_RUNS_DIR') or 'runs')
 jobs=resolve_path(jobs_dir or settings.get('AIS_JOBS_DIR') or 'jobs')
 for output in (outputs,jobs):
  if output.is_relative_to(dataset):raise ValueError('Output directories must be outside the source dataset')
 return dict(mode=mode,release=str(dataset),groups=groups,seed=42,device=device,
  runs_dir=str(outputs),jobs_dir=str(jobs),env_file=loaded_env,
  epochs={'smoke':1,'pilot':15,'full':100}[mode],batch_size=positive_int(batch_size if batch_size is not None else settings.get('AIS_BATCH_SIZE') or 64,'batch_size'),learning_rate=0.0005,
  width=16 if mode=='smoke' else 128,dropout=0.2,patience=12,
  max_shards={'smoke':2,'pilot':32,'full':None}[mode],
  rows_per_shard={'smoke':64,'pilot':256,'full':None}[mode],
  coverage=0.90,target_scale_m=1000.,sigma_floor_m=1.,beta_nll=0.,beta_nll_by_group=dict(BETA_NLL_BY_GROUP),
  use_spatial_context=True,
  probabilistic=True,verify_checksums=True)

def release_info(cfg):
 release=Path(cfg['release']).resolve();manifest=json.loads((release/'manifest.json').read_text())
 if manifest['status']!='complete':raise ValueError('Release is not complete')
 if manifest['summary']['windows']!=20599126:raise ValueError('This workflow was built for release full_5_60_20260913T163619Z; review a changed release explicitly')
 index=pd.read_csv(release/'shard_index.csv')
 if int(index.windows.sum())!=manifest['summary']['windows']:raise ValueError('Index totals do not reconcile')
 if not set(cfg['groups']).issubset(GROUPS):raise ValueError('Unknown group')
 return release,index,manifest

def select_plan(index,group,role,cfg):
 frame=index[(index['group']==group)&(index['split']==role)].sort_values('file')
 # Same source cohort across architectures. This sampling seed never depends on model name.
 key=int(hashlib.sha256(f'{cfg["seed"]}:{group}:{role}'.encode()).hexdigest()[:8],16)
 rng=np.random.default_rng(key)
 if cfg['max_shards'] is not None and len(frame)>cfg['max_shards']:
  frame=frame.iloc[np.sort(rng.choice(len(frame),cfg['max_shards'],replace=False))]
 plan=[]
 for row in frame.to_dict('records'):
  count=int(row['windows']);cap=cfg['rows_per_shard']
  rows=None if cap is None or count<=cap else np.sort(rng.choice(count,cap,replace=False)).tolist()
  plan.append(dict(file=row['file'],sha256=row['sha256'],source_windows=count,rows=rows,n=count if rows is None else len(rows)))
 return plan

def plan_hash(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()

def iter_batches(release,plan,batch_size=64,shuffle=False,seed=42,keys=KEYS):
 rng=np.random.default_rng(seed);order=np.arange(len(plan))
 if shuffle:rng.shuffle(order)
 for pi in order:
  item=plan[pi];p=(Path(release)/item['file']).resolve()
  if not p.is_relative_to(Path(release).resolve()):raise ValueError('Shard escapes release')
  with np.load(p,allow_pickle=False) as z:
   selected=np.arange(item['source_windows']) if item['rows'] is None else np.asarray(item['rows'])
   arrays={k:z[k][selected] for k in keys}
  if arrays['X'].shape[1:]!=(20,8) or arrays['y'].shape[1:]!=(12,2):raise ValueError('Incompatible shard shapes')
  if arrays['X_mask'].dtype!=bool or not np.isfinite(arrays['X']).all() or not np.isfinite(arrays['y']).all():raise ValueError('Invalid arrays')
  if not np.all(arrays['X'][~arrays['X_mask']]==0):raise ValueError('Invalid missing-value fill')
  row_order=np.arange(len(selected))
  if shuffle:rng.shuffle(row_order)
  for start in range(0,len(selected),batch_size):
   ix=row_order[start:start+batch_size]
   yield {k:v[ix] for k,v in arrays.items()}

def fit_scaler(release,train_plan,batch_size=512):
 """Population moments from valid TRAIN entries only; masks are never scaled."""
 count=np.zeros(8);mean=np.zeros(8);m2=np.zeros(8)
 context_n=0;context_mean=np.zeros(2);context_m2=np.zeros(2)
 for b in iter_batches(release,train_plan,batch_size):
  for j in range(8):
   v=b['X'][...,j][b['X_mask'][...,j]].astype('float64');n=len(v)
   if n:
    delta=v.mean()-mean[j];total=count[j]+n
    m2[j]+=((v-v.mean())**2).sum()+delta**2*count[j]*n/total
    mean[j]+=delta*n/total;count[j]=total
  v=b['origin_xy_m'];n=len(v);delta=v.mean(0)-context_mean;total=context_n+n
  context_m2+=((v-v.mean(0))**2).sum(0)+delta**2*context_n*n/total
  context_mean+=delta*n/total;context_n=total
 if context_n==0:raise ValueError('No training examples')
 scale=np.sqrt(m2/np.maximum(count,1));scale[scale<1e-6]=1
 cs=np.sqrt(context_m2/context_n);cs[cs<1]=1
 return dict(mean=mean.tolist(),scale=scale.tolist(),valid_counts=count.tolist(),context_mean=context_mean.tolist(),context_scale=cs.tolist(),train_windows=context_n)

def model_input(batch,scaler,context=True):
 mask=batch['X_mask'];x=(batch['X']-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])
 x=np.where(mask,x,0);parts=[x,mask.astype('float32')]
 if context:
  c=(batch['origin_xy_m']-scaler['context_mean'])/scaler['context_scale']
  parts.append(np.repeat(c[:,None,:],20,axis=1))
 return np.concatenate(parts,axis=-1).astype('float32')

def velocity_baseline(batch,name):
 """Grid-metre extrapolation from the last actual observation, including its age."""
 n=len(batch['X']);velocity=np.zeros((n,2))
 if name=='constant_position':return np.zeros((n,12,2))
 if name not in BASELINES:raise ValueError(name)
 for i in range(n):
  t=batch['input_observation_time_ns'][i]
  _,ix=np.unique(t,return_index=True);ix=np.sort(ix)
  tt=(t[ix]-t[ix][-1]).astype('float64')/1e9;xy=batch['X'][i,ix,:2].astype('float64')
  steps=np.diff(xy,axis=0)/np.diff(tt)[:,None] if len(ix)>1 else np.empty((0,2))
  if len(steps):
   if name in ('cv_last_step','sog_cog_last'):velocity[i]=steps[-1]
   elif name=='cv_linear_fit_20':
    centered=tt-tt.mean();velocity[i]=(centered[:,None]*(xy-xy.mean(0))).sum(0)/(centered**2).sum()
   elif name=='cv_mean_last_3':velocity[i]=steps[-3:].mean(0)
   elif name=='cv_mean_last_5':velocity[i]=steps[-5:].mean(0)
   else:velocity[i]=np.median(steps[-5:],axis=0)
 if name=='sog_cog_last':
  # COG references true north; project a one-second geodesic step into the dataset CRS.
  from pyproj import Transformer,Geod
  valid=batch['X_mask'][:,-1,2:5].all(1)
  xy=batch['origin_xy_m'][valid];x=batch['X'][valid,-1]
  if len(xy):
   to_ll=Transformer.from_crs(32630,4326,always_xy=True);to_xy=Transformer.from_crs(4326,32630,always_xy=True)
   lon,lat=to_ll.transform(xy[:,0],xy[:,1]);bearing=np.degrees(np.arctan2(x[:,3],x[:,4]))
   lo,la,_=Geod(ellps='WGS84').fwd(lon,lat,bearing,x[:,2])
   east,north=to_xy.transform(lo,la);velocity[valid]=np.column_stack([east,north])-xy
 age=(batch['origin_time_ns']-batch['input_observation_time_ns'][:,-1])/1e9
 elapsed=HORIZONS[None,:]*60.+age[:,None]
 return velocity[:,None,:]*elapsed[:,:,None]

def score_distances(y,mu,sigma):return np.sqrt(np.sum(((y-mu)/sigma)**2,axis=-1))

def calibrate(release,plan,predict,out,cfg):
 """Empirical per-horizon radial adjustment. No exchangeability guarantee is claimed."""
 n=sum(p['n'] for p in plan)
 if n==0:raise ValueError('Calibration partition empty')
 path=out/'calibration_scores.npy';scores=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(n,12))
 start=0
 for b in iter_batches(release,plan,cfg['batch_size']):
  mu,sigma=predict(b);s=score_distances(b['y'],mu,sigma)
  if not np.isfinite(s).all():raise ValueError('Non-finite calibration scores')
  scores[start:start+len(s)]=s;start+=len(s)
 scores.flush();rank=int(math.ceil((n+1)*cfg['coverage']))
 if rank>n:raise ValueError('Too few calibration examples for requested quantile')
 q=[float(np.partition(np.asarray(scores[:,h]).copy(),rank-1)[rank-1]) for h in range(12)]
 result=dict(n=n,nominal_coverage=cfg['coverage'],rank=rank,q=q,scope='per_horizon_2d_region',guarantee='Empirical only: overlapping windows and temporal shift violate ordinary IID assumptions',plan_hash=plan_hash(plan))
 write_json(out/'calibration.json',result);return np.asarray(q)

def evaluate(release,plan,predict,out,cfg,role,q=None):
 total=0;error_sum=np.zeros(12);covered_sum=np.zeros(12);area_sum=np.zeros(12);by_vessel={};by_month={};examples=[]
 for b in iter_batches(release,plan,cfg['batch_size']):
  mu,sigma=predict(b);err=np.linalg.norm(mu-b['y'],axis=-1)
  if not np.isfinite(err).all() or not np.isfinite(sigma).all() or not (sigma>0).all():raise ValueError('Invalid prediction')
  covered=score_distances(b['y'],mu,sigma)<=q if q is not None else np.zeros_like(err)
  area=np.pi*sigma[...,0]*sigma[...,1]*q**2 if q is not None else np.zeros_like(err)
  error_sum+=err.sum(0);covered_sum+=covered.sum(0);area_sum+=area.sum(0);total+=len(err)
  months=pd.to_datetime(b['origin_time_ns'],utc=True).strftime('%Y-%m').to_numpy()
  for values,agg in [(b['MMSI'],by_vessel),(months,by_month)]:
   for value in np.unique(values):
    ix=values==value;key=str(value)
    a=agg.setdefault(key,[0,np.zeros(12),np.zeros(12),np.zeros(12)])
    a[0]+=int(ix.sum());a[1]+=err[ix].sum(0);a[2]+=covered[ix].sum(0);a[3]+=area[ix].sum(0)
  if len(examples)<3:
   for i in range(min(3-len(examples),len(mu))):
    examples.append(dict(X=b['X'][i],y=b['y'][i],mu=mu[i],sigma=sigma[i],origin_xy_m=b['origin_xy_m'][i],MMSI=b['MMSI'][i],origin_time_ns=b['origin_time_ns'][i]))
 if total==0:raise ValueError(f'No {role} examples')
 result=dict(role=role,windows=total,ade_m=float(error_sum.mean()/total),fde_60_m=float(error_sum[-1]/total),plan_hash=plan_hash(plan),mode=cfg['mode'])
 horizons=pd.DataFrame(dict(horizon_min=HORIZONS,mean_error_m=error_sum/total))
 if q is not None:
  horizons['empirical_coverage']=covered_sum/total;horizons['mean_region_area_m2']=area_sum/total
  result['mean_horizon_coverage']=float(covered_sum.mean()/total)
 horizons.to_csv(out/f'{role}_horizons.csv',index=False)
 for label,agg in [('vessel',by_vessel),('month',by_month)]:
  rows=[]
  for key,(n,es,cs,ars) in agg.items():
   for j,h in enumerate(HORIZONS):
    row={label:key,'windows':n,'horizon_min':h,'mean_error_m':es[j]/n}
    if q is not None:row.update(empirical_coverage=cs[j]/n,mean_region_area_m2=ars[j]/n)
    rows.append(row)
  pd.DataFrame(rows).to_csv(out/f'{role}_by_{label}.csv',index=False)
 if role=='test':
  np.savez_compressed(out/'test_examples.npz',**{k:np.stack([e[k] for e in examples]) for k in examples[0]})
 write_json(out/f'{role}_metrics.json',result);return result

def group_beta(cfg,group):
 """beta-NLL weight for one group: its per-group entry when set, otherwise the run-wide default."""
 beta=cfg.get('beta_nll_by_group',{}).get(group,cfg.get('beta_nll',0.))
 if beta<0:raise ValueError('beta_nll must not be negative')
 return float(beta)

def run_experiment(architecture,cfg):
 if architecture not in ('baselines','bilstm_attention','bilstm_only','transformer'):raise ValueError(architecture)
 cfg=copy.deepcopy(cfg);release,index,manifest=release_info(cfg)
 tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
 run=Path(cfg.get('runs_dir',ROOT/'runs'))/f'{cfg["mode"]}_{architecture}_{tag}';run.mkdir(parents=True,exist_ok=False)
 identity=dict(architecture=architecture,config=cfg,release_manifest_sha256=sha(release/'manifest.json'),source_fingerprint=manifest['fingerprint'],workflow_sha256=sha(__file__),models_sha256=sha(ROOT/'models.py'),status='running')
 write_json(run/'run.json',identity);rows=[];skips=[]
 try:
  identity['support_code_sha256']={name:sha(ROOT/name) for name in ('settings.py','runtime_control.py','run_training.py')}
  if architecture!='baselines':
   from runtime_control import configure_runtime
   configure_runtime(cfg)
   write_json(run/'runtime.json',cfg['runtime'])
  write_json(run/'run.json',identity)
  for group in cfg['groups']:
   out=run/group;out.mkdir();plans={role:select_plan(index,group,role,cfg) for role in ROLES}
   write_json(out/'cohorts.json',plans)
   if any(not plans[role] for role in ROLES):
    skips.append(dict(group=group,reason='At least one required role has no shards'));continue
   if cfg['verify_checksums']:
    for plan in plans.values():
     for item in plan:
      if sha(release/item['file'])!=item['sha256']:raise ValueError('Source checksum mismatch')
   if group in ('Fishing','Research_Offshore','Unknown'):print(f'{group}: exploratory group; review vessel support and inherited mapping.')
   print(f'{group}: '+', '.join(f'{role}={sum(p["n"] for p in plan):,}' for role,plan in plans.items()),flush=True)
   if architecture=='baselines':
    validation=[]
    for name in BASELINES:
     model_out=out/name;model_out.mkdir()
     predict=lambda b,name=name:(velocity_baseline(b,name),np.ones_like(b['y'],dtype='float64'))
     val=evaluate(release,plans['validation'],predict,model_out,cfg,'validation');validation.append(dict(model=name,**val))
    best=min(validation,key=lambda r:r['ade_m'])['model']
    write_json(out/'selected_baseline.json',dict(selected_on='validation ADE',model=best))
    pd.DataFrame(validation).to_csv(out/'baseline_validation_comparison.csv',index=False)
    for name in BASELINES:
     model_out=out/name
     predict=lambda b,name=name:(velocity_baseline(b,name),np.ones_like(b['y'],dtype='float64'))
     q=calibrate(release,plans['calibration'],predict,model_out,cfg)
     result=evaluate(release,plans['test'],predict,model_out,cfg,'test',q)
     rows.append(dict(group=group,model=name,selected_baseline=name==best,**result))
   else:
    from models import train_model,make_predictor
    # beta is resolved per group; everything else in the configuration is shared across the run.
    group_cfg=dict(cfg,beta_nll=group_beta(cfg,group))
    scaler=fit_scaler(release,plans['train']);write_json(out/'scaler.json',scaler)
    model=train_model(architecture,release,plans,scaler,out,group_cfg)
    predict=make_predictor(model,scaler,group_cfg)
    evaluate(release,plans['validation'],predict,out,cfg,'validation')
    q=calibrate(release,plans['calibration'],predict,out,cfg)
    result=evaluate(release,plans['test'],predict,out,cfg,'test',q)
    rows.append(dict(group=group,model=architecture,selected_baseline=False,**result))
   print(f'{group} complete.',flush=True)
  pd.DataFrame(rows).to_csv(run/'comparison.csv',index=False)
  write_json(run/'skipped_groups.json',skips)
  identity['status']='complete';identity['completed_at']=datetime.now(timezone.utc).isoformat();write_json(run/'run.json',identity)
 except Exception as e:
  identity.update(status='failed',error=repr(e));write_json(run/'run.json',identity);raise
 return run

def plot_run(run):
 import matplotlib.pyplot as plt
 from matplotlib.patches import Ellipse
 run=Path(run);report=pd.read_csv(run/'comparison.csv');figs=[]
 for group,g in report.groupby('group'):
  fig,axes=plt.subplots(1,2,figsize=(12,4))
  for _,row in g.iterrows():
   p=run/group/(row['model'] if (run/group/row['model']).is_dir() else '')
   h=pd.read_csv(p/'test_horizons.csv')
   axes[0].plot(h.horizon_min,h.mean_error_m,label=row['model'])
   axes[1].plot(h.horizon_min,h.empirical_coverage,label=row['model'])
  axes[0].set(xlabel='Look-ahead (minutes)',ylabel='Mean position error (metres)')
  axes[1].axhline(json.loads((run/'run.json').read_text())['config']['coverage'],color='black',ls='--',label='Nominal level')
  axes[1].set(xlabel='Look-ahead (minutes)',ylabel='Empirical per-horizon coverage',ylim=(0,1.05))
  axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
  fig.suptitle(f'{group} | {report.iloc[0]["mode"].upper()} run | exploratory test');fig.tight_layout()
  path=run/f'{group}_errors_coverage.png';fig.savefig(path,dpi=140);figs.append(fig)
  first=g.iloc[0];p=run/group/(first['model'] if (run/group/first['model']).is_dir() else '')
  with np.load(p/'test_examples.npz',allow_pickle=False) as z:
   x,y,mu,sig=z['X'][0,:,:2],z['y'][0],z['mu'][0],z['sigma'][0]
  q=np.asarray(json.loads((p/'calibration.json').read_text())['q'])
  fig,ax=plt.subplots(figsize=(6,5));ax.plot(x[:,0],x[:,1],'.-',label='Observed inputs');ax.plot(y[:,0],y[:,1],'o-',label='AIS target labels');ax.plot(mu[:,0],mu[:,1],'x--',label='Predicted mean')
  for h in [0,5,11]:ax.add_patch(Ellipse(mu[h],2*q[h]*sig[h,0],2*q[h]*sig[h,1],fill=False,alpha=.45))
  ax.set(xlabel='East from last observation (m)',ylabel='North from last observation (m)',title=f'{group}: {first["model"]}\nActual saved example; regions at 5, 30, 60 min');ax.axis('equal');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(run/f'{group}_example.png',dpi=140);figs.append(fig)
 return figs

def compare_runs(paths):
 frames=[];identity=None;cohorts={}
 for p in map(Path,paths):
  meta=json.loads((p/'run.json').read_text())
  if meta['status']!='complete':raise ValueError('Cannot compare incomplete runs')
  signature=(meta['release_manifest_sha256'],meta['config']['mode'],meta['config']['probabilistic'],meta['config']['use_spatial_context'])
  if identity is None:identity=signature
  if identity!=signature:raise ValueError('Dataset/mode/output/context mismatch')
  for group in meta['config']['groups']:
   c=json.loads((p/group/'cohorts.json').read_text());hs={r:plan_hash(v) for r,v in c.items()}
   if group in cohorts and cohorts[group]!=hs:raise ValueError('Mismatched training/evaluation cohorts')
   cohorts[group]=hs
  f=pd.read_csv(p/'comparison.csv');f['run']=str(p);frames.append(f)
 return pd.concat(frames,ignore_index=True)

def run_isolated(architecture,cfg):
 """Keep TensorFlow's native runtime out of the interactive notebook kernel.

 The same runner is used as the CLI. Progress is streamed into the cell and a
 local job log. Arguments/configuration are passed via JSON, never a shell.
 """
 import subprocess,sys
 tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
 job=Path(cfg.get('jobs_dir',ROOT/'jobs'))/f'{architecture}_{tag}';job.mkdir(parents=True,exist_ok=False)
 write_json(job/'request.json',dict(architecture=architecture,config=cfg))
 env=os.environ.copy();env['MPLBACKEND']='Agg';env['PYTHONUNBUFFERED']='1'
 with open(job/'worker.log','w') as log:
  process=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),str(job)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  try:
   for line in process.stdout:
    log.write(line);log.flush();print(line,end='',flush=True)
   code=process.wait()
   if code:raise RuntimeError(f'Training worker failed (exit {code}); inspect {job}/worker.log')
  except BaseException:
   process.terminate()
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   raise
 return Path(json.loads((job/'result.json').read_text())['run'])
