import unittest,tempfile,json
from pathlib import Path
import numpy as np
from workflow import velocity_baseline,model_input,fit_scaler,select_plan,score_distances,calibrate,HORIZONS,iter_batches,env_width,ENV_BLOCKS,HORIZON_SPLIT_MIN
import pandas as pd

class ContractTests(unittest.TestCase):
 def batch(self,n=2):
  times=(np.arange(20)*60).astype('int64')*10**9
  b=dict(X=np.zeros((n,20,8),dtype='float32'),X_mask=np.ones((n,20,8),bool),y=np.zeros((n,12,2),dtype='float32'),origin_xy_m=np.tile([500000.,5900000.],(n,1)),origin_time_ns=np.full(n,times[-1]+120*10**9),input_observation_time_ns=np.tile(times,(n,1)),MMSI=np.ones(n,dtype='int64'),segment_id=np.array(['test']*n))
  b['X'][:,:,0]=(np.arange(20)-19)*600;b['X'][:,:,2]=10;b['X'][:,:,3]=1
  return b
 def test_age_adjusted_velocity(self):
  b=self.batch();pred=velocity_baseline(b,'cv_last_step')
  np.testing.assert_allclose(pred[:,:,0],np.tile((HORIZONS*60+120)*10,(2,1)));np.testing.assert_allclose(pred[:,:,1],0)
 def test_duplicate_observations_are_not_zero_speed(self):
  b=self.batch();b['input_observation_time_ns'][:,-1]=b['input_observation_time_ns'][:,-2];b['X'][:,-1,:2]=b['X'][:,-2,:2]
  pred=velocity_baseline(b,'cv_last_step');self.assertAlmostEqual(pred[0,0,0],4800)
 def test_stationary_baseline(self):self.assertFalse(velocity_baseline(self.batch(),'constant_position').any())
 def test_velocity_fit(self):
  b=self.batch();np.testing.assert_allclose(velocity_baseline(b,'cv_linear_fit_20'),velocity_baseline(b,'cv_last_step'))
 def test_missing_sog_uses_observed_velocity(self):
  b=self.batch();b['X_mask'][:,-1,2]=False
  np.testing.assert_allclose(velocity_baseline(b,'sog_cog_last'),velocity_baseline(b,'cv_last_step'))
 def test_feature_mask_is_preserved(self):
  b=self.batch();b['X_mask'][0,0,2]=False;b['X'][0,0,2]=0
  scaler=dict(mean=[5]*8,scale=[2]*8,context_mean=[500000.,5900000.],context_scale=[1,1])
  x=model_input(b,scaler);self.assertEqual(x.shape,(2,20,18));self.assertEqual(x[0,0,2],0);self.assertEqual(x[0,0,10],0);self.assertEqual(x[1,0,10],1)
 def test_scaler_ignores_missing_and_other_roles(self):
  b=self.batch();b['X_mask'][0,:,2]=False;b['X'][0,:,2]=0
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'qa') as d:
   np.savez(Path(d)/'a.npz',**b)
   plan=[dict(file='a.npz',source_windows=2,rows=None,n=2)]
   scaler=fit_scaler(d,plan)
   self.assertEqual(scaler['mean'][2],10);self.assertEqual(scaler['valid_counts'][2],20)
 def env_fixture(self,d,aligned=True):
  """A one-shard release plus a row-aligned sidecar, optionally with a mismatched key."""
  b=self.batch(2);shard=Path(d)/'groups/Cargo/train';shard.mkdir(parents=True)
  np.savez(shard/'s.npz',**b)
  side=Path(d)/'side/groups/Cargo/train';side.mkdir(parents=True)
  pd.DataFrame(dict(segment_id=b['segment_id'] if aligned else np.array(['elsewhere']*2),
   origin_time_ns=b['origin_time_ns'],cur_east=[0.5,9.0],cur_north=[0.,9.],
   cur_rel_sin=[0.,9.],cur_rel_cos=[1.,9.],current_valid=[True,False])).to_parquet(side/'s.parquet')
  return [dict(file='groups/Cargo/train/s.npz',source_windows=2,rows=None,n=2)],(Path(d)/'side',('CUR',))
 def test_env_block_width_matches_model_input(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'qa') as d:
   plan,env=self.env_fixture(d)
   b=next(iter_batches(d,plan,2,env=env))
   self.assertEqual(b['env'].shape[1]+b['env_valid'].shape[1],env_width(('CUR',)))
   scaler=dict(mean=[0]*8,scale=[1]*8,context_mean=[0.,0.],context_scale=[1,1],
    env_mean=[0]*4,env_scale=[1]*4,env_blocks=['CUR'])
   x=model_input(b,scaler);self.assertEqual(x.shape,(2,20,18+env_width(('CUR',))))
 def test_env_invalid_rows_are_zeroed_and_flagged(self):
  """An unavailable source must read as nothing known, not as whatever filler the sidecar carried."""
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'qa') as d:
   plan,env=self.env_fixture(d)
   b=next(iter_batches(d,plan,2,env=env))
   scaler=dict(mean=[0]*8,scale=[1]*8,context_mean=[0.,0.],context_scale=[1,1],
    env_mean=[0]*4,env_scale=[1]*4,env_blocks=['CUR'])
   x=model_input(b,scaler)
   self.assertTrue((x[1,0,18:22]==0).all())      # invalid row: features zeroed despite 9.0 in the file
   self.assertEqual(x[1,0,22],0)                  # and its validity flag is off
   self.assertEqual(x[0,0,18],0.5);self.assertEqual(x[0,0,22],1)
 def test_env_sidecar_misalignment_is_rejected(self):
  """The sidecar is row-aligned rather than key-joined, so the keys must be checked, not trusted."""
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'qa') as d:
   plan,env=self.env_fixture(d,aligned=False)
   with self.assertRaises(ValueError):next(iter_batches(d,plan,2,env=env))
 def test_horizon_bands_partition_the_horizons(self):
  """Both bands must be non-empty and together cover every horizon exactly once.

  The bands are pre-registered, so a change to HORIZON_SPLIT_MIN that silently emptied one of them
  would invalidate the registration rather than merely alter a report.
  """
  short=HORIZONS<=HORIZON_SPLIT_MIN
  self.assertTrue(short.any());self.assertTrue((~short).any())
  self.assertEqual(short.sum()+(~short).sum(),len(HORIZONS))
  self.assertEqual(sorted([*HORIZONS[short],*HORIZONS[~short]]),sorted(HORIZONS))
  self.assertEqual(int(HORIZONS[short].max()),HORIZON_SPLIT_MIN)
 def test_role_plan_is_deterministic(self):
  frame=pd.DataFrame([dict(group='Cargo',split=r,file=f'{r}/{i}',windows=100,sha256='a') for r in ['train','test'] for i in range(8)])
  cfg=dict(seed=42,max_shards=2,rows_per_shard=10)
  p=select_plan(frame,'Cargo','train',cfg);self.assertEqual(p,select_plan(frame,'Cargo','train',cfg));self.assertTrue(all(x['file'].startswith('train/') for x in p));self.assertEqual(sum(x['n'] for x in p),20)
 def test_calibration_rank_and_units(self):
  b=self.batch(20);b['y'][:,:,0]=np.arange(1,21)[:,None]
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'qa') as d:
   np.savez(Path(d)/'a.npz',**b);plan=[dict(file='a.npz',source_windows=20,rows=None,n=20)]
   q=calibrate(d,plan,lambda b:(np.zeros_like(b['y']),np.ones_like(b['y'])),Path(d),dict(batch_size=4,coverage=.9))
   np.testing.assert_allclose(q,19)
 def test_two_dimensional_region(self):
  y=np.array([[[3.,4.]]]);np.testing.assert_allclose(score_distances(y,np.zeros_like(y),np.ones_like(y)),5)
 def test_motion_regime_uses_valid_sog_only(self):
  from workflow import motion_regime,KNOTS_PER_MS
  b={'X':np.zeros((4,20,8)),'X_mask':np.ones((4,20,8),bool)}
  b['X'][0,:,2]=2.0                      # clearly under way
  b['X'][1,:,2]=0.05                     # clearly not
  b['X'][2,:,2]=0.51/KNOTS_PER_MS        # just above the threshold. Exactly ON it is left
                                         # unspecified: averaging 20 identical floats need not
                                         # return that float, so the boundary case is arbitrary.
  b['X'][3,:,2]=9.9;b['X_mask'][3,:,2]=False   # no valid SOG at all
  r=motion_regime(b,0.5)
  self.assertEqual(list(r),['under_way','not_under_way','under_way','sog_unknown'])
 def test_motion_regime_ignores_masked_steps(self):
  from workflow import motion_regime
  b={'X':np.zeros((1,20,8)),'X_mask':np.ones((1,20,8),bool)}
  b['X'][0,:,2]=0.05;b['X'][0,:5,2]=50.;b['X_mask'][0,:5,2]=False  # masked outliers must not count
  self.assertEqual(motion_regime(b,0.5)[0],'not_under_way')

if __name__=='__main__':unittest.main(verbosity=2)
