import unittest,tempfile,json
from pathlib import Path
import numpy as np
from workflow import velocity_baseline,model_input,fit_scaler,select_plan,score_distances,calibrate,HORIZONS
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

if __name__=='__main__':unittest.main(verbosity=2)
