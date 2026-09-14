"""Configuration precedence, relocation and device-policy regression checks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from settings import ROOT, read_settings
from workflow import config, GROUPS
import runtime_control as runtime

class SettingsTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'qa');self.addCleanup(self.temp.cleanup)
  self.env=Path(self.temp.name)/'settings.env'
  self.env.write_text('AIS_DATASET_DIR=./qa/example_release\nAIS_DEVICE=auto\n')
  self.environment=patch.dict(os.environ,{},clear=True);self.environment.start();self.addCleanup(self.environment.stop)
 def test_env_does_not_require_old_pointer(self):
  cfg=config('full',['all'],env_file=self.env)
  self.assertEqual(cfg['groups'],list(GROUPS));self.assertEqual(cfg['release'],str(ROOT/'qa/example_release'))
  self.assertIsNone(cfg['max_shards']);self.assertIsNone(cfg['rows_per_shard'])
 def test_cli_over_environment_over_file(self):
  with patch.dict(os.environ,{'AIS_DEVICE':'gpu','AIS_BATCH_SIZE':'32'}):
   self.assertEqual(config(env_file=self.env)['device'],'gpu')
   cfg=config(env_file=self.env,device='cpu',batch_size=8,release='qa/override')
  self.assertEqual(cfg['device'],'cpu');self.assertEqual(cfg['batch_size'],8);self.assertEqual(cfg['release'],str(ROOT/'qa/override'))
 def test_quoted_paths_and_no_shell_execution(self):
  self.env.write_text('AIS_DATASET_DIR="./qa/a # b" # comment\nAIS_RUNS_DIR="./qa/$(touch NOPE)"\n')
  cfg=config(env_file=self.env)
  self.assertEqual(cfg['release'],str(ROOT/'qa/a # b'));self.assertIn('$(touch NOPE)',cfg['runs_dir'])
 def test_relative_paths_independent_of_working_directory(self):
  before=Path.cwd()
  try:
   os.chdir(self.temp.name);cfg=config(env_file=self.env,runs_dir='qa/test_runs')
  finally:os.chdir(before)
  self.assertEqual(cfg['runs_dir'],str(ROOT/'qa/test_runs'))
 def test_relative_pointer_output_resolves_beside_pointer(self):
  pointer=Path(self.temp.name)/'pointer.json';pointer.write_text(json.dumps({'output':'release'}))
  self.env.write_text(f'AIS_RELEASE_POINTER={pointer}\n')
  self.assertEqual(config(env_file=self.env)['release'],str(pointer.parent/'release'))
 def test_bad_config_rejected_before_training(self):
  for kwargs in ({'groups':['all','Cargo']},{'groups':['Cargo','Cargo']},{'batch_size':0},{'device':'cuda'},{'runs_dir':'qa/example_release/outputs'}):
   with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):config(env_file=self.env,**kwargs)
  self.env.write_text('AIS_DEVCIE=cpu\n')
  with self.assertRaises(ValueError):read_settings(self.env)
  with self.assertRaises(FileNotFoundError):read_settings(self.env.parent/'absent.env')

class RuntimeTests(unittest.TestCase):
 def test_selection_policy(self):
  for system in ('Linux','Darwin','Windows'):
   self.assertEqual(runtime.choose_device('cpu',system,True),'cpu');self.assertEqual(runtime.choose_device('auto',system,False),'cpu')
  self.assertEqual(runtime.choose_device('auto','Darwin',True),'cpu');self.assertEqual(runtime.choose_device('auto','Linux',True),'gpu')
  self.assertEqual(runtime.choose_device('gpu','Linux',True),'gpu')
  with self.assertRaises(RuntimeError):runtime.choose_device('gpu','Linux',False)
 def test_gpu_respects_visible_allocation_and_probes(self):
  tf=MagicMock();gpu=MagicMock();gpu.name='/physical_device:GPU:0'
  tf.config.list_physical_devices.return_value=[gpu];tf.matmul.return_value.device='/job:localhost/device:GPU:0';tf.__version__='test'
  with patch.dict('sys.modules',{'tensorflow':tf}),patch.object(runtime,'_STATE',None),patch.object(runtime.platform,'system',return_value='Linux'):
   cfg={'device':'gpu'};state=runtime.configure_runtime(cfg)
   tf.config.set_visible_devices.assert_called_once_with([gpu],'GPU');tf.config.experimental.set_memory_growth.assert_called_once_with(gpu,True)
   tf.matmul.return_value.numpy.assert_called_once();self.assertEqual(state['selected_device'],'gpu')
   with self.assertRaises(ValueError):runtime.configure_runtime({'device':'cpu'})
 def test_gpu_silent_cpu_fallback_is_rejected(self):
  tf=MagicMock();tf.config.list_physical_devices.return_value=[MagicMock()];tf.matmul.return_value.device='/device:CPU:0'
  with patch.dict('sys.modules',{'tensorflow':tf}),patch.object(runtime,'_STATE',None),patch.object(runtime.platform,'system',return_value='Linux'):
   with self.assertRaises(RuntimeError):runtime.configure_runtime({'device':'gpu'})

if __name__=='__main__':unittest.main(verbosity=2)
