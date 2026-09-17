"""Three neural architectures with a shared, metre-consistent training contract."""
import math
import numpy as np
from workflow import iter_batches,model_input,write_json
import tensorflow as tf
from tensorflow import keras
L=keras.layers
from runtime_control import configure_runtime

@keras.utils.register_keras_serializable(package='ais_5_60')
class PositionEncoding(L.Layer):
 def call(self,x):
  width=x.shape[-1];position=tf.cast(tf.range(tf.shape(x)[1])[:,None],x.dtype)
  channel=tf.range(width)[None,:];rate=tf.pow(10000.,-2*tf.cast(channel//2,x.dtype)/width)
  angle=position*rate;pe=tf.where(channel%2==0,tf.sin(angle),tf.cos(angle))
  return x+pe[None,:,:]

@keras.utils.register_keras_serializable(package='ais_5_60')
class GaussianHead(L.Layer):
 def __init__(self,floor=0.001,**kwargs):super().__init__(**kwargs);self.floor=float(floor)
 def call(self,x):return tf.concat([x[...,:2],tf.nn.softplus(x[...,2:])+self.floor],axis=-1)
 def get_config(self):return dict(super().get_config(),floor=self.floor)

def make_gaussian_nll(beta=0.):
 """beta-NLL (Seitzer et al., ICLR 2022, arXiv:2203.09168).

 Plain Gaussian NLL scales each sample's gradient by 1/sigma^2, so regions the model
 currently fits badly lose weight as training proceeds and the fit converges prematurely.
 Our targets make this acute where they are zero-inflated: 36.9% of Port_Service's 60-minute
 displacements are under one metre while the tail reaches 14.9 km, so sigma collapses onto the
 stationary mass and a single tail example then costs thousands. Weighting each sample by its
 own detached sigma^(2*beta) cancels that scaling. beta=0 reproduces the original loss exactly.
 See BETA_NLL_BY_GROUP in workflow.py for which groups this is applied to and why.
 """
 def loss(y,p):
  mu=p[...,:2];sigma=p[...,2:]
  nll=tf.math.log(sigma)+.5*((y-mu)/sigma)**2+.5*math.log(2*math.pi)
  if beta:nll=tf.stop_gradient(sigma**(2*beta))*nll  # detached: the weight is not itself differentiated
  return tf.reduce_mean(tf.reduce_sum(nll,axis=-1))
 return loss

@keras.utils.register_keras_serializable(package='ais_5_60')
def gaussian_nll(y,p):
 """Retained under its original name for models saved by earlier runs."""
 return make_gaussian_nll(0.)(y,p)

def make_ade_metric(scale):
 """Mean displacement error in metres, logged next to the loss so a diverging run is visible."""
 def ade_m(y,p):return tf.reduce_mean(tf.norm((p[...,:2]-y)*scale,axis=-1))
 return ade_m

def build_model(architecture,cfg):
 configure_runtime(cfg)
 width=cfg['width'];features=18 if cfg['use_spatial_context'] else 16
 inp=L.Input((20,features),name='history_features_masks_context')
 x=L.Dense(width,activation='relu',name='input_projection')(inp)
 if architecture in ('bilstm_only','bilstm_attention'):
  # Identical encoder and pooling; attention is the only architecture-specific addition.
  for i in range(2):
   x=L.Bidirectional(L.LSTM(width,return_sequences=True,dropout=cfg['dropout']/2),name=f'bilstm_{i+1}')(x)
   x=L.LayerNormalization()(x)
  if architecture=='bilstm_attention':
   attention=L.MultiHeadAttention(num_heads=4,key_dim=width//4,dropout=cfg['dropout']/2)(x,x)
   x=L.LayerNormalization()(L.Add()([x,attention]))
 elif architecture=='transformer':
  x=PositionEncoding()(x)
  for i in range(3):
   attention=L.MultiHeadAttention(num_heads=4,key_dim=width//4,dropout=cfg['dropout']/2)(x,x)
   x=L.LayerNormalization()(L.Add()([x,attention]))
   ff=L.Dense(width*2,activation='relu')(x);ff=L.Dropout(cfg['dropout'])(ff);ff=L.Dense(width)(ff)
   x=L.LayerNormalization()(L.Add()([x,ff]))
 else:raise ValueError(architecture)
 x=L.GlobalAveragePooling1D()(x);x=L.Dense(width*2,activation='relu')(x);x=L.Dropout(cfg['dropout'])(x);x=L.Dense(width,activation='relu')(x)
 raw=L.Dense(12*(4 if cfg['probabilistic'] else 2))(x)
 out=L.Reshape((12,4 if cfg['probabilistic'] else 2))(raw)
 if cfg['probabilistic']:out=GaussianHead(cfg['sigma_floor_m']/cfg['target_scale_m'])(out)
 model=keras.Model(inp,out,name=architecture)
 loss=make_gaussian_nll(cfg.get('beta_nll',0.)) if cfg['probabilistic'] else keras.losses.Huber(delta=1.)
 model.compile(optimizer=keras.optimizers.Adam(cfg['learning_rate'],clipnorm=1.),loss=loss,
               metrics=[make_ade_metric(cfg['target_scale_m'])])
 return model

def dataset(release,plan,scaler,cfg,training):
 epoch=[0]
 def generate():
  seed=cfg['seed']+epoch[0];epoch[0]+=1
  for b in iter_batches(release,plan,cfg['batch_size'],shuffle=training,seed=seed):
   yield model_input(b,scaler,cfg['use_spatial_context']),b['y']/cfg['target_scale_m']
 features=18 if cfg['use_spatial_context'] else 16
 ds=tf.data.Dataset.from_generator(generate,output_signature=(tf.TensorSpec((None,20,features),tf.float32),tf.TensorSpec((None,12,2),tf.float32)))
 batches=sum(math.ceil(p['n']/cfg['batch_size']) for p in plan)
 ds=ds.apply(tf.data.experimental.assert_cardinality(batches))
 options=tf.data.Options();options.threading.private_threadpool_size=1;options.threading.max_intra_op_parallelism=1;options.experimental_deterministic=True
 return ds.with_options(options).prefetch(1)

def train_model(architecture,release,plans,scaler,out,cfg):
 configure_runtime(cfg)
 keras.backend.clear_session();keras.utils.set_random_seed(cfg['seed'])
 model=build_model(architecture,cfg)
 lines=[];model.summary(print_fn=lambda s,**kw:lines.append(s));(out/'architecture.txt').write_text('\n'.join(lines))
 write_json(out/'model_contract.json',dict(architecture=architecture,parameters=model.count_params(),input_features=18 if cfg['use_spatial_context'] else 16,output_shape=list(model.output_shape),target_scale_m=cfg['target_scale_m'],probabilistic=cfg['probabilistic'],sigma_floor_m=cfg['sigma_floor_m'],beta_nll=cfg.get('beta_nll',0.),monitor=cfg.get('monitor','val_ade_m')))
 # Select on metre error, not likelihood: a single heavy-tailed validation example can swing the NLL
 # several-fold, so val_loss selected checkpoints up to 55 m worse than the run's best. Both are logged.
 monitor=cfg.get('monitor','val_ade_m')
 callbacks=[keras.callbacks.EarlyStopping(monitor=monitor,mode='min',patience=cfg['patience'],restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(monitor=monitor,mode='min',factor=.5,patience=4,min_lr=1e-6),keras.callbacks.ModelCheckpoint(str(out/'best_model.keras'),monitor=monitor,mode='min',save_best_only=True),keras.callbacks.CSVLogger(str(out/'training_history.csv')),keras.callbacks.TerminateOnNaN()]
 history=model.fit(dataset(release,plans['train'],scaler,cfg,True),validation_data=dataset(release,plans['validation'],scaler,cfg,False),epochs=cfg['epochs'],callbacks=callbacks,verbose=2)
 if not all(np.isfinite(v).all() for v in history.history.values()):raise ValueError('Non-finite training history')
 write_json(out/'history.json',history.history)
 # Always reload the validation-selected checkpoint; also tests custom-layer serialization.
 loaded=keras.models.load_model(out/'best_model.keras',compile=False)
 sample=next(iter_batches(release,plans['validation'],2));x=model_input(sample,scaler,cfg['use_spatial_context'])
 if not np.isfinite(loaded(x,training=False).numpy()).all():raise ValueError('Reloaded model failed')
 return loaded

def make_predictor(model,scaler,cfg):
 def predict(b):
  p=model(model_input(b,scaler,cfg['use_spatial_context']),training=False).numpy().astype('float64')
  mu=p[...,:2]*cfg['target_scale_m']
  # Deterministic models use metre-distance residual calibration (circular regions).
  sigma=p[...,2:]*cfg['target_scale_m'] if cfg['probabilistic'] else np.ones_like(mu)
  return mu,sigma
 return predict
