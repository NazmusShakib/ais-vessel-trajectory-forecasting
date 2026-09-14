"""Select TensorFlow hardware before creating tensors or model state."""
import platform

_STATE = None


def choose_device(requested, system, has_gpu):
    if requested not in ('cpu', 'gpu', 'auto'):
        raise ValueError('device must be cpu, gpu or auto')
    if requested == 'gpu' and not has_gpu:
        raise RuntimeError('GPU requested, but TensorFlow detects no GPU. Check the job allocation and TensorFlow/CUDA environment, or select --device cpu.')
    if requested == 'auto':
        return 'cpu' if system == 'Darwin' or not has_gpu else 'gpu'
    return requested


def configure_runtime(cfg):
    global _STATE
    requested = cfg.get('device', 'cpu')
    if _STATE is not None:
        if _STATE['requested_device'] != requested:
            raise ValueError('Device changes require a fresh Python process or training worker')
        cfg['runtime'] = dict(_STATE)
        return cfg['runtime']
    import tensorflow as tf
    system = platform.system()
    gpus = tf.config.list_physical_devices('GPU')
    selected = choose_device(requested, system, bool(gpus))
    if selected == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        if system == 'Darwin':
            tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
        visible = []
    else:
        # Respect scheduler visibility; use one allocated GPU, no distribution.
        tf.config.set_visible_devices(gpus[:1], 'GPU')
        if system != 'Darwin':
            tf.config.experimental.set_memory_growth(gpus[0], True)
        visible = [gpus[0].name]
        with tf.device('/GPU:0'):
            probe = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        probe.numpy()
        if 'GPU:' not in probe.device.upper():
            raise RuntimeError('GPU was detected but the computation probe ran on CPU')
    _STATE = dict(requested_device=requested, selected_device=selected,
                  platform=system, tensorflow_version=tf.__version__,
                  detected_gpus=[g.name for g in gpus], visible_gpus=visible,
                  macos_cpu_workaround=(system == 'Darwin' and selected == 'cpu'))
    cfg['runtime'] = dict(_STATE)
    print(f"TensorFlow device: {selected.upper()} (requested: {requested}, platform: {system})", flush=True)
    return cfg['runtime']
