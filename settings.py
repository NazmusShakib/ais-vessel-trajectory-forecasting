"""Portable settings. Read .env as data; never execute it as shell code."""
from pathlib import Path
import os
import re
import shlex

ROOT = Path(__file__).resolve().parent
KEYS = {'AIS_DEVICE', 'AIS_DATASET_DIR', 'AIS_RELEASE_POINTER',
        'AIS_RUNS_DIR', 'AIS_JOBS_DIR', 'AIS_BATCH_SIZE'}


def read_settings(env_file=None):
    path = Path(env_file).expanduser().resolve() if env_file else ROOT / '.env'
    values = {}
    if env_file and not path.is_file():
        raise FileNotFoundError(f'Environment file does not exist: {path}')
    if path.is_file():
        for number, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            key, sep, value = line.partition('=')
            key = key.strip()
            if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
                raise ValueError(f'{path}:{number}: expected KEY=value')
            if key not in KEYS:
                raise ValueError(f'{path}:{number}: unsupported setting {key}')
            tokens = shlex.split(value, comments=True, posix=True)
            if len(tokens) > 1:
                raise ValueError(f'{path}:{number}: quote paths containing spaces')
            values[key] = tokens[0] if tokens else ''
    values.update({k: os.environ[k] for k in KEYS if k in os.environ})
    return values, str(path) if path.is_file() else None


def resolve_path(value):
    """Relative setting/CLI paths start at the training folder, not the cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def positive_int(value, name):
    result = int(value)
    if result < 1:
        raise ValueError(f'{name} must be a positive integer')
    return result
