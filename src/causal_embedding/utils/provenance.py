"""Seeding, thread limits and run metadata."""

import os
import platform
import random
import subprocess
from datetime import datetime, timezone


def limit_threads(n_threads):
    """Cap BLAS/OpenMP threads; call before numpy / TensorFlow are imported."""
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = str(n_threads)


def configure_tensorflow(n_threads, deterministic=True):
    """Thread counts and op determinism; call before the first TF op."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def set_seed(seed):
    """Seed Python, NumPy's global RNG and TensorFlow's global RNG."""
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def git_revision(repo_dir):
    def run(*args):
        return subprocess.run(['git', '-C', str(repo_dir), *args],
                              capture_output=True, text=True).stdout.strip()
    rev = run('rev-parse', 'HEAD')
    dirty = bool(run('status', '--porcelain', '--', 'src', 'configs', 'scripts'))
    return {'commit': rev or None, 'dirty': dirty}


def environment():
    import numpy, pandas, scipy, sklearn, tensorflow
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'hostname': platform.node(),
        'tensorflow': tensorflow.__version__,
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'scikit-learn': sklearn.__version__,
        'pandas': pandas.__version__,
        'devices': [d.device_type for d in tensorflow.config.list_physical_devices()],
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
