"""YAML configuration: defaults, dataset files, ablation arms.

Resolution order (later wins):  configs/default.yaml
                                <- configs/experiments/<dataset>.yaml (dgp + overrides)
                                <- arm overrides of an ablation file.

``causal_embedding/default.yaml`` is a symlink to ``configs/default.yaml`` so
the defaults ship with the installed package.
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'default.yaml'


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def default_config():
    return load_yaml(DEFAULT_CONFIG)


def pipeline_defaults():
    """Pipeline section of default.yaml (no DGP)."""
    return default_config()['pipeline']


def evaluation_criteria():
    return default_config()['evaluation']


def _experiments_dir(path):
    for parent in Path(path).resolve().parents:
        if parent.name == 'experiments':
            return parent
    raise FileNotFoundError(f"{path} is not inside a configs/experiments folder")


def dataset_config(name, experiments_dir):
    """default.yaml merged with <experiments_dir>/<name>.yaml."""
    ds = load_yaml(Path(experiments_dir) / f'{name}.yaml')
    cfg = deep_merge(pipeline_defaults(), ds.get('overrides', {}))
    cfg['dgp'] = ds['dgp']
    return cfg, ds


def expand_experiment(path):
    """Experiment file -> (name, [(dataset, arm, resolved_cfg, seeds)], spec).

    A dataset file (has 'dgp') is a one-arm experiment named 'main'.
    An ablation file lists 'datasets' and 'arms' (overrides per arm) and may
    replace the datasets' seed lists with its own 'seeds'.
    """
    spec = load_yaml(path)
    if 'dgp' in spec:
        datasets, arms = [Path(path).stem], {'main': {}}
    else:
        datasets, arms = spec['datasets'], spec['arms']
    exp_dir = _experiments_dir(path)
    jobs = []
    for ds_name in datasets:
        base, ds = dataset_config(ds_name, exp_dir)
        for arm, override in arms.items():
            jobs.append((ds_name, arm, deep_merge(base, override),
                         spec.get('seeds', ds['seeds'])))
    return spec.get('name', Path(path).stem), jobs, spec
