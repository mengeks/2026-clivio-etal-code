"""Run Monte Carlo experiments of the score-ratio pipeline.

    python scripts/run_simulation.py configs/experiments/linear_rank1.yaml
    python scripts/run_simulation.py configs/experiments/ablations/*.yaml --jobs 12
    python scripts/run_simulation.py configs/experiments/linear_rank1.yaml --fast
    python scripts/run_simulation.py configs/experiments/ablations/*.yaml \
        --seeds $(seq 1 20) --name-suffix _20seeds     # seed-robustness runs

Each (resolved config, seed) is one job, cached in
outputs/results/_jobs/<config_hash>/seed<k>.json, so arms shared by several
experiments run once and interrupted runs resume. Per experiment the script
writes outputs/results/<name>/results.csv (one row per dataset x arm x seed)
and run_info.json (resolved configs and hashes, seeds, git revision,
environment, runtimes, medians).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FAST_OVERRIDES = {
    'treatment': {'bce': {'max_epochs': 20, 'patience': 5},
                  'ism': {'max_epochs': 20, 'patience': 5}},
    'joint': {'max_epochs': 20, 'patience': 5},
    'diagnostics': {'n_rep': 5},
}

SUMMARY_COLS = ['beta_sr_cos2', 'alpha_sr_cos2', 'beta_init_cos2', 'alpha_init_cos2']


def _job_path(out_root, cfg_hash, seed):
    return out_root / '_jobs' / cfg_hash / f'seed{seed}.json'


def _worker_init(n_threads):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import warnings
    warnings.filterwarnings('ignore', message='Liblinear failed to converge')
    from causal_embedding.utils.provenance import configure_tensorflow, limit_threads
    limit_threads(n_threads)
    configure_tensorflow(n_threads)


def _prefetch(job):
    """Compute the cached baseline directions for one (config, seed)."""
    cfg, seed, cache_dir = job
    from causal_embedding.models.pipeline import cached_baselines
    cached_baselines(cfg, seed, cache_dir)
    return seed


def _run_job(job):
    """Run one job in a fresh process; returns (key, ok, message)."""
    cfg, cfg_hash, seed, path, n_threads, cache_dir = job
    try:
        from causal_embedding.models.pipeline import run_pipeline
        from causal_embedding.utils.provenance import (environment, git_revision,
                                                       set_seed, utc_now)
        started = utc_now()
        set_seed(seed)
        row = run_pipeline(cfg, seed, cache_dir)
        record = dict(config_hash=cfg_hash, seed=seed, started=started,
                      finished=utc_now(), threads=n_threads,
                      git=git_revision(REPO), environment=environment(),
                      config=cfg, metrics=row)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=1, default=float))
        tmp.replace(path)
        return (cfg_hash, seed), True, (f"beta={row['beta_sr_cos2']:.3f} "
                                        f"alpha={row['alpha_sr_cos2']:.3f} "
                                        f"({row['runtime_sec']:.0f}s)")
    except Exception:
        return (cfg_hash, seed), False, traceback.format_exc()


def _summarise(df):
    import numpy as np
    out = {}
    for (ds, arm), g in df.groupby(['dataset', 'arm'], sort=False):
        out[f'{ds}/{arm}'] = {c: {'median': float(np.median(g[c])), 'min': float(g[c].min()),
                                  'max': float(g[c].max())}
                              for c in SUMMARY_COLS if c in g}
    return out


def write_experiment(name, jobs, spec, out_root, seeds_override, suffix=''):
    import pandas as pd
    from causal_embedding.utils.config import config_hash
    from causal_embedding.utils.provenance import utc_now
    rows, configs, gits, envs, runtime = [], {}, set(), None, 0.0
    for ds, arm, cfg, seeds in jobs:
        h = config_hash(cfg)
        configs[f'{ds}/{arm}'] = {'config_hash': h, 'config': cfg}
        for seed in seeds_override or seeds:
            path = _job_path(out_root, h, seed)
            if not path.exists():
                continue
            rec = json.loads(path.read_text())
            rows.append(dict(dataset=ds, arm=arm, config_hash=h, **rec['metrics']))
            gits.add(f"{rec['git']['commit']}{'+dirty' if rec['git']['dirty'] else ''}")
            envs = rec['environment']
            runtime += rec['metrics']['runtime_sec']
    if not rows:
        return None
    df = pd.DataFrame(rows)
    exp_dir = out_root / f'{name}{suffix}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(exp_dir / 'results.csv', index=False)
    info = dict(name=name, spec=spec, written=utc_now(), n_rows=len(df),
                git_revisions=sorted(gits), environment=envs,
                total_job_runtime_sec=round(runtime, 1),
                summary=_summarise(df), configs=configs)
    (exp_dir / 'run_info.json').write_text(json.dumps(info, indent=1, default=str))
    return exp_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('configs', nargs='+', help='experiment or ablation YAML files')
    ap.add_argument('--jobs', type=int, default=1, help='parallel worker processes')
    ap.add_argument('--threads', type=int, default=2, help='threads per worker')
    ap.add_argument('--seeds', type=int, nargs='*', help='override the seed list')
    ap.add_argument('--fast', action='store_true',
                    help='first seed only, 20 epochs; writes to outputs/results/_fast')
    ap.add_argument('--force', action='store_true', help='recompute cached jobs')
    ap.add_argument('--out', default=str(REPO / 'outputs' / 'results'))
    ap.add_argument('--name-suffix', default='',
                    help='append to experiment names in outputs/results/ (e.g. _20seeds)')
    ap.add_argument('--list', action='store_true', help='list jobs and exit')
    args = ap.parse_args()

    from causal_embedding.utils.config import config_hash, deep_merge, expand_experiment

    out_root = Path(args.out) / '_fast' if args.fast else Path(args.out)
    cache_dir = str(out_root / '_cache' / 'baselines')
    experiments, pending, seen = [], [], set()
    for path in args.configs:
        name, jobs, spec = expand_experiment(path)
        if args.fast:
            jobs = [(ds, arm, deep_merge(cfg, FAST_OVERRIDES), seeds[:1])
                    for ds, arm, cfg, seeds in jobs]
        experiments.append((name, jobs, spec))
        for ds, arm, cfg, seeds in jobs:
            h = config_hash(cfg)
            for seed in args.seeds or seeds:
                key = (h, seed)
                if key in seen:
                    continue
                seen.add(key)
                path_j = _job_path(out_root, h, seed)
                if args.force or not path_j.exists():
                    pending.append((cfg, h, seed, path_j, args.threads, cache_dir,
                                    f'{name}:{ds}/{arm}'))

    print(f"{len(seen)} unique jobs, {len(pending)} to run "
          f"({args.jobs} workers x {args.threads} threads)", flush=True)
    if args.list:
        for _, _, seed, *_, label in pending:
            print(f"  {label} seed={seed}")
        return

    t0, failures = time.time(), []
    labels = {(j[1], j[2]): j[-1] for j in pending}
    if pending:
        ctx = mp.get_context('spawn')
        from causal_embedding.utils.config import config_hash as _h
        prefetch = {}
        for cfg, _, seed, *_ in pending:
            prefetch.setdefault(_h({'dgp': cfg['dgp'], 'baselines': cfg['baselines'],
                                    'seed': seed}), (cfg, seed, cache_dir))
        # Workers are reused: every job reseeds all RNGs, and results were checked to be
        # bit-identical to running each job in a fresh process.
        with ctx.Pool(args.jobs, initializer=_worker_init, initargs=(args.threads,)) as pool:
            list(pool.imap_unordered(_prefetch, list(prefetch.values())))
            print(f"baselines ready ({len(prefetch)} datasets x seeds)  t={time.time() - t0:.0f}s",
                  flush=True)
            for i, (key, ok, msg) in enumerate(
                    pool.imap_unordered(_run_job, [j[:6] for j in pending]), 1):
                status = 'ok ' if ok else 'ERR'
                print(f"[{i}/{len(pending)}] {status} {labels[key]} seed={key[1]}  "
                      f"{msg if ok else ''}  t={time.time() - t0:.0f}s", flush=True)
                if not ok:
                    failures.append((labels[key], key[1], msg))

    for name, jobs, spec in experiments:
        exp_dir = write_experiment(name, jobs, spec, out_root, args.seeds, args.name_suffix)
        if exp_dir:
            print(f"wrote {exp_dir.relative_to(REPO) if exp_dir.is_relative_to(REPO) else exp_dir}/results.csv",
                  flush=True)
    for label, seed, msg in failures:
        print(f"\nFAILED {label} seed={seed}\n{msg}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
