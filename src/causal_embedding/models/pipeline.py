"""Score-ratio prognostic direction pipeline (one dataset, one config).

1. Warm start (alpha_0, beta_0): SAVE / LASSO / random / oracle.
2. Treatment score w_A: BCE propensity network (paper) or ISM network.
   beta_hat = leading eigenvector of H_A = E[w_A w_A^T].
3. Joint score w_AY: ISM bottleneck network, W_v initialised to orth[beta_0, alpha_0].
4. alpha_hat = leading eigenvector of H_0 built from controls.

Every run also records the H_0 alternatives of research note A09 and the
trace-estimator spread of note A10, which need no extra training.
Oracle quantities are used only for evaluation.
"""

import os
import time
from pathlib import Path

import numpy as np

from causal_embedding.models.baselines import baseline_directions, standardize
from causal_embedding.models.ism import (score_vectors, train_ism,
                                         trace_estimator_spread)
from causal_embedding.models.propensity import (PropensityNet, train_propensity,
                                                treatment_scores)
from causal_embedding.models.score_ratio_reduction import ScoreRatioNet
from causal_embedding.simulations.dgp import simulate
from causal_embedding.utils.config import config_hash
from causal_embedding.utils.provenance import set_seed
from causal_embedding.utils.metrics import (cosine_sq, score_mse, second_moment,
                                            top_eigvecs)


def _orth_init(first_cols, rank, p, seed):
    """Orthonormal p x rank matrix whose span starts with ``first_cols``.

    Extra columns (rank > len(first_cols)) are Gaussian, seeded.
    """
    cols = [np.asarray(c, dtype=np.float64).reshape(-1, 1) for c in first_cols]
    n_extra = rank - len(cols)
    if n_extra > 0:
        cols.append(np.random.default_rng(seed + 5555).standard_normal((p, n_extra)))
    Q, _ = np.linalg.qr(np.hstack(cols).astype(np.float32))
    return Q[:, :rank].astype(np.float32)


def cached_baselines(cfg, seed, cache_dir=None):
    """Baseline directions for (dgp, baselines config, seed), cached on disk.

    The estimators are deterministic; the L1-logistic fit takes ~2 min on the
    linear DGP, so experiments sharing a dataset compute it once.
    """
    key = config_hash({'dgp': cfg['dgp'], 'baselines': cfg['baselines'], 'seed': seed})
    path = Path(cache_dir) / f'{key}.npz' if cache_dir else None
    if path is not None and path.exists():
        z = np.load(path)
        return {m: (z[f'{m}_alpha'], z[f'{m}_beta']) for m in ('save', 'sir', 'lasso')}
    data = simulate(cfg['dgp'], seed)
    out = baseline_directions(standardize(data['X']), data['A'], data['Y'], cfg['baselines'])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
        np.savez(tmp, **{f'{m}_{k}': v for m, (a, b) in out.items()
                         for k, v in (('alpha', a), ('beta', b))})
        tmp.replace(path)
    return out


def initial_directions(method, p, baselines, seed, oracle=None):
    """Warm start (alpha_0, beta_0) for the networks."""
    if method in ('save', 'lasso'):
        return baselines[method]
    if method == 'random':
        rng = np.random.default_rng(seed + 7777)
        beta0 = rng.standard_normal(p).astype(np.float32); beta0 /= np.linalg.norm(beta0)
        alpha0 = rng.standard_normal(p).astype(np.float32); alpha0 /= np.linalg.norm(alpha0)
        return alpha0, beta0
    if method == 'oracle':
        if oracle is None:
            raise ValueError("init method 'oracle' needs the true directions")
        return oracle
    raise ValueError(f"unknown init method {method!r}")


def fit_treatment(cfg, V, A, beta0, seed):
    """Returns (score_fn(V, A) -> w_A, info dict)."""
    p = V.shape[1]
    if cfg['rank'] == 1:   # W_A = beta_0 / ||beta_0||, as in the original scripts
        W0 = (beta0 / np.linalg.norm(beta0)).reshape(-1, 1).astype(np.float32)
    else:
        W0 = _orth_init([beta0], cfg['rank'], p, seed)
    if cfg['objective'] == 'bce':
        net = PropensityNet(p, rank=cfg['rank'], hidden=tuple(cfg['hidden']))
        net.W_A.assign(W0, read_value=False)
        net.normalize()
        b = cfg['bce']
        ep, val, ran = train_propensity(
            net, V, A, val_frac=b['val_frac'], max_epochs=b['max_epochs'],
            lr=b['lr'], l2=b['l2'], patience=b['patience'],
            batch_size=b['batch_size'], clip_norm=b['clip_norm'],
            normalize=cfg['normalize'], seed=seed)
        info = dict(selected_epoch_A=ep, val_loss_A=val, epochs_run_A=ran,
                    norm_W_A=float(np.linalg.norm(net.W_A.numpy())))
        return (lambda V_, A_: treatment_scores(net, V_, A_)), info
    if cfg['objective'] == 'ism':
        m = cfg['ism']
        net = ScoreRatioNet(p, obs_dim=1, r_prime=cfg['rank'], s_prime=1,
                            hidden_units=tuple(m['hidden']), name='A_ism')
        net.W_v.assign(W0, read_value=False)
        ep, val, ran = train_ism(
            net, V, A.reshape(-1, 1), val_frac=m['val_frac'],
            max_epochs=m['max_epochs'], lr=m['lr'], lam1=0.0, lam2=m['lam2'],
            lam_l2=m['lam_l2'], patience=m['patience'], clip_norm=m['clip_norm'],
            stiefel=cfg['normalize'], trace=m['trace'], n_hutch=m['n_hutch'],
            seed=seed, split_offset=999)
        info = dict(selected_epoch_A=ep, val_loss_A=val, epochs_run_A=ran,
                    norm_W_A=float(np.linalg.norm(net.W_v.numpy())))
        return (lambda V_, A_: score_vectors(net, V_, A_.reshape(-1, 1))), info
    raise ValueError(f"unknown treatment objective {cfg['objective']!r}")


def fit_directions(X, A, Y, cfg, seed=0, baselines=None, oracle=None):
    """Score-ratio estimates of the treatment and prognostic directions.

    X: (n, p) covariates, A: (n,) binary treatment, Y: (n,) outcome.
    cfg: pipeline configuration (the 'pipeline' section of default.yaml).
    Directions are unit vectors in standardised-covariate coordinates.
    Returns a dict with 'beta' (treatment), 'alpha' (prognostic, from H_0),
    the warm start, the H_0 matrices of all constructions, score vectors and
    training information. ``baselines`` reuses precomputed SAVE/SIR/LASSO
    directions; ``oracle`` = (alpha, beta) is needed only for init 'oracle'.

    Python's, NumPy's global and TensorFlow's RNGs are seeded with ``seed``
    before training, so results do not depend on how the baselines were
    obtained (sklearn may draw from NumPy's global RNG).
    """
    Xs = standardize(np.asarray(X))
    V = Xs.astype(np.float32)
    A_obs = np.asarray(A)
    Af = A_obs.astype(np.float32)
    Y = np.asarray(Y)
    AY = np.column_stack([A_obs, Y]).astype(np.float32)
    ctrl = A_obs == 0
    p = V.shape[1]
    info = {}

    # 1. Warm start
    if baselines is None:
        baselines = baseline_directions(Xs, A_obs, Y, cfg['baselines'])
    alpha0, beta0 = initial_directions(cfg['init']['method'], p, baselines, seed, oracle)
    set_seed(seed)

    # 2. Treatment score -> beta_hat
    wA_fn, tr_info = fit_treatment(cfg['treatment'], V, Af, beta0, seed)
    info.update(tr_info)
    wA_all = wA_fn(V, Af)
    beta_hat = top_eigvecs(second_moment(wA_all), 1)[0].ravel()

    # 3. Joint score
    js = cfg['joint']
    net_AY = ScoreRatioNet(p, obs_dim=2, r_prime=js['rank'], s_prime=js['s_prime'],
                           hidden_units=tuple(js['hidden']), name='AY')
    net_AY.W_v.assign(_orth_init([beta0, alpha0], js['rank'], p, seed), read_value=False)
    ep, val, ran = train_ism(
        net_AY, V, AY, val_frac=js['val_frac'], max_epochs=js['max_epochs'],
        lr=js['lr'], lam1=js['lam1'], lam2=js['lam2'], lam_l2=js['lam_l2'],
        patience=js['patience'], clip_norm=js['clip_norm'], stiefel=js['stiefel'],
        trace=js['trace'], n_hutch=js['n_hutch'], seed=seed)
    info.update(selected_epoch_AY=ep, val_loss_AY=val, epochs_run_AY=ran)

    # 4. H_0 on controls: per-sample subtraction (paper) and the A09 alternatives
    n0 = int(ctrl.sum())
    V0, Y0 = V[ctrl], Y[ctrl]
    AY0 = np.column_stack([np.zeros(n0), Y0]).astype(np.float32)
    wAY0 = score_vectors(net_AY, V0, AY0)
    wA0 = wA_fn(V0, np.zeros(n0, dtype=np.float32))
    d = wAY0 - wA0
    wAY_all = score_vectors(net_AY, V, AY)
    H0 = {
        'per_sample': second_moment(d),
        'difference_ctrl': second_moment(wAY0) - second_moment(wA0),
        'difference_all': second_moment(wAY_all) - second_moment(wA_all),
    }
    alpha_hat = top_eigvecs(H0[cfg['h0']['construction']], 1)[0].ravel()
    return dict(alpha=alpha_hat, beta=beta_hat, alpha0=alpha0, beta0=beta0,
                H0=H0, w_A=wA_all, d=d, w_AY_ctrl=wAY0, V=V, ctrl=ctrl,
                net_AY=net_AY, AY=AY, info=info)


def _h0_readout(H, alpha):
    a_hat, eigs = top_eigvecs(H, 1)
    return dict(cos2=cosine_sq(a_hat, alpha), min_eig=float(eigs[-1]),
                eig_ratio=float(eigs[0] / (abs(eigs[1]) + 1e-12)))


def run_pipeline(cfg, seed, cache_dir=None):
    """Simulate one dataset (cfg['dgp'], seed), fit, and evaluate against the truth.

    Returns a flat dict of metrics. ``cache_dir`` stores the deterministic
    baseline directions (optional).
    """
    t0 = time.time()
    data = simulate(cfg['dgp'], seed)
    alpha, beta = data['alpha'], data['beta']
    row = dict(seed=seed, n_treated=int(data['A'].sum()),
               n_control=int((data['A'] == 0).sum()))

    baselines = cached_baselines(cfg, seed, cache_dir)
    for name, (a_hat, b_hat) in baselines.items():
        row[f'alpha_{name}_cos2'] = cosine_sq(a_hat, alpha)
        row[f'beta_{name}_cos2'] = cosine_sq(b_hat, beta)

    fit = fit_directions(data['X'], data['A'], data['Y'], cfg, seed, baselines,
                         oracle=(alpha, beta))
    V, ctrl, n0 = fit['V'], fit['ctrl'], int(fit['ctrl'].sum())
    Af = data['A'].astype(np.float32)
    row['alpha_init_cos2'] = cosine_sq(fit['alpha0'], alpha)
    row['beta_init_cos2'] = cosine_sq(fit['beta0'], beta)
    row.update({k: v for k, v in fit['info'].items() if 'AY' not in k})
    row['beta_sr_cos2'] = cosine_sq(fit['beta'], beta)
    row['score_error_A'] = score_mse(fit['w_A'], data['oracle_w_A'](V, Af))
    row['norm_wA'] = float(np.mean(np.linalg.norm(fit['w_A'], axis=1)))
    row.update({k: v for k, v in fit['info'].items() if 'AY' in k})

    for name, H in fit['H0'].items():
        for k, v in _h0_readout(H, alpha).items():
            row[f'h0_{name}_{k}'] = v
    row['alpha_sr_cos2'] = cosine_sq(fit['alpha'], alpha)
    row['score_error_YA'] = score_mse(
        fit['d'], data['oracle_w_YA'](V[ctrl], np.zeros(n0), data['Y'][ctrl]))
    row['norm_wAY'] = float(np.mean(np.linalg.norm(fit['w_AY_ctrl'], axis=1)))

    if cfg['diagnostics']['trace_spread']:
        row.update(trace_estimator_spread(
            fit['net_AY'], V, fit['AY'],
            n_hutch_list=tuple(cfg['diagnostics']['n_hutch']),
            n_rep=cfg['diagnostics']['n_rep']))

    row['runtime_sec'] = round(time.time() - t0, 1)
    return row
