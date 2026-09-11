"""
Nonlinear DR experiments — p=100, n=500.

Experiment 2 (rank-1):
  Y(0) = (alpha'V)^2 + eps
  logit e(V) = s_T[(beta'V)^2 - 1]        (centred so E[logit]=0)

Experiment 3 (rank-2):
  Y(0) = sin(alpha1'V) + 0.5*(alpha2'V)^2 + eps
  logit e(V) = s_T[sin(beta1'V) + (beta2'V)^2 - 1]

Key design choice:
  Treatment score:  propensity NN trained via BCE → wA = (A-e)*grad_eta  (NOT ISM)
  Outcome score:    generic ScoreRatioNet (ISM, Hutchinson K=8, r'=5)
  Competitors:      SIR, SAVE, LASSO/Ridge (linear negative controls)

SIR is expected to fail for both DGPs (symmetry kills odd moments).
SAVE should succeed for both (quadratic structures change conditional variance).
LASSO/Ridge should fail (quadratic signal, linear method).

Saves:
  results/sr_nn_nonlinear_rank1.csv
  results/sr_nn_nonlinear_rank2.csv
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import expit

from score_ratio_reduction import ScoreRatioNet, _ism_step
from score_ratio_rescue import (
    construct_beta, cosine_sq, sklearn_directions, SKLEARN_COMBOS, _get_vecs,
)
from score_ratio_bernoulli import BernoulliScoreNet

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

P = 30; QA = 5; N = 500
NOISE = 0.5          # outcome noise std; SNR≈4 so SAVE and ISM can both work
S_T = 1.5            # quadratic propensity scale
DATA_SEEDS = [1, 2, 3]
N_PROP_INITS = 5     # random restarts for propensity NN
N_EPOCHS_PROP = 400
N_EPOCHS_SR = 400
LR_PROP = 3e-3
L2_PROP = 1e-3       # L2 weight regularization on propensity BCE (prevents overfit)
LR_SR = 1e-3
BATCH_SR = 64
R_PRIME_AY = 3       # p=30: 3 latent dims sufficient
N_HUTCH = 8
N_SLICES = 5         # SIR/SAVE slices for continuous Y
HIDDEN = (16, 16)    # smaller MLP for p=30
RESULT_DIR = os.path.join(_here, 'results')

# ═══════════════════════════════════════════════════════════════════════════════
# Direction construction
# ═══════════════════════════════════════════════════════════════════════════════

def make_rank1_directions(p=P, q=QA, ab_dp=0.75, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(p)
    alpha[:q] = rng.standard_normal(q)
    alpha /= np.linalg.norm(alpha)
    beta = construct_beta(alpha, ab_dp, rng)
    return alpha.astype(np.float32), beta.astype(np.float32)


def make_rank2_directions(p=P, q=QA, seed=0):
    """Two orthogonal pairs: (alpha1,alpha2) supported on [:q] and [q:2q]."""
    rng = np.random.default_rng(seed)
    a1 = np.zeros(p); a1[:q]    = rng.standard_normal(q); a1 /= np.linalg.norm(a1)
    a2 = np.zeros(p); a2[q:2*q] = rng.standard_normal(q); a2 /= np.linalg.norm(a2)
    a2 -= (a2 @ a1) * a1; a2 /= np.linalg.norm(a2)
    b1 = construct_beta(a1, 0.75, rng)
    b2 = construct_beta(a2, 0.75, rng)
    b2 -= (b2 @ b1) * b1; b2 /= np.linalg.norm(b2)
    return tuple(v.astype(np.float32) for v in (a1, a2, b1, b2))

# ═══════════════════════════════════════════════════════════════════════════════
# DGPs
# ═══════════════════════════════════════════════════════════════════════════════

def gen_rank1(n, alpha, beta, s_T=S_T, noise=NOISE, seed=0):
    """Y(0)=(a'V)^2+eps,  logit e=s_T[(b'V)^2-1]."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, len(alpha))).astype(np.float32)
    e = expit(s_T * ((V @ beta)**2 - 1.0))
    A = rng.binomial(1, e).astype(np.float32)
    Y = ((V @ alpha)**2 + noise * rng.standard_normal(n)).astype(np.float32)
    return V, A, Y, e


def gen_rank2(n, alpha1, alpha2, beta1, beta2, s_T=S_T, noise=NOISE, seed=0):
    """Y(0)=sin(a1'V)+0.5(a2'V)^2+eps,  logit e=s_T[sin(b1'V)+(b2'V)^2-1]."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, len(alpha1))).astype(np.float32)
    e = expit(s_T * (np.sin(V @ beta1) + (V @ beta2)**2 - 1.0))
    A = rng.binomial(1, e).astype(np.float32)
    Y = (np.sin(V @ alpha1) + 0.5*(V @ alpha2)**2
         + noise * rng.standard_normal(n)).astype(np.float32)
    return V, A, Y, e

# ═══════════════════════════════════════════════════════════════════════════════
# Propensity NN via BCE  (treatment score — replaces ISM)
# ═══════════════════════════════════════════════════════════════════════════════

def _bce_loss(net, V_tf, A_tf):
    eta = net.eta(V_tf)
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(A_tf, tf.float32), logits=eta))


def train_propensity_bce(net, V_tf, A_tf,
                         n_epochs=N_EPOCHS_PROP, lr=LR_PROP, l2=L2_PROP):
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(n_epochs):
        with tf.GradientTape() as tape:
            bce = _bce_loss(net, V_tf, A_tf)
            l2_pen = l2 * tf.add_n(
                [tf.reduce_sum(v ** 2) for v in net.trainable_variables])
            loss = bce + l2_pen
        opt.apply_gradients(zip(
            tape.gradient(loss, net.trainable_variables),
            net.trainable_variables))
    return float(bce)  # report bare BCE (not penalised loss)


def get_wA(net, V_tf, A_tf):
    """w_A(v,a) = (a - sigma(eta(v))) * grad_v eta(v)  →  (n, p)."""
    V_tf = tf.cast(V_tf, tf.float32)
    with tf.GradientTape() as t:
        t.watch(V_tf)
        eta_sum = tf.reduce_sum(net.eta(V_tf))
    g = t.gradient(eta_sum, V_tf)              # (n, p)
    e = tf.sigmoid(net.eta(V_tf))
    a = tf.cast(A_tf, tf.float32)
    return ((a - e)[:, None] * g).numpy()


def fit_propensity(V_std, A, r_prime=R_PRIME_AY, hidden=None):
    """Best-of-N_PROP_INITS random restarts by final BCE loss."""
    V_tf = tf.constant(V_std, dtype=tf.float32)
    A_tf = tf.constant(A, dtype=tf.float32)
    if hidden is None:
        hidden = HIDDEN
    best_loss, best_net = np.inf, None
    for _ in range(N_PROP_INITS):
        net = BernoulliScoreNet(V_std.shape[1], r_prime=r_prime,
                                hidden_units=hidden)
        loss = train_propensity_bce(net, V_tf, A_tf)
        if loss < best_loss:
            best_loss, best_net = loss, net
    wA = get_wA(best_net, V_tf, A_tf)
    print(f"    propensity BCE loss={best_loss:.4f}")
    return wA, best_net

# ═══════════════════════════════════════════════════════════════════════════════
# Joint score SR-NN (ISM) — flexible r'
# ═══════════════════════════════════════════════════════════════════════════════

def fit_H_AY(V_std, AY, r_prime=R_PRIME_AY, lam1=0.02, lam2=0.01,
             n_hutch=N_HUTCH, n_epochs=N_EPOCHS_SR, hidden=HIDDEN):
    n, p = V_std.shape
    net = ScoreRatioNet(p, obs_dim=2, r_prime=r_prime, s_prime=2,
                        hidden_units=hidden, name='H_AY')
    opt = tf.keras.optimizers.Adam(LR_SR)
    V_tf  = tf.constant(V_std, dtype=tf.float32)
    AY_tf = tf.constant(AY,    dtype=tf.float32)
    ds = (tf.data.Dataset.from_tensor_slices((V_tf, AY_tf))
          .shuffle(n, seed=0).batch(BATCH_SR))
    for ep in range(n_epochs):
        for bv, bo in ds:
            _ism_step(net, opt, bv, bo, lam1, lam2, n_hutch)
        if (ep + 1) % max(1, n_epochs // 4) == 0:
            print(f"    [H_AY ISM] ep {ep+1}/{n_epochs}")
    return net

# ═══════════════════════════════════════════════════════════════════════════════
# H0 via per-sample subtraction (F1 fix)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_H0(wAY_ctrl, wA_ctrl):
    d = wAY_ctrl - wA_ctrl
    return d.T @ d / len(d)

# ═══════════════════════════════════════════════════════════════════════════════
# SIR and SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def _stdV(V):
    return (V - V.mean(0)) / (V.std(0) + 1e-8)


def sir(V, response, n_slices=N_SLICES, n_dirs=1):
    """
    Sliced Inverse Regression.
    Binary response: two-class mean difference (SIR ≡ LDA direction).
    Symmetric quadratic DGP: E[V|A] collapses to 0 by symmetry → expected to fail.
    """
    Vs = _stdV(V); n = len(response)
    uniq = np.unique(response)
    if len(uniq) == 2:
        M = sum((response == lab).sum()/n
                * np.outer(Vs[response == lab].mean(0),
                            Vs[response == lab].mean(0))
                for lab in uniq)
    else:
        order = np.argsort(response)
        M = sum(len(sl)/n * np.outer(Vs[sl].mean(0), Vs[sl].mean(0))
                for sl in np.array_split(order, n_slices))
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1][:, :n_dirs]


def save_sdr(V, response, n_slices=N_SLICES, n_dirs=1):
    """
    Sliced Average Variance Estimation with Ledoit-Wolf shrinkage.
    Captures directions where conditional variance changes → works for quadratic.
    Shrinkage corrects for rank-deficiency when n_slice < p.
    """
    from sklearn.covariance import LedoitWolf
    Vs = _stdV(V); n, p = Vs.shape; Ip = np.eye(p)
    uniq = np.unique(response)
    if len(uniq) == 2:
        masks = [response == lab for lab in uniq]
    else:
        order = np.argsort(response)
        idx = np.array_split(order, n_slices)
        masks = [np.isin(np.arange(n), sl) for sl in idx]
    M = np.zeros((p, p))
    for mask in masks:
        wt = mask.sum() / n
        n_h = mask.sum()
        if n_h > 1:
            Sh = LedoitWolf().fit(Vs[mask]).covariance_
        else:
            Sh = Ip
        diff = Ip - Sh
        M += wt * (diff @ diff)
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1][:, :n_dirs]

# ═══════════════════════════════════════════════════════════════════════════════
# Subspace accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def subspace_cos2(U_hat, U_true):
    """
    Mean squared cosine of principal angles between column spans.
    Rank-1: equals standard cosine_sq.  Rank-k: mean over k singular values^2.
    Both inputs: (p,) vectors or (p, k) matrices.
    """
    U_hat  = np.atleast_2d(U_hat).T  if U_hat.ndim  == 1 else U_hat
    U_true = np.atleast_2d(U_true).T if U_true.ndim == 1 else U_true
    Q_h, _ = np.linalg.qr(U_hat)
    Q_t, _ = np.linalg.qr(U_true)
    k = min(Q_h.shape[1], Q_t.shape[1])
    sv = np.linalg.svd(Q_h[:, :k].T @ Q_t[:, :k], compute_uv=False)
    return float(np.mean(np.clip(sv, 0, 1)**2))

# ═══════════════════════════════════════════════════════════════════════════════
# Our method: propensity-BCE + ISM-AY + F1 subtraction
# ═══════════════════════════════════════════════════════════════════════════════

def our_method(V, A, Y, n_alpha_dirs=1):
    """
    Returns (alpha_hat, beta_hat) column-matrices.
    - beta from H_A = E[wA wA^T], top eigenvector(s)
    - alpha from H0 = E_ctrl[(wAY - wA)(wAY - wA)^T], top n_alpha_dirs eigenvectors
    """
    V_std = ((V - V.mean(0)) / (V.std(0) + 1e-8)).astype(np.float32)
    AY = np.column_stack([A, Y]).astype(np.float32)

    print("  [propensity BCE]")
    wA_all, net_prop = fit_propensity(V_std, A)

    print("  [SR-NN ISM for wAY]")
    net_AY = fit_H_AY(V_std, AY)

    # evaluate at controls
    ctrl = A == 0; n0 = int(ctrl.sum())
    V_c  = tf.constant(V_std[ctrl], dtype=tf.float32)
    A_c  = tf.constant(A[ctrl],     dtype=tf.float32)
    AY_c = tf.constant(AY[ctrl],    dtype=tf.float32)

    wAY_ctrl = _get_vecs(net_AY, V_c, AY_c)
    wA_ctrl  = get_wA(net_prop, V_c, A_c)

    # outcome direction
    H0 = compute_H0(wAY_ctrl, wA_ctrl)
    _, vecs_H0 = np.linalg.eigh(H0)
    alpha_hat = vecs_H0[:, ::-1][:, :n_alpha_dirs]

    # treatment direction
    H_A = wA_all.T @ wA_all / len(A)
    _, vecs_HA = np.linalg.eigh(H_A)
    beta_hat = vecs_HA[:, ::-1][:, :1]

    return alpha_hat, beta_hat

# ═══════════════════════════════════════════════════════════════════════════════
# All-methods runner for one dataset
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(V, A, Y, alpha_true, beta_true, n_alpha_dirs=1):
    """
    alpha_true: (p,) vector [rank-1] or (p, k) matrix [rank-2].
    beta_true:  (p,) vector.
    Returns list of result dicts.
    """
    rows = []

    # ── Our method ──────────────────────────────────────────────────────────
    alpha_hat, beta_hat = our_method(V, A, Y, n_alpha_dirs)
    rows.append(dict(
        method='SR-NN (prop-BCE)',
        cos2_alpha=subspace_cos2(alpha_hat, alpha_true),
        cos2_beta =subspace_cos2(beta_hat,  beta_true),
    ))

    # ── SIR, SAVE ───────────────────────────────────────────────────────────
    V_std = ((V - V.mean(0)) / (V.std(0) + 1e-8)).astype(np.float32)
    ctrl = A == 0
    V_ctrl = V[ctrl]; Y_ctrl = Y[ctrl]

    for label, fn in [('SIR', sir), ('SAVE', save_sdr)]:
        # alpha: 3 slices so each has n_ctrl/3 ≥ p=100 obs (avoids rank-deficiency)
        a_hat = fn(V_ctrl, Y_ctrl, n_slices=3, n_dirs=n_alpha_dirs)
        # beta: binary A → 2 slices (one per class)
        b_hat = fn(V, A, n_slices=2, n_dirs=1)
        rows.append(dict(
            method=label,
            cos2_alpha=subspace_cos2(a_hat, alpha_true),
            cos2_beta =subspace_cos2(b_hat, beta_true),
        ))

    # ── LASSO / Ridge ────────────────────────────────────────────────────────
    for combo in SKLEARN_COMBOS:
        ah, bh = sklearn_directions(V, A, Y,
                                    l1_alpha=combo['l1_alpha'],
                                    l1_T=combo['l1_T'])
        rows.append(dict(
            method=combo['label'],
            cos2_alpha=subspace_cos2(ah, alpha_true),
            cos2_beta =subspace_cos2(bh, beta_true),
        ))

    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_rank1(alpha, beta, seeds=DATA_SEEDS):
    print(f"\nDGP rank-1: Y=(a'V)^2+eps,  logit e=s_T[(b'V)^2-1],  s_T={S_T}")
    all_rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n  Rank-1  seed={seed}\n{'='*60}")
        V, A, Y, e = gen_rank1(N, alpha, beta, seed=seed)
        print(f"  n_treat={int(A.sum())},  e=[{e.min():.3f},{e.max():.3f}]")
        rows = run_one(V, A, Y, alpha_true=alpha, beta_true=beta, n_alpha_dirs=1)
        for r in rows: r['seed'] = seed
        all_rows.extend(rows)
        _print_seed_summary(rows)
    df = pd.DataFrame(all_rows)
    print("\n── RANK-1 MEDIAN ACROSS SEEDS ──")
    _print_median(df)
    return df


def run_rank2(a1, a2, b1, b2, seeds=DATA_SEEDS):
    alpha_true = np.column_stack([a1, a2])
    beta_true  = np.column_stack([b1, b2])
    print(f"\nDGP rank-2: Y=sin(a1'V)+0.5(a2'V)^2+eps,  s_T={S_T}")
    all_rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n  Rank-2  seed={seed}\n{'='*60}")
        V, A, Y, e = gen_rank2(N, a1, a2, b1, b2, seed=seed)
        print(f"  n_treat={int(A.sum())},  e=[{e.min():.3f},{e.max():.3f}]")
        rows = run_one(V, A, Y, alpha_true=alpha_true, beta_true=beta_true,
                       n_alpha_dirs=2)
        for r in rows: r['seed'] = seed
        all_rows.extend(rows)
        _print_seed_summary(rows)
    df = pd.DataFrame(all_rows)
    print("\n── RANK-2 MEDIAN ACROSS SEEDS ──")
    _print_median(df)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _print_seed_summary(rows):
    print()
    for r in rows:
        print(f"  {r['method']:<30s}  cos2α={r['cos2_alpha']:.3f}  "
              f"cos2β={r['cos2_beta']:.3f}")


def _print_median(df):
    for method, sub in df.groupby('method', sort=False):
        print(f"  {method:<30s}  "
              f"cos2α={sub.cos2_alpha.median():.3f}  "
              f"cos2β={sub.cos2_beta.median():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Rank-1
    alpha, beta = make_rank1_directions()
    print(f"Rank-1 DGP: p={P}, q={QA}, alpha·beta={float(alpha @ beta):.4f}")
    df1 = run_rank1(alpha, beta)
    out1 = os.path.join(RESULT_DIR, 'sr_nn_nonlinear_rank1.csv')
    df1.to_csv(out1, index=False)
    print(f"\nSaved: {out1}")

    # Rank-2
    a1, a2, b1, b2 = make_rank2_directions()
    print(f"\nRank-2 DGP: a1·a2={float(a1@a2):.4f}, b1·b2={float(b1@b2):.4f}")
    df2 = run_rank2(a1, a2, b1, b2)
    out2 = os.path.join(RESULT_DIR, 'sr_nn_nonlinear_rank2.csv')
    df2.to_csv(out2, index=False)
    print(f"\nSaved: {out2}")


if __name__ == '__main__':
    main()
