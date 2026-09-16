"""
score_ratio_nonlinear_rank1_stable.py

Canonical fully data-driven nonlinear rank-1 experiment.

DGP: quadratic single-index, p=30, n=500
  Y(0) = (α'V)² + ε,   ε ~ N(0, 0.5²)
  logit e(V) = s_T[(β'V)² − 1],   s_T=1.5

Pipeline:
  1. SAVE warm start → β̂₀ (binary A, 2 slices), α̂₀ (controls Y, 3 slices)
  2. BCE propensity NN → ŵ_A, W_A ∈ R^p normalized each step (rank-1)
  3. ISM joint score   → ŵ_AY, W_v ∈ Stiefel(p,2) projected each step
  4. Per-sample d_i = ŵ_AY − ŵ_A on controls → Ĥ₀ → α̂

Baselines: SAVE, SIR, LASSO-Y/logistic-LASSO-T

Saves: results/sr_nn_nonlinear_rank1_stable.csv
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
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from score_ratio_reduction import ScoreRatioNet, compute_subspace

# ── Hyperparameters ────────────────────────────────────────────────────────────

P, QA, N  = 30, 5, 500
NOISE     = 0.5      # outcome noise std
S_T       = 1.5      # propensity quadratic scale
N_SAVE_SLICES_Y = 3  # continuous Y slices for outcome SAVE
N_SAVE_SLICES_A = 2  # binary A slices (one per class)

# BCE propensity (same as linear stable)
R_A            = 1
L2_A           = 1e-3
LR_A           = 1e-3
MAX_EPOCHS_A   = 600
PATIENCE_A     = 40
VAL_FRAC_A     = 0.2
BATCH_SIZE_A   = 64
CLIP_NORM_A    = 1.0

# ISM joint score (same as linear stable)
R_AY           = 2
S_AY           = 2
HIDDEN_AY      = (16, 16)
LAM2_AY        = 1e-2
LAM_L2_AY      = 1e-3
LR_AY          = 3e-3
MAX_EPOCHS_AY  = 600
PATIENCE_AY    = 40
VAL_FRAC_AY    = 0.2
CLIP_NORM_AY   = 1.0

VERBOSE_INTERVAL = 50  # finer reporting for p=30

# ── Utility functions ──────────────────────────────────────────────────────────

def construct_beta(alpha, target_dp, rng):
    """Generate β with α·β = target_dp, unit norm."""
    p = len(alpha)
    raw = rng.standard_normal(p)
    raw -= (raw @ alpha) * alpha
    raw /= np.linalg.norm(raw)
    beta = target_dp * alpha + np.sqrt(1.0 - target_dp**2) * raw
    return beta / np.linalg.norm(beta)


def cosine_sq(u, v):
    """Sign-invariant cos²(u, v)."""
    u = u.ravel(); v = v.ravel()
    return float((u @ v)**2 / (np.dot(u, u) * np.dot(v, v) + 1e-15))


def score_mse(w_hat, w_true):
    """Normalised score MSE: E[‖ŵ−w‖²] / E[‖w‖²]."""
    num   = np.mean(np.sum((w_hat - w_true)**2, axis=1))
    denom = np.mean(np.sum(w_true**2,           axis=1)) + 1e-10
    return float(num / denom)

# ── DGP ───────────────────────────────────────────────────────────────────────

def make_rank1_directions(p=P, q=QA, ab_dp=0.75, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(p)
    alpha[:q] = rng.standard_normal(q)
    alpha /= np.linalg.norm(alpha)
    beta = construct_beta(alpha, ab_dp, rng)
    return alpha.astype(np.float32), beta.astype(np.float32)


def gen_rank1(n, alpha, beta, s_T=S_T, noise=NOISE, seed=0):
    """Y(0)=(α'V)²+ε,  logit e=s_T[(β'V)²−1]."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, len(alpha))).astype(np.float32)
    e = expit(s_T * ((V @ beta)**2 - 1.0))
    A = rng.binomial(1, e).astype(np.float32)
    Y = ((V @ alpha)**2 + noise * rng.standard_normal(n)).astype(np.float32)
    return V, A, Y, e

# ── Oracle score functions (eval-only) ────────────────────────────────────────

def true_w_A_nl(V, A, beta, s_T=S_T):
    """w_A(v,a) = (a−e(v)) · 2s_T(β'v) · β."""
    z = V @ beta                                     # (n,)
    e = expit(s_T * (z**2 - 1.0))
    g = 2.0 * s_T * z[:, None] * beta[None, :]      # (n,p)
    return (A - e)[:, None] * g


def true_w_YA_nl(V_ctrl, Y_ctrl, alpha, noise=NOISE):
    """w_{Y|A=0}(v,y) = (y−(α'v)²) · 2(α'v) · α / noise²."""
    z     = V_ctrl @ alpha                           # (n0,)
    coeff = (Y_ctrl - z**2) * 2.0 * z / (noise**2)  # (n0,)
    return coeff[:, None] * alpha[None, :]           # (n0,p)

# ── SAVE / SIR baselines ──────────────────────────────────────────────────────

def _stdV(V):
    return (V - V.mean(0)) / (V.std(0) + 1e-8)


def save_sdr(V, response, n_slices=N_SAVE_SLICES_Y, n_dirs=1):
    """SAVE with Ledoit-Wolf shrinkage on slice covariances."""
    Vs = _stdV(V); n, p = Vs.shape; Ip = np.eye(p)
    uniq = np.unique(response)
    if len(uniq) == 2:
        masks = [response == lab for lab in uniq]
    else:
        order = np.argsort(response)
        idx   = np.array_split(order, n_slices)
        masks = [np.isin(np.arange(n), sl) for sl in idx]
    M = np.zeros((p, p))
    for mask in masks:
        wt  = mask.sum() / n
        n_h = mask.sum()
        Sh  = LedoitWolf().fit(Vs[mask]).covariance_ if n_h > 1 else Ip
        diff = Ip - Sh
        M += wt * (diff @ diff)
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1][:, :n_dirs]


def sir(V, response, n_slices=N_SAVE_SLICES_Y, n_dirs=1):
    """SIR (fails for symmetric quadratic DGPs)."""
    Vs = _stdV(V); n = len(response)
    uniq = np.unique(response)
    if len(uniq) == 2:
        M = sum((response == lab).sum() / n
                * np.outer(Vs[response == lab].mean(0),
                           Vs[response == lab].mean(0))
                for lab in uniq)
    else:
        order = np.argsort(response)
        M = sum(len(sl) / n * np.outer(Vs[sl].mean(0), Vs[sl].mean(0))
                for sl in np.array_split(order, n_slices))
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1][:, :n_dirs]


def lasso_directions(V, A, Y):
    """LASSO-Y for α, logistic-LASSO for β (expected to fail for quadratic DGP)."""
    V_std = _stdV(V)
    ctrl  = A == 0
    # outcome: LASSO on controls
    m = LassoCV(cv=5, max_iter=2000).fit(V_std[ctrl], Y[ctrl])
    alpha_hat = m.coef_.copy()
    nrm = np.linalg.norm(alpha_hat)
    if nrm < 1e-10:
        alpha_hat = np.random.default_rng(42).standard_normal(V.shape[1])
    alpha_hat /= np.linalg.norm(alpha_hat)
    # treatment: logistic LASSO
    lm = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear',
                               max_iter=2000).fit(V_std, A.astype(int))
    beta_hat = lm.coef_.ravel().copy()
    nrm = np.linalg.norm(beta_hat)
    if nrm < 1e-10:
        beta_hat = np.random.default_rng(43).standard_normal(V.shape[1])
    beta_hat /= np.linalg.norm(beta_hat)
    return alpha_hat.astype(np.float32), beta_hat.astype(np.float32)

# ── SAVE warm start ────────────────────────────────────────────────────────────

def save_warm_start(V, A, Y):
    """Fully data-driven SAVE initialisation (no oracle).

    beta_init:  SAVE on (V, A) with 2 slices (binary)
    alpha_init: SAVE on (V_ctrl, Y_ctrl) with 3 equal-frequency slices
    """
    ctrl  = A == 0
    # treatment direction
    b_hat = save_sdr(V, A, n_slices=N_SAVE_SLICES_A, n_dirs=1).ravel()
    b_hat /= np.linalg.norm(b_hat)
    # outcome direction from controls
    a_hat = save_sdr(V[ctrl], Y[ctrl], n_slices=N_SAVE_SLICES_Y, n_dirs=1).ravel()
    a_hat /= np.linalg.norm(a_hat)
    return a_hat.astype(np.float32), b_hat.astype(np.float32)

# ── Propensity network ─────────────────────────────────────────────────────────

class PropensityNet(tf.Module):
    """η(v) = h(W_A^T v), W_A ∈ R^p normalised to unit norm after each step."""

    def __init__(self, p, hidden=(8,), name='prop_net'):
        super().__init__(name=name)
        self.p = p
        init = tf.initializers.GlorotUniform()
        self.W_A = tf.Variable(init(shape=(p, 1)), dtype=tf.float32, name='W_A')
        layers = []
        for h_dim in hidden:
            layers.append(tf.keras.layers.Dense(h_dim, activation='tanh'))
        layers.append(tf.keras.layers.Dense(1))
        self.h = tf.keras.Sequential(layers, name='inner_h')
        self.h.build((None, 1))

    @property
    def trainable_variables(self):
        return [self.W_A] + list(self.h.trainable_variables)

    def eta(self, V):
        V = tf.cast(V, tf.float32)
        z = V @ self.W_A
        return tf.squeeze(self.h(z), axis=-1)

    def normalize(self):
        w = self.W_A.numpy()
        self.W_A.assign(w / (np.linalg.norm(w) + 1e-8), read_value=False)


def _bce_loss(net, V_tf, A_tf):
    eta = net.eta(V_tf)
    A   = tf.cast(A_tf, tf.float32)
    return -tf.reduce_mean(
        A * tf.math.log_sigmoid(eta) + (1. - A) * tf.math.log_sigmoid(-eta))


def train_propensity(net, V_np, A_np, val_frac=VAL_FRAC_A,
                     n_epochs=MAX_EPOCHS_A, lr=LR_A, l2=L2_A,
                     patience=PATIENCE_A, batch_size=BATCH_SIZE_A,
                     clip_norm=CLIP_NORM_A, seed=0, verbose=True):
    """BCE + L2 + validation early stopping; W_A unit-normalised each step."""
    n = len(A_np)
    rng = np.random.default_rng(seed + 999)
    perm = rng.permutation(n)
    n_val = max(10, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    V_tr  = tf.constant(V_np[tr_idx], dtype=tf.float32)
    A_tr  = tf.constant(A_np[tr_idx], dtype=tf.float32)
    V_val = tf.constant(V_np[val_idx], dtype=tf.float32)
    A_val = tf.constant(A_np[val_idx], dtype=tf.float32)
    n_tr  = len(tr_idx)

    opt            = tf.keras.optimizers.Adam(lr)
    best_val_bce   = float('inf')
    best_weights   = [v.numpy().copy() for v in net.trainable_variables]
    patience_count = 0
    selected_epoch = 0

    for epoch in range(1, n_epochs + 1):
        perm_ep = np.random.permutation(n_tr)
        for start in range(0, n_tr, batch_size):
            b_idx = perm_ep[start:start + batch_size]
            Vb    = tf.gather(V_tr, b_idx)
            Ab    = tf.gather(A_tr, b_idx)
            with tf.GradientTape() as tape:
                bce  = _bce_loss(net, Vb, Ab)
                l2_p = l2 * tf.add_n([tf.reduce_sum(v**2)
                                       for v in net.trainable_variables])
                loss = bce + l2_p
            grads = tape.gradient(loss, net.trainable_variables)
            grads = [tf.clip_by_norm(g, clip_norm) if g is not None else g
                     for g in grads]
            opt.apply_gradients(zip(grads, net.trainable_variables))
            net.normalize()

        val_bce = float(_bce_loss(net, V_val, A_val))

        if verbose and epoch % VERBOSE_INTERVAL == 0:
            print(f"    ep {epoch:>4}/{n_epochs}  val_BCE={val_bce:.5f}")

        if val_bce < best_val_bce - 1e-6:
            best_val_bce   = val_bce
            best_weights   = [v.numpy().copy() for v in net.trainable_variables]
            selected_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    for v, w in zip(net.trainable_variables, best_weights):
        v.assign(w, read_value=False)
    net.normalize()
    return selected_epoch, best_val_bce


def get_wA(net, V_np, A_np):
    """ŵ_A(V_i,A_i) = (A_i−ê_i)∇_v η(V_i), shape (n,p)."""
    V_tf = tf.constant(V_np, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(V_tf)
        eta_sum = tf.reduce_sum(net.eta(V_tf))
    g = t.gradient(eta_sum, V_tf).numpy()
    e = tf.sigmoid(net.eta(V_tf)).numpy()
    return ((A_np - e)[:, None] * g)

# ── ISM joint score ────────────────────────────────────────────────────────────

def stiefel_project(net):
    W = net.W_v.numpy()
    Q, _ = np.linalg.qr(W)
    net.W_v.assign(Q[:, :net.r_prime].astype(np.float32), read_value=False)


def _ism_step_qr(net, optimizer, V_tf, obs_tf,
                 lam2=LAM2_AY, lam_l2=LAM_L2_AY, clip_norm=CLIP_NORM_AY):
    W_v_np = net.W_v.numpy()
    Q_np, _ = np.linalg.qr(W_v_np)
    n, r_prime = V_tf.shape[0], net.r_prime

    with tf.GradientTape() as outer:
        trace_est = tf.constant(0.0)
        for k in range(r_prime):
            z_k   = tf.constant(Q_np[:, k].astype(np.float32))
            z_k_b = tf.tile(z_k[None, :], [n, 1])
            with tf.GradientTape() as inner:
                inner.watch(V_tf)
                w_z = net(V_tf, obs_tf)
                sw  = tf.reduce_sum(z_k_b * w_z, axis=1)
            dw_dv     = inner.gradient(sw, V_tf)
            trace_est = trace_est + tf.reduce_mean(
                tf.reduce_sum(dw_dv * z_k_b, axis=1))

        w    = net(V_tf, obs_tf)
        half = tf.reduce_mean(0.5 * tf.reduce_sum(w * w, axis=1))
        ref  = tf.reduce_mean(-tf.reduce_sum(V_tf * w, axis=1))

        with tf.device('/CPU:0'):
            nuc_obs = tf.reduce_sum(tf.linalg.svd(net.W_obs, compute_uv=False))
        psi_vars = net.psi.trainable_variables
        l2_psi   = (lam_l2 * tf.add_n([tf.reduce_sum(v**2) for v in psi_vars])
                    if psi_vars else tf.constant(0.0))

        loss = half + trace_est + ref + lam2 * nuc_obs + l2_psi

    grads = outer.gradient(loss, net.trainable_variables)
    grads = [tf.clip_by_norm(g, clip_norm) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return float(loss)


def _ism_val_loss(net, V_tf, obs_tf):
    W_v_np = net.W_v.numpy()
    Q_np, _ = np.linalg.qr(W_v_np)
    n, r_prime = V_tf.shape[0], net.r_prime

    trace_val = 0.0
    for k in range(r_prime):
        z_k   = tf.constant(Q_np[:, k].astype(np.float32))
        z_k_b = tf.tile(z_k[None, :], [n, 1])
        with tf.GradientTape() as inner:
            inner.watch(V_tf)
            w_z = net(V_tf, obs_tf)
            sw  = tf.reduce_sum(z_k_b * w_z, axis=1)
        dw_dv     = inner.gradient(sw, V_tf)
        trace_val += float(tf.reduce_mean(tf.reduce_sum(dw_dv * z_k_b, axis=1)))

    w    = net(V_tf, obs_tf).numpy()
    half = float(np.mean(0.5 * np.sum(w * w, axis=1)))
    ref  = float(np.mean(-np.sum(V_tf.numpy() * w, axis=1)))
    return half + trace_val + ref


def train_joint_score(net, V_np, AY_np, val_frac=VAL_FRAC_AY,
                      n_epochs=MAX_EPOCHS_AY, lr=LR_AY,
                      lam2=LAM2_AY, lam_l2=LAM_L2_AY,
                      patience=PATIENCE_AY, seed=0, verbose=True):
    """ISM + Stiefel projection + val early stopping."""
    n = len(V_np)
    rng = np.random.default_rng(seed + 888)
    perm = rng.permutation(n)
    n_val = max(10, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    V_tr   = tf.constant(V_np[tr_idx],  dtype=tf.float32)
    AY_tr  = tf.constant(AY_np[tr_idx], dtype=tf.float32)
    V_val  = tf.constant(V_np[val_idx], dtype=tf.float32)
    AY_val = tf.constant(AY_np[val_idx], dtype=tf.float32)

    _ = net(V_tr[:1], AY_tr[:1])   # build Keras Sequential

    opt            = tf.keras.optimizers.Adam(lr)
    best_val_ism   = float('inf')
    best_weights   = [v.numpy().copy() for v in net.trainable_variables]
    patience_count = 0
    selected_epoch = 0

    for epoch in range(1, n_epochs + 1):
        _ism_step_qr(net, opt, V_tr, AY_tr, lam2=lam2, lam_l2=lam_l2)
        stiefel_project(net)

        val_ism = _ism_val_loss(net, V_val, AY_val)

        if verbose and epoch % VERBOSE_INTERVAL == 0:
            print(f"    ep {epoch:>4}/{n_epochs}  val_ISM={val_ism:.5f}")

        if val_ism < best_val_ism - 1e-5:
            best_val_ism   = val_ism
            best_weights   = [v.numpy().copy() for v in net.trainable_variables]
            selected_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    for v, w in zip(net.trainable_variables, best_weights):
        v.assign(w, read_value=False)
    stiefel_project(net)
    return selected_epoch, best_val_ism


def get_wAY(net, V_np, obs_np, batch_size=64):
    n     = len(V_np)
    V_tf  = tf.constant(V_np,  dtype=tf.float32)
    obs_tf = tf.constant(obs_np, dtype=tf.float32)
    return np.concatenate([
        net(V_tf[s:min(s+batch_size, n)],
            obs_tf[s:min(s+batch_size, n)]).numpy()
        for s in range(0, n, batch_size)], axis=0)

# ── Full pipeline for one seed ─────────────────────────────────────────────────

def run_one_seed(seed, alpha_true, beta_true, init_mode='save', verbose=True):
    """Run pipeline for one data seed.

    init_mode: 'save' (SAVE warm start) or 'random' (Glorot, no warm start).
    Returns a dict with all CSV columns.
    """
    tag = f"seed={seed}, init={init_mode}"
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {tag}")
        print(f"{'─'*60}")

    # ── 1. Data ───────────────────────────────────────────────────────────
    V, A, Y, e_true = gen_rank1(N, alpha_true, beta_true, seed=seed)
    mu  = V.mean(0); sig = V.std(0) + 1e-8
    V   = ((V - mu) / sig).astype(np.float32)
    AY  = np.column_stack([A, Y]).astype(np.float32)
    if verbose:
        print(f"  n_treat={int(A.sum())}, n_ctrl={int((A==0).sum())}, "
              f"e=[{e_true.min():.3f},{e_true.max():.3f}]")

    # ── 2. Baselines ─────────────────────────────────────────────────────
    ctrl = A == 0

    alpha_save = save_sdr(V[ctrl], Y[ctrl],
                          n_slices=N_SAVE_SLICES_Y, n_dirs=1).ravel()
    alpha_save /= np.linalg.norm(alpha_save)
    beta_save  = save_sdr(V, A,
                          n_slices=N_SAVE_SLICES_A, n_dirs=1).ravel()
    beta_save  /= np.linalg.norm(beta_save)

    alpha_sir  = sir(V[ctrl], Y[ctrl],
                     n_slices=N_SAVE_SLICES_Y, n_dirs=1).ravel()
    alpha_sir  /= (np.linalg.norm(alpha_sir) + 1e-10)
    beta_sir   = sir(V, A,
                     n_slices=N_SAVE_SLICES_A, n_dirs=1).ravel()
    beta_sir   /= (np.linalg.norm(beta_sir) + 1e-10)

    alpha_lasso, beta_lasso = lasso_directions(V, A, Y)

    a_save_cos2  = cosine_sq(alpha_save, alpha_true)
    b_save_cos2  = cosine_sq(beta_save,  beta_true)
    a_sir_cos2   = cosine_sq(alpha_sir,  alpha_true)
    b_sir_cos2   = cosine_sq(beta_sir,   beta_true)
    a_lasso_cos2 = cosine_sq(alpha_lasso, alpha_true)
    b_lasso_cos2 = cosine_sq(beta_lasso,  beta_true)

    if verbose:
        print(f"  SAVE:  cos²β={b_save_cos2:.3f},  cos²α={a_save_cos2:.3f}")
        print(f"  SIR:   cos²β={b_sir_cos2:.3f},  cos²α={a_sir_cos2:.3f}")
        print(f"  LASSO: cos²β={b_lasso_cos2:.3f}, cos²α={a_lasso_cos2:.3f}")

    # ── 3. Initialisation ────────────────────────────────────────────────
    if init_mode == 'save':
        alpha0 = alpha_save.copy()
        beta0  = beta_save.copy()
        beta_init_cos2  = b_save_cos2
        alpha_init_cos2 = a_save_cos2
    else:
        rng_init = np.random.default_rng(seed + 7777)
        beta0  = rng_init.standard_normal(P).astype(np.float32)
        beta0  /= np.linalg.norm(beta0)
        alpha0 = rng_init.standard_normal(P).astype(np.float32)
        alpha0 /= np.linalg.norm(alpha0)
        beta_init_cos2  = cosine_sq(beta0,  beta_true)
        alpha_init_cos2 = cosine_sq(alpha0, alpha_true)

    if verbose:
        print(f"  Init ({init_mode}): cos²β={beta_init_cos2:.3f}, "
              f"cos²α={alpha_init_cos2:.3f}")

    # ── 4. BCE propensity ────────────────────────────────────────────────
    net_A = PropensityNet(P, hidden=(8,))
    net_A.W_A.assign(beta0.reshape(-1, 1).astype(np.float32), read_value=False)
    net_A.normalize()

    if verbose:
        print(f"  [BCE propensity, hidden=(8,)] ...")
    ep_A, val_bce = train_propensity(net_A, V, A, seed=seed, verbose=verbose)
    if verbose:
        print(f"  → stopped ep={ep_A}, val_BCE={val_bce:.6f}")

    beta_hat = net_A.W_A.numpy().ravel()
    beta_sr_cos2 = cosine_sq(beta_hat, beta_true)

    wA_all  = get_wA(net_A, V, A)
    norm_wA = float(np.mean(np.linalg.norm(wA_all, axis=1)))

    wA_true = true_w_A_nl(V, A, beta_true)
    eps_A   = score_mse(wA_all, wA_true)

    if verbose:
        print(f"  cos²β={beta_sr_cos2:.3f}  εA={eps_A:.4f}  ‖ŵA‖={norm_wA:.4f}")

    # ── 5. ISM joint score ───────────────────────────────────────────────
    net_AY = ScoreRatioNet(P, obs_dim=2, r_prime=R_AY, s_prime=S_AY,
                           hidden_units=HIDDEN_AY, name='AY_r2_nl_stable')

    col0   = beta0.reshape(-1, 1).astype(np.float32)
    col1   = alpha0.reshape(-1, 1).astype(np.float32)
    W_init = np.hstack([col0, col1])
    Q_init, _ = np.linalg.qr(W_init)
    net_AY.W_v.assign(Q_init[:, :R_AY].astype(np.float32), read_value=False)

    if verbose:
        print(f"  [ISM joint score, r'={R_AY}] ...")
    ep_AY, val_ism = train_joint_score(net_AY, V, AY, seed=seed, verbose=verbose)
    if verbose:
        print(f"  → stopped ep={ep_AY}, val_ISM={val_ism:.6f}")

    # ── 6. Per-sample subtraction → α̂ ───────────────────────────────────
    V_ctrl  = V[ctrl]
    Y_ctrl  = Y[ctrl]
    n0      = ctrl.sum()
    A0_np   = np.zeros(n0, dtype=np.float32)
    AY0     = np.column_stack([np.zeros(n0), Y_ctrl]).astype(np.float32)

    wAY_ctrl = get_wAY(net_AY, V_ctrl, AY0)
    wA_ctrl  = get_wA(net_A, V_ctrl, A0_np)

    d     = wAY_ctrl - wA_ctrl
    H0    = 0.5 * (d.T @ d / n0 + (d.T @ d / n0).T)
    alpha_hat, _ = compute_subspace(H0, 1)
    alpha_hat    = alpha_hat.ravel()
    alpha_sr_cos2 = cosine_sq(alpha_hat, alpha_true)

    norm_wAY = float(np.mean(np.linalg.norm(wAY_ctrl, axis=1)))

    wYA_true = true_w_YA_nl(V_ctrl, Y_ctrl, alpha_true)
    eps_AY   = score_mse(d, wYA_true)

    if verbose:
        print(f"  cos²α={alpha_sr_cos2:.3f}  εAY={eps_AY:.4f}  ‖ŵAY‖={norm_wAY:.4f}")
        print(f"  ► seed {seed} ({init_mode}): cos²β={beta_sr_cos2:.3f}, "
              f"cos²α={alpha_sr_cos2:.3f}")

    return dict(
        seed                       = seed,
        init_mode                  = init_mode,
        beta_init_cos2             = round(float(beta_init_cos2),   4),
        alpha_init_cos2            = round(float(alpha_init_cos2),  4),
        beta_save_cos2             = round(float(b_save_cos2),      4),
        alpha_save_cos2            = round(float(a_save_cos2),      4),
        beta_sir_cos2              = round(float(b_sir_cos2),       4),
        alpha_sir_cos2             = round(float(a_sir_cos2),       4),
        beta_lasso_cos2            = round(float(b_lasso_cos2),     4),
        alpha_lasso_cos2           = round(float(a_lasso_cos2),     4),
        beta_sr_cos2               = round(float(beta_sr_cos2),     4),
        alpha_sr_cos2              = round(float(alpha_sr_cos2),    4),
        selected_epoch_A           = ep_A,
        selected_epoch_AY          = ep_AY,
        val_BCE                    = round(float(val_bce),          6),
        val_ISM                    = round(float(val_ism),          6),
        norm_wA                    = round(norm_wA,                 4),
        norm_wAY                   = round(norm_wAY,                4),
        score_error_A_oracle_eval  = round(float(eps_A),            4),
        score_error_YA_oracle_eval = round(float(eps_AY),           4),
    )

# ── Summary and success criterion ─────────────────────────────────────────────

def print_summary(df, label, n_seeds=None):
    n_seeds = n_seeds or len(df)
    print(f"\n{'═'*60}")
    print(f"  {label} — {len(df)} seeds")
    print(f"{'═'*60}")
    for col in ('beta_sr_cos2', 'alpha_sr_cos2',
                'beta_save_cos2', 'alpha_save_cos2',
                'score_error_A_oracle_eval', 'score_error_YA_oracle_eval',
                'norm_wA', 'norm_wAY'):
        if col not in df.columns:
            continue
        v = df[col].values
        print(f"  {col:<38s}  med={np.median(v):.3f}  "
              f"[{v.min():.3f}, {v.max():.3f}]")

    b = df['beta_sr_cos2'].values
    a = df['alpha_sr_cos2'].values
    med_b, med_a   = np.median(b), np.median(a)
    n_ok           = int(np.sum((b >= 0.50) & (a >= 0.50)))
    success = (med_b >= 0.70 and med_a >= 0.80 and n_ok == len(df))
    print(f"\n  SUCCESS CRITERION: med_β≥0.70: {med_b:.3f}, "
          f"med_α≥0.80: {med_a:.3f}, "
          f"no catastrophic (<0.50): {n_ok}/{len(df)}")
    print(f"  → {'PASS ✓' if success else 'FAIL ✗'}")
    return success


def main():
    OUT_CSV = os.path.join(_here, 'results', 'sr_nn_nonlinear_rank1_stable.csv')
    os.makedirs(os.path.join(_here, 'results'), exist_ok=True)

    alpha_true, beta_true = make_rank1_directions()
    print(f"DGP: p={P}, n={N}, q={QA}, α·β={float(alpha_true @ beta_true):.4f}, "
          f"s_T={S_T}, noise={NOISE}")

    # ── Phase 1: 3-seed canonical (SAVE init) ────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# Phase 1: SAVE-init pipeline, 3 seeds")
    print(f"{'#'*60}")
    rows_save = []
    for seed in [1, 2, 3]:
        row = run_one_seed(seed, alpha_true, beta_true,
                           init_mode='save', verbose=True)
        rows_save.append(row)

    df3 = pd.DataFrame(rows_save)
    pass3 = print_summary(df3, "SAVE-init, 3 seeds")

    # ── Phase 2: extend to 5 seeds if Phase 1 passes ─────────────────────
    if pass3:
        print(f"\n{'─'*60}")
        print(f"  Phase 1 PASSED → extending to 5 seeds")
        print(f"{'─'*60}")
        for seed in [4, 5]:
            row = run_one_seed(seed, alpha_true, beta_true,
                               init_mode='save', verbose=True)
            rows_save.append(row)
        df_save = pd.DataFrame(rows_save)
        print_summary(df_save, "SAVE-init, 5 seeds (primary)")
    else:
        print("\n  Phase 1 FAILED — stopping at 3 seeds. Check diagnostics.")
        df_save = df3

    # ── Phase 3: exploratory — random init on seeds 1, 2 ─────────────────
    print(f"\n{'#'*60}")
    print(f"# Exploratory: random init vs SAVE init (seeds 1, 2)")
    print(f"{'#'*60}")
    rows_rand = []
    for seed in [1, 2]:
        row = run_one_seed(seed, alpha_true, beta_true,
                           init_mode='random', verbose=True)
        rows_rand.append(row)

    df_rand = pd.DataFrame(rows_rand)
    print_summary(df_rand, "Random-init (seeds 1, 2, exploratory)")

    # ── Save combined CSV ─────────────────────────────────────────────────
    all_rows = rows_save + rows_rand
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # ── Side-by-side comparison for seeds 1, 2 ───────────────────────────
    print(f"\n{'─'*60}")
    print("  SAVE init vs random init (seeds 1–2)")
    print(f"{'─'*60}")
    comp = df_all[df_all['seed'].isin([1, 2])][
        ['seed', 'init_mode', 'beta_sr_cos2', 'alpha_sr_cos2',
         'beta_save_cos2', 'alpha_save_cos2']
    ].sort_values(['seed', 'init_mode'])
    print(comp.to_string(index=False))


if __name__ == '__main__':
    main()
