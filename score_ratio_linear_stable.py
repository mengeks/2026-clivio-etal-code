"""
score_ratio_linear_stable.py

Canonical fully data-driven linear DR experiment.

Pipeline:
  1. LASSO warm start  → β̂₀ (logistic), α̂₀ (OLS on controls)
  2. BCE propensity NN → ŵ_A,  W_A ∈ R^p normalized each step (rank-1)
  3. ISM joint score   → ŵ_AY, W_v ∈ Stiefel(p,2) projected each step
  4. Per-sample d_i = ŵ_AY − ŵ_A on controls → Ĥ₀ → α̂

No oracle quantities used in training or model selection.
Early stopping: validation BCE (propensity) and validation ISM (joint).

DGP: Clivio et al. setting 4, p=100, n=500.
Success criterion: median cos²(β̂,β) ≥ 0.80, median cos²(α̂,α) ≥ 0.80, ≥4/5 both > 0.70.

Saves: results/sr_nn_linear_stable.csv
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
_diag_dir = os.path.join(_here, 'score_ratio_linear_diagnosis_and_exploration')
if os.path.isdir(_diag_dir):
    sys.path.insert(0, _diag_dir)

import numpy as np
import pandas as pd
import tensorflow as tf

from score_ratio_reduction import ScoreRatioNet, compute_subspace
from score_ratio_rescue import (
    construct_beta, gen_simulated,
    true_w_A, true_w_YA, score_mse, cosine_sq, sklearn_directions,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_SEEDS        = 5
SETTING        = 4       # s_T=4, s_Y=5, p=100, q_α=20
P, QA          = 100, 20

# BCE propensity
L2_A           = 1e-3    # L2 penalty on all PropensityNet weights
LR_A           = 1e-3    # Adam learning rate
MAX_EPOCHS_A   = 600
PATIENCE_A     = 40      # stop if val BCE hasn't improved for this many epochs
VAL_FRAC_A     = 0.2     # fraction of data held out for val
BATCH_SIZE_A   = 64
CLIP_NORM_A    = 1.0     # per-gradient clip norm

# ISM joint score
R_AY           = 2       # W_v ∈ Stiefel(p, 2)   — span(α,β)
S_AY           = 2       # W_obs rank for (A, Y) observations
HIDDEN_AY      = (16, 16)
LAM2_AY        = 1e-2    # nuclear norm on W_obs
LAM_L2_AY      = 1e-3    # L2 on inner ψ weights
LR_AY          = 3e-3
MAX_EPOCHS_AY  = 600
PATIENCE_AY    = 40
VAL_FRAC_AY    = 0.2
CLIP_NORM_AY   = 1.0

VERBOSE_INTERVAL = 100   # print every N epochs during training


# ── Propensity network ─────────────────────────────────────────────────────────

class PropensityNet(tf.Module):
    """η(v) = h(W_A^T v),  W_A ∈ R^p normalised to unit norm after each step.

    hidden=()    → linear inner model  (η = w·(W_A^T v) + b)
    hidden=(8,)  → one tanh hidden layer
    """

    def __init__(self, p, hidden=(8,), name='prop_net'):
        super().__init__(name=name)
        self.p = p
        init = tf.initializers.GlorotUniform()
        self.W_A = tf.Variable(init(shape=(p, 1)), dtype=tf.float32, name='W_A')
        layers = []
        for h_dim in hidden:
            layers.append(tf.keras.layers.Dense(h_dim, activation='tanh'))
        layers.append(tf.keras.layers.Dense(1))          # scalar output
        self.h = tf.keras.Sequential(layers, name='inner_h')
        self.h.build((None, 1))

    @property
    def trainable_variables(self):
        return [self.W_A] + list(self.h.trainable_variables)

    def eta(self, V):
        """η(v) = h(W_A^T v), shape (n,)."""
        V = tf.cast(V, tf.float32)
        z = V @ self.W_A                     # (n, 1)
        return tf.squeeze(self.h(z), axis=-1)  # (n,)

    def normalize(self):
        """Project W_A to the unit sphere."""
        w = self.W_A.numpy()
        self.W_A.assign(w / (np.linalg.norm(w) + 1e-8), read_value=False)


def _bce_loss(net, V_tf, A_tf):
    """BCE loss without L2 penalty (used for validation)."""
    eta = net.eta(V_tf)
    A   = tf.cast(A_tf, tf.float32)
    return -tf.reduce_mean(
        A * tf.math.log_sigmoid(eta) + (1. - A) * tf.math.log_sigmoid(-eta))


def train_propensity(net, V_np, A_np, val_frac=VAL_FRAC_A,
                     n_epochs=MAX_EPOCHS_A, lr=LR_A, l2=L2_A,
                     patience=PATIENCE_A, batch_size=BATCH_SIZE_A,
                     clip_norm=CLIP_NORM_A, seed=0, verbose=True):
    """Train PropensityNet with BCE + L2 + validation early stopping.

    W_A is normalised to unit norm after every mini-batch step.
    Returns: (selected_epoch, val_bce_at_selected_epoch)
    """
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
        # Mini-batch training
        perm_ep = np.random.permutation(n_tr)
        for start in range(0, n_tr, batch_size):
            b_idx = perm_ep[start:start + batch_size]
            Vb = tf.gather(V_tr, b_idx)
            Ab = tf.gather(A_tr, b_idx)
            with tf.GradientTape() as tape:
                bce   = _bce_loss(net, Vb, Ab)
                l2_p  = l2 * tf.add_n([tf.reduce_sum(v**2)
                                        for v in net.trainable_variables])
                loss  = bce + l2_p
            grads = tape.gradient(loss, net.trainable_variables)
            grads = [tf.clip_by_norm(g, clip_norm) if g is not None else g
                     for g in grads]
            opt.apply_gradients(zip(grads, net.trainable_variables))
            net.normalize()   # project W_A to unit sphere after every step

        # Validation (bare BCE, no penalty)
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
    net.normalize()   # ensure unit norm after weight restoration
    return selected_epoch, best_val_bce


def get_wA(net, V_np, A_np):
    """ŵ_A(V_i, A_i) = (A_i − ê_i) ∇_v η(V_i),  shape (n, p)."""
    V_tf = tf.constant(V_np, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(V_tf)
        eta_sum = tf.reduce_sum(net.eta(V_tf))
    g    = t.gradient(eta_sum, V_tf).numpy()            # (n, p)
    e    = tf.sigmoid(net.eta(V_tf)).numpy()             # (n,)
    return ((A_np - e)[:, None] * g)                    # (n, p)


# ── ISM joint score: exact QR trace + Stiefel projection ─────────────────────

def stiefel_project(net):
    """Project net.W_v to the Stiefel manifold (orthonormal columns)."""
    W = net.W_v.numpy()
    Q, _ = np.linalg.qr(W)
    net.W_v.assign(Q[:, :net.r_prime].astype(np.float32), read_value=False)


def _ism_step_qr(net, optimizer, V_tf, obs_tf,
                 lam2=LAM2_AY, lam_l2=LAM_L2_AY, clip_norm=CLIP_NORM_AY):
    """ISM gradient step with exact QR trace (r' forward passes).

    Does NOT include nuclear norm on W_v (Stiefel projection handles it).
    Regularizers: nuclear norm on W_obs + L2 on inner ψ weights.
    """
    W_v_np = net.W_v.numpy()
    Q_np, _ = np.linalg.qr(W_v_np)          # probes are constants wrt the graph
    n, r_prime = V_tf.shape[0], net.r_prime

    with tf.GradientTape() as outer:
        # Exact trace via r' QR ambient probes
        trace_est = tf.constant(0.0)
        for k in range(r_prime):
            z_k   = tf.constant(Q_np[:, k].astype(np.float32))   # (p,)
            z_k_b = tf.tile(z_k[None, :], [n, 1])                # (n, p)
            with tf.GradientTape() as inner:
                inner.watch(V_tf)
                w_z = net(V_tf, obs_tf)
                sw  = tf.reduce_sum(z_k_b * w_z, axis=1)          # (n,)
            dw_dv     = inner.gradient(sw, V_tf)                   # (n, p)
            trace_est = trace_est + tf.reduce_mean(
                tf.reduce_sum(dw_dv * z_k_b, axis=1))

        # ½‖w‖² + ref = −v^T w
        w    = net(V_tf, obs_tf)
        half = tf.reduce_mean(0.5 * tf.reduce_sum(w * w, axis=1))
        ref  = tf.reduce_mean(-tf.reduce_sum(V_tf * w, axis=1))

        # Regularization
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
    """Bare ISM loss (no regularization) on a held-out set. Returns float."""
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
        dw_dv    = inner.gradient(sw, V_tf)
        trace_val += float(tf.reduce_mean(tf.reduce_sum(dw_dv * z_k_b, axis=1)))

    w    = net(V_tf, obs_tf).numpy()
    half = float(np.mean(0.5 * np.sum(w * w, axis=1)))
    ref  = float(np.mean(-np.sum(V_tf.numpy() * w, axis=1)))
    return half + trace_val + ref


def train_joint_score(net, V_np, AY_np, val_frac=VAL_FRAC_AY,
                      n_epochs=MAX_EPOCHS_AY, lr=LR_AY,
                      lam2=LAM2_AY, lam_l2=LAM_L2_AY,
                      patience=PATIENCE_AY, seed=0, verbose=True):
    """Train joint score (ISM) with Stiefel projection + val early stopping.

    Full-batch training on n_tr = n*(1−val_frac) samples.
    Returns: (selected_epoch, val_ism_at_selected_epoch)
    """
    n = len(V_np)
    rng = np.random.default_rng(seed + 888)
    perm = rng.permutation(n)
    n_val = max(10, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    V_tr  = tf.constant(V_np[tr_idx],  dtype=tf.float32)
    AY_tr = tf.constant(AY_np[tr_idx], dtype=tf.float32)
    V_val = tf.constant(V_np[val_idx],  dtype=tf.float32)
    AY_val= tf.constant(AY_np[val_idx], dtype=tf.float32)

    # Build the Keras Sequential inside ScoreRatioNet before accessing trainable_variables
    _ = net(V_tr[:1], AY_tr[:1])

    opt            = tf.keras.optimizers.Adam(lr)
    best_val_ism   = float('inf')
    best_weights   = [v.numpy().copy() for v in net.trainable_variables]
    patience_count = 0
    selected_epoch = 0

    for epoch in range(1, n_epochs + 1):
        _ism_step_qr(net, opt, V_tr, AY_tr, lam2=lam2, lam_l2=lam_l2)
        stiefel_project(net)     # W_v stays on Stiefel manifold

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
    stiefel_project(net)     # ensure orthonormal after restoration
    return selected_epoch, best_val_ism


def get_wAY(net, V_np, obs_np, batch_size=64):
    """Forward-pass ŵ_AY vectors, shape (n, p)."""
    n     = len(V_np)
    V_tf  = tf.constant(V_np,  dtype=tf.float32)
    obs_tf= tf.constant(obs_np, dtype=tf.float32)
    return np.concatenate([
        net(V_tf[s:min(s+batch_size,n)], obs_tf[s:min(s+batch_size,n)]).numpy()
        for s in range(0, n, batch_size)], axis=0)


# ── Full pipeline for one seed ─────────────────────────────────────────────────

def run_one_seed(seed, alpha_true, beta_true, hidden_A=(8,), verbose=True):
    """Run the full stable pipeline for one data seed.

    Returns a dict with all CSV columns.
    Oracle quantities (escale, mscale, true directions) are used only for
    post-hoc evaluation diagnostics, never for training or model selection.
    """
    tag = f"seed={seed}, hidden_A={hidden_A}"
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {tag}")
        print(f"{'─'*60}")

    # ── 1. Generate data ─────────────────────────────────────────────────
    X, T, Y, escale, mscale = gen_simulated(SETTING, seed, alpha_true, beta_true)
    mu  = X.mean(0); sig = X.std(0) + 1e-8
    V   = ((X - mu) / sig).astype(np.float32)
    A   = T.astype(np.float32)
    AY  = np.column_stack([T, Y]).astype(np.float32)
    if verbose:
        print(f"  n_treat={T.sum()}, n_ctrl={(T==0).sum()}, "
              f"escale={escale}, mscale={mscale}")

    # ── 2. LASSO warm start (data-driven; no oracle) ──────────────────────
    alpha0, beta0 = sklearn_directions(X, T, Y, l1_alpha=1.0, l1_T=1.0)
    beta_init_cos2  = cosine_sq(beta0,  beta_true)
    alpha_init_cos2 = cosine_sq(alpha0, alpha_true)
    if verbose:
        print(f"  LASSO init: cos²β={beta_init_cos2:.3f}, "
              f"cos²α={alpha_init_cos2:.3f}")

    # ── 3. BCE propensity ─────────────────────────────────────────────────
    net_A = PropensityNet(P, hidden=hidden_A)
    net_A.W_A.assign(beta0.reshape(-1, 1).astype(np.float32), read_value=False)
    net_A.normalize()

    if verbose:
        print(f"  [BCE propensity, hidden={hidden_A}] ...")
    ep_A, val_bce = train_propensity(net_A, V, A, seed=seed, verbose=verbose)
    if verbose:
        print(f"  → stopped ep={ep_A}, val_BCE={val_bce:.6f}")

    beta_hat = net_A.W_A.numpy().ravel()   # already unit norm
    beta_final_cos2 = cosine_sq(beta_hat, beta_true)

    wA_all  = get_wA(net_A, V, A)          # (n, p)
    norm_wA = float(np.mean(np.linalg.norm(wA_all, axis=1)))

    # Oracle diagnostic (eval only)
    wA_true = true_w_A(V, A, beta_true, escale)
    eps_A   = score_mse(wA_all, wA_true)

    if verbose:
        print(f"  cos²β={beta_final_cos2:.3f}  εA={eps_A:.4f}  ‖ŵA‖={norm_wA:.4f}")

    # ── 4. ISM joint score ────────────────────────────────────────────────
    net_AY = ScoreRatioNet(P, obs_dim=2, r_prime=R_AY, s_prime=S_AY,
                           hidden_units=HIDDEN_AY, name='AY_r2_stable')

    # Init W_v = orth([β̂₀, α̂₀])  (data-driven)
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

    # ── 5. Per-sample subtraction on controls → α̂ ───────────────────────
    ctrl   = T == 0
    n0     = ctrl.sum()
    V_ctrl = V[ctrl]
    Y_ctrl = Y[ctrl]
    A0_np  = np.zeros(n0, dtype=np.float32)
    AY0    = np.column_stack([np.zeros(n0), Y_ctrl]).astype(np.float32)

    wAY_ctrl = get_wAY(net_AY, V_ctrl, AY0)   # (n0, p)
    wA_ctrl  = get_wA(net_A,  V_ctrl, A0_np)  # (n0, p) evaluated at a=0

    d    = wAY_ctrl - wA_ctrl                  # (n0, p), per-sample F1 correction
    H0   = 0.5 * (d.T @ d / n0 + (d.T @ d / n0).T)
    alpha_hat, _ = compute_subspace(H0, 1)
    alpha_hat    = alpha_hat.ravel()
    alpha_final_cos2 = cosine_sq(alpha_hat, alpha_true)

    norm_wAY = float(np.mean(np.linalg.norm(wAY_ctrl, axis=1)))

    # Oracle diagnostic (eval only)
    wYA_true = true_w_YA(V_ctrl, A0_np, Y_ctrl, alpha_true, mscale)
    eps_AY   = score_mse(d, wYA_true)

    if verbose:
        print(f"  cos²α={alpha_final_cos2:.3f}  εAY={eps_AY:.4f}  ‖ŵAY‖={norm_wAY:.4f}")

    return dict(
        seed                       = seed,
        hidden_A                   = str(hidden_A),
        beta_init_cos2             = round(float(beta_init_cos2),  4),
        alpha_init_cos2            = round(float(alpha_init_cos2), 4),
        beta_final_cos2            = round(float(beta_final_cos2), 4),
        alpha_final_cos2           = round(float(alpha_final_cos2),4),
        selected_epoch_A           = ep_A,
        selected_epoch_AY          = ep_AY,
        validation_BCE             = round(float(val_bce), 6),
        validation_ISM             = round(float(val_ism), 6),
        score_error_A_oracle_eval  = round(float(eps_A),  4),
        score_error_AY_oracle_eval = round(float(eps_AY), 4),
        norm_wA                    = round(norm_wA,  4),
        norm_wAY                   = round(norm_wAY, 4),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def print_summary(df, label):
    print(f"\n{'═'*60}")
    print(f"  {label} — {len(df)} seeds")
    print(f"{'═'*60}")
    for col in ('beta_final_cos2', 'alpha_final_cos2',
                'score_error_A_oracle_eval', 'score_error_AY_oracle_eval',
                'norm_wA', 'norm_wAY'):
        v = df[col].values
        print(f"  {col:<38s}  med={np.median(v):.3f}  "
              f"[{v.min():.3f}, {v.max():.3f}]")

    b = df['beta_final_cos2'].values
    a = df['alpha_final_cos2'].values
    n_pass = int(np.sum((b > 0.70) & (a > 0.70)))
    med_b, med_a = np.median(b), np.median(a)
    success = (med_b >= 0.80 and med_a >= 0.80 and n_pass >= 4)
    print(f"\n  SUCCESS CRITERION: med_β={med_b:.3f} (≥0.80), "
          f"med_α={med_a:.3f} (≥0.80), "
          f"seeds≥0.70: {n_pass}/5 (need 4)")
    print(f"  → {'PASS ✓' if success else 'FAIL ✗'}")


def main():
    OUT_CSV = 'results/sr_nn_linear_stable.csv'
    os.makedirs('results', exist_ok=True)

    # Fixed DGP directions (same as rescue.py / bernoulli.py)
    rng0 = np.random.default_rng(0)
    alpha_true = np.concatenate([rng0.standard_normal(QA) / QA,
                                 np.zeros(P - QA)])
    alpha_true /= np.linalg.norm(alpha_true)
    beta_true   = construct_beta(alpha_true, 0.75, rng0)
    print(f"DGP: p={P}, qa={QA}, α·β={alpha_true @ beta_true:.4f}")

    all_rows = []

    for hidden_A in [(8,), ()]:
        label = f"PropensityNet hidden={hidden_A}"
        print(f"\n{'#'*60}")
        print(f"# {label}")
        print(f"{'#'*60}")

        rows = []
        for seed in range(1, N_SEEDS + 1):
            row = run_one_seed(seed, alpha_true, beta_true,
                               hidden_A=hidden_A, verbose=True)
            rows.append(row)
            all_rows.append(row)
            print(f"  ► seed {seed}: cos²β={row['beta_final_cos2']:.3f}, "
                  f"cos²α={row['alpha_final_cos2']:.3f}")

        df_h = pd.DataFrame(rows)
        print_summary(df_h, label)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Final summary for primary hidden=(8,) configuration
    df8 = df[df['hidden_A'] == str((8,))]
    print_summary(df8, "PRIMARY: hidden=(8,)")


if __name__ == '__main__':
    main()
