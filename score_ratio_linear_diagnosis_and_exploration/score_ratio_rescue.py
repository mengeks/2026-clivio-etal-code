"""
Score-ratio NN rescue experiment — p=100, setting 4, 20 iterations.

Four correctness fixes vs. score_ratio_nn_diagnostic.py:

  F1. d_i subtraction BEFORE outer product:
        d_i = ŵ_{AY}(V_i, 0, Y_i) − ŵ_A(V_i, 0)   [controls only]
        H_0 = (1/n_0) Σ d_i d_i^T  →  u_1(H_0) ≈ α
      (original: H_YA = H_Y − H_X, i.e. subtraction of p×p matrices,
       which ≠ E[(w_AY−w_A)(w_AY−w_A)^T] in general)

  F2. H_A architecture: r'=1, s'_A=1
      (original: r'=5, s'=2 — over-parameterised; truth is rank-1)

  F3. H_A regularisation: lam2_A = 0
      (original: lam2=0.01 nuclear norm on 1×1 W_obs shrinks treatment signal)

  F4. H_A trace: exact via probe z = W_v/‖W_v‖ (no Hutchinson variance)
      With r'=1:  z^T ∇_v w z = (∂ψ/∂z_V)·‖W_v‖²  =  Tr(∇_v w)  exactly.
      (original: n_hutch=4 random probes, Var ∝ p²/K/B ≈ 39 >> signal)

Additional diagnostic: true-score relative MSE  ε_A, ε_{Y|A}.

Saves: results/sr_nn_rescue_p100.csv
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
from sklearn.linear_model import LogisticRegressionCV, LassoCV, RidgeCV

from score_ratio_reduction import ScoreRatioNet, compute_subspace, _ism_step

# ─────────────────────────────────────────────────────────────────────────────
# DGP
# ─────────────────────────────────────────────────────────────────────────────

def construct_beta(alpha, ab_dp, rng):
    S = np.where(np.abs(alpha) > 0)[0]
    alpha_s = alpha[S]
    v = rng.standard_normal(len(S))
    v -= (v @ alpha_s) / (alpha_s @ alpha_s) * alpha_s
    v /= np.linalg.norm(v)
    beta_s = ab_dp * alpha_s + np.sqrt(1 - ab_dp**2) * v
    beta = np.zeros(len(alpha))
    beta[S] = beta_s
    beta /= np.linalg.norm(beta)
    return beta


def gen_simulated(setting, seed, alpha, beta):
    escale = [1., 1., 4., 4.][setting - 1]
    mscale = [2., 5., 2., 5.][setting - 1]
    rng = np.random.default_rng(seed)
    n = 500
    X = rng.standard_normal((n, len(alpha)))
    e = expit(escale * (X @ beta))
    T = rng.binomial(1, e)
    Y = mscale * (X @ alpha) + rng.standard_normal(n)
    return X, T, Y, escale, mscale


# ─────────────────────────────────────────────────────────────────────────────
# True score functions — approximate in standardised V space (μ≈0, σ≈1)
# ─────────────────────────────────────────────────────────────────────────────

def true_w_A(V, A, beta, escale):
    """w_A^*(v,a) = escale·(a − sigmoid(escale·v@β))·β  in standardised V space."""
    e = expit(escale * (V @ beta))
    return (escale * (A - e))[:, None] * beta[None, :]   # (n, p)


def true_w_YA(V, A, Y, alpha, mscale, tau=0.0):
    """w_{Y|A}^*(v,a,y) = mscale·(y − mscale·v@α − τ·a)·α."""
    mu = mscale * (V @ alpha) + tau * A
    return (mscale * (Y - mu))[:, None] * alpha[None, :]  # (n, p)


def score_mse(w_hat, w_true):
    return float(np.sum((w_hat - w_true) ** 2) / (np.sum(w_true ** 2) + 1e-10))


def cosine_sq(u, v):
    dp = float(u @ v)
    return dp ** 2 / (float(u @ u) * float(v @ v) + 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# F4: Exact-trace training step for r'=1
#     Probe z = W_v/‖W_v‖ is constant (detached from graph) each step.
#     Then:  z^T J_v(w) z = (∂ψ/∂z_V)·‖W_v‖²  =  Tr(J_v(w))  exactly.
# ─────────────────────────────────────────────────────────────────────────────

def _ism_step_exact1(net, optimizer, v, obs, lam1):
    """ISM training step with exact divergence for rank-1 network.

    Uses a single deterministic probe z = W_v/‖W_v‖ instead of Hutchinson.
    F3 (lam2=0): W_obs nuclear-norm term is omitted.
    """
    w_v_np  = net.W_v.numpy().ravel()
    norm_wv = float(np.linalg.norm(w_v_np)) + 1e-8
    z_np    = (w_v_np / norm_wv).astype(np.float32)
    z_probe = tf.constant(np.tile(z_np, (v.shape[0], 1)))  # (batch, p)

    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            inner.watch(v)
            w_z = net(v, obs)
            sw  = tf.reduce_sum(z_probe * w_z, axis=1)
        dw_dv       = inner.gradient(sw, v)
        trace_exact = tf.reduce_mean(tf.reduce_sum(dw_dv * z_probe, axis=1))

        w    = net(v, obs)
        half = tf.reduce_mean(0.5 * tf.reduce_sum(w * w, axis=1))
        ref  = tf.reduce_mean(-tf.reduce_sum(v * w, axis=1))

        with tf.device('/CPU:0'):
            nuc_v = tf.reduce_sum(tf.linalg.svd(net.W_v, compute_uv=False))

        # lam2 = 0: no W_obs nuclear-norm term
        loss = half + trace_exact + ref + lam1 * nuc_v

    grads = outer.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Fit H_A: r'=1, exact trace, lam2=0  (F2 + F3 + F4)
# ─────────────────────────────────────────────────────────────────────────────

def fit_H_A_rank1(V_std, A,
                  n_epochs=200, batch_size=64, lam1=0.01,
                  hidden=(16, 16), verbose=True):
    """Returns H_A (p×p), fitted net, score-vector matrix (n×p)."""
    n, p = V_std.shape
    obs  = A.reshape(-1, 1).astype(np.float32)
    net  = ScoreRatioNet(p, obs_dim=1, r_prime=1, s_prime=1,
                         hidden_units=hidden, name='H_A_r1')
    opt  = tf.keras.optimizers.Adam(1e-3)

    V_tf   = tf.constant(V_std, dtype=tf.float32)
    obs_tf = tf.constant(obs,   dtype=tf.float32)
    ds     = (tf.data.Dataset.from_tensor_slices((V_tf, obs_tf))
              .shuffle(n, seed=0).batch(batch_size))

    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for bv, bo in ds:
            epoch_loss += float(_ism_step_exact1(net, opt, bv, bo, lam1))
            n_batches  += 1
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"    [H_A r1-exact] epoch {epoch+1:>4}/{n_epochs}"
                  f"  loss={epoch_loss/n_batches:.4f}")

    W = _get_vecs(net, V_tf, obs_tf, batch_size)
    H = 0.5 * ((W.T @ W) / n + (W.T @ W).T / n)
    return H, net, W


# ─────────────────────────────────────────────────────────────────────────────
# Fit H_AY: r'=2, Hutchinson K=8
# ─────────────────────────────────────────────────────────────────────────────

def fit_H_AY_r2(V_std, AY,
                n_epochs=200, batch_size=64, lam1=0.02, lam2=0.01, n_hutch=8,
                hidden=(32, 32), verbose=True):
    n, p = V_std.shape
    net  = ScoreRatioNet(p, obs_dim=2, r_prime=2, s_prime=2,
                         hidden_units=hidden, name='H_AY_r2')
    opt  = tf.keras.optimizers.Adam(1e-3)

    V_tf  = tf.constant(V_std, dtype=tf.float32)
    AY_tf = tf.constant(AY,   dtype=tf.float32)
    ds    = (tf.data.Dataset.from_tensor_slices((V_tf, AY_tf))
             .shuffle(n, seed=0).batch(batch_size))

    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for bv, bo in ds:
            epoch_loss += float(_ism_step(net, opt, bv, bo, lam1, lam2, n_hutch))
            n_batches  += 1
        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"    [H_AY r2]  epoch {epoch+1:>4}/{n_epochs}"
                  f"  loss={epoch_loss/n_batches:.4f}")
    return net


def _get_vecs(net, V_tf, obs_tf, batch_size=64):
    n = V_tf.shape[0]
    return np.concatenate(
        [net(V_tf[s:min(s+batch_size,n)],
             obs_tf[s:min(s+batch_size,n)]).numpy()
         for s in range(0, n, batch_size)], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# LASSO / Ridge baselines
# ─────────────────────────────────────────────────────────────────────────────

def sklearn_directions(X, T, Y, l1_alpha=1.0, l1_T=1.0):
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
    solver = 'liblinear' if l1_T == 1.0 else 'lbfgs'
    penalty = 'l1' if l1_T == 1.0 else 'l2'
    prop = LogisticRegressionCV(penalty=penalty, solver=solver,
                                cv=5, max_iter=1000, random_state=0).fit(Xc, T)
    bh = prop.coef_.ravel()
    bh /= np.linalg.norm(bh) + 1e-10

    ctrl = T == 0
    Xc0, Y0 = Xc[ctrl], Y[ctrl]
    if l1_alpha == 1.0:
        ah = LassoCV(cv=5, max_iter=5000, random_state=0).fit(Xc0, Y0).coef_
    else:
        ah = RidgeCV(cv=5).fit(Xc0, Y0).coef_
    ah /= np.linalg.norm(ah) + 1e-10
    return ah, bh


# ─────────────────────────────────────────────────────────────────────────────
# SR-NN rescue: all four fixes
# ─────────────────────────────────────────────────────────────────────────────

def sr_rescue(X, T, Y, escale, mscale, alpha_true, beta_true, verbose=True):
    """Apply F1–F4.  Returns alpha_hat, beta_hat, diag dict."""
    mu  = X.mean(0); sig = X.std(0) + 1e-8
    V   = ((X - mu) / sig).astype(np.float32)
    A   = T.astype(np.float32)
    AY  = np.column_stack([T, Y]).astype(np.float32)
    n   = len(T)

    # --- F2 + F3 + F4: H_A (rank-1, exact trace, lam2=0) ---
    if verbose: print("  [H_A] r'=1, exact trace, lam2=0 ...")
    H_A, net_A, w_A_all = fit_H_A_rank1(
        V, A, n_epochs=200, batch_size=64, lam1=0.01,
        hidden=(16, 16), verbose=verbose)
    beta_hat, eigs_A = compute_subspace(H_A, 1)
    beta_hat = beta_hat.ravel()

    # --- H_AY (r'=2, K=8 Hutchinson) ---
    if verbose: print("  [H_AY] r'=2, Hutchinson K=8 ...")
    net_AY = fit_H_AY_r2(
        V, AY, n_epochs=200, batch_size=64,
        lam1=0.02, lam2=0.01, n_hutch=8, hidden=(32, 32), verbose=verbose)

    # --- F1: per-sample d_i subtraction on controls ---
    ctrl  = T == 0
    n0    = int(ctrl.sum())
    V_c   = tf.constant(V[ctrl],  dtype=tf.float32)
    A0_c  = tf.constant(np.zeros((n0, 1), dtype=np.float32))   # evaluate at a=0
    AY_c  = tf.constant(AY[ctrl], dtype=tf.float32)

    w_AY_c = _get_vecs(net_AY, V_c, AY_c)   # (n0, p)
    w_A_c  = _get_vecs(net_A,  V_c, A0_c)   # (n0, p)
    d      = w_AY_c - w_A_c                  # F1 correction

    H0 = d.T @ d / n0
    H0 = 0.5 * (H0 + H0.T)
    alpha_hat, eigs_H0 = compute_subspace(H0, 1)
    alpha_hat = alpha_hat.ravel()

    # --- true-score MSE diagnostic ---
    V_ctrl_np = V[ctrl]
    A_ctrl_np = A[ctrl]
    Y_ctrl_np = Y[ctrl]

    w_A_true  = true_w_A(V_ctrl_np,  A_ctrl_np, beta_true,  escale)
    w_YA_true = true_w_YA(V_ctrl_np, A_ctrl_np, Y_ctrl_np,  alpha_true, mscale)
    eps_A  = score_mse(w_A_c,  w_A_true)
    eps_YA = score_mse(d,       w_YA_true)

    diag = dict(
        eig1_A  = float(eigs_A[0]),
        eig2_A  = float(eigs_A[1] if len(eigs_A) > 1 else 0.0),
        eig1_H0 = float(eigs_H0[0]),
        eig2_H0 = float(eigs_H0[1] if len(eigs_H0) > 1 else 0.0),
        eps_A   = eps_A,
        eps_YA  = eps_YA,
    )
    return alpha_hat, beta_hat, diag


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

SKLEARN_COMBOS = [
    dict(l1_alpha=1., l1_T=1., label='LASSO Y / LASSO T'),
    dict(l1_alpha=1., l1_T=0., label='LASSO Y / Ridge T'),
    dict(l1_alpha=0., l1_T=1., label='Ridge Y / LASSO T'),
    dict(l1_alpha=0., l1_T=0., label='Ridge Y / Ridge T'),
]

NAN_COLS = dict(eps_A=np.nan, eps_YA=np.nan,
                eig1_A=np.nan, eig2_A=np.nan,
                eig1_H0=np.nan, eig2_H0=np.nan)


def main():
    SETTING = 4; N_ITERS = 20; P = 100; QA = 20
    OUT_CSV = 'results/sr_nn_rescue_p100.csv'
    os.makedirs('results', exist_ok=True)

    rng0 = np.random.default_rng(0)
    alpha_true = np.concatenate([rng0.standard_normal(QA) / QA, np.zeros(P - QA)])
    alpha_true /= np.linalg.norm(alpha_true)
    beta_true  = construct_beta(alpha_true, 0.75, rng0)
    print(f"DGP: p={P}, qa={QA}, alpha·beta={alpha_true @ beta_true:.4f}")
    print(f"True alpha non-zero entries: {(alpha_true!=0).sum()}, "
          f"True beta non-zero entries: {(beta_true!=0).sum()}")

    rows = []
    for it in range(1, N_ITERS + 1):
        print(f"\n{'='*65}")
        print(f"  Iteration {it}/{N_ITERS}  (setting={SETTING}, p={P})")
        print('='*65)
        X, T, Y, escale, mscale = gen_simulated(SETTING, it, alpha_true, beta_true)
        print(f"  n={len(T)}, n_treat={T.sum()}, n_ctrl={(T==0).sum()}")

        # SR-NN rescue (all four fixes)
        print("[ SR-NN rescue: F1+F2+F3+F4 ]")
        a_sr, b_sr, diag = sr_rescue(
            X, T, Y, escale, mscale, alpha_true, beta_true, verbose=True)
        rows.append(dict(
            iter=it, method='SR-NN rescue',
            cos2_alpha=cosine_sq(a_sr,  alpha_true),
            cos2_beta =cosine_sq(b_sr,  beta_true),
            ab_dp     =float(a_sr @ b_sr),
            **diag,
        ))

        # LASSO / Ridge baselines
        for c in SKLEARN_COMBOS:
            print(f"[ {c['label']} ]")
            ah, bh = sklearn_directions(X, T, Y, c['l1_alpha'], c['l1_T'])
            rows.append(dict(
                iter=it, method=c['label'],
                cos2_alpha=cosine_sq(ah, alpha_true),
                cos2_beta =cosine_sq(bh, beta_true),
                ab_dp     =float(ah @ bh),
                **NAN_COLS,
            ))

        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

        # ── per-iteration summary ────────────────────────────────────
        print()
        sr_row = rows[-5]
        print(f"  SR-NN rescue:  cos²(α)={sr_row['cos2_alpha']:.3f}  "
              f"cos²(β)={sr_row['cos2_beta']:.3f}  "
              f"ε_A={sr_row['eps_A']:.3f}  ε_Y|A={sr_row['eps_YA']:.3f}")
        for row in rows[-4:]:
            print(f"  {row['method']:26s} cos²(α)={row['cos2_alpha']:.3f}  "
                  f"cos²(β)={row['cos2_beta']:.3f}")

    # ── final summary ────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*65}")
    print(f"Final results saved to {OUT_CSV}")
    print('='*65)
    print("\nMedian cos² and score MSE across iterations:")
    cols = ['cos2_alpha', 'cos2_beta', 'ab_dp', 'eps_A', 'eps_YA']
    print(df.groupby('method')[cols].median().round(4).to_string())
    print()
    print("H_A eigenvalue ratio (SR-NN rescue):")
    sr_df = df[df.method == 'SR-NN rescue']
    print(f"  median eig1_A/eig2_A = "
          f"{(sr_df.eig1_A / (sr_df.eig2_A + 1e-8)).median():.1f}")
    print(f"  median eig1_H0/eig2_H0 = "
          f"{(sr_df.eig1_H0 / (sr_df.eig2_H0 + 1e-8)).median():.1f}")


if __name__ == '__main__':
    main()
