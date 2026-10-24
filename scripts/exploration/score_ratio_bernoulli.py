"""
Bernoulli-structured score-ratio model for treatment score.

Model:
  η_θ(v) = h_θ(W_v^T v)   scalar logit,  h_θ: R^{r'} → R (MLP)
  e_θ(v) = σ(η_θ(v))
  w_A(v,a) = {a − e_θ(v)} ∇_v η_θ(v)

The score lies in span(W_v) by construction.  The true Bernoulli score is
  w_A*(v,a) = s × (a − σ(s v^T β)) × β
so at convergence ∇_v η → s β and e → σ(s v^T β).

ISM objective (ρ = N(0, I)):
  J = E[ ½‖w_A‖² + ∇·w_A − v^T w_A ]

Analytical divergence (product rule + σ' = e(1−e)):
  ∇·w_A = (a−e) Δ_v η − e(1−e) ‖∇_v η‖²

Exact Laplacian:
  Δ_v η = Tr(H_h G),  G = W_v^T W_v,  H_h = Hessian of h at z = W_v^T v
Computed via r' double-tape HVPs using QR ambient probes (watch V_tf pattern).
Proof: with W_v = QR,  Σ_k Q[:,k]^T H_v(η) Q[:,k] = Tr(R H_h R^T) = Tr(H_h G).

Comparison:
  Bernoulli SR-NN (r'=5, exact Laplacian)
  Generic SR-NN rescue  (r'=1, exact trace, F2+F3+F4)
  LASSO/Ridge baselines

Saves:
  outputs/results/exploration/sr_nn_bernoulli_p100.csv      — per-seed summary comparison
  outputs/results/exploration/sr_nn_bernoulli_training.csv  — epoch diagnostics (Bernoulli model)
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)   # sibling exploration scripts
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_here)), 'outputs', 'results', 'exploration')

import numpy as np
import pandas as pd
import tensorflow as tf

from causal_embedding.models.score_ratio_reduction import compute_subspace
from score_ratio_rescue import (
    construct_beta, gen_simulated,
    true_w_A, true_w_YA, score_mse, cosine_sq,
    fit_H_A_rank1, fit_H_AY_r2, sklearn_directions,
    _get_vecs, SKLEARN_COMBOS, NAN_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BernoulliScoreNet(tf.Module):
    """Scalar-logit treatment score: η(v) = h(W_v^T v), w_A = (a−e) ∇_v η."""

    def __init__(self, p, r_prime, hidden_units=(32, 32), name='bern_srnet'):
        super().__init__(name=name)
        self.p       = p
        self.r_prime = r_prime

        init = tf.initializers.GlorotUniform(seed=None)(shape=(p, r_prime))
        self.W_v = tf.Variable(init, dtype=tf.float32, name='W_v')

        layers = []
        for h_dim in hidden_units:
            layers.append(tf.keras.layers.Dense(h_dim, activation='tanh'))
        layers.append(tf.keras.layers.Dense(1))   # scalar output, no activation
        self.h = tf.keras.Sequential(layers, name='h_theta')
        self.h.build((None, r_prime))

    @property
    def trainable_variables(self):
        return [self.W_v] + self.h.trainable_variables

    def eta(self, V):
        """η(v) = h(W_v^T v), shape (n,)."""
        return tf.squeeze(self.h(V @ self.W_v), axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Exact Laplacian Δ_v η = Tr(H_h G) via r' double-tape QR HVPs
# ─────────────────────────────────────────────────────────────────────────────

def _exact_laplacian_qr(net, V_tf):
    """Per-sample Δ_v η = Σ_k z_k^T H_v(η) z_k, where z_k = Q[:,k] ∈ R^p.

    With W_v = QR (thin QR):
      Σ_k Q[:,k]^T H_v(η) Q[:,k] = Tr(R H_h R^T) = Tr(H_h G),  G = R^T R.

    For each probe k:
      1. t1 computes ∂η/∂V (1st derivative), watching V_tf.
      2. t2 computes ∂(proj_k)/∂V (2nd derivative, Hessian-vector product).
      3. lap_k = z_k · (H_v η) z_k per sample.

    Must be called inside the training-step outer GradientTape so that
    outer.gradient(loss, params) correctly backprops through the Laplacian.
    """
    W_v_np = net.W_v.numpy()
    Q_np, _ = np.linalg.qr(W_v_np)   # Q: (p, r'), probes treated as constants
    n = V_tf.shape[0]
    lap = tf.zeros((n,))

    for k in range(net.r_prime):
        z_k   = tf.constant(Q_np[:, k].astype(np.float32))   # (p,)
        z_k_b = tf.tile(z_k[None, :], [n, 1])                # (n, p)

        with tf.GradientTape() as t2:
            t2.watch(V_tf)                 # explicit: V_tf is not a tf.Variable
            with tf.GradientTape() as t1:
                t1.watch(V_tf)
                eta_k     = net.eta(V_tf)              # (n,), inside t1 & t2
                eta_k_sum = tf.reduce_sum(eta_k)       # scalar, inside t1
            deta_dV = t1.gradient(eta_k_sum, V_tf)     # (n, p), inside t2
            proj      = tf.reduce_sum(deta_dV * z_k_b, axis=1)   # (n,)
            proj_sum  = tf.reduce_sum(proj)             # scalar, inside t2
        Hvk = t2.gradient(proj_sum, V_tf)              # (n, p) = H_v(η) z_k

        lap = lap + tf.reduce_sum(Hvk * z_k_b, axis=1)   # (n,) z_k^T H z_k

    return lap


# ─────────────────────────────────────────────────────────────────────────────
# ISM training step
# ─────────────────────────────────────────────────────────────────────────────

def bern_ism_step(net, optimizer, V_tf, A_tf, lam1=1e-3):
    """Full-batch ISM step for the Bernoulli score model.

    Loss = E[ ½(a−e)²‖g‖² + (a−e)Δη − e(1−e)‖g‖² − (a−e)v·g ] + λ1 ‖W_v‖_*
    g = ∇_v η = W_v ∇h,  Δη = Tr(H_h G) via _exact_laplacian_qr.

    The outer tape records all operations so that gradients propagate through
    g (1st-order), Δη (2nd-order), and their products with (a−e) to both
    W_v and h_theta's parameters.
    """
    a = A_tf   # (n,) float32

    with tf.GradientTape() as outer:
        # ── compute g = ∇_v η and e = σ(η) via a single inner tape ──────────
        with tf.GradientTape() as t_grad:
            t_grad.watch(V_tf)
            eta     = net.eta(V_tf)             # (n,)
            eta_sum = tf.reduce_sum(eta)         # scalar, inside t_grad
        g = t_grad.gradient(eta_sum, V_tf)      # (n, p) = ∇_v η, inside outer

        e      = tf.sigmoid(eta)                 # (n,)
        g_sq   = tf.reduce_sum(g * g, axis=1)   # (n,) ‖∇η‖²
        v_dotg = tf.reduce_sum(V_tf * g, axis=1)# (n,) v · ∇η

        # ── exact Laplacian Δ_v η (inside outer, captures param dependency) ──
        lap = _exact_laplacian_qr(net, V_tf)    # (n,)

        # ── ISM loss per sample ──────────────────────────────────────────────
        half    = 0.5 * (a - e) ** 2 * g_sq     # ½‖w_A‖²
        div_ae  = (a - e) * lap                  # (a−e) Δη term
        div_var = -e * (1. - e) * g_sq           # −e(1−e)‖g‖² term
        ref     = -(a - e) * v_dotg              # −v^T w_A term

        loss = tf.reduce_mean(half + div_ae + div_var + ref)

        with tf.device('/CPU:0'):
            nuc_v = tf.reduce_sum(tf.linalg.svd(net.W_v, compute_uv=False))
        loss = loss + lam1 * nuc_v

    grads = outer.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Score evaluation
# ─────────────────────────────────────────────────────────────────────────────

def bern_get_scores(net, V_tf, a_np):
    """Compute w_A(V_i, a_i) = (a_i − e_i) ∇_v η(V_i).

    a_np : (n,) numpy array of treatment values (0.0 for a=0 evaluation).

    Returns:
      H_A   (p×p) symmetrised score outer-product matrix
      w_A   (n×p) score vectors
      e_np  (n,)  estimated propensities
    """
    with tf.GradientTape() as t:
        t.watch(V_tf)
        eta     = net.eta(V_tf)          # (n,)
        eta_sum = tf.reduce_sum(eta)
    g = t.gradient(eta_sum, V_tf)        # (n, p)

    e_np  = tf.sigmoid(eta).numpy()      # (n,)
    g_np  = g.numpy()                    # (n, p)

    w_A = ((a_np - e_np)[:, None] * g_np)   # (n, p)
    H   = w_A.T @ w_A / len(a_np)
    return 0.5 * (H + H.T), w_A, e_np


# ─────────────────────────────────────────────────────────────────────────────
# W_v warm start
# ─────────────────────────────────────────────────────────────────────────────

def warm_start_Wv(beta_true, r_prime, seed=0):
    """W_v with first column aligned to β*, remaining columns orthogonal."""
    p   = len(beta_true)
    rng = np.random.default_rng(seed)
    col0 = beta_true.astype(np.float32).reshape(-1, 1)
    rest = rng.standard_normal((p, r_prime - 1)).astype(np.float32)
    Q, _ = np.linalg.qr(np.hstack([col0, rest]))
    return Q[:, :r_prime].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop with epoch diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_one_bern(net, optimizer, V_tf, A_tf_1d,
                 V_np, A_np, beta_true, escale,
                 n_epochs, ckpt_every, data_seed, init_seed, init_type,
                 lam1=1e-3):
    """Train for n_epochs.  Returns list of checkpoint records."""
    records = []
    for epoch in range(1, n_epochs + 1):
        loss = bern_ism_step(net, optimizer, V_tf, A_tf_1d, lam1=lam1)

        if epoch % ckpt_every == 0 or epoch == 1:
            H_A, w_A, _ = bern_get_scores(net, V_tf, A_np)
            beta_hat, eigs = compute_subspace(H_A, 1)
            c2b    = cosine_sq(beta_hat.ravel(), beta_true)
            w_true = true_w_A(V_np, A_np, beta_true, escale)
            eps_A  = score_mse(w_A, w_true)

            svs     = tf.linalg.svd(net.W_v, compute_uv=False).numpy()
            norm_wv = float(np.linalg.norm(svs))
            mean_wa = float(np.mean(np.sum(w_A ** 2, axis=1)))

            rec = dict(
                data_seed=data_seed, init_seed=init_seed, init_type=init_type,
                epoch=epoch, loss=round(loss, 4),
                cos2_beta=round(c2b, 4), eps_A=round(eps_A, 4),
                norm_wv=round(norm_wv, 4), mean_wA_sq=round(mean_wa, 4),
                **{f'sv{j+1}': round(float(svs[j]), 4) for j in range(len(svs))},
            )
            records.append(rec)
            print(f"  ep{epoch:>4}  loss={loss:>10.3f}  "
                  f"cos²β={c2b:.3f}  εA={eps_A:.3f}  ‖Wv‖={norm_wv:.3f}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Bernoulli α-recovery
# ─────────────────────────────────────────────────────────────────────────────

def bern_alpha_recovery(net_bern, net_AY, V_ctrl_tf, AY_ctrl_tf, batch_size=64):
    """d_i = w_AY(V_i, 0, Y_i) - w_A_Bernoulli(V_i, 0) on controls.

    AY_ctrl_tf has shape (n0, 2): controls already have T=0 in column 0.

    Returns H0 (p×p), alpha direction, score-diff matrix d (n0×p), eigs.
    """
    n0    = V_ctrl_tf.shape[0]
    a0_np = np.zeros(n0, dtype=np.float32)

    _, w_A_ctrl, _ = bern_get_scores(net_bern, V_ctrl_tf, a0_np)
    w_AY_ctrl = _get_vecs(net_AY, V_ctrl_tf, AY_ctrl_tf, batch_size)

    d  = w_AY_ctrl - w_A_ctrl
    H0 = 0.5 * (d.T @ d / n0 + (d.T @ d).T / n0)
    alpha_hat, eigs_H0 = compute_subspace(H0, 1)
    return H0, alpha_hat.ravel(), d, eigs_H0


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SETTING      = 4
    R_PRIME      = 5
    LAM1         = 1e-3
    N_EPOCHS     = 500
    CKPT_EVERY   = 25
    LR           = 3e-3
    HIDDEN       = (32, 32)
    DATA_SEEDS   = [1, 2]
    N_RAND_INITS = 5
    OUT_CSV      = os.path.join(OUT_DIR, 'sr_nn_bernoulli_p100.csv')
    TRAIN_CSV    = os.path.join(OUT_DIR, 'sr_nn_bernoulli_training.csv')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Fixed DGP
    rng0 = np.random.default_rng(0)
    p, qa = 100, 20
    alpha_true = np.concatenate([rng0.standard_normal(qa) / qa, np.zeros(p - qa)])
    alpha_true /= np.linalg.norm(alpha_true)
    beta_true  = construct_beta(alpha_true, 0.75, rng0)
    print(f"DGP: p={p}, qa={qa}, alpha·beta={alpha_true @ beta_true:.4f}")

    rows      = []
    train_log = []

    for data_seed in DATA_SEEDS:
        print(f"\n{'='*70}")
        print(f"  Data seed {data_seed}")
        print('='*70)
        X, T, Y, escale, mscale = gen_simulated(SETTING, data_seed,
                                                  alpha_true, beta_true)
        print(f"  n={len(T)}, n_treat={T.sum()}, n_ctrl={(T==0).sum()}")

        # Standardise X (consistent with rescue; X~N(0,I) so ≈ identity)
        mu_x  = X.mean(0); sig_x = X.std(0) + 1e-8
        V_std = ((X - mu_x) / sig_x).astype(np.float32)
        A     = T.astype(np.float32)
        AY    = np.column_stack([T, Y]).astype(np.float32)

        V_tf  = tf.constant(V_std)
        A_tf  = tf.constant(A)          # (n,) 1-D for Bernoulli model
        AY_tf = tf.constant(AY)

        ctrl  = T == 0;  n0 = ctrl.sum()
        V_c   = tf.constant(V_std[ctrl])
        AY_c  = tf.constant(AY[ctrl])

        # ── Fit shared H_AY (generic, for α recovery) ───────────────────────
        print("\n[H_AY: r'=2, Hutchinson K=8, 200 epochs ...]")
        net_AY = fit_H_AY_r2(V_std, AY, n_epochs=200, batch_size=64,
                              lam1=0.02, lam2=0.01, n_hutch=8,
                              hidden=(32, 32), verbose=False)

        # ── Bernoulli SR-NN: r'=5, exact Laplacian ──────────────────────────
        print(f"\n[Bernoulli SR-NN: r'={R_PRIME}, exact Laplacian, {N_EPOCHS} epochs]")
        best_bern = None;  best_bern_c2b = -1.0

        for init_seed in range(N_RAND_INITS):
            print(f"\n  -- Random init {init_seed} --")
            tf.random.set_seed(init_seed * 1000 + data_seed * 7)
            np.random.seed(init_seed * 1000 + data_seed * 7)
            net_b = BernoulliScoreNet(p, R_PRIME, HIDDEN,
                                      name=f'bern_d{data_seed}_i{init_seed}')
            opt_b = tf.keras.optimizers.Adam(LR)
            recs  = run_one_bern(
                net_b, opt_b, V_tf, A_tf,
                V_std, A, beta_true, escale,
                N_EPOCHS, CKPT_EVERY,
                data_seed, init_seed, 'random', lam1=LAM1)
            train_log.extend(recs)
            if recs[-1]['cos2_beta'] > best_bern_c2b:
                best_bern_c2b = recs[-1]['cos2_beta']
                best_bern = net_b

        # Warm start: W_v[:,0] = β*
        print(f"\n  -- Warm start (W_v[:,0] = β*) --")
        net_bw = BernoulliScoreNet(p, R_PRIME, HIDDEN,
                                   name=f'bern_d{data_seed}_warm')
        net_bw.W_v.assign(warm_start_Wv(beta_true, R_PRIME, seed=data_seed))
        opt_bw = tf.keras.optimizers.Adam(LR)
        recs_w = run_one_bern(
            net_bw, opt_bw, V_tf, A_tf,
            V_std, A, beta_true, escale,
            N_EPOCHS, CKPT_EVERY,
            data_seed, 99, 'warm_start', lam1=LAM1)
        train_log.extend(recs_w)
        if recs_w[-1]['cos2_beta'] > best_bern_c2b:
            best_bern_c2b = recs_w[-1]['cos2_beta']
            best_bern = net_bw

        pd.DataFrame(train_log).to_csv(TRAIN_CSV, index=False)
        print(f"\n  Best Bernoulli cos²β = {best_bern_c2b:.4f}")

        # α recovery for best Bernoulli model
        H_A_b, w_A_b, e_b = bern_get_scores(best_bern, V_c, np.zeros(n0, np.float32))
        beta_hat_b, eigs_Ab = compute_subspace(H_A_b, 1)
        beta_hat_b = beta_hat_b.ravel()

        _, alpha_hat_b, d_b, eigs_H0b = bern_alpha_recovery(
            best_bern, net_AY, V_c, AY_c)

        w_A_true_c = true_w_A(V_std[ctrl], A[ctrl], beta_true, escale)
        eps_A_b    = score_mse(w_A_b, w_A_true_c)

        rows.append(dict(
            iter=data_seed, method='Bernoulli SR-NN',
            cos2_alpha=cosine_sq(alpha_hat_b, alpha_true),
            cos2_beta =cosine_sq(beta_hat_b,  beta_true),
            ab_dp     =float(alpha_hat_b @ beta_hat_b),
            eps_A     =eps_A_b,
            eig1_A    =float(eigs_Ab[0]),
            eig2_A    =float(eigs_Ab[1] if len(eigs_Ab) > 1 else 0.),
            eig1_H0   =float(eigs_H0b[0]),
            eig2_H0   =float(eigs_H0b[1] if len(eigs_H0b) > 1 else 0.),
        ))

        # ── Generic SR-NN rescue (r'=1, F2+F3+F4) ──────────────────────────
        print("\n[Generic SR-NN rescue: r'=1, exact trace, 200 epochs]")
        H_A_g, net_A_g, _ = fit_H_A_rank1(
            V_std, A, n_epochs=200, batch_size=64, lam1=0.01,
            hidden=(16, 16), verbose=True)
        beta_hat_g, eigs_Ag = compute_subspace(H_A_g, 1)
        beta_hat_g = beta_hat_g.ravel()

        A0_c = tf.constant(np.zeros((n0, 1), dtype=np.float32))
        w_AY_c_g  = _get_vecs(net_AY, V_c, AY_c)
        w_A_g_c   = _get_vecs(net_A_g, V_c, A0_c)
        d_g = w_AY_c_g - w_A_g_c
        H0_g = 0.5 * (d_g.T @ d_g / n0 + (d_g.T @ d_g).T / n0)
        alpha_hat_g, eigs_H0g = compute_subspace(H0_g, 1)
        alpha_hat_g = alpha_hat_g.ravel()

        eps_A_g = score_mse(w_A_g_c, w_A_true_c)

        rows.append(dict(
            iter=data_seed, method='Generic SR-NN (rescue r1)',
            cos2_alpha=cosine_sq(alpha_hat_g, alpha_true),
            cos2_beta =cosine_sq(beta_hat_g,  beta_true),
            ab_dp     =float(alpha_hat_g @ beta_hat_g),
            eps_A     =eps_A_g,
            eig1_A    =float(eigs_Ag[0]),
            eig2_A    =float(eigs_Ag[1] if len(eigs_Ag) > 1 else 0.),
            eig1_H0   =float(eigs_H0g[0]),
            eig2_H0   =float(eigs_H0g[1] if len(eigs_H0g) > 1 else 0.),
        ))

        # ── LASSO / Ridge baselines ──────────────────────────────────────────
        for c in SKLEARN_COMBOS:
            print(f"[ {c['label']} ]")
            ah, bh = sklearn_directions(X, T, Y, c['l1_alpha'], c['l1_T'])
            rows.append(dict(
                iter=data_seed, method=c['label'],
                cos2_alpha=cosine_sq(ah, alpha_true),
                cos2_beta =cosine_sq(bh, beta_true),
                ab_dp     =float(ah @ bh),
                eps_A=np.nan,
                eig1_A=np.nan, eig2_A=np.nan, eig1_H0=np.nan, eig2_H0=np.nan,
            ))

        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

        # per-seed summary
        sr_b = rows[-6]; sr_g = rows[-5]
        print(f"\n  Bernoulli SR-NN:         cos²β={sr_b['cos2_beta']:.3f}  "
              f"cos²α={sr_b['cos2_alpha']:.3f}  εA={sr_b['eps_A']:.3f}")
        print(f"  Generic SR-NN rescue:    cos²β={sr_g['cos2_beta']:.3f}  "
              f"cos²α={sr_g['cos2_alpha']:.3f}  εA={sr_g['eps_A']:.3f}")

    # ── Final summary ────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"Results → {OUT_CSV}")
    print(f"Training log → {TRAIN_CSV}")
    print('='*70)
    print("\nMedian across data seeds:")
    cols = ['cos2_alpha', 'cos2_beta', 'eps_A', 'ab_dp']
    print(df.groupby('method')[cols].median().round(4).to_string())


if __name__ == '__main__':
    main()
