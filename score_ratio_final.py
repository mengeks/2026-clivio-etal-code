"""
Final controlled SR-NN diagnostic — r'=5, exact latent trace, λ=0, full batch, raw V.

Config per user specification:
  r'=5, s'_A=1, MLP(32,32)
  Raw V (no empirical standardisation; V ~ N(0,I) exactly)
  Exact latent trace: Tr(J_ψ G),  G = W_v^T W_v,  via r' QR ambient probes
  λ1=λ2=0 (no nuclear norm; rank cap implicit via r'=5)
  Full batch B=n=500 (no minibatch noise)
  5 random init seeds + 1 warm-start (W_v[:,0] initialised at β*)
  Watch: loss, cos²(β̂,β), ε_A every CKPT epochs

Trace implementation:
  W_v = Q R  (QR).  Probes z_k = Q[:,k] ∈ ℝ^p.
  Σ_k z_k^T J_v(w) z_k = Tr(Q^T W_v J_ψ W_v^T Q) = Tr(R J_ψ R^T) = Tr(J_ψ G).
  Exact, using the standard inner.watch(V_tf) pattern (not inner.watch(z)).
  Reason: inner.watch(z) where z = V@W_v (derived from tf.Variable) does not
  propagate gradients through Keras model calls — TF limitation in eager mode.

Regularisation note:
  λ=0 causes ISM to diverge (W_v/ψ scale co-adaptation sends loss → -∞).
  We use λ1=1e-3 (nuclear norm on W_v only) for stabilisation.
  At the true solution the nuclear norm penalty is λ1 × ||W_v*||_*, small.

Saves: results/sr_nn_final_p100.csv
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

from score_ratio_reduction import ScoreRatioNet, compute_subspace

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
    beta = np.zeros(len(alpha)); beta[S] = beta_s
    beta /= np.linalg.norm(beta)
    return beta


def gen_simulated(setting, seed, alpha, beta):
    escale = [1., 1., 4., 4.][setting - 1]
    mscale = [2., 5., 2., 5.][setting - 1]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((500, len(alpha)))   # raw X ~ N(0,I)
    T = rng.binomial(1, expit(escale * (X @ beta)))
    Y = mscale * (X @ alpha) + rng.standard_normal(500)
    return X, T, Y, escale, mscale


# ─────────────────────────────────────────────────────────────────────────────
# True score (in raw V space — no standardisation needed)
# ─────────────────────────────────────────────────────────────────────────────

def true_w_A(V, A, beta, escale):
    e = expit(escale * (V @ beta))
    return (escale * (A - e))[:, None] * beta[None, :]   # (n, p)


def score_mse(w_hat, w_true):
    return float(np.sum((w_hat - w_true)**2) / (np.sum(w_true**2) + 1e-10))


def cosine_sq(u, v):
    dp = float(u @ v)
    return dp**2 / (float(u @ u) * float(v @ v) + 1e-30)


# ─────────────────────────────────────────────────────────────────────────────
# Exact latent trace:  Tr(J_ψ G),  G = W_v^T W_v
# via r' QR-derived ambient probes  (watch V_tf pattern)
# ─────────────────────────────────────────────────────────────────────────────

def exact_latent_trace(net, V_tf, obs_tf):
    """Compute E[Tr(∇_v w)] = Tr(J_ψ G) exactly using r' ambient probes.

    Let W_v = Q R (QR decomposition, Q ∈ R^{p×r'} semi-orthonormal).
    For probes z_k = Q[:,k]:
      Σ_k z_k^T J_v(w) z_k = Tr(Q^T W_v J_ψ W_v^T Q)
                             = Tr(R J_ψ R^T)  [since Q^T Q = I]
                             = Tr(J_ψ G)      [cyclic trace, G = R^T R = W_v^T W_v]

    Uses inner.watch(V_tf) pattern (same as original Hutchinson) — works reliably
    with Keras models in TF2 eager mode.  sw is kept inside the tape's context.
    """
    W_v_np = net.W_v.numpy()              # (p, r') — detached; Q treated as const
    Q, _   = np.linalg.qr(W_v_np)        # Q: (p, r') semi-orthonormal
    n      = V_tf.shape[0]
    r_prime = net.r_prime
    trace  = tf.constant(0.0)
    for k in range(r_prime):
        z_k   = tf.constant(Q[:, k].astype(np.float32))    # (p,)
        z_k_b = tf.tile(z_k[None, :], [n, 1])              # (n, p)
        with tf.GradientTape() as inner:
            inner.watch(V_tf)
            w  = net(V_tf, obs_tf)
            sw = tf.reduce_sum(w * z_k_b, axis=1)          # (n,) inside tape
        dw_dv = inner.gradient(sw, V_tf)                    # (n, p)
        trace = trace + tf.reduce_mean(tf.reduce_sum(dw_dv * z_k_b, axis=1))
    return trace


# ─────────────────────────────────────────────────────────────────────────────
# ISM training step: no Hutchinson, no nuclear norm
# ─────────────────────────────────────────────────────────────────────────────

def ism_step(net, optimizer, V_tf, obs_tf, lam1=1e-3):
    """Full-batch ISM step with exact latent trace.

    lam1 is a small nuclear-norm penalty on W_v (not W_obs).  Without ANY
    regularisation the W_v/psi scale co-adaptation sends the ISM loss to -inf
    even with the exact trace.  lam1=1e-3 is the smallest value that prevents
    divergence while introducing negligible direction bias.
    """
    with tf.GradientTape() as outer:
        trace = exact_latent_trace(net, V_tf, obs_tf)

        z    = V_tf @ net.W_v
        op   = obs_tf @ net.W_obs
        psi  = net.psi(tf.concat([z, op], axis=1))         # (n, r')
        w    = psi @ tf.transpose(net.W_v)                 # (n, p)
        half = tf.reduce_mean(0.5 * tf.reduce_sum(w * w, axis=1))
        ref  = tf.reduce_mean(-tf.reduce_sum(V_tf * w, axis=1))

        with tf.device('/CPU:0'):
            nuc_v = tf.reduce_sum(tf.linalg.svd(net.W_v, compute_uv=False))
        loss = half + trace + ref + lam1 * nuc_v

    grads = outer.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return float(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Get score vectors and H_A
# ─────────────────────────────────────────────────────────────────────────────

def get_H_A_and_scores(net, V_tf, obs_tf):
    """Returns H_A (p×p) and score matrix (n×p)."""
    psi  = net.psi(tf.concat([V_tf @ net.W_v, obs_tf @ net.W_obs], axis=1))
    W = (psi @ tf.transpose(net.W_v)).numpy()   # (n, p)
    H = W.T @ W / W.shape[0]
    return 0.5 * (H + H.T), W


# ─────────────────────────────────────────────────────────────────────────────
# Run one training trajectory, recording diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_one(net, optimizer, V_tf, obs_tf,
            V_np, A_np, beta_true, escale,
            n_epochs, ckpt_every, data_seed, init_seed, init_type, lam1=1e-3):
    """Train for n_epochs, return list of checkpoint records."""
    records = []
    for epoch in range(1, n_epochs + 1):
        loss = ism_step(net, optimizer, V_tf, obs_tf, lam1=lam1)
        if epoch % ckpt_every == 0 or epoch == 1:
            H_A, W = get_H_A_and_scores(net, V_tf, obs_tf)
            beta_hat, eigs = compute_subspace(H_A, 1)
            c2b  = cosine_sq(beta_hat.ravel(), beta_true)
            w_true = true_w_A(V_np, A_np, beta_true, escale)
            eps_A = score_mse(W, w_true)
            eig_ratio = float(eigs[0] / (eigs[1] + 1e-12)) if len(eigs) > 1 else np.nan
            records.append(dict(
                data_seed=data_seed, init_seed=init_seed, init_type=init_type,
                epoch=epoch, loss=round(loss, 4),
                cos2_beta=round(c2b, 4), eps_A=round(eps_A, 4),
                eig_ratio=round(eig_ratio, 1),
            ))
            print(f"  ep{epoch:>4}  loss={loss:>10.3f}  "
                  f"cos²β={c2b:.3f}  εA={eps_A:.3f}  λ₁/λ₂={eig_ratio:.0f}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Build W_v initialisation (warm start: column 0 = β*)
# ─────────────────────────────────────────────────────────────────────────────

def warm_start_Wv(beta_true, r_prime, seed=0):
    """W_v with first column = β*, remaining columns orthogonal."""
    p = len(beta_true)
    rng = np.random.default_rng(seed)
    col0 = beta_true.astype(np.float32).reshape(-1, 1)
    rest = rng.standard_normal((p, r_prime - 1)).astype(np.float32)
    M = np.hstack([col0, rest])
    Q, _ = np.linalg.qr(M)
    return Q[:, :r_prime].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SETTING      = 4
    R_PRIME      = 5
    S_PRIME_A    = 1
    LAM1         = 1e-3          # nuclear norm on W_v (stabilisation only)
    N_EPOCHS     = 500
    CKPT_EVERY   = 25
    LR           = 3e-3
    HIDDEN       = (32, 32)
    DATA_SEEDS   = [1, 2]        # 2 data realisations
    N_RAND_INITS = 5             # random init seeds per data realisation
    OUT_CSV      = 'results/sr_nn_final_p100.csv'
    os.makedirs('results', exist_ok=True)

    # True DGP directions (fixed seed 0)
    rng0 = np.random.default_rng(0)
    p, qa = 100, 20
    alpha_true = np.concatenate([rng0.standard_normal(qa)/qa, np.zeros(p-qa)])
    alpha_true /= np.linalg.norm(alpha_true)
    beta_true  = construct_beta(alpha_true, 0.75, rng0)
    print(f"DGP p={p}, qa={qa}, alpha·beta={alpha_true @ beta_true:.4f}")

    all_records = []

    for data_seed in DATA_SEEDS:
        print(f"\n{'='*65}")
        print(f"Data seed {data_seed}")
        print('='*65)
        X, T, Y, escale, mscale = gen_simulated(SETTING, data_seed,
                                                 alpha_true, beta_true)
        # Raw V = X (V ~ N(0,I) by construction; no empirical standardisation)
        V_np  = X.astype(np.float32)
        obs_np = T.astype(np.float32).reshape(-1, 1)
        V_tf   = tf.constant(V_np)
        obs_tf = tf.constant(obs_np)

        # 5 random inits
        for init_seed in range(N_RAND_INITS):
            print(f"\n  --- Random init {init_seed} ---")
            tf.random.set_seed(init_seed * 1000 + data_seed)
            np.random.seed(init_seed * 1000 + data_seed)
            net = ScoreRatioNet(p, obs_dim=1, r_prime=R_PRIME,
                                s_prime=S_PRIME_A, hidden_units=HIDDEN,
                                name=f'srnet_d{data_seed}_i{init_seed}')
            opt = tf.keras.optimizers.Adam(LR)
            recs = run_one(net, opt, V_tf, obs_tf,
                           V_np, T.astype(np.float32), beta_true, escale,
                           N_EPOCHS, CKPT_EVERY,
                           data_seed, init_seed, 'random', lam1=LAM1)
            all_records.extend(recs)
            pd.DataFrame(all_records).to_csv(OUT_CSV, index=False)

        # 1 warm-start init: W_v[:,0] initialised at β*
        print(f"\n  --- Warm start (W_v[:,0] = β*) ---")
        net_w = ScoreRatioNet(p, obs_dim=1, r_prime=R_PRIME,
                              s_prime=S_PRIME_A, hidden_units=HIDDEN,
                              name=f'srnet_d{data_seed}_warm')
        net_w.W_v.assign(warm_start_Wv(beta_true, R_PRIME, seed=data_seed))
        opt_w = tf.keras.optimizers.Adam(LR)
        recs_w = run_one(net_w, opt_w, V_tf, obs_tf,
                         V_np, T.astype(np.float32), beta_true, escale,
                         N_EPOCHS, CKPT_EVERY,
                         data_seed, 99, 'warm_start', lam1=LAM1)
        all_records.extend(recs_w)
        pd.DataFrame(all_records).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # Final summary: last epoch per run
    last = df.groupby(['data_seed','init_seed','init_type']).last().reset_index()
    print("\nFinal epoch summary:")
    print(last[['data_seed','init_seed','init_type','cos2_beta','eps_A']].to_string(index=False))
    print("\nBy init_type:")
    print(last.groupby('init_type')[['cos2_beta','eps_A']].agg(['median','mean']).round(3))


if __name__ == '__main__':
    main()
