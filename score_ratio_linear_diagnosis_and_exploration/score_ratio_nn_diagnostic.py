"""
Neural-network score-ratio dimension reduction diagnostic (p=100 DGP).

DGP: same as oracle_utils.R gen_dataset("simulated", setting, iter)
but with p=100 (20 active features, same 20% sparsity ratio).
p=100 keeps Hutchinson trace variance tractable on CPU (variance ~ p^2).

For each of N_ITERS simulated datasets (setting 4):
  - Fit H_X  via ISM on (V, A)      → top eigvec ≈ beta
  - Fit H_Y  via ISM on (V, [A,Y])  → top-2 eigvecs ≈ span(alpha, beta)
  - H_YA = H_Y - H_X               → top eigvec ≈ alpha

Compare to LASSO/Ridge (sklearn) on the same data.
Saves results/sr_nn_diagnostic_p100.csv.
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))  # parent dir has score_ratio_reduction.py

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegressionCV, LassoCV, RidgeCV

from score_ratio_reduction import fit_score_ratio, compute_subspace

# ---------------------------------------------------------------------------
# DGP — mirrors oracle_utils.R gen_dataset("simulated", setting, iter)
# ---------------------------------------------------------------------------

def construct_beta(alpha, ab_dp, rng):
    S = np.where(np.abs(alpha) > 0)[0]
    alpha_s = alpha[S]
    a, b_val = ab_dp, np.sqrt(1 - ab_dp**2)
    v = rng.standard_normal(len(S))
    v -= (v @ alpha_s) / (alpha_s @ alpha_s) * alpha_s
    v /= np.linalg.norm(v)
    beta_s = a * alpha_s + b_val * v
    beta = np.zeros(len(alpha))
    beta[S] = beta_s
    beta /= np.linalg.norm(beta)
    return beta


def gen_simulated(setting, iter_seed, alpha_true, beta_true):
    escale = [1.0, 1.0, 4.0, 4.0][setting - 1]
    mscale = [2.0, 5.0, 2.0, 5.0][setting - 1]
    n, tau = 500, 0.0
    rng = np.random.default_rng(iter_seed)
    X = rng.standard_normal((n, len(alpha_true)))
    e = expit(escale * (X @ beta_true))
    T = rng.binomial(1, e)
    Y = mscale * (X @ alpha_true) + tau * T + rng.standard_normal(n)
    return X, T, Y


# ---------------------------------------------------------------------------
# LASSO / Ridge direction estimates (sklearn)
# ---------------------------------------------------------------------------

def sklearn_directions(X, T, Y, l1_ratio_alpha=1.0, l1_ratio_T=1.0):
    """
    l1_ratio=1 → Lasso/LogisticL1, l1_ratio=0 → Ridge/LogisticL2
    Returns alpha_hat_n, beta_hat_n (normalised direction vectors).
    """
    n = len(T)
    # Propensity: logistic L1 or L2
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
    if l1_ratio_T == 1.0:
        prop = LogisticRegressionCV(
            penalty='l1', solver='liblinear', cv=5,
            max_iter=1000, random_state=0
        ).fit(Xc, T)
    else:
        prop = LogisticRegressionCV(
            penalty='l2', solver='lbfgs', cv=5,
            max_iter=1000, random_state=0
        ).fit(Xc, T)
    beta_hat = prop.coef_.ravel()
    nrm = np.linalg.norm(beta_hat)
    beta_hat_n = beta_hat / nrm if nrm > 1e-10 else beta_hat

    # Outcome: Lasso or Ridge on controls
    ctrl = T == 0
    Xc0 = Xc[ctrl]; Y0 = Y[ctrl]
    if l1_ratio_alpha == 1.0:
        out = LassoCV(cv=5, max_iter=5000, random_state=0).fit(Xc0, Y0)
        alpha_hat = out.coef_
    else:
        out = RidgeCV(cv=5).fit(Xc0, Y0)
        alpha_hat = out.coef_
    nrm = np.linalg.norm(alpha_hat)
    alpha_hat_n = alpha_hat / nrm if nrm > 1e-10 else alpha_hat

    return alpha_hat_n, beta_hat_n


# ---------------------------------------------------------------------------
# Score-ratio NN direction estimates
# ---------------------------------------------------------------------------

def sr_nn_directions(X, T, Y,
                     r_prime=5, s_prime_X=2, s_prime_Y=3,
                     hidden_units=(64, 64), n_epochs=200,
                     batch_size=64, lam1_X=0.05, lam1_Y=0.02,
                     lam2=0.01, n_hutch=4, verbose=True):
    """
    Fit H_X and H_Y via ISM, compute H_YA = H_Y - H_X.
    Returns:
        alpha_hat_n  — top eigvec of H_YA (≈ alpha direction)
        beta_hat_n   — top eigvec of H_X  (≈ beta direction)
        eigs_X       — eigenvalues of H_X (all p)
        eigs_Y       — eigenvalues of H_Y
        eigs_YA      — eigenvalues of H_YA
    """
    # Standardise X → approximate N(0,I) reference
    mu  = X.mean(0)
    sig = X.std(0) + 1e-8
    Vs  = ((X - mu) / sig).astype(np.float32)

    A  = T.astype(np.float32).reshape(-1, 1)
    AY = np.column_stack([T, Y]).astype(np.float32)

    # H_X: score-ratio of p(V|A) / ρ(V)
    if verbose: print("  Fitting H_X ...")
    H_X, _ = fit_score_ratio(Vs, A,
                              r_prime=r_prime, s_prime=s_prime_X,
                              hidden_units=hidden_units,
                              n_epochs=n_epochs, batch_size=batch_size,
                              lam1=lam1_X, lam2=lam2, n_hutch=n_hutch,
                              verbose=verbose, label='H_X')

    # H_Y: score-ratio of p(V|A,Y) / ρ(V)
    if verbose: print("  Fitting H_Y ...")
    H_Y, _ = fit_score_ratio(Vs, AY,
                              r_prime=r_prime, s_prime=s_prime_Y,
                              hidden_units=hidden_units,
                              n_epochs=n_epochs, batch_size=batch_size,
                              lam1=lam1_Y, lam2=lam2, n_hutch=n_hutch,
                              verbose=verbose, label='H_Y')

    # H_YA = H_Y - H_X  →  concentrates on alpha direction
    H_YA = H_Y - H_X
    H_YA = 0.5 * (H_YA + H_YA.T)

    beta_hat_n,  eigs_X  = compute_subspace(H_X,  1)
    alpha_hat_n, eigs_YA = compute_subspace(H_YA, 1)

    _, eigs_Y = compute_subspace(H_Y, 2)

    return (alpha_hat_n.ravel(), beta_hat_n.ravel(),
            eigs_X, eigs_Y, eigs_YA)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def cosine_sq(u, v):
    dp = float(u @ v)
    return dp**2 / (float(u @ u) * float(v @ v))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

SKLEARN_COMBOS = [
    dict(l1_alpha=1., l1_T=1., label='LASSO Y / LASSO T'),
    dict(l1_alpha=1., l1_T=0., label='LASSO Y / Ridge T'),
    dict(l1_alpha=0., l1_T=1., label='Ridge Y / LASSO T'),
    dict(l1_alpha=0., l1_T=0., label='Ridge Y / Ridge T'),
]


def main():
    SETTING  = 4
    N_ITERS  = 20
    OUT_CSV  = "results/sr_nn_diagnostic_p100.csv"
    os.makedirs("results", exist_ok=True)

    # True DGP directions — p=100, 20 active features (same 20% sparsity)
    rng0 = np.random.default_rng(0)
    qa = 20; p = 100
    alpha_true = np.concatenate([rng0.standard_normal(qa) / qa, np.zeros(p - qa)])
    alpha_true /= np.linalg.norm(alpha_true)
    beta_true = construct_beta(alpha_true, 0.75, rng0)
    print(f"DGP: p={p}, qa={qa}, alpha.beta={alpha_true @ beta_true:.4f}")

    rows = []
    for it in range(1, N_ITERS + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {it}/{N_ITERS}  (setting={SETTING}, p={p})")
        print('='*60)

        X, T, Y = gen_simulated(SETTING, it, alpha_true, beta_true)

        # --- score-ratio NN ---
        print("[ Score-Ratio NN ]")
        a_sr, b_sr, eigs_X, eigs_Y, eigs_YA = sr_nn_directions(
            X, T, Y, verbose=True)

        rows.append({
            'iter': it, 'method': 'SR-NN',
            'cos2_alpha': cosine_sq(a_sr, alpha_true),
            'cos2_beta':  cosine_sq(b_sr, beta_true),
            'ab_dp':      float(a_sr @ b_sr),
            'eig1_X':  float(eigs_X[0]),
            'eig2_X':  float(eigs_X[1]) if len(eigs_X) > 1 else 0.0,
            'eig1_YA': float(eigs_YA[0]),
            'eig2_YA': float(eigs_YA[1]) if len(eigs_YA) > 1 else 0.0,
        })

        # --- LASSO / Ridge combos ---
        for combo in SKLEARN_COMBOS:
            print(f"[ {combo['label']} ]")
            a_hat, b_hat = sklearn_directions(
                X, T, Y, l1_ratio_alpha=combo['l1_alpha'], l1_ratio_T=combo['l1_T'])
            rows.append({
                'iter': it, 'method': combo['label'],
                'cos2_alpha': cosine_sq(a_hat, alpha_true),
                'cos2_beta':  cosine_sq(b_hat, beta_true),
                'ab_dp':      float(a_hat @ b_hat),
                'eig1_X': np.nan, 'eig2_X': np.nan,
                'eig1_YA': np.nan, 'eig2_YA': np.nan,
            })

        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

        # Quick per-iter summary
        df_now = pd.DataFrame(rows)
        print()
        for m in ['SR-NN'] + [c['label'] for c in SKLEARN_COMBOS]:
            r = df_now[df_now.method == m].iloc[-1]
            print(f"  {m:25s}  cos²(α)={r.cos2_alpha:.3f}  cos²(β)={r.cos2_beta:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nFinal results saved to {OUT_CSV}")

    summary = (df.groupby('method')[['cos2_alpha', 'cos2_beta', 'ab_dp']]
                 .median().round(4))
    print("\nMedian cos² across iterations:")
    print(summary.to_string())


if __name__ == '__main__':
    main()
