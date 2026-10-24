"""Data-generating processes for the score-ratio experiments.

Each generator reproduces the random-number stream of the script it was
extracted from, so a given seed gives the same data as the original scripts:

  linear  - Clivio et al. (2026) sparse linear Gaussian DGP
            (``score_ratio_rescue.py``: ``construct_beta``, ``gen_simulated``)
  rank1   - quadratic single-index DGP
            (``score_ratio_nonlinear_rank1_stable.py``)
  rank2   - sin + quadratic multi-index DGP
            (``score_ratio_nonlinear.py``; not used by the paper pipeline yet)

Oracle score functions are used only for post-hoc evaluation, never for
training or model selection.
"""

import numpy as np
from scipy.special import expit


# ── Direction construction ────────────────────────────────────────────────────

def construct_beta_sparse(alpha, ab_dp, rng):
    """Unit beta on the support of alpha with alpha·beta = ab_dp."""
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


def construct_beta_dense(alpha, ab_dp, rng):
    """Unit beta in R^p with alpha·beta = ab_dp."""
    p = len(alpha)
    raw = rng.standard_normal(p)
    raw -= (raw @ alpha) * alpha
    raw /= np.linalg.norm(raw)
    beta = ab_dp * alpha + np.sqrt(1.0 - ab_dp**2) * raw
    return beta / np.linalg.norm(beta)


def make_linear_directions(p=100, q=20, ab_dp=0.75, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.concatenate([rng.standard_normal(q) / q, np.zeros(p - q)])
    alpha /= np.linalg.norm(alpha)
    beta = construct_beta_sparse(alpha, ab_dp, rng)
    return alpha, beta


def make_rank1_directions(p=30, q=5, ab_dp=0.75, seed=0):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(p)
    alpha[:q] = rng.standard_normal(q)
    alpha /= np.linalg.norm(alpha)
    beta = construct_beta_dense(alpha, ab_dp, rng)
    return alpha.astype(np.float32), beta.astype(np.float32)


def make_rank2_directions(p=30, q=5, ab_dp=0.75, seed=0):
    """Two orthogonal pairs: (alpha1, alpha2) supported on [:q] and [q:2q]."""
    rng = np.random.default_rng(seed)
    a1 = np.zeros(p); a1[:q] = rng.standard_normal(q); a1 /= np.linalg.norm(a1)
    a2 = np.zeros(p); a2[q:2*q] = rng.standard_normal(q); a2 /= np.linalg.norm(a2)
    a2 -= (a2 @ a1) * a1; a2 /= np.linalg.norm(a2)
    b1 = construct_beta_sparse(a1, ab_dp, rng)
    b2 = construct_beta_sparse(a2, ab_dp, rng)
    b2 -= (b2 @ b1) * b1; b2 /= np.linalg.norm(b2)
    return tuple(v.astype(np.float32) for v in (a1, a2, b1, b2))


# ── Generators ────────────────────────────────────────────────────────────────

def gen_linear(n, alpha, beta, s_T=4.0, s_Y=5.0, seed=0):
    """A|V ~ Bern(sigmoid(s_T V'beta)),  Y = s_Y V'alpha + N(0,1)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(alpha)))
    e = expit(s_T * (X @ beta))
    T = rng.binomial(1, e)
    Y = s_Y * (X @ alpha) + rng.standard_normal(n)
    return X, T, Y, e


def gen_rank1(n, alpha, beta, s_T=1.5, noise=0.5, seed=0):
    """Y(0) = (alpha'V)^2 + eps,  logit e = s_T[(beta'V)^2 - 1]."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, len(alpha))).astype(np.float32)
    e = expit(s_T * ((V @ beta)**2 - 1.0))
    A = rng.binomial(1, e).astype(np.float32)
    Y = ((V @ alpha)**2 + noise * rng.standard_normal(n)).astype(np.float32)
    return V, A, Y, e


def gen_rank2(n, alpha1, alpha2, beta1, beta2, s_T=1.5, noise=0.5, seed=0):
    """Y(0) = sin(a1'V) + 0.5(a2'V)^2 + eps,  logit e = s_T[sin(b1'V) + (b2'V)^2 - 1]."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, len(alpha1))).astype(np.float32)
    e = expit(s_T * (np.sin(V @ beta1) + (V @ beta2)**2 - 1.0))
    A = rng.binomial(1, e).astype(np.float32)
    Y = (np.sin(V @ alpha1) + 0.5*(V @ alpha2)**2
         + noise * rng.standard_normal(n)).astype(np.float32)
    return V, A, Y, e


# ── Oracle scores (evaluation only, in standardised-V space) ─────────────────

def true_w_A_linear(V, A, beta, s_T):
    """w_A(v,a) = s_T (a - sigmoid(s_T v'beta)) beta."""
    e = expit(s_T * (V @ beta))
    return (s_T * (A - e))[:, None] * beta[None, :]


def true_w_YA_linear(V, A, Y, alpha, s_Y, tau=0.0):
    """w_{Y|A}(v,a,y) = s_Y (y - s_Y v'alpha - tau a) alpha."""
    mu = s_Y * (V @ alpha) + tau * A
    return (s_Y * (Y - mu))[:, None] * alpha[None, :]


def true_w_A_rank1(V, A, beta, s_T):
    """w_A(v,a) = (a - e(v)) 2 s_T (beta'v) beta."""
    z = V @ beta
    e = expit(s_T * (z**2 - 1.0))
    g = 2.0 * s_T * z[:, None] * beta[None, :]
    return (A - e)[:, None] * g


def true_w_YA_rank1(V, Y, alpha, noise):
    """w_{Y|A=0}(v,y) = (y - (alpha'v)^2) 2 (alpha'v) alpha / noise^2."""
    z = V @ alpha
    coeff = (Y - z**2) * 2.0 * z / (noise**2)
    return coeff[:, None] * alpha[None, :]


# ── Dispatcher used by the pipeline ──────────────────────────────────────────

def make_directions(dgp):
    """True (alpha, beta) for a DGP config dict (fixed across data seeds)."""
    family = dgp['family']
    if family == 'linear':
        return make_linear_directions(dgp['p'], dgp['q_alpha'], dgp['ab_dp'],
                                      dgp['direction_seed'])
    if family == 'rank1':
        return make_rank1_directions(dgp['p'], dgp['q_alpha'], dgp['ab_dp'],
                                     dgp['direction_seed'])
    raise ValueError(f"unknown DGP family for the pipeline: {family!r}")


def simulate(dgp, seed):
    """Draw one dataset.

    Returns a dict with raw covariates ``X``, treatment ``A`` (0/1), outcome
    ``Y``, true directions, and oracle score callables (evaluation only).
    """
    alpha, beta = make_directions(dgp)
    family = dgp['family']
    if family == 'linear':
        X, A, Y, e = gen_linear(dgp['n'], alpha, beta, dgp['s_T'], dgp['s_Y'], seed)
        s_T, s_Y = dgp['s_T'], dgp['s_Y']
        w_A = lambda V, A_: true_w_A_linear(V, A_, beta, s_T)
        w_YA = lambda V, A_, Y_: true_w_YA_linear(V, A_, Y_, alpha, s_Y)
    elif family == 'rank1':
        X, A, Y, e = gen_rank1(dgp['n'], alpha, beta, dgp['s_T'], dgp['noise'], seed)
        s_T, noise = dgp['s_T'], dgp['noise']
        w_A = lambda V, A_: true_w_A_rank1(V, A_, beta, s_T)
        w_YA = lambda V, A_, Y_: true_w_YA_rank1(V, Y_, alpha, noise)
    else:
        raise ValueError(f"unknown DGP family: {family!r}")
    return dict(X=X, A=A, Y=Y, e=e, alpha=alpha, beta=beta,
                oracle_w_A=w_A, oracle_w_YA=w_YA)
