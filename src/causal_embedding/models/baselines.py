"""Classical direction estimators: SAVE, SIR and LASSO / logistic-LASSO.

They serve as baselines and as data-driven warm starts for the score-ratio
networks. Extracted from ``score_ratio_rescue.py`` (``sklearn_directions``)
and ``score_ratio_nonlinear_rank1_stable.py`` (``save_sdr``, ``sir``,
``lasso_directions``).
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV, LogisticRegressionCV


def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _slice_masks(response, n_slices):
    n = len(response)
    if len(np.unique(response)) == 2:
        return [response == lab for lab in np.unique(response)]
    order = np.argsort(response)
    return [np.isin(np.arange(n), sl) for sl in np.array_split(order, n_slices)]


def save_directions(X, response, n_slices=3, n_dirs=1):
    """SAVE with Ledoit-Wolf shrinkage of the slice covariances.

    A binary response uses one slice per class; a continuous response uses
    ``n_slices`` equal-frequency slices.
    """
    Xs = standardize(X); n, p = Xs.shape; Ip = np.eye(p)
    M = np.zeros((p, p))
    for mask in _slice_masks(response, n_slices):
        n_h = mask.sum()
        Sh = LedoitWolf().fit(Xs[mask]).covariance_ if n_h > 1 else Ip
        diff = Ip - Sh
        M += n_h / n * (diff @ diff)
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1][:, :n_dirs]


def sir_directions(X, response, n_slices=3, n_dirs=1):
    """Sliced inverse regression (fails for symmetric DGPs: E[X|Y] = 0)."""
    Xs = standardize(X); n = len(response)
    M = sum(mask.sum() / n * np.outer(Xs[mask].mean(0), Xs[mask].mean(0))
            for mask in _slice_masks(response, n_slices))
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1][:, :n_dirs]


def _unit(v, fallback_seed):
    nrm = np.linalg.norm(v)
    if nrm < 1e-10 and fallback_seed is not None:
        v = np.random.default_rng(fallback_seed).standard_normal(len(v))
        nrm = np.linalg.norm(v)
    return v / (nrm + 1e-10)


def lasso_directions(X, A, Y, lasso_max_iter=5000, logistic_max_iter=1000,
                     random_state=0, zero_fallback=False):
    """alpha_hat: LassoCV of Y on X among controls; beta_hat: L1-logistic of A on X.

    ``zero_fallback`` replaces an all-zero coefficient vector by a fixed random
    direction (seeds 42/43), as the nonlinear script did.
    """
    Xs = standardize(X)
    ctrl = A == 0
    lasso = LassoCV(cv=5, max_iter=lasso_max_iter, random_state=random_state)
    a_hat = lasso.fit(Xs[ctrl], Y[ctrl]).coef_.copy()
    logit = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5,
                                 max_iter=logistic_max_iter,
                                 random_state=random_state)
    b_hat = logit.fit(Xs, A.astype(int)).coef_.ravel().copy()
    a_hat = _unit(a_hat, 42 if zero_fallback else None)
    b_hat = _unit(b_hat, 43 if zero_fallback else None)
    return a_hat, b_hat


def baseline_directions(X, A, Y, cfg):
    """All baseline estimates as {method: (alpha_hat, beta_hat)}.

    Treatment directions use (X, A) on all units; outcome directions use the
    controls only.
    """
    ctrl = A == 0
    out = {}
    for name, fn in (('save', save_directions), ('sir', sir_directions)):
        a = fn(X[ctrl], Y[ctrl], n_slices=cfg['n_slices_y']).ravel()
        b = fn(X, A, n_slices=cfg['n_slices_a']).ravel()
        out[name] = (_unit(a, None), _unit(b, None))
    out['lasso'] = lasso_directions(X, A, Y, **cfg['lasso'])
    return out
