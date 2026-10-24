"""Direction-recovery and score-error metrics."""

import numpy as np


def cosine_sq(u, v):
    """Sign-invariant squared cosine between two vectors."""
    u = np.ravel(u).astype(np.float64)
    v = np.ravel(v).astype(np.float64)
    return float((u @ v)**2 / ((u @ u) * (v @ v) + 1e-30))


def score_mse(w_hat, w_true):
    """Relative score error  sum||w_hat - w||^2 / sum||w||^2."""
    return float(np.sum((w_hat - w_true)**2) / (np.sum(w_true**2) + 1e-10))


def top_eigvecs(H, r):
    """Leading r eigenvectors (descending) and all eigenvalues of symmetric H."""
    eigs, vecs = np.linalg.eigh(H)
    idx = np.argsort(eigs)[::-1]
    return vecs[:, idx[:r]], eigs[idx]


def second_moment(W):
    """Symmetrised (1/n) W^T W for an (n, p) matrix of score vectors."""
    H = W.T @ W / len(W)
    return 0.5 * (H + H.T)
