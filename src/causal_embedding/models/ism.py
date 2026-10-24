"""Implicit score matching (ISM) training for bottleneck score-ratio networks.

J(w) = E[ 1/2 ||w||^2 + tr(grad_v w) - v^T w ]   (reference rho = N(0, I))

for w(v, obs) = W_v psi(W_v^T v, W_obs^T obs) (``ScoreRatioNet``). Extracted
from the joint-score code of ``score_ratio_linear_stable.py``:

- trace='exact_qr': with W_v = QR, the r' deterministic probes Q[:, k] give
  the trace exactly (w lies in span(W_v)).
- trace='hutchinson': K Gaussian probes per sample, as in the original
  ``score_ratio_reduction._ism_step`` (unbiased, positive variance).
- Stiefel projection of W_v after every step removes the
  (W_v, psi) -> (c W_v, psi / c) scale co-adaptation.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import numpy as np
import tensorflow as tf


def stiefel_project(net):
    """Orthonormalise the columns of net.W_v."""
    Q, _ = np.linalg.qr(net.W_v.numpy())
    net.W_v.assign(Q[:, :net.r_prime].astype(np.float32), read_value=False)


def _probe_trace(net, V_tf, obs_tf, probes):
    """mean_i sum_k z_ik^T (grad_v w) z_ik for a list of (n, p) probe tensors."""
    trace = tf.constant(0.0)
    for z in probes:
        with tf.GradientTape() as inner:
            inner.watch(V_tf)
            sw = tf.reduce_sum(z * net(V_tf, obs_tf), axis=1)
        dw_dv = inner.gradient(sw, V_tf)
        trace = trace + tf.reduce_mean(tf.reduce_sum(dw_dv * z, axis=1))
    return trace


def _probes(net, n, trace, n_hutch):
    """Probe tensors and the factor that turns their sum into the trace."""
    if trace == 'exact_qr':
        Q = np.linalg.qr(net.W_v.numpy())[0]        # constants wrt the graph
        return [tf.tile(tf.constant(Q[:, k].astype(np.float32))[None, :], [n, 1])
                for k in range(net.r_prime)], 1.0
    if trace == 'hutchinson':
        return [tf.random.normal((n, net.v_dim)) for _ in range(n_hutch)], 1.0 / n_hutch
    raise ValueError(f"unknown trace estimator {trace!r}")


def ism_step(net, optimizer, V_tf, obs_tf, *, trace='exact_qr', n_hutch=4,
             lam1=0.0, lam2=1e-2, lam_l2=1e-3, clip_norm=1.0):
    """One gradient step on ISM + lam1 ||W_v||_* + lam2 ||W_obs||_* + lam_l2 ||psi||^2."""
    probes, scale = _probes(net, V_tf.shape[0], trace, n_hutch)
    with tf.GradientTape() as outer:
        trace_est = _probe_trace(net, V_tf, obs_tf, probes) * scale
        w = net(V_tf, obs_tf)
        half = tf.reduce_mean(0.5 * tf.reduce_sum(w * w, axis=1))
        ref = tf.reduce_mean(-tf.reduce_sum(V_tf * w, axis=1))
        with tf.device('/CPU:0'):
            nuc_obs = tf.reduce_sum(tf.linalg.svd(net.W_obs, compute_uv=False))
        psi_vars = net.psi.trainable_variables
        l2_psi = (lam_l2 * tf.add_n([tf.reduce_sum(v**2) for v in psi_vars])
                  if psi_vars else tf.constant(0.0))
        loss = half + trace_est + ref + lam2 * nuc_obs + l2_psi
        if lam1:
            with tf.device('/CPU:0'):
                loss = loss + lam1 * tf.reduce_sum(
                    tf.linalg.svd(net.W_v, compute_uv=False))
    grads = outer.gradient(loss, net.trainable_variables)
    grads = [tf.clip_by_norm(g, clip_norm) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return float(loss)


def ism_loss(net, V_tf, obs_tf, *, trace='exact_qr', n_hutch=4):
    """Bare ISM loss (no regularisation), e.g. on a validation set."""
    probes, scale = _probes(net, V_tf.shape[0], trace, n_hutch)
    trace_val = float(_probe_trace(net, V_tf, obs_tf, probes)) * scale
    w = net(V_tf, obs_tf).numpy()
    half = float(np.mean(0.5 * np.sum(w * w, axis=1)))
    ref = float(np.mean(-np.sum(V_tf.numpy() * w, axis=1)))
    return half + trace_val + ref


def train_ism(net, V_np, obs_np, *, val_frac, max_epochs, lr, lam1, lam2, lam_l2,
              patience, clip_norm, stiefel=True, trace='exact_qr', n_hutch=4,
              seed=0, split_offset=888, min_delta=1e-5, verbose_every=0):
    """Full-batch ISM training with validation early stopping.

    The validation loss uses the same trace estimator as training.
    Returns (selected_epoch, best_val_ism, epochs_run).
    """
    n = len(V_np)
    perm = np.random.default_rng(seed + split_offset).permutation(n)
    n_val = max(10, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    V_tr = tf.constant(V_np[tr_idx], dtype=tf.float32)
    O_tr = tf.constant(obs_np[tr_idx], dtype=tf.float32)
    V_val = tf.constant(V_np[val_idx], dtype=tf.float32)
    O_val = tf.constant(obs_np[val_idx], dtype=tf.float32)

    _ = net(V_tr[:1], O_tr[:1])          # build psi before reading its variables
    opt = tf.keras.optimizers.Adam(lr)
    best_val = float('inf')
    best_weights = [v.numpy().copy() for v in net.trainable_variables]
    patience_count, selected_epoch, epoch = 0, 0, 0
    kw = dict(trace=trace, n_hutch=n_hutch)

    for epoch in range(1, max_epochs + 1):
        ism_step(net, opt, V_tr, O_tr, lam1=lam1, lam2=lam2, lam_l2=lam_l2,
                 clip_norm=clip_norm, **kw)
        if stiefel:
            stiefel_project(net)
        val = ism_loss(net, V_val, O_val, **kw)
        if verbose_every and epoch % verbose_every == 0:
            print(f"    [ISM] ep {epoch:>4}/{max_epochs}  val_ISM={val:.5f}")
        if val < best_val - min_delta:
            best_val, selected_epoch, patience_count = val, epoch, 0
            best_weights = [v.numpy().copy() for v in net.trainable_variables]
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    for v, w in zip(net.trainable_variables, best_weights):
        v.assign(w, read_value=False)
    if stiefel:
        stiefel_project(net)
    return selected_epoch, best_val, epoch


def score_vectors(net, V_np, obs_np, batch_size=64):
    """Forward-pass w(V_i, obs_i), shape (n, p)."""
    n = len(V_np)
    V_tf = tf.constant(V_np, dtype=tf.float32)
    O_tf = tf.constant(obs_np, dtype=tf.float32)
    return np.concatenate([
        net(V_tf[s:min(s + batch_size, n)], O_tf[s:min(s + batch_size, n)]).numpy()
        for s in range(0, n, batch_size)], axis=0)


def trace_estimator_spread(net, V_np, obs_np, n_hutch_list=(4, 8), n_rep=100):
    """Exact trace vs repeated Hutchinson estimates at fixed network weights.

    Returns {'trace_exact': ..., 'trace_hutch{K}_sd': ...} where sd is the
    standard deviation of the K-probe estimate over n_rep probe draws.
    """
    V_tf = tf.constant(V_np, dtype=tf.float32)
    O_tf = tf.constant(obs_np, dtype=tf.float32)
    n = V_tf.shape[0]
    probes, _ = _probes(net, n, 'exact_qr', 0)
    out = {'trace_exact': float(_probe_trace(net, V_tf, O_tf, probes))}
    for K in n_hutch_list:
        est = [float(_probe_trace(net, V_tf, O_tf, _probes(net, n, 'hutchinson', K)[0])) / K
               for _ in range(n_rep)]
        out[f'trace_hutch{K}_mean'] = float(np.mean(est))
        out[f'trace_hutch{K}_sd'] = float(np.std(est, ddof=1))
    return out
