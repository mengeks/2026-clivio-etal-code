# Copied verbatim from Causal-DR-Coding/score_ratio_reduction.py (2026-10-24).
# The SR-NN scripts previously imported it from the parent folder via sys.path.

"""
Score-ratio dimension reduction for causal inference (upstream module for CausalBGM-DR).

Implements Algorithm 1 from:
  Baptista, Brennan, Marzouk (2024). "Dimension reduction via score ratio matching."
  TMLR 04/2025.

Causal adaptation (from causalbgm_DR_writeup.tex):
  - Run score-ratio DR on (V, X)   → H_X ∈ R^{p×p} → eigenvectors U_X
  - Run score-ratio DR on (V, X,Y) → H_Y ∈ R^{p×p} → eigenvectors U_Y
  - Partition: U_0 = U_X ∩ U_Y,  U_1 = U_Y ⊖ U_0,  U_2 = U_X ⊖ U_0
  - Reduced covariates: R_0 = U_0^T V,  R_1 = U_1^T V,  R_2 = U_2^T V

Usage:
  from score_ratio_reduction import ScoreRatioDR
  dr = ScoreRatioDR(r_X=6, r_Y=6).fit(V, X, Y)
  R_0, R_1, R_2, R = dr.transform(V)
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Score-ratio network  w_Θ(v, obs) = W_v ψ_θ(W_v^T v, W_obs^T obs)
# ---------------------------------------------------------------------------

class ScoreRatioNet(tf.keras.Model):
    """Structure-exploiting score-ratio network (Baptista et al. §4).

    Parameterizes w_Θ(v, obs) = W_v ψ_θ(W_v^T v, W_obs^T obs)
    where:
      W_v   ∈ R^{v_dim × r'}   — low-rank parameter projection
      W_obs ∈ R^{obs_dim × s'} — low-rank observation projection
      ψ_θ   : R^{r'+s'} → R^{r'} — MLP

    Nuclear-norm regularization on W_v and W_obs promotes low rank.
    """
    def __init__(self, v_dim, obs_dim, r_prime=10, s_prime=5,
                 hidden_units=(64, 64), name='srnet'):
        super().__init__(name=name)
        self.v_dim   = v_dim
        self.obs_dim = obs_dim
        self.r_prime = r_prime
        self.s_prime = s_prime

        # Learnable projection matrices — orthogonal init via numpy SVD so the
        # initialization works on machines where TF's GPU sign op needs libdevice.
        def _ortho(rows, cols):
            a = np.random.standard_normal((max(rows, cols), min(rows, cols))).astype(np.float32)
            u, _, _ = np.linalg.svd(a, full_matrices=False)
            m = u if rows >= cols else u.T
            return m[:rows, :cols]
        self.W_v   = tf.Variable(_ortho(v_dim,   r_prime), name='W_v',   trainable=True)
        self.W_obs = tf.Variable(_ortho(obs_dim, s_prime), name='W_obs', trainable=True)

        # Inner MLP  ψ_θ: R^{r'+s'} → R^{r'}
        layers = []
        for h in hidden_units:
            layers.append(tf.keras.layers.Dense(h, activation='tanh'))
        layers.append(tf.keras.layers.Dense(r_prime))
        self.psi = tf.keras.Sequential(layers)

    def call(self, v, obs):
        """
        v:   (batch, v_dim)
        obs: (batch, obs_dim)
        returns w: (batch, v_dim)
        """
        v_proj   = v   @ self.W_v              # (batch, r')
        obs_proj = obs @ self.W_obs            # (batch, s')
        inp      = tf.concat([v_proj, obs_proj], axis=1)  # (batch, r'+s')
        psi_out  = self.psi(inp)               # (batch, r')
        return psi_out @ tf.transpose(self.W_v)  # (batch, v_dim)


# ---------------------------------------------------------------------------
# ISM training step with nested tapes (Baptista Eq. 8)
# ---------------------------------------------------------------------------

def _ism_step(net, optimizer, v, obs, lam1, lam2, n_hutch):
    """One gradient step minimizing the ISM + nuclear-norm objective.

    J(w_θ) = E[½||w||² + Tr(∇_v w) + ∇log ρ(v)^T w]
              + λ₁||W_v||_* + λ₂||W_obs||_*

    Reference density ρ = N(0,I), so ∇log ρ(v) = -v.

    Tr(∇_v w) estimated via Hutchinson: E_z[z^T ∇_v w z] = E_z[∂(z^T w)/∂v · z].
    Nested GradientTapes give the correct second-order gradient w.r.t. net params.
    """
    # Draw Hutchinson probes outside the outer tape (fixed across the step)
    z_probes = [tf.random.normal(tf.shape(v)) for _ in range(n_hutch)]

    with tf.GradientTape() as outer:
        # --- trace term via Hutchinson ---------------------------------
        trace_est = tf.constant(0.0)
        for z in z_probes:
            with tf.GradientTape() as inner:
                inner.watch(v)
                w_z = net(v, obs)
                sw  = tf.reduce_sum(z * w_z, axis=1)   # (batch,)  z^T w
            dw_dv = inner.gradient(sw, v)               # (batch, p) = ∂(z^T w)/∂v
            trace_est = trace_est + tf.reduce_mean(
                tf.reduce_sum(dw_dv * z, axis=1))       # E[z^T J z]
        trace_est = trace_est / float(n_hutch)

        # --- remaining terms ------------------------------------------
        w     = net(v, obs)
        half  = tf.reduce_mean(0.5 * tf.reduce_sum(w * w, axis=1))
        ref   = tf.reduce_mean(-tf.reduce_sum(v * w, axis=1))   # Gaussian ref

        # nuclear norms — force CPU to avoid cuSolverDN (broken on Aries for small matrices)
        with tf.device('/CPU:0'):
            nuc_v   = tf.reduce_sum(tf.linalg.svd(net.W_v,   compute_uv=False))
            nuc_obs = tf.reduce_sum(tf.linalg.svd(net.W_obs, compute_uv=False))

        loss = half + trace_est + ref + lam1 * nuc_v + lam2 * nuc_obs

    grads = outer.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss


# ---------------------------------------------------------------------------
# Fit one score-ratio network → H_CDR diagnostic matrix
# ---------------------------------------------------------------------------

def fit_score_ratio(V_std, obs, r_prime=10, s_prime=5, hidden_units=(64, 64),
                    n_epochs=300, batch_size=256, lr=1e-3,
                    lam1=5e-3, lam2=5e-3, n_hutch=2, verbose=True, label=''):
    """Fit a ScoreRatioNet and return the H_CDR matrix (Baptista Algorithm 1).

    H_CDR = (1/N) Σ_j w(v_j, obs_j) w(v_j, obs_j)^T  ∈ R^{p×p}

    Args:
        V_std:  (n, p) standardized covariates
        obs:    (n, m) observations (X or [X, Y])
    Returns:
        H: (p, p) diagnostic matrix (symmetric, PSD)
        net: trained ScoreRatioNet
    """
    n, v_dim  = V_std.shape
    obs_dim   = obs.shape[1]

    net       = ScoreRatioNet(v_dim, obs_dim, r_prime=r_prime, s_prime=s_prime,
                              hidden_units=hidden_units, name=f'srnet_{label}')
    optimizer = tf.keras.optimizers.Adam(lr)

    V_tf   = tf.constant(V_std, dtype=tf.float32)
    obs_tf = tf.constant(obs,   dtype=tf.float32)

    dataset = (tf.data.Dataset
               .from_tensor_slices((V_tf, obs_tf))
               .shuffle(n, seed=0)
               .batch(batch_size))

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for bv, bo in dataset:
            loss      = _ism_step(net, optimizer, bv, bo, lam1, lam2, n_hutch)
            epoch_loss += float(loss)
            n_batches  += 1
        if verbose and (epoch + 1) % max(1, n_epochs // 6) == 0:
            print(f"    [{label}] epoch {epoch+1:>4}/{n_epochs}  "
                  f"loss={epoch_loss/n_batches:.4f}")

    # Compute H_CDR = (1/n) Σ w w^T  in batches
    p = v_dim
    H = np.zeros((p, p), dtype=np.float64)
    for start in range(0, n, batch_size):
        end    = min(start + batch_size, n)
        w_b    = net(V_tf[start:end], obs_tf[start:end]).numpy()  # (b, p)
        H     += w_b.T @ w_b
    H /= n
    H  = 0.5 * (H + H.T)   # symmetrize numerically

    return H, net


# ---------------------------------------------------------------------------
# Eigendecomposition helpers
# ---------------------------------------------------------------------------

def compute_subspace(H, r):
    """Leading r eigenvectors of symmetric matrix H (descending eigenvalue order).

    Returns:
        U:    (p, r) orthonormal columns
        eigs: (p,) eigenvalues descending
    """
    eigs, vecs = np.linalg.eigh(H)           # ascending order
    idx  = np.argsort(eigs)[::-1]
    return vecs[:, idx[:r]], eigs[idx]


def subspace_intersection(U_X, U_Y, tol=0.90):
    """Partition U_X, U_Y into shared (U_0), outcome-only (U_1), treatment-only (U_2).

    Method: SVD of the cross-Gram matrix U_X^T U_Y.
    Singular values σ_k ≈ 1 ⟺ direction is in both subspaces.

    Args:
        U_X: (p, r_X) orthonormal columns
        U_Y: (p, r_Y) orthonormal columns
        tol: singular value threshold for calling a direction "shared"

    Returns:
        U_0: (p, r0)  confounding  — in span(U_X) ∩ span(U_Y)
        U_1: (p, r1)  outcome-only — in span(U_Y) ⊖ span(U_0)
        U_2: (p, r2)  treat.-only  — in span(U_X) ⊖ span(U_0)
        sigma: singular values of U_X^T U_Y (all of them)
    """
    M     = U_X.T @ U_Y           # (r_X, r_Y)
    P, sigma, Qt = np.linalg.svd(M, full_matrices=False)
    shared = sigma >= tol
    r0     = int(shared.sum())

    if r0 == 0:
        return (np.zeros((U_X.shape[0], 0)),
                U_Y, U_X, sigma)

    # U_0: average the two representations and re-orthogonalise
    U0_from_X = U_X @ P[:, :r0]          # (p, r0)
    U0_from_Y = U_Y @ Qt.T[:, :r0]       # (p, r0)
    U_0_raw   = 0.5 * (U0_from_X + U0_from_Y)
    U_0, _    = np.linalg.qr(U_0_raw)
    U_0       = U_0[:, :r0]

    def _orthogonal_complement(U_base, U_full):
        """Columns of U_full orthogonalised against U_base."""
        proj   = U_base @ (U_base.T @ U_full)  # (p, r_full)
        resid  = U_full - proj                  # (p, r_full)
        rk     = np.linalg.matrix_rank(resid, tol=1e-6)
        if rk == 0:
            return np.zeros((U_base.shape[0], 0))
        Q, _   = np.linalg.qr(resid)
        return Q[:, :rk]

    U_1 = _orthogonal_complement(U_0, U_Y)
    U_2 = _orthogonal_complement(U_0, U_X)

    return U_0, U_1, U_2, sigma


# ---------------------------------------------------------------------------
# Main DR class
# ---------------------------------------------------------------------------

class ScoreRatioDR:
    """Score-ratio dimension reduction upstream module for CausalBGM-DR.

    Learns:
        U_X  from (V, X)    — directions in V informative for treatment
        U_Y  from (V, X, Y) — directions in V informative for outcome|treatment
    Then partitions into U_0, U_1, U_2 and projects V.

    Usage:
        dr = ScoreRatioDR(r_X=6, r_Y=6)
        dr.fit(V, X, Y)
        R_0, R_1, R_2, R = dr.transform(V)
        # Feed R (n × r_total) as 'v' to CausalBGM with v_dim=dr.r_total
    """

    def __init__(
        self,
        r_X=6, r_Y=6,                        # subspace dims for H_X, H_Y
        r_prime=15, s_prime_X=3, s_prime_Y=5, # inner network ranks
        hidden_units=(64, 64),
        n_epochs=300, batch_size=256, lr=1e-3,
        lam1=5e-3, lam2=5e-3, n_hutch=2,
        intersection_tol=0.90,
        verbose=True,
    ):
        self.r_X              = r_X
        self.r_Y              = r_Y
        self.r_prime          = r_prime
        self.s_prime_X        = s_prime_X
        self.s_prime_Y        = s_prime_Y
        self.hidden_units     = hidden_units
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.lr               = lr
        self.lam1             = lam1
        self.lam2             = lam2
        self.n_hutch          = n_hutch
        self.intersection_tol = intersection_tol
        self.verbose          = verbose

        # Set after fit()
        self.U_0 = self.U_1 = self.U_2 = None
        self.H_X = self.H_Y = None
        self.sigma_shared = None
        self.V_mean = self.V_std_scale = None

    # ------------------------------------------------------------------
    def fit(self, V, X, Y):
        """Fit DR from training data.

        Args:
            V: (n, p)  covariates
            X: (n,) or (n,1)  binary treatment
            Y: (n,) or (n,1)  outcome
        Returns self.
        """
        V = np.asarray(V, dtype=np.float32)
        X = np.asarray(X, dtype=np.float32).reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1, 1)

        # Standardise V → approximate Gaussian reference ρ = N(0,I)
        self.V_mean      = V.mean(axis=0)
        self.V_std_scale = V.std(axis=0) + 1e-8
        V_std            = (V - self.V_mean) / self.V_std_scale

        # --- H_X: treatment-relevant -----------------------------------
        if self.verbose:
            print("=== Fitting H_X (treatment-relevant directions) ===")
        self.H_X, self.net_X = fit_score_ratio(
            V_std, X,
            r_prime=self.r_prime, s_prime=self.s_prime_X,
            hidden_units=self.hidden_units, n_epochs=self.n_epochs,
            batch_size=self.batch_size, lr=self.lr,
            lam1=self.lam1, lam2=self.lam2, n_hutch=self.n_hutch,
            verbose=self.verbose, label='H_X')

        U_X, eigs_X = compute_subspace(self.H_X, self.r_X)
        if self.verbose:
            print(f"  H_X top-{self.r_X} eigenvalues: {eigs_X[:self.r_X].round(5)}")

        # --- H_Y: outcome-relevant -------------------------------------
        XY = np.concatenate([X, Y], axis=1)   # (n, 2)
        if self.verbose:
            print("=== Fitting H_Y (outcome-relevant directions) ===")
        self.H_Y, self.net_Y = fit_score_ratio(
            V_std, XY,
            r_prime=self.r_prime, s_prime=self.s_prime_Y,
            hidden_units=self.hidden_units, n_epochs=self.n_epochs,
            batch_size=self.batch_size, lr=self.lr,
            lam1=self.lam1, lam2=self.lam2, n_hutch=self.n_hutch,
            verbose=self.verbose, label='H_Y')

        U_Y, eigs_Y = compute_subspace(self.H_Y, self.r_Y)
        if self.verbose:
            print(f"  H_Y top-{self.r_Y} eigenvalues: {eigs_Y[:self.r_Y].round(5)}")

        # --- Subspace partition ----------------------------------------
        self.U_0, self.U_1, self.U_2, self.sigma_shared = subspace_intersection(
            U_X, U_Y, tol=self.intersection_tol)

        if self.verbose:
            print(f"\n=== Subspace partition (tol={self.intersection_tol}) ===")
            print(f"  r0 (confounding/shared):  {self.r0}")
            print(f"  r1 (outcome-only):         {self.r1}")
            print(f"  r2 (treatment-only):       {self.r2}")
            print(f"  r_total:                   {self.r_total}")
            print(f"  Cross-Gram singular values: {self.sigma_shared.round(3)}")

        return self

    # ------------------------------------------------------------------
    def transform(self, V):
        """Project V → (R_0, R_1, R_2, R_concat).

        Returns:
            R_0: (n, r0)  — confounding summaries
            R_1: (n, r1)  — outcome-only summaries
            R_2: (n, r2)  — treatment-only summaries
            R:   (n, r_total) — concatenated [R_0, R_1, R_2]
        """
        V_std = (np.asarray(V, dtype=np.float32) - self.V_mean) / self.V_std_scale
        R_0 = V_std @ self.U_0 if self.r0 > 0 else np.zeros((len(V), 0), dtype=np.float32)
        R_1 = V_std @ self.U_1 if self.r1 > 0 else np.zeros((len(V), 0), dtype=np.float32)
        R_2 = V_std @ self.U_2 if self.r2 > 0 else np.zeros((len(V), 0), dtype=np.float32)
        R   = np.concatenate([R_0, R_1, R_2], axis=1).astype(np.float32)
        return R_0, R_1, R_2, R

    # ------------------------------------------------------------------
    @property
    def r0(self):
        return self.U_0.shape[1] if self.U_0 is not None else 0

    @property
    def r1(self):
        return self.U_1.shape[1] if self.U_1 is not None else 0

    @property
    def r2(self):
        return self.U_2.shape[1] if self.U_2 is not None else 0

    @property
    def r_total(self):
        return self.r0 + self.r1 + self.r2

    def summary(self):
        print(f"ScoreRatioDR: r0={self.r0}, r1={self.r1}, r2={self.r2} "
              f"(total={self.r_total}, original p={self.V_std_scale.shape[0]})")
        if self.sigma_shared is not None:
            print(f"  Cross-Gram σ: {self.sigma_shared.round(3)}")
