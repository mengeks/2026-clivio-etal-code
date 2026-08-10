###############################################################################
# Analytical score-ratio dimension reduction for the linear Gaussian DGP.
#
# Translation of score_ratio_reduction.py to R for the linear Gaussian setting.
# For nonlinear DGPs, use reticulate::source_python("score_ratio_reduction.py").
#
# In the linear Gaussian DGP (X ~ N(0,I), logistic propensity, linear outcome):
#
#   w(v_i, a_i)        = (a_i - ehat_i)  * beta_hat      [p-vector, ∝ beta]
#   w_{Y|A}(v_i,a_i,y_i) = (y_i - yhat_i) * alpha_hat    [p-vector, ∝ alpha]
#
# H_X   = (1/n) Σ w(v_i,a_i)   w(v_i,a_i)^T  → rank-1, top eigvec = beta_hat/‖beta_hat‖
# H_YA  = (1/n) Σ w_{Y|A,i} w_{Y|A,i}^T    → rank-1, top eigvec = alpha_hat/‖alpha_hat‖
#
# Dimension choice: r=1 for both H_X and H_YA.
# Justification: the linear DGP guarantees rank-1 structure (Clivio et al. Sec. 4).
###############################################################################

cosine_sq <- function(u, v) {
  # Squared cosine similarity (= squared correlation of direction) between unit vectors.
  # Works with unnormalized inputs.
  dp <- sum(u * v)
  (dp / (sqrt(sum(u^2)) * sqrt(sum(v^2))))^2
}

# ---------------------------------------------------------------------------
# Core: compute score-ratio diagnostic matrices and top eigenvectors
# ---------------------------------------------------------------------------

score_ratio_dr <- function(X, T, Y, estimand, Y_ALPHA, T_ALPHA,
                            prop_ests = NULL, out_ests = NULL) {
  # Fit models if not supplied
  if (is.null(prop_ests))
    prop_ests <- estimate_propensity(X, T, alpha = T_ALPHA)
  if (is.null(out_ests))
    out_ests  <- estimate_outcome(X, T, Y, estimand, alpha = Y_ALPHA)

  ehat      <- as.numeric(prop_ests$ehat)
  beta_hat  <- prop_ests$beta_hat     # raw glmnet coef (p-vector)
  yhat      <- as.numeric(out_ests$preds)
  alpha_hat <- out_ests$alpha_hat     # raw glmnet coef (p-vector)

  resid_T <- T - ehat        # (T_i - ehat_i)  for each observation
  resid_Y <- Y - yhat        # (Y_i - yhat_i)

  # H_X  = mean((T-ehat)^2) * (beta_hat beta_hat^T)
  #   → eigenvalue lambda_X = mean((T-ehat)^2) * ||beta_hat||^2
  #   → top eigenvector     = beta_hat / ||beta_hat||
  lambda_X  <- mean(resid_T^2) * sum(beta_hat^2)
  beta_norm <- sqrt(sum(beta_hat^2))

  # H_YA = mean((Y-yhat)^2) * (alpha_hat alpha_hat^T)
  #   → eigenvalue lambda_YA = mean((Y-yhat)^2) * ||alpha_hat||^2
  #   → top eigenvector      = alpha_hat / ||alpha_hat||
  lambda_YA  <- mean(resid_Y^2) * sum(alpha_hat^2)
  alpha_norm <- sqrt(sum(alpha_hat^2))

  beta_hat_normalized  <- if (beta_norm  > 1e-10) beta_hat  / beta_norm  else beta_hat
  alpha_hat_normalized <- if (alpha_norm > 1e-10) alpha_hat / alpha_norm else alpha_hat

  # H_Y has contributions from both propensity and outcome.
  # The cross-term w_X^T w_YA = (T-ehat)(Y-yhat) * beta_hat^T alpha_hat.
  # For ATT, outcome fit is on controls only, so cross-term is small but non-zero.
  ab_dp <- sum(beta_hat_normalized * alpha_hat_normalized)

  list(
    alpha_hat_normalized = alpha_hat_normalized,
    beta_hat_normalized  = beta_hat_normalized,
    lambda_X             = lambda_X,
    lambda_YA            = lambda_YA,
    # Eigenvalue ratio: fraction of variance in top eigenvector direction.
    # For a rank-1 matrix this is 1; values < 1 indicate contamination.
    lambda_ratio_X       = 1.0,   # always 1 in linear case (H_X is exactly rank-1)
    lambda_ratio_YA      = 1.0,
    ab_dp_estimated      = ab_dp  # inner product between estimated directions
  )
}


# ---------------------------------------------------------------------------
# Evaluate direction recovery for one dataset
# ---------------------------------------------------------------------------

eval_directions <- function(X, T, Y, estimand, Y_ALPHA, T_ALPHA,
                             alpha_true, beta_true) {
  prop_ests <- estimate_propensity(X, T, alpha = T_ALPHA)
  out_ests  <- estimate_outcome(X, T, Y, estimand, alpha = Y_ALPHA)

  sr <- score_ratio_dr(X, T, Y, estimand, Y_ALPHA, T_ALPHA,
                        prop_ests = prop_ests, out_ests = out_ests)

  alpha_hat_n <- sr$alpha_hat_normalized
  beta_hat_n  <- sr$beta_hat_normalized

  # Cosine² similarity to true directions
  cos2_alpha <- cosine_sq(alpha_hat_n, alpha_true)
  cos2_beta  <- cosine_sq(beta_hat_n,  beta_true)

  # Dot product of estimated directions (used in gamma computation)
  ab_dp <- sum(alpha_hat_n * beta_hat_n)

  # True dot product
  ab_dp_true <- sum(alpha_true * beta_true)

  list(
    cos2_alpha   = cos2_alpha,
    cos2_beta    = cos2_beta,
    ab_dp        = ab_dp,
    ab_dp_true   = ab_dp_true,
    lambda_X     = sr$lambda_X,
    lambda_YA    = sr$lambda_YA
  )
}
