library(glmnet)
library(tidyverse)
library(R.utils)

# Base-R replacement for rstiefel::NullC
# Returns orthonormal basis for the orthogonal complement of col(M)
NullC <- function(M) {
  qr_decomp <- qr(M)
  n <- nrow(M)
  r <- qr_decomp$rank
  if (r >= n) stop("No null space: matrix has full row rank.")
  Q <- qr.Q(qr_decomp, complete = TRUE)
  Q[, (r + 1):n, drop = FALSE]
}

get_attr_default <- function(thelist, attrname, default) {
  if (!is.null(thelist[[attrname]])) thelist[[attrname]] else default
}

logistic <- function(x) {
  exp(x) / (1 + exp(x))
}

construct_beta <- function(alpha, ab_dp) {
  p <- length(alpha)
  alpha_norm <- sqrt(sum(alpha^2))
  if (abs(alpha_norm - 1) > 1e-12)
    stop("alpha must have ||alpha|| ≈ 1.  Found: ", alpha_norm)
  if (abs(ab_dp) > 1)
    stop("Infeasible: ab_dp must lie in [-1, 1]. Got ab_dp = ", ab_dp)

  S <- which(abs(alpha) > 0)
  s <- length(S)
  if (s == 0) stop("alpha is the zero vector; cannot construct a unit beta.")

  if (s == 1) {
    if (abs(ab_dp - 1) < 1e-12) return(alpha)
    if (abs(ab_dp + 1) < 1e-12) return(-alpha)
    stop("Infeasible: alpha is 1-sparse, so the dot product with beta can only be ±1.")
  }

  alpha_s <- alpha[S]
  a     <- ab_dp
  b_val <- sqrt(1 - a^2)

  v <- rnorm(s)
  proj_scale <- sum(v * alpha_s) / sum(alpha_s^2)
  v <- v - proj_scale * alpha_s

  v_norm <- sqrt(sum(v^2))
  tries <- 0
  while (v_norm < 1e-15 && tries < 10) {
    v <- rnorm(s)
    proj_scale <- sum(v * alpha_s) / sum(alpha_s^2)
    v <- v - proj_scale * alpha_s
    v_norm <- sqrt(sum(v^2))
    tries <- tries + 1
  }
  if (v_norm < 1e-15) stop("Failed to find an orthonormal direction in alpha's support.")
  v <- v / v_norm

  beta_s <- a * alpha_s + b_val * v
  beta <- numeric(p)
  beta[S] <- beta_s

  beta_norm <- sqrt(sum(beta^2))
  if (abs(beta_norm - 1) > 1e-7) beta <- beta / beta_norm

  dp <- sum(alpha * beta)
  if (abs(dp - ab_dp) > 1e-6)
    stop(sprintf("Constructed dot product %.6g differs from ab_dp=%.6g", dp, ab_dp))

  beta
}

compute_gammas <- function(ab_dot_prod, mvals, mvecs, w2, times) {
  p <- dim(mvecs)[1]
  null_vecs <- NullC(mvecs)[, sample(p - ncol(mvecs),
                                     size = min(p - ncol(mvecs), times),
                                     replace = FALSE)]
  w1 <- sqrt(as.numeric((ab_dot_prod - mvals[2] * w2^2) / mvals[1]))
  stopifnot(1 - w1^2 - w2^2 > -1e-6)
  w1 * mvecs[, 1] + w2 * mvecs[, 2] + sqrt(max(1 - w1^2 - w2^2, 0)) * null_vecs
}

gen_linearY_logisticT <- function(n, p, tau, alpha, beta, mscale, escale, sigma2_y,
                                   y_intercept = 0) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  m  <- mscale * X %*% alpha
  xb <- X %*% beta
  e  <- logistic(escale * xb)
  T  <- rbinom(n, 1, e)
  Y  <- y_intercept + m + tau * T + rnorm(n, 0, sigma2_y)
  list(X = X, T = T, Y = Y, m = m, xb = xb, e = e)
}

gen_dataset <- function(dataset, setting, iter) {
  if (dataset == 'simulated') {
    tau <- 0
    n   <- 500
    p   <- 1000
    AB_DP <- 0.75

    set.seed(0)
    qa    <- 20
    alpha <- c(rnorm(qa) / qa, rep(0, p - qa))
    alpha <- alpha / sqrt(sum(alpha^2))
    beta  <- construct_beta(alpha, ab_dp = AB_DP)

    escale   <- c(1.0, 1.0, 4.0, 4.0)[setting]
    mscale   <- c(2.0, 5.0, 2.0, 5.0)[setting]
    sigma2_y <- 1
    set.seed(iter)
    simdat <- gen_linearY_logisticT(n, p, tau, alpha, beta, mscale, escale, sigma2_y)

    list(X = simdat$X, T = simdat$T, Y = simdat$Y, true_ate = tau)
  } else {
    stop("oracle experiment only supports dataset='simulated'")
  }
}

ipw_est <- function(e, T, Y, estimand, hajek = FALSE, epsilon = .Machine$double.eps) {
  if (estimand == "ATT") {
    one_minus_e <- 1 - e
    one_minus_e[one_minus_e < epsilon] <- epsilon
    wts <- (T - (1 - T) * e / one_minus_e) / sum(T)
  } else if (estimand == "ATC") {
    wts <- (T * (1 - e) / e - (1 - T)) / sum(1 - T)
  } else {
    wts <- (T / e) - ((1 - T) / (1 - e)) / (sum(T) + sum(1 - T))
  }

  if (hajek) {
    sum(wts[T == 1] * Y[T == 1]) / sum(wts[T == 1]) -
      sum(wts[T == 0] * Y[T == 0]) / sum(wts[T == 0])
  } else {
    sum(wts * Y)
  }
}

estimate_outcome <- function(X, T, Y, estimand, alpha = 0,
                              pred_policy = 'lambda.min',
                              coef_policy = 'lambda.min',
                              lambda = NULL, cv = TRUE) {
  if (estimand == "ATT") {
    Xfit  <- X[T == 0, ]
    Yfit  <- Y[T == 0]
  } else {
    stop("Only ATT supported")
  }
  balance_target <- colMeans(X[T == 1, ])

  if (cv) {
    cvglm <- glmnet::cv.glmnet(Xfit, Yfit, alpha = alpha, lambda = lambda)
    lambdas <- list(
      'lambda.min'   = cvglm$lambda.min,
      'lambda.1se'   = cvglm$lambda.1se,
      'undersmooth'  = cvglm$lambda[max(which(cvglm$cvlo < min(cvglm$cvm)))]
    )
    pred_lam <- lambdas[[pred_policy]]
    coef_lam <- lambdas[[coef_policy]]
    mu_pred        <- predict(cvglm, newx = matrix(balance_target, 1, length(balance_target)), s = pred_lam)
    fitted_values  <- predict(cvglm, newx = X, s = pred_lam)
    intercept      <- coef(cvglm, s = coef_lam)[1]
    alpha_hat      <- coef(cvglm, s = coef_lam)[-1]
  } else {
    glm_fit        <- glmnet::glmnet(Xfit, Yfit, lambda = lambda, alpha = alpha)
    mu_pred        <- predict(glm_fit, newx = matrix(balance_target, 1, length(balance_target)))
    fitted_values  <- predict(glm_fit, newx = X)
    intercept      <- coef(glm_fit)[1]
    alpha_hat      <- coef(glm_fit)[-1]
  }

  alpha_hat_normalized <- alpha_hat / sqrt(sum(alpha_hat^2))
  tau_hat <- mean(Y[T == 1]) - mu_pred

  list(
    alpha_hat            = alpha_hat,
    alpha_hat_normalized = alpha_hat_normalized,
    tau_hat              = tau_hat,
    intercept_alpha_hat  = intercept,
    mhat0                = mu_pred,
    mhat1                = mean(Y[T == 1]),
    preds                = fitted_values
  )
}

estimate_propensity <- function(X, T, cv = TRUE, T_lambda_min = 0, alpha = 0) {
  p <- ncol(X)
  if (cv) {
    cvglm <- glmnet::cv.glmnet(cbind(X), T, family = "binomial",
                               alpha = alpha, penalty.factor = rep(1, p), intercept = TRUE)
    T_lambda_min <- cvglm$lambda.min
  }

  propensity_fit <- glmnet::glmnet(cbind(X), T, family = "binomial",
                                   alpha = alpha, penalty.factor = rep(1, p),
                                   intercept = TRUE, lambda = T_lambda_min)
  beta_hat            <- coef(propensity_fit)[-1]
  intercept_beta_hat  <- coef(propensity_fit)[1]
  escale_hat          <- sqrt(sum(beta_hat^2))
  beta_hat_normalized <- if (escale_hat > 0) beta_hat / escale_hat else beta_hat
  ehat                <- predict(propensity_fit, type = "response", newx = X)

  list(
    beta_hat            = beta_hat,
    beta_hat_normalized = beta_hat_normalized,
    intercept_beta_hat  = intercept_beta_hat,
    escale_hat          = escale_hat,
    ehat                = ehat,
    lambda              = T_lambda_min
  )
}
