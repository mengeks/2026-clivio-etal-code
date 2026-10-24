
library(RColorBrewer)
library(colorspace)
library(patchwork)
library(sl3)
library(data.table)
library(glmnet)

library(aciccomp2016)
library(caret)
library(npci)        

library(torch)
library(torchvision)

construct_beta <- function(alpha, ab_dp) {
  p <- length(alpha)
  
  alpha_norm <- sqrt(sum(alpha^2))
  if (abs(alpha_norm - 1) > 1e-12) {
    stop("alpha must have ||alpha|| ≈ 1.  Found: ", alpha_norm)
  }
  
  if (abs(ab_dp) > 1) {
    stop("Infeasible: ab_dp must lie in [-1, 1]. Got ab_dp = ", ab_dp)
  }
  
  S <- which(abs(alpha) > 0)     
  s <- length(S)              
  
  if (s == 0) {
    stop("alpha is the zero vector; cannot construct a unit beta.")
  }
  
  if (s == 1) {
    if (abs(ab_dp - 1) < 1e-12) {
      beta <- alpha
    } else if (abs(ab_dp + 1) < 1e-12) {
      beta <- -alpha
    } else {
      stop("Infeasible: alpha is 1-sparse, so the dot product with beta can only be ±1.")
    }
    return(beta)
  }
  
  alpha_s <- alpha[S]
  
  a <- ab_dp
  b_sq <- 1 - a^2
  b_val <- sqrt(b_sq)
  
  v <- rnorm(s)
  proj_scale <- sum(v * alpha_s) / sum(alpha_s^2)
  v <- v - proj_scale * alpha_s
  
  v_norm <- sqrt(sum(v^2))
  tries <- 0
  while (v_norm < 1e-15 && tries < 10) {
    print("WARNING : low v_norm, retrying")
    v <- rnorm(s)
    proj_scale <- sum(v * alpha_s) / sum(alpha_s^2)
    v <- v - proj_scale * alpha_s
    v_norm <- sqrt(sum(v^2))
    tries <- tries + 1
  }
  if (v_norm < 1e-15) {
    stop("Failed to find an orthonormal direction in alpha's support (unexpected).")
  }
  v <- v / v_norm
  
  beta_s <- a * alpha_s + b_val * v
  
  beta <- numeric(p)
  beta[S] <- beta_s
  
  beta_norm <- sqrt(sum(beta^2))
  if (abs(beta_norm - 1) > 1e-7) {
    beta <- beta / beta_norm
  }
  
  dp <- sum(alpha * beta)
  if (abs(dp - ab_dp) > 1e-6) {
    stop("ERROR : Constructed dot product = %.6g differs from ab_dp=%.6g by >1e-6", dp, ab_dp)
  }
  
  return(beta)
}



compute_gammas <- function(ab_dot_prod, mvals, mvecs, w2, times){
  p <- dim(mvecs)[1]

  null_vecs <- NullC(mvecs)[, sample(p-ncol(mvecs), size=min(p-ncol(mvecs), times), replace=FALSE)]
  w1 <- sqrt(as.numeric((ab_dot_prod - mvals[2] * w2^2)/mvals[1]))

  stopifnot(1 - w1^2 - w2^2 > -1e-6)

  gammas <- w1 * mvecs[, 1] + w2 * mvecs[, 2] + sqrt(max(1 - w1^2 - w2^2, 0)) * null_vecs
  gammas
}




load_mnist_torch <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) stop("install.packages('torch')")
  if (!requireNamespace("torchvision", quietly = TRUE)) stop("install.packages('torchvision')")

  ds_train <- torchvision::mnist_dataset(train = TRUE, download = TRUE)
  ds_test  <- torchvision::mnist_dataset(train = FALSE, download = TRUE)

  to_array <- function(ds) {
    n <- ds$.length()
    x <- array(0, dim = c(n, 28, 28))
    y <- integer(n)
    for (i in seq_len(n)) {
      item <- ds$.getitem(i)         
      x[i, , ] <- as.array(item[[1]])  
      y[i]     <- as.integer(as.array(item[[2]])) 
    }
    list(x = x, y = y)
  }

  tr <- to_array(ds_train)
  te <- to_array(ds_test)

  list(
    x = abind::abind(tr$x, te$x, along = 1), 
    y = c(tr$y, te$y)
  )
}

linear_normalization <- function(x, new_min, new_max) {
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || diff(rng) == 0) {
    return(rep((new_min + new_max)/2, length(x)))
  }
  (x - rng[1]) * (new_max - new_min) / (rng[2] - rng[1]) + new_min
}

alpha_fn <- function(pi, lambda_) {
  (pi * lambda_)^(-1) + 1.0 - lambda_^(-1)
}

beta_fn <- function(pi, lambda_) {
  lambda_ * (pi)^(-1) + 1.0 - lambda_
}

complete_propensity <- function(x, u, gamma, beta = 0.75) {
  x <- as.numeric(x); u <- as.numeric(u)
  stopifnot(length(x) == length(u))
  logit   <- beta * x + 0.5
  nominal <- 1 / (1 + exp(-logit)) 
  a <- alpha_fn(nominal, gamma)
  b <- beta_fn(nominal, gamma)
  as.numeric(u / a + (1 - u) / b)
}

f_mu <- function(x, t, u, theta = 4.0) {
  x <- as.numeric(x)
  N <- length(x)
  if (length(t) == 1L) t <- rep(t, N)
  u <- as.numeric(u); if (length(u) == 1L) u <- rep(u, N)
  (2 * t - 1) * x +
    (2.0 * t - 1) -
    2 * sin((4 * t - 2) * x) -
    (theta * u - 2) * (1 + 0.5 * x)
}

policy_risk <- function(pi, y1, y0) {
  mean(pi * y1 + (1 - pi) * y0)
}

lambda_top_func <- function(mu, k, y, alpha) {
  y <- as.numeric(y)
  m <- length(y)
  if (k < 0 || k > m) stop("k must be in [0, m] (0-based).")
  r <- if (k == m) 0 else sum(y[(k + 1):m] - mu)
  mu + r / (m * (alpha + 1) - k)
}

lambda_bottom_func <- function(mu, k, y, alpha) {
  y <- as.numeric(y)
  m <- length(y)
  if (k < 0 || k > m) stop("k must be in [0, m] (0-based).")
  r <- if (k == 0) 0 else sum(y[1:k] - mu)
  mu + r / (m * alpha + k)
}

fit_phi_model_from_data <- function(X_img, y_lab, edges) {
  N <- dim(X_img)[1]; H <- dim(X_img)[2]; W <- dim(X_img)[3]
  stopifnot(H == 28, W == 28)

  X <- matrix(as.numeric(X_img) / 255, nrow = N, ncol = H * W)
  X <- sweep(X, 2, 0.1307, `-`)
  X <- X / 0.3081

  per_img_means <- rowMeans(X)
  digits <- sort(unique(y_lab)) 
  if (length(edges) < length(digits) + 1)
    stop("`edges` must have at least 11 values for 10 digits.")

  model <- list()
  for (i in seq_along(digits)) {
    d <- digits[i]
    idx <- which(y_lab == d)
    means_d <- per_img_means[idx]
    mu_d <- mean(means_d)
    sigma_d <- stats::sd(means_d)
    lo <- edges[i]
    hi <- edges[i + 1]
    model[[as.character(d)]] <- list(mu = mu_d, sigma = sigma_d, lo = lo, hi = hi)
  }
  model
}

HCMNIST <- function(
  root = NULL,
  gamma_star,
  mode = "mu",          
  p_u = "bernoulli",  
  theta = 4.0,
  beta = 0.75,
  sigma_y = 1.0,
  domain = 2.0,
  seed = 1331,
  download = TRUE
) {
  set.seed(seed)

  mn <- load_mnist_torch()
  X_img  <- mn$x
  targets <- mn$y
  N <- dim(X_img)[1]
  W <- 28
  H <- 28
  data_flat <- matrix(as.numeric(X_img), nrow = N, ncol = W*H)

  dim_input     <- c(1L, 28L, 28L)
  dim_treatment <- 1L
  dim_output    <- 1L

  step  <- (2 * domain) / 10
  edges <- seq(from = -domain, to = domain + 1e-8, by = step)
  phi_model <- fit_phi_model_from_data(X_img, targets, edges)

  u <- switch(
    p_u,
    "bernoulli" = rbinom(N, size = 1, prob = 0.5),
    "uniform"   = runif(N),
    "beta_bi"   = rbeta(N, shape1 = 0.5, shape2 = 0.5),
    "beta_uni"  = rbeta(N, shape1 = 2,   shape2 = 5),
    stop(sprintf("%s is not a supported distribution", p_u))
  )

  x_norm <- sweep(matrix(as.numeric(data_flat) / 255, nrow = N, ncol = H * W),
                  2, 0.1307, `-`)
  x_norm <- x_norm / 0.3081

  means <- rowMeans(x_norm)

  z_vec <- numeric(N)
  for (d in names(phi_model)) {
    idx <- which(targets == as.integer(d))
    if (length(idx) == 0) next
    mu_d <- phi_model[[d]]$mu
    sg_d <- phi_model[[d]]$sigma
    lo_d <- phi_model[[d]]$lo
    hi_d <- phi_model[[d]]$hi
    z_d <- if (!is.finite(sg_d) || sg_d <= 0) rep(0, length(idx)) else (means[idx] - mu_d) / sg_d
    z_d <- pmax(-1.4, pmin(1.4, z_d))
    z_vec[idx] <- linear_normalization(z_d, lo_d, hi_d)
  }
  phi <- matrix(z_vec, ncol = 1)


  pi <- complete_propensity(x = as.numeric(phi), u = u, gamma = gamma_star, beta = beta)
  if (any(!is.finite(pi)) || any(pi < 0) || any(pi > 1)) {
    stop("complete_propensity must return finite probabilities in [0,1].")
  }
  t_vec <- rbinom(N, size = 1, prob = pmin(pmax(pi, 0), 1))
  eps   <- rnorm(N, mean = 0, sd = sigma_y)

  mu0 <- f_mu(x = as.numeric(phi), t = 0.0, u = u, theta = theta)
  mu1 <- f_mu(x = as.numeric(phi), t = 1.0, u = u, theta = theta)

  y0 <- mu0 + eps
  y1 <- mu1 + eps
  y  <- t_vec * y1 + (1 - t_vec) * y0
  tau <- mu1 - mu0

  y_mean <- c(0.0)
  y_std  <- c(1.0)

  x_prop <- x_norm

  list(
    data = data_flat,                  
    targets = targets,                 
    mode = mode,
    dim_input = dim_input,
    dim_treatment = dim_treatment,
    dim_output = dim_output,
    phi_model = phi_model,             
    u = u,                             
    pi = pi,                           
    t = as.numeric(t_vec),             
    mu0 = mu0,                         
    mu1 = mu1,                         
    y0 = y0,                           
    y1 = y1,                           
    y  = y,                            
    tau = tau,                         
    y_mean = y_mean,
    y_std = y_std,
    x = x_prop,                        
    phi = phi                          
  )
}

get_item <- function(ds, index) {
  i <- as.integer(index)
  x_i <- ds$x[i, , drop = TRUE]
  t_i <- ds$t[i]
  if (ds$mode == "pi") {
    list(x = x_i, t = t_i)
  } else if (ds$mode == "mu") {
    xt <- cbind(matrix(x_i, nrow = 1), matrix(t_i, nrow = 1))
    y_i <- ds$y[i]
    list(X = xt, y = y_i)
  } else {
    stop(sprintf("%s not supported. Choose 'pi' or 'mu'.", ds$mode))
  }
}



gen_dataset <- function(dataset, setting, iter){
  if (dataset=='simulated'){
    tau <- 0
    n <- 500
    p <- 1000
    AB_DP <- 0.75

    set.seed(0)
    qa <- 20
    alpha <- c(rnorm(qa) / qa, rep(0, p - qa))
    alpha <- alpha / sqrt(sum(alpha^2))
    beta <- construct_beta(alpha, ab_dp=AB_DP)

    escale <- c(1.0, 1.0, 4.0, 4.0)[setting]
    mscale <- c(2.0, 5.0, 2.0, 5.0)[setting]
    sigma2_y <- 1
    set.seed(iter)
    simdat <- gen_linearY_logisticT(n, p, tau, alpha, beta, mscale, escale, sigma2_y)

    X <- simdat$X
    T <- simdat$T
    Y <- simdat$Y
    true_ate <- tau

    list(X=X, T=T, Y=Y,true_ate=true_ate)
  }
  else if (dataset=='acic2016'){
    simdat <- dgp_2016(input_2016, setting, iter)

    X <- model.matrix(~ . - 1, data = input_2016)
    T <- simdat$z
    Y <- simdat$y
    true_ate <- mean((simdat$mu.1 - simdat$mu.0)[T==1])

    list(X=X, T=T, Y=Y,true_ate=true_ate)
  }

  else if (dataset=='hcmnist'){
    simdat <-  HCMNIST(gamma_star=1, seed=iter)
    T <- simdat$t 
    X <- simdat$x
    Y <- simdat$y
    true_ate <- mean((simdat$mu1 - simdat$mu0)[T==1])

    list(X=X, T=T, Y=Y,true_ate=true_ate)
  }

  else if (dataset == 'ihdp'){

    setting_letter = c("A", "B", "C")[setting %% 3 + 1]
    overlap = c(TRUE, FALSE)[setting %% 2 + 1]

    

    loadDataInCurrentEnvironment(covariates = "all") 

    generateDataForIterInCurrentEnvironment(
              iter       = iter,
              x          = x,
              z          = z,
              w          = 0.5,
              overlap    = overlap,
              covariates = "full",
              setting    = setting_letter,    
              p.score    = "none")

      X   <- do.call(cbind, x)                        
      T   <- z                               
      Y   <- y                           
      mu0 <- mu.0                        
      mu1 <- mu.1


      n <- nrow(X)
      p <- ncol(X)


      true_ate <- mean(mu1[T == 1] - mu0[T == 1])
                  

      list(X=X, T=T, Y=Y,true_ate=true_ate)
      
  }
  else{
    stop("Wrong dataset type!")
  }
}

gen_linearY_logisticT <- function(n, p, tau, alpha, beta, mscale, escale, sigma2_y,
                                  y_intercept=0){

  X <- matrix(rnorm(n * p), nrow=n, ncol=p)

  m <- mscale * X %*% alpha
  xb <- X %*% beta

  e <- logistic(escale * xb)

  T <- rbinom(n, 1, e)
  Y <- y_intercept + m +  tau * T + rnorm(n, 0, sigma2_y)

  list(X=X, T=T, Y=Y, m=m, xb=xb, e=e)
}


estimate_outcome <- function(X, T, Y, estimand, alpha=0,
                             scale_X = FALSE,
                             pred_policy = 'lambda.min',
                             coef_policy = 'lambda.min',
                             lambda=NULL, cv=TRUE) {
  if (scale_X) {
    scl = apply(X, 2, sd, na.rm = TRUE)
    is.binary = apply(X, 2, function(xx) sum(xx == 0) + sum(xx == 1) == length(xx))
    scl[is.binary] = 1
    Xscl = scale(X, center = FALSE, scale = scl)
  } else {
    Xscl = X
  }

  ## estimand <- "ATE"
  if (estimand == "ATT"){
    Xfit <- X[T==0, ]
    Yfit <- Y[T==0]
    Xpred <- X[T==1, ]

  } else if (estimand == "ATC") {

    Xfit <- X[T==1, ]
    Yfit <- Y[T==1]
    Xpred <- X[T==0, ]
    stop("Not supported in this branch")
    
  } else {
    Xfit <- cbind(T, X)
    Yfit <- Y
    stop("Not supported in this branch")
  }

      balance_target  <-  colMeans(X[T==1,])


  if (cv){

    cvglm <- glmnet::cv.glmnet(Xfit, Yfit, alpha = alpha, lambda=lambda)
    lambdas <- list('lambda.min' = cvglm$lambda.min,
                    'lambda.1se' = cvglm$lambda.1se,
                    'undersmooth' = cvglm$lambda[max(which(cvglm$cvlo < min(cvglm$cvm)))])
    pred_lam <- lambdas[[pred_policy]]
    coef_lam <- lambdas[[coef_policy]]
    mu_pred <- predict(cvglm,
                     newx = matrix(balance_target, 1,
                                   length(balance_target)),
                     s=pred_lam)
    fitted_values <- predict(cvglm, newx=X, s=pred_lam)


    intercept   <- coef(cvglm, s=coef_lam)[1]
    alpha_hat <- coef(cvglm, s=coef_lam)[-1]
    alpha_hat_normalized <- alpha_hat / sqrt(sum(alpha_hat^2))
  } else {

    glm <- glmnet(Xfit, Yfit, lambda=lambda, alpha=alpha)
    mu_pred <- predict(glm,
                     newx = matrix(balance_target, 1,
                                   length(balance_target)),
                     )
    fitted_values <- predict(glm, newx=X)


    intercept   <- coef(glm)[1]
    alpha_hat <- coef(glm)[-1]
  }

  alpha_hat_normalized <- alpha_hat / sqrt(sum(alpha_hat^2))  
  tau_hat  <- mean(Y[T==1]) - mu_pred

  mhat0  <- mu_pred
  mhat1  <- mean(Y[T==1])

  list(alpha_hat=alpha_hat,
       alpha_hat_normalized=alpha_hat_normalized,
       tau_hat=tau_hat,
       intercept_alpha_hat = intercept,
       mhat0=mhat0,
       mhat1=mhat1,
       preds=fitted_values)
}

estimate_propensity <- function(X, T, cv=TRUE, T_lambda_min=0, alpha=0){
  p <- ncol(X)
  if(cv){
    cvglm <- cv.glmnet(cbind(X), T, family="binomial",
                       alpha=alpha, penalty.factor = rep(1, p), intercept=TRUE)
    T_lambda_min <- cvglm$lambda.min
  }


  propensity_fit <- glmnet(cbind(X), T, family="binomial",
                           alpha=alpha, penalty.factor = rep(1, p),intercept=TRUE,
                           lambda=T_lambda_min)
  beta_hat <- coef(propensity_fit)[-1]
  intercept_beta_hat <- coef(propensity_fit)[1]
  escale_hat <- sqrt(sum(beta_hat^2))

  beta_hat_normalized <- if(escale_hat > 0) {
                           beta_hat / escale_hat
                         } else {
                           beta_hat
                         }
  ehat <- predict(propensity_fit, type="response", newx=X)

  if (cv){
  propensity_fit_all <- glmnet(cbind(X), T, family="binomial",
                               alpha=alpha, penalty.factor = rep(1, p),intercept=TRUE,
                               lambda=cvglm$lambda)
    ehat_all <- predict(propensity_fit_all, type="response", newx=X)

  }
  else{
    ehat_all <- NULL
  }


  list(beta_hat=beta_hat,
       beta_hat_normalized=beta_hat_normalized,
       intercept_beta_hat=intercept_beta_hat,
       escale_hat=escale_hat,
       ehat=ehat,
       ehat_all = ehat_all,
       lambda=T_lambda_min)
}


get_attr_default <- function(thelist, attrname, default){
  if(!is.null(thelist[[attrname]])) thelist[[attrname]] else default
}


invlogit <- function(x, a, b) {
  exp(a+x*b) / (1+exp(a+x*b))
}

logistic <- function(x){
  exp(x) / (1 + exp(x))
                                      
}

ipw_est <- function(e, T, Y, estimand, hajek=FALSE, epsilon=.Machine$double.eps){

  if(estimand == "ATT") {
    one_minus_e = 1-e 
    one_minus_e[one_minus_e < epsilon] = epsilon
    wts <- (T - (1-T) * e / one_minus_e) / sum(T)
  } else if (estimand == "ATC") {
    wts <- (T * (1-e) / e - (1-T)) / sum(1-T)
  } else {
    wts <- (T / e) - ((1 - T) / (1 - e)) / (sum(T)+sum(1-T))
  }

  if(hajek) {
    sum(wts[T == 1] * Y[T == 1]) / sum(wts[T == 1]) - sum(wts[T == 0] * Y[T == 0]) / sum(wts[T == 0])
  } else {
    sum(wts * Y)
  }
}

make_bias_var_plot  <- function(results_array, true_ate, subtitle="Title", cols_vec=NULL, lty_vec=NULL) {
  if (is.null(cols_vec)) {
    cols <- brewer.pal(9, "Set1")
    cols_vec <- cols[c(1, 1, 1, 7, 7, 8, 8, 6, 6, 2, 2, 2, 3, 4, 5, 9, 9)]
  }
  if (is.null(lty_vec))
    lty_vec <- c("solid", "dashed", "dotted")[c(2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 1, 1, 2, 1)]


  rmse_mat <- sqrt(apply((results_array - true_ate)^2, c(2, 3), function(x) mean(x, na.rm=TRUE)))
  bias_mat <- abs(apply(results_array - true_ate, c(2, 3), function(x) mean(x, na.rm=TRUE)))
  var_mat  <- rmse_mat^2 - bias_mat^2

  tib <- as_tibble(t(rmse_mat))
  tib$W <- as.numeric(colnames(rmse_mat))
  tib2 <- tib %>% gather(key=Type, value=RMSE, -W)

  rmse_plot <- tib %>% gather(key=Type, value=RMSE, -W) %>%
    ggplot() + geom_line(aes(x=W, y=RMSE, col=Type, linetype=Type)) +
    theme_bw() + theme(legend.position="none") +
    scale_color_manual(values=cols_vec, labels=latex2exp::TeX) +
    scale_linetype_manual(values=lty_vec, labels=latex2exp::TeX) +
    xlab(expression(w))

  rmse_plot <- tib2 %>% ggplot() +
    geom_line(aes(x=W, y=RMSE, col=Type, linetype=Type)) +
    theme_bw() + theme(legend.position="none") +
    scale_color_manual(values=cols_vec, labels=latex2exp::TeX) +
    scale_linetype_manual(values=lty_vec, labels=latex2exp::TeX) +
    xlab(expression(w))

  tib <- as_tibble(t(bias_mat))
  tib$W <- as.numeric(colnames(bias_mat))
  bias_plot <- tib %>% gather(key=Type, value=Bias, -W) %>%
    ggplot() + geom_line(aes(x=W, y=Bias, col=Type, linetype=Type)) +
    theme_bw() + theme(legend.position="none") + geom_hline(yintercept=0, linetype=2) +
    scale_color_manual(values=cols_vec, labels=latex2exp::TeX) +
    scale_linetype_manual(values=lty_vec, labels=latex2exp::TeX) +
    xlab(expression(w)) + ylab("Absolute Bias")

  tib <- as_tibble(t(sqrt(var_mat)))
  tib$W <- as.numeric(colnames(bias_mat))
  sd_plot <- tib %>% gather(key=Type, value=SD, -W) %>%
    ggplot() + geom_line(aes(x=W, y=SD, col=Type, linetype=Type)) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5)) +
    scale_color_manual(values=cols_vec, labels=latex2exp::TeX) +
    scale_linetype_manual(values=lty_vec, labels=latex2exp::TeX) +
    xlab(expression(w))

  rmse_plot + bias_plot + sd_plot +
    plot_annotation(
      title = subtitle,
      theme = theme(plot.title = element_text(size = 11, hjust = 0.5))
    ) &
    theme(
      legend.text  = element_text(size = 10),
      legend.title = element_text(size = 10)
    )
}


shrinkage_path <- function(tib, nms=NULL) {

  df <- tib
  df$observation = 1:nrow(df)

  df %>% gather(w, `Propensity Score`, -observation) %>%
    mutate(w = factor(w, levels=colnames(df))) %>%
    arrange(desc(w)) %>%
    ggplot(aes(y = `Propensity Score`, x = w)) +
    geom_vline(xintercept=ncol(df)/2, col="red", size=2) +
    geom_point() +
    geom_path(aes(group=observation)) +
    theme_bw(base_size=16) +
    scale_x_discrete(labels=nms)
}

double_shrinkage_plot <- function(baseline_e, e_top, e_bottom, nms=NULL) {


  df <- tibble(baseline=baseline_e, top=e_top, bottom=e_bottom)

  if(!is.null(nms))
    colnames(df) <- nms
  else
    nms <- colnames(df)

  df$observation = 1:nrow(df)

  df %>% gather(Type, `Propensity Score`, -observation) %>%
    mutate(Type = factor(Type, levels=nms[c(2, 1, 3)])) %>%
    arrange(desc(Type)) %>%
    ggplot(aes(y = `Propensity Score`, x = Type)) +
    geom_point() +
    geom_path(aes(group=observation)) +
    theme_bw()
}


shrinkage_plot <- function(a, b) {
  plot(x=a, y=rep(1, length(a)), ylim=c(-0.1, 1.1))
  points(x=b, y=rep(0, length(b)))
  segments(x0=a, x1=b, y0=rep(1, length(a)), y1=rep(0, length(b)))
  abline(v=1/2, lty=2)
  abline(h=1)
  abline(h=0)
}
