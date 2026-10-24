library(tidyverse)
library(mvtnorm)
library(rstiefel)
library(glmnet)
library(lubridate)
library(cobalt)
library(R.utils)
source("utilities.R")
library(aciccomp2016)
library(caret)
library(npci)        
source("data.R")


options(warn=1)

argv <- R.utils::commandArgs(trailingOnly=TRUE, asValues=TRUE)

Y_ALPHA <- as.numeric(get_attr_default(argv, "y_alpha", 1))

T_ALPHA <- as.numeric(get_attr_default(argv, "t_alpha", 1))


estimand <- as.character(get_attr_default(argv, "estimand", "ATT"))

setting        <- as.numeric(get_attr_default(argv, "setting", 1)) 
dataset <- as.character(get_attr_default(argv, "dataset", "simulated"))

iters <- as.numeric(get_attr_default(argv, "iters", 2))
times <- as.numeric(get_attr_default(argv, "times", 1))


FULL_ALPHA_SET <- as.logical(get_attr_default(argv, "full_alpha_set", FALSE))



file_name = sprintf("results_%s_setting%s_yalpha%i_talpha%i_iters%s_times%s_fullalphaset%s.RData", dataset, setting, Y_ALPHA, T_ALPHA, iters, times, FULL_ALPHA_SET)

if (file.exists(paste("results", file_name, sep="/"))) {
  stop(paste("File", file_name, "already exists. Stopping the script."))
}




if(FULL_ALPHA_SET) {
    w2_scale_vec <- c(seq(1, -1, by=-.1))
} else {
    w2_scale_vec <- c(1, 0, -1)
}

all_methods <- c("Naive",
                  "Regression", "Regression_d",
                  "IPW", "IPW_d",
                  "AIPW", "AIPW_d"
                  )


results_array_4d <- array(dim=c(iters, length(all_methods), length(w2_scale_vec), times))
w2lim_true_vec <- numeric(iters)


dimnames(results_array_4d) <- list(1:iters, all_methods, w2_scale_vec, 1:times)
true_ate <- array(dim=c(iters, length(all_methods), length(w2_scale_vec)))
dimnames(true_ate) <- list(1:iters, all_methods, w2_scale_vec)


set.seed(0)

warning_lines <- character(0)
for(iter  in 1:iters) {

  set.seed(iter)


  dat <- gen_dataset(dataset, setting, iter)
  X <- dat$X 
  T <- dat$T 
  Y <- dat$Y 
  true_ate[iter, , ] <- dat$true_ate


  out_ests <- estimate_outcome(X, T, Y, estimand, alpha=Y_ALPHA, coef_policy="lambda.min", pred_policy="lambda.min")

  alpha_hat <- get_attr_default(out_ests, "alpha_hat", NA)
  outcome_intercept <- get_attr_default(out_ests, "intercept_alpha_hat", NA)
  alpha_hat_normalized <- get_attr_default(out_ests, "alpha_hat_normalized",
                                           NA)
  mhat0 <- get_attr_default(out_ests, "mhat0", NA)
  mhat1 <- get_attr_default(out_ests, "mhat1", NA)
  tau_hat <- get_attr_default(out_ests, "tau_hat", NA)
  outcome_preds <- get_attr_default(out_ests, "preds", NA)

  prop_ests <- estimate_propensity(X, T, alpha=T_ALPHA)

  beta_hat <- get_attr_default(prop_ests, "beta_hat", NA)
  beta_hat_normalized <- get_attr_default(prop_ests, "beta_hat_normalized", NA)
  escale_hat <- get_attr_default(prop_ests, "escale_hat", NA)
  ehat <- get_attr_default(prop_ests, "ehat", NA)
  xb <- X %*% beta_hat_normalized


  results_array_4d[iter, "Regression", , ] <- tau_hat

  naive <- mean(Y[T == 1]) - mean(Y[T == 0])
  results_array_4d[iter, "Naive", , ] <- naive


  results_array_4d[iter, "IPW", , ] <- ipw_est(ehat, T, Y, estimand, hajek=TRUE)



  if(estimand == "ATT") {
    residual <- 0 + (1-T) * (Y - outcome_preds)
  }
  else {
    warning("ATE not currently supported")
    residual <- Y - X %*% alpha_hat - T * tau_hat - outcome_intercept
  }

  results_array_4d[iter, "AIPW", , ] <- tau_hat +
    ipw_est(ehat, T, residual, estimand, hajek=TRUE)



  ab_dot_prod <- as.numeric(t(alpha_hat_normalized) %*% beta_hat_normalized)
  if(max(abs(alpha_hat)) < 1e-10 | max(abs(beta_hat)) < 1e-10)
  {
        results_array_4d[iter, "IPW_d", , ] <- results_array_4d[iter, "IPW", , ]
        results_array_4d[iter, "AIPW_d", , ] <- results_array_4d[iter, "AIPW", , ]
        results_array_4d[iter, "Regression_d", , ] <- results_array_4d[iter, "Regression", , ]
        if (max(abs(alpha_hat)) < 1e-10) {warning_lines <- c(warning_lines, sprintf("WARNING (%s %s %s %s) (iter %s): near-zero alpha", dataset, setting, Y_ALPHA, T_ALPHA, iter))}
        if (max(abs(beta_hat)) < 1e-10) {warning_lines <- c(warning_lines, sprintf("WARNING (%s %s %s %s) (iter %s): near-zero beta", dataset, setting, Y_ALPHA, T_ALPHA, iter))}
        next
  }
  if(ab_dot_prod < 0)
  {
      beta_hat_normalized <- -beta_hat_normalized
      ab_dot_prod <- as.numeric(t(alpha_hat_normalized) %*% beta_hat_normalized)

  } 
  mvecs <- cbind((alpha_hat_normalized + beta_hat_normalized)/sqrt(2 + 2 * ab_dot_prod),
  (alpha_hat_normalized - beta_hat_normalized)/sqrt(2 - 2 * ab_dot_prod))
  mvals <- c((ab_dot_prod + 1)/2, (ab_dot_prod - 1)/2)


  for(j in 1:length(w2_scale_vec)) {
    w2scale <- w2_scale_vec[j]

    w2_lim <- -sqrt(abs(mvals[2]))
    w2 <- w2scale * w2_lim

    gammas <- compute_gammas(ab_dot_prod=ab_dot_prod, mvals=mvals, mvecs=mvecs, w2=w2, times=times)
    dd <- X %*% gammas


    for (i in 1:ncol(dd)){
        if(var(dd[,i]) < 1e-10 | var(dd[,i][T==0]) < 1e-10)
        {
              results_array_4d[iter, "IPW_d", j, i] <- results_array_4d[iter, "IPW", j, i]
              results_array_4d[iter, "AIPW_d", j, i] <- results_array_4d[iter, "AIPW", j, i]
              results_array_4d[iter, "Regression_d", j, i] <- results_array_4d[iter, "Regression", j, i]
              if (var(dd[,i]) < 1e-10) {warning_lines <- c(warning_lines, sprintf("WARNING (%s %s %s %s) (iter %s) (point %s %s): near-zero variance of deconf score", dataset, setting, Y_ALPHA, T_ALPHA, iter, j, i))}
              if (var(dd[,i][T==0]) < 1e-10) {warning_lines <- c(warning_lines, sprintf("WARNING (%s %s %s %s) (iter %s) (point %s %s): near-zero *control* variance of deconf score", dataset, setting, Y_ALPHA, T_ALPHA, iter, j, i))} 
              next
        }
        ddi <- cbind(dd[,i], 0)

        ehat_ddi <- estimate_propensity(ddi, T, cv=FALSE, T_lambda_min=0, alpha=0)$ehat
        out_ests_ddi <- estimate_outcome(ddi, T, Y, estimand, alpha=0, lambda=0, cv=FALSE)
        tau_hat_ddi <- out_ests_ddi$tau_hat
        outcome_preds_ddi <- out_ests_ddi$preds
        if(estimand == "ATT") {
          residual_ddi <- 0 + (1-T) * (Y - outcome_preds_ddi)
        }
        else {
          stop("ATE not currently supported")
        }


        results_array_4d[iter, "Regression_d", j, i] <- tau_hat_ddi

        results_array_4d[iter, "IPW_d", j, i] <- ipw_est(ehat_ddi, T, Y, estimand, hajek=TRUE)
        results_array_4d[iter, "AIPW_d", j, i] <- tau_hat_ddi + ipw_est(ehat_ddi, T, residual_ddi, estimand, hajek=TRUE)
        



    }

  }
  results_array <- apply(results_array_4d, c(1, 2, 3), function(x) mean(x, na.rm=TRUE))

}
if (length(warning_lines) > 0) {
  warn_file <- paste("results", sub("\\.RData$", "_warnings.txt", file_name), sep="/")
  writeLines(warning_lines, warn_file)
}
results_array <- apply(results_array_4d, c(1, 2, 3), function(x) mean(x, na.rm=TRUE))

na_indices <- which(is.na(results_array), arr.ind = TRUE)
if (nrow(na_indices) > 0) {
  na_file <- paste("results", sub("\\.RData$", "_NA.txt", file_name), sep="/")
  na_lines <- sapply(1:nrow(na_indices), function(k) {
    sprintf("OBS! %s %s %s %s : NA at (iter=%s, method=%s, j=%s)",
            dataset, setting, Y_ALPHA, T_ALPHA,
            dimnames(results_array)[[1]][na_indices[k, 1]],
            dimnames(results_array)[[2]][na_indices[k, 2]],
            dimnames(results_array)[[3]][na_indices[k, 3]])
  })
  writeLines(na_lines, na_file)
}


sqrt(apply((results_array - true_ate)^2, c(2, 3), function(x) mean(x, na.rm=TRUE)))

apply(abs(results_array - true_ate), c(2, 3), function(x) median(x, na.rm=TRUE))

save(results_array_4d, results_array, true_ate, w2lim_true_vec, file=paste("results", file_name, sep="/"))
