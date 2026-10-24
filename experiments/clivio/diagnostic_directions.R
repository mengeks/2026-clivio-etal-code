###############################################################################
# Diagnostic: how well do different methods recover true alpha and beta?
#
# For each of 100 simulation iterations (setting 4, ATT):
#   • LASSO/Ridge outcome × LASSO/Ridge propensity (4 combos)
#   • Metric: cosine²(estimated direction, true direction)
#   • Also: α·β (inner product of estimated directions vs truth = 0.75)
#   • Signal strength: lambda_X, lambda_YA
#
# Produces:
#   results/diagnostic_directions.pdf
###############################################################################

source("oracle_utils.R")
source("score_ratio_dr.R")

library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)
library(RColorBrewer)

set.seed(42)

SETTING  <- 4
ITERS    <- 100
ESTIMAND <- "ATT"

REG_COMBOS <- list(
  list(y_alpha = 1, t_alpha = 1, label = "LASSO Y / LASSO T"),
  list(y_alpha = 1, t_alpha = 0, label = "LASSO Y / Ridge T"),
  list(y_alpha = 0, t_alpha = 1, label = "Ridge Y / LASSO T"),
  list(y_alpha = 0, t_alpha = 0, label = "Ridge Y / Ridge T")
)

# True DGP directions (same across all iterations)
set.seed(0)
qa_true    <- 20
p_true     <- 1000
alpha_true <- c(rnorm(qa_true) / qa_true, rep(0, p_true - qa_true))
alpha_true <- alpha_true / sqrt(sum(alpha_true^2))
beta_true  <- construct_beta(alpha_true, ab_dp = 0.75)

results <- list()

for (reg in REG_COMBOS) {
  cat(sprintf("Running: %s\n", reg$label))

  cos2_alpha_vec <- numeric(ITERS)
  cos2_beta_vec  <- numeric(ITERS)
  ab_dp_vec      <- numeric(ITERS)
  lambda_X_vec   <- numeric(ITERS)
  lambda_YA_vec  <- numeric(ITERS)

  for (iter in seq_len(ITERS)) {
    dat <- gen_dataset("simulated", SETTING, iter)
    X   <- dat$X; T <- dat$T; Y <- dat$Y

    ev <- eval_directions(
      X, T, Y, ESTIMAND,
      Y_ALPHA    = reg$y_alpha,
      T_ALPHA    = reg$t_alpha,
      alpha_true = alpha_true,
      beta_true  = beta_true
    )

    cos2_alpha_vec[iter] <- ev$cos2_alpha
    cos2_beta_vec[iter]  <- ev$cos2_beta
    ab_dp_vec[iter]      <- ev$ab_dp
    lambda_X_vec[iter]   <- ev$lambda_X
    lambda_YA_vec[iter]  <- ev$lambda_YA
  }

  results[[reg$label]] <- data.frame(
    label      = reg$label,
    iter       = seq_len(ITERS),
    cos2_alpha = cos2_alpha_vec,
    cos2_beta  = cos2_beta_vec,
    ab_dp      = ab_dp_vec,
    lambda_X   = lambda_X_vec,
    lambda_YA  = lambda_YA_vec
  )
}

df <- bind_rows(results)

# Also add oracle row for reference (cos² = 1 by definition)
df_oracle <- data.frame(
  label      = "Oracle (true α, β)",
  iter       = seq_len(ITERS),
  cos2_alpha = 1,
  cos2_beta  = 1,
  ab_dp      = 0.75,
  lambda_X   = NA,
  lambda_YA  = NA
)
df <- bind_rows(df, df_oracle)

col_labels <- c(
  "LASSO Y / LASSO T", "LASSO Y / Ridge T",
  "Ridge Y / LASSO T", "Ridge Y / Ridge T",
  "Oracle (true α, β)"
)
cols <- c(brewer.pal(4, "Set1"), "black")
names(cols) <- col_labels

df$label <- factor(df$label, levels = col_labels)

# ---------------------------------------------------------------------------
# Plot 1: cos²(alpha_hat, alpha_true) — how well is α recovered?
# ---------------------------------------------------------------------------

p_alpha <- df %>%
  filter(label != "Oracle (true α, β)") %>%
  ggplot(aes(x = label, y = cos2_alpha, fill = label)) +
  geom_boxplot(width = 0.5, outlier.size = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "black", linewidth = 0.5) +
  scale_fill_manual(values = cols) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1)) +
  labs(x = NULL,
       y = bquote(cos^2*(hat(alpha)*","*alpha)),
       title = "Direction recovery: alpha (outcome)")

# ---------------------------------------------------------------------------
# Plot 2: cos²(beta_hat, beta_true)
# ---------------------------------------------------------------------------

p_beta <- df %>%
  filter(label != "Oracle (true α, β)") %>%
  ggplot(aes(x = label, y = cos2_beta, fill = label)) +
  geom_boxplot(width = 0.5, outlier.size = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "black", linewidth = 0.5) +
  scale_fill_manual(values = cols) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1)) +
  labs(x = NULL,
       y = bquote(cos^2*(hat(beta)*","*beta)),
       title = "Direction recovery: beta (propensity)")

# ---------------------------------------------------------------------------
# Plot 3: estimated α·β inner product vs truth (0.75)
# ---------------------------------------------------------------------------

p_dp <- df %>%
  filter(label != "Oracle (true α, β)") %>%
  ggplot(aes(x = label, y = ab_dp, fill = label)) +
  geom_boxplot(width = 0.5, outlier.size = 0.8) +
  geom_hline(yintercept = 0.75, linetype = "dashed", colour = "black", linewidth = 0.5) +
  scale_fill_manual(values = cols) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1)) +
  labs(x = NULL,
       y = bquote(hat(alpha) %.% hat(beta)),
       title = "Estimated alpha.beta inner product (truth = 0.75)")

# ---------------------------------------------------------------------------
# Plot 4: signal strength lambda_X and lambda_YA
# ---------------------------------------------------------------------------

df_lambda <- df %>%
  filter(!is.na(lambda_X)) %>%
  select(label, iter, lambda_X, lambda_YA) %>%
  pivot_longer(c(lambda_X, lambda_YA),
               names_to  = "matrix",
               values_to = "lambda") %>%
  mutate(matrix = recode(matrix,
                         "lambda_X"  = "H_X  (beta direction)",
                         "lambda_YA" = "H[Y|A]  (alpha direction)"))

p_lambda <- df_lambda %>%
  ggplot(aes(x = label, y = lambda, fill = label)) +
  geom_boxplot(width = 0.5, outlier.size = 0.8) +
  facet_wrap(~ matrix, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = cols) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1),
        strip.text = element_text(size = 9)) +
  labs(x = NULL, y = "Leading eigenvalue",
       title = "Score-ratio signal strength (rank-1 eigenvalue)")

# ---------------------------------------------------------------------------
# Assemble and save
# ---------------------------------------------------------------------------

final <- (p_alpha | p_beta) / p_dp / p_lambda +
  plot_annotation(
    title    = "Diagnostic: direction recovery vs. regularisation (setting 4, 100 iters)",
    subtitle = "Dashed line = oracle / truth; score-ratio top eigenvector = normalised LASSO/Ridge coef"
  )

ggsave("results/diagnostic_directions.pdf", final, width = 10, height = 12)
cat("Saved results/diagnostic_directions.pdf\n")

# ---------------------------------------------------------------------------
# Summary table: median cos² per method
# ---------------------------------------------------------------------------

cat("\n--- Median cos²(direction, truth) across 100 iterations ---\n")
summary_tbl <- df %>%
  filter(label != "Oracle (true α, β)") %>%
  group_by(label) %>%
  summarise(
    median_cos2_alpha = round(median(cos2_alpha), 4),
    median_cos2_beta  = round(median(cos2_beta),  4),
    median_ab_dp      = round(median(ab_dp),       4),
    .groups = "drop"
  )
print(as.data.frame(summary_tbl))
