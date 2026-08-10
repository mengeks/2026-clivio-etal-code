###############################################################################
# Plot score-ratio NN diagnostic vs LASSO/Ridge.
# Reads results/sr_nn_diagnostic.csv written by score_ratio_nn_diagnostic.py
###############################################################################

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(RColorBrewer)

csv_path <- "results/sr_nn_diagnostic.csv"
if (!file.exists(csv_path)) stop("Run score_ratio_nn_diagnostic.py first.")
df <- read.csv(csv_path)

method_levels <- c("LASSO Y / LASSO T", "Ridge Y / Ridge T", "SR-NN")
method_labels <- c("LASSO", "Ridge", "Score-ratio NN")
cols <- setNames(c(brewer.pal(3, "Set1")[c(1,2)], "#1a9641"), method_levels)
df$method <- factor(df$method, levels = method_levels, labels = method_labels)

# ---------------------------------------------------------------------------
# Panel A/B: cos²(direction, truth)
# ---------------------------------------------------------------------------

p_alpha <- ggplot(df, aes(x = method, y = cos2_alpha, fill = method)) +
  geom_boxplot(width = 0.5, outlier.size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = setNames(cols, method_labels)) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, .25)) +
  theme_bw() + theme(legend.position = "none") +
  labs(x = NULL,
       y = bquote(cos^2*(hat(alpha)*","*alpha)),
       title = "alpha recovery (outcome direction)")

p_beta <- ggplot(df, aes(x = method, y = cos2_beta, fill = method)) +
  geom_boxplot(width = 0.5, outlier.size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = setNames(cols, method_labels)) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, .25)) +
  theme_bw() + theme(legend.position = "none") +
  labs(x = NULL,
       y = bquote(cos^2*(hat(beta)*","*beta)),
       title = "beta recovery (propensity direction)")

# ---------------------------------------------------------------------------
# Panel C: alpha.beta inner product vs truth 0.75
# ---------------------------------------------------------------------------

p_dp <- ggplot(df, aes(x = method, y = ab_dp, fill = method)) +
  geom_boxplot(width = 0.5, outlier.size = 1) +
  geom_hline(yintercept = 0.75, linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = setNames(cols, method_labels)) +
  theme_bw() + theme(legend.position = "none") +
  labs(x = NULL,
       y = bquote(hat(alpha) %.% hat(beta)),
       title = bquote("Estimated " * hat(alpha) %.% hat(beta) * "  (truth = 0.75)"))

# ---------------------------------------------------------------------------
# Panel D: H_X and H_YA eigenvalue rank-1 signal (SR-NN only)
# ---------------------------------------------------------------------------

df_eig <- df %>%
  filter(method == "Score-ratio NN") %>%
  select(iter, eig1_X, eig2_X, eig1_YA, eig2_YA) %>%
  pivot_longer(-iter, names_to = "which", values_to = "eig") %>%
  mutate(matrix   = ifelse(grepl("_X$",  which), "H_X  (beta)", "H_YA (alpha)"),
         rank     = ifelse(grepl("eig1", which), "lambda_1", "lambda_2"))

p_eig <- ggplot(df_eig, aes(x = rank, y = eig, fill = matrix)) +
  geom_boxplot(width = 0.4, outlier.size = 1) +
  facet_wrap(~ matrix, scales = "free_y") +
  scale_fill_manual(values = c("H_X  (beta)" = "#4dac26",
                                "H_YA (alpha)" = "#d01c8b")) +
  theme_bw() + theme(legend.position = "none") +
  labs(x = NULL, y = "Eigenvalue",
       title = "Score-ratio NN: eigenvalue spectrum (rank-1 signal)")

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

n_iters <- max(df$iter)
final <- (p_alpha | p_beta) / (p_dp | p_eig) +
  plot_annotation(
    title = sprintf(
      "Score-ratio NN vs LASSO/Ridge — direction recovery (setting 4, %d iters)", n_iters),
    subtitle = "Dashed line = oracle/truth.  H_YA = H_Y - H_X isolates alpha direction."
  )

ggsave("results/sr_nn_diagnostic.pdf", final, width = 10, height = 8)
cat("Saved results/sr_nn_diagnostic.pdf\n")

cat("\nMedian cos² per method:\n")
df %>%
  group_by(method) %>%
  summarise(cos2_alpha = round(median(cos2_alpha), 4),
            cos2_beta  = round(median(cos2_beta),  4),
            ab_dp      = round(median(ab_dp),       4),
            .groups    = "drop") %>%
  print()
