###############################################################################
# Plot SR-NN rescue experiment vs LASSO/Ridge.
# Reads results/sr_nn_rescue_p100.csv
# Saves results/sr_nn_rescue_p100.pdf
###############################################################################

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(RColorBrewer)

csv  <- "results/sr_nn_rescue_p100.csv"
if (!file.exists(csv)) stop("Run score_ratio_rescue.py first.")
df   <- read.csv(csv)

METHODS <- c("SR-NN rescue",
             "LASSO Y / LASSO T", "LASSO Y / Ridge T",
             "Ridge Y / LASSO T", "Ridge Y / Ridge T")
LABELS  <- c("SR-NN (rescue)",
             "LASSO-Y/LASSO-T", "LASSO-Y/Ridge-T",
             "Ridge-Y/LASSO-T", "Ridge-Y/Ridge-T")
pal <- c("#1a9641", brewer.pal(4, "Set1"))
cols <- setNames(pal, LABELS)

df$method <- factor(df$method, levels = METHODS, labels = LABELS)

# ── Panel A: cos²(alpha) ────────────────────────────────────────────────────
p_alpha <- ggplot(df, aes(x = method, y = cos2_alpha, fill = method)) +
  geom_boxplot(width = 0.5, outlier.size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = cols) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, .25)) +
  theme_bw() + theme(legend.position = "none",
                     axis.text.x = element_text(angle = 25, hjust = 1, size = 8)) +
  labs(x = NULL, y = bquote(cos^2*(hat(alpha)*","*alpha)),
       title = "alpha recovery (outcome)")

# ── Panel B: cos²(beta) ─────────────────────────────────────────────────────
p_beta <- ggplot(df, aes(x = method, y = cos2_beta, fill = method)) +
  geom_boxplot(width = 0.5, outlier.size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = cols) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, .25)) +
  theme_bw() + theme(legend.position = "none",
                     axis.text.x = element_text(angle = 25, hjust = 1, size = 8)) +
  labs(x = NULL, y = bquote(cos^2*(hat(beta)*","*beta)),
       title = "beta recovery (propensity)")

# ── Panel C: true score MSE (SR-NN rescue only) ─────────────────────────────
df_eps <- df %>%
  filter(method == "SR-NN (rescue)") %>%
  select(iter, eps_A, eps_YA) %>%
  pivot_longer(-iter, names_to = "which", values_to = "eps") %>%
  mutate(which = recode(which,
                        eps_A  = "epsilon[A] (H_A score)",
                        eps_YA = "epsilon[Y|A] (d[i] score)"))

p_eps <- ggplot(df_eps, aes(x = which, y = eps, fill = which)) +
  geom_boxplot(width = 0.4, outlier.size = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4) +
  scale_fill_manual(values = c("epsilon[A] (H_A score)"  = "#4dac26",
                                "epsilon[Y|A] (d[i] score)" = "#d01c8b")) +
  scale_y_log10() +
  theme_bw() + theme(legend.position = "none") +
  labs(x = NULL, y = "Relative MSE (log scale)",
       title = "True score MSE (SR-NN rescue)")

# ── Panel D: H_A and H_0 eigenvalue rank-1 signal ───────────────────────────
df_eig <- df %>%
  filter(method == "SR-NN (rescue)") %>%
  select(iter, eig1_A, eig2_A, eig1_H0, eig2_H0) %>%
  pivot_longer(-iter, names_to = "which", values_to = "eig") %>%
  mutate(matrix = case_when(grepl("_A$",  which) ~ "H_A (beta, r=1)",
                             grepl("_H0$", which) ~ "H_0 (alpha, d_i)"),
         rank   = ifelse(grepl("eig1", which), "lambda[1]", "lambda[2]"))

p_eig <- ggplot(df_eig, aes(x = rank, y = eig, fill = matrix)) +
  geom_boxplot(width = 0.4, outlier.size = 1) +
  facet_wrap(~ matrix, scales = "free_y") +
  scale_fill_manual(values = c("H_A (beta, r=1)" = "#4dac26",
                                "H_0 (alpha, d_i)" = "#d01c8b")) +
  theme_bw() + theme(legend.position = "none") +
  labs(x = NULL, y = "Eigenvalue",
       title = "Eigenvalue spectrum (SR-NN rescue)")

# ── Assemble ─────────────────────────────────────────────────────────────────
n_iters <- max(df$iter, na.rm = TRUE)
final <- (p_alpha | p_beta) / (p_eps | p_eig) +
  plot_annotation(
    title    = paste0("SR-NN rescue vs LASSO/Ridge -- direction recovery ",
                      "(setting 4, p=100, ", n_iters, " iters)"),
    subtitle = paste0("Fixes: d_i subtraction before outer product | ",
                      "r'=1 for H_A | lam2_A=0 | exact trace (no Hutchinson)")
  )

out <- "results/sr_nn_rescue_p100.pdf"
ggsave(out, final, width = 12, height = 8)
cat("Saved", out, "\n")

cat("\nMedian cos^2 per method:\n")
df %>%
  group_by(method) %>%
  summarise(cos2_alpha = round(median(cos2_alpha, na.rm=TRUE), 4),
            cos2_beta  = round(median(cos2_beta,  na.rm=TRUE), 4),
            ab_dp      = round(median(ab_dp,       na.rm=TRUE), 4),
            eps_A      = round(median(eps_A,       na.rm=TRUE), 4),
            eps_YA     = round(median(eps_YA,      na.rm=TRUE), 4),
            .groups = "drop") %>%
  print()
