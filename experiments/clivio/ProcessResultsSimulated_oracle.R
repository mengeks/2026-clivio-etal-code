library(tidyverse)
library(ggridges)
library(patchwork)
library(RColorBrewer)
library(kableExtra)
library(modelr)
library(cowplot)
library(latex2exp)

make_bias_var_plot <- function(results_array, true_ate, subtitle="Title", cols_vec=NULL, lty_vec=NULL) {
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

  legend_theme <- theme(
    legend.text  = element_text(size = 10),
    legend.title = element_text(size = 10)
  )
  (rmse_plot + legend_theme) + (bias_plot + legend_theme) + (sd_plot + legend_theme) +
    plot_annotation(
      title = subtitle,
      theme = theme(plot.title = element_text(size = 11, hjust = 0.5))
    )
}

regs <- c("yalpha1_talpha1","yalpha1_talpha0","yalpha0_talpha1","yalpha0_talpha0")
regs_name <- c('LASSO outcome and propensity', "LASSO outcome and Ridge propensity", "Ridge outcome and LASSO propensity", "Ridge outcome and propensity")
regs_name_latex <- lapply(regs_name, function(s){paste('\\multicolumn{5}{c}{', s, '}')} )
rmse_tables <- list(NA, NA, NA, NA)
bias_var_plots <- list(NA, NA, NA, NA)
file_names_plots <- lapply(regs, function(s){paste("results_simulated_oracle_setting4_", s, "_iters100_times1_fullalphasetTRUE.RData")})

for (idx in c(1:4)){

  reg <- regs[idx]


  results_dir <- "results"
  results_files <- dir(results_dir)


  results_files  <- results_files[grepl(reg, results_files)]
  file_matches <- results_files[grepl("simulated_oracle", results_files)]
  file_matches <- file_matches[grepl(".RData", file_matches)]


  METHODS_no_d <- c("IPW", "AIPW","Regression")

  METHODS_d <- c("IPW_d", "AIPW_d","Regression_d")

  rmse_table <- matrix(nrow=length(METHODS_no_d) + 3*length(METHODS_d), ncol=length(file_matches))
  count <- 1



  for(fn in file_matches){

      file_path <- paste(results_dir, fn, sep="/")

      load(file_path)



      rmse_mat <- sqrt(apply((results_array - true_ate)^2, c(2, 3), function(x) mean(x, na.rm=TRUE)))
      rmse_table[, count] <- c(rmse_mat[METHODS_no_d, 1],
                              rmse_mat[METHODS_d, "1"],
                              rmse_mat[METHODS_d, "0"],
                              rmse_mat[METHODS_d, "-1"])


      count <- count + 1
  }


  METHODS_no_d <- gsub('Regression', 'Regr', METHODS_no_d)
  METHODS_d <- gsub('Regression', 'Regr', METHODS_d)
  rownames(rmse_table) <- c(gsub("_", "-", METHODS_no_d), gsub("-d", "-$\\hat{\\beta}$", gsub("_", "-", METHODS_d), fixed=TRUE), gsub("-d", "-$\\hat{\\delta}$", gsub("_", "-", METHODS_d), fixed=TRUE), gsub("-d", "-$\\hat{\\alpha}$", gsub("_", "-", METHODS_d), fixed=TRUE))

  rmse_tables[[idx]] <- apply(rmse_table, 2, function(x) {


      x <- round(x, 2)
      min_index <- which.min(x)
      for (method in gsub("_", "-", METHODS_no_d)){
        names_here <- names(x)[grepl(paste("^", method, "-", sep=""), names(x))]
        for (name in names_here){
          if (x[name] < x[method]) {x[name] <- paste0("\\green{", x[name], "}")} else {x[name] <- paste0("\\red{", x[name], "}")}
        }
      }

      x[min_index] <- paste0("\\underline{", x[min_index], "}")
      x
  })


  file_name <- file_names_plots[idx]
  file_name
  file_path <- paste0(results_dir, "/", file_name)


  METHODS <- c("AIPW", "AIPW_d", "IPW", "IPW_d", "Regression", "Regression_d")
  results_array <- results_array[, METHODS, ]
  true_ate <- true_ate[, METHODS, ]
  dimnames(results_array)[[2]] <- c(
    "AIPW",
    "AIPW-$\\hat{\\gamma}$",
    "IPW",
    "IPW-$\\hat{\\gamma}$",
    "Regr",
    "Regr-$\\hat{\\gamma}$"
  )
  cols <- brewer.pal(8, "Set1")
  cols_vec <- cols[c(1, 1, 2, 2, 4, 4)]
  lty_vec <- c("solid", "dashed", "dotted")[c(2, 1, 2, 1, 2, 1)]


  bias_var_plots[[idx]]  <- make_bias_var_plot(results_array, true_ate, subtitle=regs_name[idx], cols_vec=cols_vec, lty_vec=lty_vec)

}

# Create merged table


rmse_table_combined <- rbind(
  matrix(rep(NA,4), nrow=1, dimnames = list(regs_name_latex[1], NULL)),
  rmse_tables[[1]],
  matrix(rep(NA,4), nrow=1, dimnames = list(regs_name_latex[2], NULL)),
  rmse_tables[[2]],
  matrix(rep(NA,4), nrow=1, dimnames = list(regs_name_latex[3], NULL)),
  rmse_tables[[3]],
  matrix(rep(NA,4), nrow=1, dimnames = list(regs_name_latex[4], NULL)),
  rmse_tables[[4]]
)


overlap_colname <- c("High"=2, "Low"=2)
snr_colname <- rep(c("Low"=1, "High"=1), 2)

output <- kable(rmse_table_combined, format="latex", align="c", escape=FALSE) %>%
    add_header_above(c("SNR ", snr_colname)) %>%
    add_header_above(c("Overlap", overlap_colname))

output <- gsub("\\hline", "", output, fixed=TRUE)
output <- gsub("\n\\multicolumn{5}{c}", "\n\\hline\n\\multicolumn{5}{c}", output, fixed=TRUE)

output <- gsub("& NA & NA & NA & NA",  "", output, fixed=TRUE)
output <- gsub("\nNaive", "\n\\hline\nNaive", output, fixed=TRUE)
output <- gsub("\nIPW", "\n\\hline\nIPW", output, fixed=TRUE)
output <- gsub("\\begin{tabular}{l|c|c|c|c}", "\n\\hspace{-3cm}\n\\begin{tabular}{l|c|c|c|c}", output, fixed=TRUE)
print(output)

final <- plot_grid(
  bias_var_plots[[1]],
  bias_var_plots[[2]],
  bias_var_plots[[3]],
  bias_var_plots[[4]],
  ncol = 1,
  align = "v",
  axis = "l",
  rel_heights = c(1,1,1,1)
)

ggsave("combined_plot_oracle.pdf", final, width = 7, height = 12)
