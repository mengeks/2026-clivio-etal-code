library(tidyverse)
library(ggridges)
library(superheat)
library(patchwork)
library(RColorBrewer)
library(kableExtra)
library(modelr)
library(cowplot)

source("utilities.R")

regs <- c("yalpha1_talpha1","yalpha1_talpha0","yalpha0_talpha1","yalpha0_talpha0")
regs_name <- c('LASSO outcome and propensity', "LASSO outcome and Ridge propensity", "Ridge outcome and LASSO propensity", 'Ridge outcome and propensity')
regs_name_latex <- lapply(regs_name, function(s){paste('\\multicolumn{4}{c}{', s, '}')} )
rmse_tables <- list(NA, NA, NA, NA)

bias_var_plots <- list(NA, NA, NA)

## ITERATE BY REGULARIZATION
idx_plot <- 1
for (idx_reg in c(1:4)){

  reg <- regs[idx_reg]

  ns <- list(0, 0, 0)


  ## CREATE TABLE OF RESULTS BY REGULARIZATION

  results_dir <- "results"
  results_files <- dir(results_dir)
  results_files  <- results_files[grepl("times1_", results_files)]

  results_files  <- results_files[grepl(reg, results_files)]
  results_files <- results_files[grepl(".RData", results_files)]



  METHODS_no_d <- c("IPW", "AIPW","Regression")

  METHODS_d <- c("IPW_d", "AIPW_d","Regression_d")

  rmse_table <- matrix(0.0, nrow=length(METHODS_no_d) + 3*length(METHODS_d), ncol=3)
  count <- 1




  for(fn in results_files){

      file_path <- paste(results_dir, fn, sep="/")

      load(file_path)

      rmse_mat <- sqrt(apply((results_array - true_ate)^2, c(2, 3), function(x) mean(x, na.rm=TRUE)))
      
      idx <- 0
      if (grepl("ihdp", fn, fixed = TRUE)){
        idx <- 1
      }
      if (grepl("acic", fn, fixed = TRUE)){
        idx <- 2
      }
      if (grepl("hcmnist", fn, fixed = TRUE)){
        idx <- 3
      }
      
      
      if (idx>0)
      {
        ns[[idx]] <- ns[[idx]] + 1

        rmse_table[, idx] <- rmse_table[, idx] + c(rmse_mat[METHODS_no_d, 1]**2,
                                rmse_mat[METHODS_d, "1"]**2,
                                rmse_mat[METHODS_d, "0"]**2,
                                rmse_mat[METHODS_d, "-1"]**2)
      }
      
    }

  for (idx in c(1,2,3)){
      rmse_table[,idx] <- sqrt(rmse_table[,idx] / ns[[idx]])

  }



  METHODS_no_d <- gsub('Regression', 'Regr', METHODS_no_d)
  METHODS_d <- gsub('Regression', 'Regr', METHODS_d)
  rownames(rmse_table) <- c(gsub("_", "-", METHODS_no_d), gsub("-onlyrep", "", gsub("-d", "-$\\hat{\\beta}$", gsub("_", "-", METHODS_d), fixed=TRUE)), gsub("-onlyrep", "", gsub("-d", "-$\\hat{\\delta}$", gsub("_", "-", METHODS_d), fixed=TRUE)), gsub("-onlyrep", "", gsub("-d", "-$\\hat{\\alpha}$", gsub("_", "-", METHODS_d), fixed=TRUE)))

    
  rmse_tables[[idx_reg]] <- apply(rmse_table, 2, function(x) {


        x <- round(x, 2)
        min_index <- which.min(x)

        for (method in gsub("_", "-", METHODS_no_d)){
          names_here <- names(x)[grepl(paste("^", method, "-", sep=""), gsub("-onlyrep", " ", gsub("-collaborative", " ", names(x))))]
          for (name in names_here){
            if (x[name] < x[method]) {x[name] <- paste0("\\green{", x[name], "}")} else {x[name] <- paste0("\\red{", x[name], "}")}
          }
        }

        x[min_index] <- paste0("\\underline{", x[min_index], "}")
        x
    })


}

# Create merged table


rmse_table_combined <- rbind(
  matrix(rep(NA,3), nrow=1, dimnames = list(regs_name_latex[1], NULL)),
  rmse_tables[[1]],
  matrix(rep(NA,3), nrow=1, dimnames = list(regs_name_latex[2], NULL)),
  rmse_tables[[2]],
  matrix(rep(NA,3), nrow=1, dimnames = list(regs_name_latex[3], NULL)),
  rmse_tables[[3]],
  matrix(rep(NA,3), nrow=1, dimnames = list(regs_name_latex[4], NULL)),
  rmse_tables[[4]]
)


dataset_colname <- c("IHDP"=1, "ACIC2016"=1, "HC-MNIST"=1)

output <- kable(rmse_table_combined, format="latex", align="c", escape=FALSE) %>%
    add_header_above(c("Dataset ", dataset_colname))

output <- gsub("\\hline", "", output, fixed=TRUE)
output <- gsub("\n\\multicolumn{4}{c}", "\n\\hline\n\\multicolumn{4}{c}", output, fixed=TRUE)

output <- gsub("& NA & NA & NA",  "", output, fixed=TRUE)
output <- gsub("\nNaive", "\n\\hline\nNaive", output, fixed=TRUE)
output <- gsub("\nIPW", "\n\\hline\nIPW", output, fixed=TRUE)
output <- gsub("\\begin{tabular}{l|c|c|c|c|c|c|c|c}", "\n\\hspace{-3cm}\n\\begin{tabular}{l|c|c|c|c|c|c|c|c}", output, fixed=TRUE)
print(output)
