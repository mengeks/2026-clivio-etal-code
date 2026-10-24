# Clivio et al. (2026) deconfounding-score experiments (R)

Reproducibility code of Clivio, D'Amour, Franks, Bruns-Smith, Holmes and Feller
(AISTATS 2026), with extensions by X. Meng (Aug 2026): oracle deconfounding
(`oracle_utils.R`, `ReducerSims_oracle.R`, `ProcessResultsSimulated_oracle.R`,
`run_simulated_oracle.sh`), an analytical score-ratio reduction for the linear
Gaussian DGP (`score_ratio_dr.R`) and direction diagnostics
(`diagnostic_directions.R`).

The folder is self-contained. Scripts use paths relative to it (`data/`,
`results/`) and the `run_*.sh` scripts `cd` here first, so they can be launched
from anywhere, e.g. `bash experiments/clivio/run_simulated.sh`. The `Rscript
Process*.R` steps below must be run from this folder (the semi-synthetic step
called `ProcessResults.R` below is `ProcessResultsSemiSynthetic.R`). The estimators in
`utilities.R` / `ReducerSims.R` (ATT estimation with deconfounding scores) are
the starting point for the downstream causal evaluation planned for the journal
version.

Data: `data/ihdp.RData` (IHDP, included); ACIC 2016 comes from the R package
`aciccomp2016`; HC-MNIST is downloaded through `torchvision::mnist_dataset`.

## Original instructions

1) Delete all files in `results/`; this folder contains individual runs' results and runs are not re-run if their corresponding results are already saved in the corresponding folder.
2) In each `bash run_[...].sh` file, in the line `cat "$COMBO_FILE" | parallel --line-buffer -j X --colsep ' ' '`, `X` is the number of parallel threads. Please adapt `X` to your computational resources. WARNING: HC-MNIST can be computationally intensive, thus the default lower value of `X` of 1 in `bash run_hcmnist.sh`.
3) Run each `bash run_[..].sh` file.
4) For simulated datasets, retrieve results through `Rscript ProcessResultsSimulated.R`. It prints a tabular given a table of results as in Table 1, and saves a figure as in Figure 2 in `combined_plot.pdf`. For semi-synthetic datasets, retrieve results through `Rscript ProcessResults.R`. It prints a tabular given a table of results as in Table 2. The tabulars can be visualized in LaTeX by defining `\newcommand{\red}[1]{\textcolor{red}{#1}}` and `\newcommand{\green}[1]{\textcolor{teal}{#1}}`.

Sources: 
- https://github.com/afranks86/reducer (original code)
- https://github.com/anndvision/quince (HCMNIST)
