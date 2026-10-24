# Exploration scripts (historical)

These are the development scripts behind `work-journal.tex`. They are kept
runnable for provenance. The paper pipeline does not use them: it lives in
`src/causal_embedding/` and runs through `scripts/run_simulation.py`.

Each script is self-contained. Apart from three changes, the code is the code
that produced the September 2026 results:

- `ScoreRatioNet` is imported from `causal_embedding.models.score_ratio_reduction`
  instead of through a `sys.path` entry for the parent folder.
- Sibling scripts import each other from this folder.
- Outputs go to `outputs/results/exploration/`.

The original outputs are kept in `outputs/results/legacy/`.

| Script | Journal round | What it tests | Output | Cited in |
|---|---|---|---|---|
| `score_ratio_nn_diagnostic.py` | 0 | original estimator, Ĥ_Y − Ĥ_X, Hutchinson trace | `sr_nn_diagnostic_p100.csv` | A09 (history) |
| `score_ratio_rescue.py` | 1 | ISM generic MLP, r′=1, random init, exact trace, per-sample H₀ | `sr_nn_rescue_p100.csv` | A01, A04 |
| `score_ratio_final.py` | 2 | ISM generic MLP, r′=5, random vs oracle warm start, 500 epochs | `sr_nn_final_p100.csv` | A04 |
| `score_ratio_bernoulli.py` | 3 | ISM Bernoulli MLP (exact Laplacian), r′=5, oracle warm start | `sr_nn_bernoulli_p100.csv`, `sr_nn_bernoulli_training.csv` | A04, A06 |
| `score_ratio_nonlinear.py` | 4 | nonlinear rank-1/rank-2 DGPs, random init, ISM r′=3 joint score | `sr_nn_nonlinear_rank{1,2}.csv` | A05, Exp. 3 |
| `score_ratio_linear_stable.py` | stable | original linear stable pipeline (LASSO init) | `sr_nn_linear_stable.csv` | replication check |
| `score_ratio_nonlinear_rank1_stable.py` | stable | original nonlinear stable pipeline (SAVE init) | `sr_nn_nonlinear_rank1_stable.csv` | replication check |
| `plot_sr_rescue.R`, `plot_sr_nn_diagnostic.R` | 0–1 | plots of the round 0–1 CSVs | `*.pdf` | — |

Run them all in parallel: `bash experiments/run_exploration.sh`. To run
selected scripts, name them, e.g. `bash experiments/run_exploration.sh rescue final`.
None of these scripts seed TensorFlow or the mini-batch order, so reruns
differ from each other and from the legacy CSVs. The DGP, the warm starts and
the baselines are deterministic.
