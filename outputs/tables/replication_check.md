# Replication check: package pipeline vs original stable scripts

Original: outputs/results/legacy (Sep 2026, Aries GPU, unseeded TensorFlow).
Rerun: outputs/results/<dataset>/results.csv (this repository, seeded).
The original scripts did not seed TensorFlow or the mini-batch order, so
per-seed values are not expected to match; data, warm starts and baselines
are deterministic and should match exactly.

| dataset | quantity | original | rerun |
|---|---|---|---|
| linear_rank1 | SR cos2 beta | 0.824 [0.796, 0.863] | 0.787 [0.767, 0.818] |
| linear_rank1 | SR cos2 alpha | 0.929 [0.785, 0.971] | 0.932 [0.831, 0.982] |
| linear_rank1 | beta_init_cos2 (original init) | 0.913 [0.888, 0.915] | 0.913 [0.888, 0.915] |
| linear_rank1 | alpha_init_cos2 (original init) | 0.989 [0.986, 0.992] | 0.989 [0.986, 0.992] |
| nonlinear_rank1 | SR cos2 beta | 0.854 [0.775, 0.917] | 0.870 [0.787, 0.920] |
| nonlinear_rank1 | SR cos2 alpha | 0.967 [0.542, 0.986] | 0.947 [0.180, 0.980] |
| nonlinear_rank1 | beta_init_cos2 (original init) | 0.869 [0.758, 0.894] | 0.869 [0.758, 0.894] |
| nonlinear_rank1 | alpha_init_cos2 (original init) | 0.393 [0.019, 0.579] | 0.393 [0.019, 0.579] |
| linear_rank1 | package, seeds 1-20, beta | - | 0.803 [0.743, 0.840] |
| linear_rank1 | package, seeds 1-20, alpha | - | 0.916 [0.792, 0.982] |
| nonlinear_rank1 | package, seeds 1-20, beta | - | 0.862 [0.520, 0.921] |
| nonlinear_rank1 | package, seeds 1-20, alpha | - | 0.955 [0.021, 0.980] |
| linear_rank1 | original script rerun here, beta | - | 0.782 [0.774, 0.831] |
| linear_rank1 | original script rerun here, alpha | - | 0.956 [0.864, 0.975] |
| nonlinear_rank1 | original script rerun here, beta | - | 0.893 [0.796, 0.935] |
| nonlinear_rank1 | original script rerun here, alpha | - | 0.960 [0.500, 0.985] |
