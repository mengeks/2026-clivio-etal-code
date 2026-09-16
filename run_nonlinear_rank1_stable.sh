#!/bin/bash
# Run canonical nonlinear rank-1 stable experiment.
# Usage: bash run_nonlinear_rank1_stable.sh [gpu_id]
# Output: results/sr_nn_nonlinear_rank1_stable.csv

GPU=${1:-0}
cd "$(dirname "$0")"

echo "=== Nonlinear rank-1 stable experiment ==="
echo "DGP: Y=(a'V)^2+eps, logit e=s_T[(b'V)^2-1], p=30, n=500, s_T=1.5"
echo "Init: SAVE warm start  |  r_A=1 BCE  |  r_AY=2 ISM Stiefel"
echo "3 seeds → if PASS extend to 5; + random-init comparison (seeds 1-2)"
date

CUDA_VISIBLE_DEVICES=$GPU python3 score_ratio_nonlinear_rank1_stable.py 2>&1

echo
echo "Done."
date
