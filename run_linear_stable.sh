#!/bin/bash
# Run canonical linear stable experiment (5 seeds, hidden=(8,) + hidden=()).
# Usage: bash run_linear_stable.sh [gpu_id]
# Output: results/sr_nn_linear_stable.csv

GPU=${1:-0}
cd "$(dirname "$0")"

echo "=== Linear stable experiment ==="
echo "p=100, n=500, setting=4  |  5 seeds  |  r_A=1, r_AY=2"
echo "BCE propensity (hidden=(8,) and linear) + ISM joint score"
date

CUDA_VISIBLE_DEVICES=$GPU python3 score_ratio_linear_stable.py 2>&1

echo
echo "Done."
date
