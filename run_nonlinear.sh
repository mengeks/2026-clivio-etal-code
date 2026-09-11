#!/bin/bash
# Run nonlinear DR experiments on Aries GPU.
# Usage: bash run_nonlinear.sh [gpu_id]
#
# Experiments 2 (rank-1) and 3 (rank-2), p=30, n=500, 3 seeds.
# Results saved to results/sr_nn_nonlinear_rank{1,2}.csv

GPU=${1:-0}

cd "$(dirname "$0")"

echo "=== Nonlinear DR experiments on GPU $GPU ==="
echo "p=30, q=5, n=500, noise=0.5, s_T=1.5"
echo "Seeds: 1,2,3  |  Prop epochs: 400  |  ISM epochs: 400"
date

CUDA_VISIBLE_DEVICES=$GPU python3 score_ratio_nonlinear.py 2>&1

echo
echo "Done."
date
