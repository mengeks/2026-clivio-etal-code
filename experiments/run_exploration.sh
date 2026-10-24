#!/usr/bin/env bash
# Re-run the historical exploration scripts (work-journal rounds 1-4 and the two
# original "stable" scripts) in parallel. They supply the historical evidence
# cited in research notes A01, A04, A05, A06 and the cross-check of the package
# pipeline against the original stable scripts.
#
#   bash experiments/run_exploration.sh            # all scripts
#   bash experiments/run_exploration.sh rescue     # selected scripts
#
# Outputs: outputs/results/exploration/*.csv   Logs: outputs/logs/exploration/
# The scripts do not seed TensorFlow globally, so reruns vary from run to run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1
SCRIPTS=${*:-rescue final bernoulli nonlinear linear_stable nonlinear_rank1_stable}
mkdir -p outputs/logs/exploration
for s in $SCRIPTS; do
  $PY scripts/exploration/score_ratio_$s.py > outputs/logs/exploration/$s.log 2>&1 &
  echo "started $s (pid $!)"
done
wait
echo "all exploration scripts finished"
