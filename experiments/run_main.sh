#!/usr/bin/env bash
# Main paper results: linear and nonlinear rank-1 experiments (5 seeds each),
# then the paper tables and LaTeX number macros.
#
#   bash experiments/run_main.sh            # full run
#   bash experiments/run_main.sh --fast     # 1 seed, 20 epochs (smoke test)
#
# JOBS (parallel workers, default 10) and THREADS (per worker, default 2)
# can be set in the environment; PYTHON selects the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/run_simulation.py \
    configs/experiments/linear_rank1.yaml configs/experiments/nonlinear_rank1.yaml \
    --jobs "${JOBS:-10}" --threads "${THREADS:-2}" "$@"
if [[ " $* " != *" --fast "* ]]; then
    $PY scripts/make_paper_tables.py
fi
