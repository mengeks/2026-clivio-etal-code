#!/usr/bin/env bash
# Controlled A-vs-B comparisons behind research-notes/A*.md: every arm is the
# main pipeline with one setting changed (configs/experiments/ablations/).
# Arms identical to the main configuration reuse its cached jobs.
#
#   bash experiments/run_research_notes.sh                      # all notes
#   bash experiments/run_research_notes.sh A03_initialization   # selected notes
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
if [[ $# -gt 0 ]]; then
    CONFIGS=$(printf 'configs/experiments/ablations/%s.yaml ' "$@")
else
    CONFIGS=$(ls configs/experiments/ablations/A*.yaml)
fi
$PY scripts/run_simulation.py $CONFIGS --jobs "${JOBS:-10}" --threads "${THREADS:-2}"
$PY scripts/make_paper_tables.py
