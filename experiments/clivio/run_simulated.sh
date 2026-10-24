#!/usr/bin/env bash
cd "$(dirname "$0")"  # Clivio scripts use paths relative to this folder

ESTIMAND="ATT"
REG_ALPHA_ARR=(0 1)

# Name of the file that will hold all parameter combos
COMBO_FILE="param_combos.txt"
rm -f "$COMBO_FILE"

# -----------------------------------------------------------------------------
# 1) Build a list of all parameter combos and store in $COMBO_FILE
# -----------------------------------------------------------------------------


for (( i=1; i<4+1; i++ )); do
  for y_alpha in "${REG_ALPHA_ARR[@]}"; do
    for t_alpha in "${REG_ALPHA_ARR[@]}"; do
        echo "simulated $i $y_alpha $t_alpha" >> "$COMBO_FILE"
    done
  done
done
      

# -----------------------------------------------------------------------------
# 2) Process these combos in parallel, using CPU cores
# -----------------------------------------------------------------------------

cat "$COMBO_FILE" | parallel --line-buffer -j 8 --colsep ' ' '
  dataset={1}
  setting={2}
  y_alpha={3}
  t_alpha={4}

  # Automatically set full_alpha_set
  if [ "$setting" -eq 4 ]; then
    full_alpha_set=TRUE
  else
    full_alpha_set=FALSE
  fi

  echo "Running: dataset=$dataset, setting=$setting, y_alpha=$y_alpha, t_alpha=$t_alpha, full_alpha=$full_alpha_set"

  Rscript ReducerSims.R \
    --dataset=$dataset \
    --setting=$setting \
    --y_alpha=$y_alpha \
    --t_alpha=$t_alpha \
    --estimand='"$ESTIMAND"' \
    --full_alpha_set=$full_alpha_set \
    --times 1 \
    --iters 100
'

# Clean up
rm -f "$COMBO_FILE"
