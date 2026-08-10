#!/usr/bin/env bash

ESTIMAND="ATT"
REG_ALPHA_ARR=(0 1)

COMBO_FILE="param_combos_oracle.txt"
rm -f "$COMBO_FILE"

for (( i=1; i<4+1; i++ )); do
  for y_alpha in "${REG_ALPHA_ARR[@]}"; do
    for t_alpha in "${REG_ALPHA_ARR[@]}"; do
        echo "simulated $i $y_alpha $t_alpha" >> "$COMBO_FILE"
    done
  done
done


cat "$COMBO_FILE" | parallel --line-buffer -j 8 --colsep ' ' '
  dataset={1}
  setting={2}
  y_alpha={3}
  t_alpha={4}

  if [ "$setting" -eq 4 ]; then
    full_alpha_set=TRUE
  else
    full_alpha_set=FALSE
  fi

  echo "Running oracle: dataset=$dataset, setting=$setting, y_alpha=$y_alpha, t_alpha=$t_alpha, full_alpha=$full_alpha_set"

  Rscript ReducerSims_oracle.R \
    --dataset=$dataset \
    --setting=$setting \
    --y_alpha=$y_alpha \
    --t_alpha=$t_alpha \
    --estimand='"$ESTIMAND"' \
    --full_alpha_set=$full_alpha_set \
    --times 1 \
    --iters 100
'

rm -f "$COMBO_FILE"
