#!/bin/bash
# run a06 over all conditions until ~150 s are used
cd "$(dirname "$0")"
GR=${FOODPREF_GRANULARITY_ROOT:-../../outputs/analyses/granularity}
start=$(date +%s)
for c in "1 42" $(for s in 42 43 44 45 46; do for k in 2 3 4 5 6; do echo "$k:$s"; done; done); do
  k=${c%%[: ]*}; s=${c##*[: ]}
  [ -f "$GR"/results/$(printf "k%02d_seed%s" $k $s)/fit_k01/weights.npy ] && continue
  now=$(date +%s); [ $((now-start)) -gt 110 ] && { echo "budget"; exit 0; }
  ${PYTHON:-python3} a06_weights.py --k $k --seed $s --work "${WORK:?set WORK to a scratch directory}" 2>&1 | grep -v RuntimeWarning | grep -v nanmean
done
echo "WEIGHTS COMPLETE"
