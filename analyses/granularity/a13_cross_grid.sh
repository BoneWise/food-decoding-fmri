#!/bin/bash
cd "$(dirname "$0")"
GR=${FOODPREF_GRANULARITY_ROOT:-../../outputs/analyses/granularity}
start=$(date +%s)
for s in 42 43 44 45 46; do for k in 1 2 3 4 5 6; do
  [ -f "$GR"/results_summary/cross/seed${s}_ktest${k}.csv ] && continue
  now=$(date +%s); [ $((now-start)) -gt 125 ] && { echo "budget"; exit 0; }
  ${PYTHON:-python3} a07_cross.py --seed $s --ktest $k 2>&1 | grep -v -i "warn\|nanmean"
done; done
echo "CROSS COMPLETE"
