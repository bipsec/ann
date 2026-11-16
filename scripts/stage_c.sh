#!/usr/bin/env bash
# Stage C driver: one cell at a time, sharded across pinned cores, with the shard count
# chosen so each shard's corpus + index fits in memory (see filter_runner.recommended_shards).
# Latency must stay single-threaded, so parallelism comes from shards, never threads.
set -u
# Measured latency is single-threaded: pin BLAS/OpenMP pools to one thread in every shard.
# (The original script set none of these, so numpy's BLAS could oversubscribe a pinned core.)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=./.venv/bin/python
CFG=${CFG:-configs/stage_c.yaml}
LOG=logs/stage_c.log

$PY -m annstudy.filter_runner --plan --config "$CFG" | tee -a "$LOG"

CFG="$CFG" $PY - <<'PLAN' > /tmp/stage_c_plan.txt
import os
from annstudy.filter_runner import plan_shards
for c in plan_shards(os.environ["CFG"]):  # was plan_shards(): always stage_c.yaml
    print(c["N"], c["D"], c["shards"])
PLAN

while read -r N D SHARDS; do
  echo "=== cell N=$N D=$D with $SHARDS shard(s) ===" | tee -a "$LOG"
  pids=()
  for i in $(seq 0 $((SHARDS-1))); do
    taskset -c "$i" $PY -m annstudy.filter_runner --config "$CFG" \
      --only-N "$N" --only-D "$D" --shard "$i" --num-shards "$SHARDS" \
      --gt-threads 16 >> "logs/stage_c_N${N}_D${D}_s${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== cell N=$N D=$D done ===" | tee -a "$LOG"
done < /tmp/stage_c_plan.txt

$PY -m annstudy.filter_runner --consolidate | tee -a "$LOG"
