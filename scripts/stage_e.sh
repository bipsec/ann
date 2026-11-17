#!/usr/bin/env bash
# Stage E driver (configs/stage_e_tradeoff.yaml): truncate vs quantize at matched bytes.
#
#   CORES="0 1 2 3 4 5 6 7" ./scripts/stage_e.sh
#
# 1) Build every arm's index and re-rank store multi-threaded (unmeasured).
# 2) Measure on pinned, single-threaded shards (one per core), arms split round-robin.
set -u
PY=./.venv/bin/python
CFG=${CFG:-configs/stage_e_tradeoff.yaml}
CORES=${CORES:-"0 1 2 3 4 5 6 7"}
BUILD_THREADS=${BUILD_THREADS:-16}
LOG=logs/stage_e.log
read -r -a core_arr <<< "$CORES"
SHARDS=${#core_arr[@]}

echo "=== $(date -Is) Stage E: $CFG, $SHARDS shards on cores: $CORES ===" | tee -a "$LOG"
OMP_NUM_THREADS=$BUILD_THREADS $PY -m annstudy.tradeoff --config "$CFG" --build-only \
  --build-threads "$BUILD_THREADS" 2>&1 | tee -a "$LOG"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  taskset -c "${core_arr[$i]}" $PY -m annstudy.tradeoff --config "$CFG" \
    --shard "$i" --num-shards "$SHARDS" >> "logs/stage_e_s${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m annstudy.tradeoff --consolidate | tee -a "$LOG"
echo "=== $(date -Is) Stage E done ===" | tee -a "$LOG"
