#!/usr/bin/env bash
# H2 rerun driver (configs/stage_c_h2.yaml): ONE campaign over every D, with a FIXED shard
# count and D-interleaved job order, so dimension is not confounded with session or with the
# number of busy cores (the original Stage C ran cells sequentially with 16/6/12/3 shards).
#
#   CORES="0 1 2 3 4 5 6 7" ./scripts/stage_c_h2.sh
#
# Index builds (stage-local, repeat_builds per cell) happen first, multi-threaded and
# unmeasured, so no measured job ever shares the machine with a build.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=./.venv/bin/python
CFG=${CFG:-configs/stage_c_h2.yaml}
CORES=${CORES:-"0 1 2 3 4 5 6 7"}
BUILD_THREADS=${BUILD_THREADS:-16}
LOG=logs/stage_c_h2.log
read -r -a core_arr <<< "$CORES"
SHARDS=${#core_arr[@]}

echo "=== $(date -Is) H2 campaign: $CFG, $SHARDS shards on cores: $CORES ===" | tee -a "$LOG"

# 1) Build every (N, D, index, repeat) index up front, sequentially, with all threads.
OMP_NUM_THREADS=$BUILD_THREADS CFG="$CFG" BT="$BUILD_THREADS" $PY - <<'BUILD' 2>&1 | tee -a "$LOG"
import os, numpy as np
from annstudy.filter_runner import load_stage_c_config, build_or_load_index
from annstudy.coordinate import passages_path
cfg = load_stage_c_config(os.environ["CFG"])
for N in cfg["cells"]["N"]:
    for D in cfg["cells"]["D"]:
        x = np.asarray(np.memmap(passages_path(D), dtype=np.float32, mode="r", shape=(N, D)))
        for it in cfg["indexes"]:
            for r in range(int(cfg.get("repeat_builds", 1))):
                build_or_load_index(cfg, N, D, it, r, x, build_threads=int(os.environ["BT"]))
BUILD

# 2) Filtered ground truth for every predicate, before any shard starts timing.
OMP_NUM_THREADS=$BUILD_THREADS $PY -m annstudy.filter_runner --config "$CFG" --precompute-gt \
  --gt-threads "$BUILD_THREADS" 2>&1 | tee -a "$LOG"

# 3) Measure: fixed shard count, each shard pinned to its own core, all D interleaved.
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  taskset -c "${core_arr[$i]}" $PY -m annstudy.filter_runner --config "$CFG" \
    --shard "$i" --num-shards "$SHARDS" --gt-threads 1 \
    >> "logs/stage_c_h2_s${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -m annstudy.filter_runner --consolidate | tee -a "$LOG"
echo "=== $(date -Is) H2 campaign done ===" | tee -a "$LOG"
