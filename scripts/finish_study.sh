#!/usr/bin/env bash
# Run the remainder of the study to completion, in order, then produce every report.
#
# Resumable and idempotent: each stage skips work already recorded, so re-running this after
# an interruption continues rather than repeating. Safe to launch with nohup.
#
#   nohup ./scripts/finish_study.sh > logs/finish_study.log 2>&1 &
set -u
PY=./.venv/bin/python
LOG=logs/finish_study.log

say() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- Stage C (filtering)
say "=== Stage C: filtered search ==="
# Wait out an already-running driver rather than starting a second one.
while pgrep -f 'scripts/stage_c.s[h]' > /dev/null; do sleep 60; done

expected=$($PY - <<'PLAN'
from annstudy.filter_runner import load_stage_c_config, enumerate_filter_jobs
print(len(enumerate_filter_jobs(load_stage_c_config())))
PLAN
)
done_now=$(ls results/raw/filter_runs/*.parquet 2>/dev/null | wc -l)
say "Stage C jobs: $done_now of $expected recorded"

attempt=0
while [ "$done_now" -lt "$expected" ] && [ "$attempt" -lt 3 ]; do
  attempt=$((attempt+1))
  say "Stage C pass $attempt (resumes; completed jobs are skipped)"
  ./scripts/stage_c.sh >> logs/stage_c_driver.log 2>&1
  done_now=$(ls results/raw/filter_runs/*.parquet 2>/dev/null | wc -l)
  say "Stage C jobs now: $done_now of $expected"
done

$PY -m annstudy.filter_runner --consolidate 2>&1 | tail -2
say "Stage C finished with $done_now/$expected jobs"

# ------------------------------------------------------------------ Stage D (updates)
say "=== Stage D: insert study ==="
$PY -m annstudy.update_runner --config configs/stage_d.yaml 2>&1 | tail -5 || \
  say "WARNING: Stage D reported a failure; continuing so reports still regenerate"

# ----------------------------------------------------------------------- all reports
say "=== Reports ==="
for cfg in configs/stage_a.yaml configs/stage_b.yaml; do
  $PY -m annstudy.coordinate --config "$cfg" --phase report 2>&1 | grep -E 'Wrote' || true
done

say "analysis (tables, figures, findings)"
$PY -m annstudy.analysis.run_all 2>&1 | grep -E '\[ok\]|\[skip\]|\[ERROR\]' || true

say "acceptance checks"
$PY -m annstudy.acceptance --checks data 2>&1 | tail -4 || true

say "budget actuals"
$PY -m annstudy.budget --cores 16 > /dev/null 2>&1 || true

say "paper"
$PY -m annstudy.paper 2>&1 | tail -3 || true

say "=== DONE: all stages run and reports regenerated ==="
