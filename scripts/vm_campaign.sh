#!/usr/bin/env bash
# One-command VM campaign: Stage E + the confound-free H2 rerun + analysis + paper.
#
#   ./scripts/vm_campaign.sh            # detaches itself; safe to close the SSH session
#   tail -f logs/vm_campaign.log        # follow progress
#
# Env: CORES (measurement cores, default "0 1 2 3 4 5 6 7"), BUILD_THREADS (default 16),
#      EXPECT_HOST (default the study VM 1d3d9b66e7d3e1ac; set EXPECT_HOST=any to override).
#
# The GB10 pilot of 2026-11-17 died ~3 h in when its launching shell closed; this script
# re-executes itself under setsid+nohup so a dropped connection cannot kill the campaign.
set -u
cd "$(dirname "$0")/.."
LOG=logs/vm_campaign.log
mkdir -p logs

if [ "${VM_CAMPAIGN_DETACHED:-0}" != "1" ]; then
  VM_CAMPAIGN_DETACHED=1 setsid nohup "$0" "$@" >> "$LOG" 2>&1 < /dev/null &
  echo "detached as PID $!; follow with: tail -f $LOG"
  exit 0
fi

PY=./.venv/bin/python
EXPECT_HOST=${EXPECT_HOST:-1d3d9b66e7d3e1ac}
export CORES=${CORES:-"0 1 2 3 4 5 6 7"}
export BUILD_THREADS=${BUILD_THREADS:-16}
step() { echo "=== $(date -Is) $* ==="; }

step "preflight"
if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev --extra hnswlib || { echo "uv sync failed"; exit 1; }
fi
git log --oneline -1
HOST=$($PY -c "from annstudy.hardware import snapshot; print(snapshot()['hardware_id'])")
echo "host=$HOST expected=$EXPECT_HOST cores=[$CORES] build_threads=$BUILD_THREADS"
if [ "$EXPECT_HOST" != "any" ] && [ "$HOST" != "$EXPECT_HOST" ]; then
  echo "REFUSING: this is not the study host (set EXPECT_HOST=any to override)"; exit 2
fi
if [ -n "${ANNSTUDY_ANALYSIS_HARDWARE_ID:-}" ]; then
  echo "REFUSING: ANNSTUDY_ANALYSIS_HARDWARE_ID is set; never set it while measuring"; exit 2
fi
free -g | head -2
nvidia-smi >/dev/null 2>&1 && echo "note: a GPU is present; check nothing else is loading the host"
uptime
$PY -m pytest -q || { echo "tests failed; aborting"; exit 1; }

step "Stage E (truncate vs quantize)"
./scripts/stage_e.sh || { echo "Stage E failed"; exit 1; }

step "H2 rerun (stage_c_h2)"
./scripts/stage_c_h2.sh || { echo "H2 rerun failed"; exit 1; }

step "analysis + paper"
$PY -m annstudy.analysis.run_all
$PY -m annstudy.paper
$PY -m annstudy.paper --check-only

step "done — commit results/ reports/ paper/ figures/ logs/ and push"
