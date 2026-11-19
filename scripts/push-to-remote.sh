#!/usr/bin/env bash
# Push LOCAL code + docs to the remote VM (companion to sync-from-remote.sh, which pulls).
#
# The sync is pull-only and the VM is authoritative for RESULTS, so this pushes ONLY source,
# docs, configs, tests and scripts — never data/, results/, figures/, logs/, reports/, or the
# generated paper/main.md + paper/tables/ (those are produced ON the VM). After pushing, run on
# the VM:  make analysis && make stage-reports && make paper && python -m annstudy.paper --check-only
# then re-sync down with scripts/sync-from-remote.sh.
#
# Usage:
#   ./scripts/push-to-remote.sh <IP|user@host>
#   ./scripts/push-to-remote.sh exouser@149.165.1.2
# Works with just ssh (tar pipe); uses rsync if installed. No --delete, so nothing on the VM is
# removed — locally-deleted files (e.g. the orphaned paper/01-09) must be removed on the VM by hand
# if they exist there (they were local-only here).

set -euo pipefail

DEFAULT_USER=exouser
HOST_ARG=""
for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '3,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) HOST_ARG="$arg" ;;
  esac
done
if [ -n "$HOST_ARG" ]; then
  case "$HOST_ARG" in *@*) REMOTE="$HOST_ARG" ;; *) REMOTE="$DEFAULT_USER@$HOST_ARG" ;; esac
else
  REMOTE="${REMOTE:-exouser@statwiseai-1}"
fi
DEST="${DEST:-/home/exouser/Project/ann}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Only these paths are pushed.
PATHS=(src tests configs scripts Makefile pyproject.toml README.md FEEDBACK.md REMAINING_WORK.md
       paper/README.md)
EXCLUDES=(__pycache__ '*.pyc')

echo "Push : $LOCAL  ->  $REMOTE:$DEST/"
echo "Paths: ${PATHS[*]}"
echo

if command -v rsync >/dev/null 2>&1; then
  rex=(); for e in "${EXCLUDES[@]}"; do rex+=(--exclude "$e"); done
  rsync -avzP "${rex[@]}" "${PATHS[@]/#/$LOCAL/}" "$REMOTE:$DEST/"
else
  echo ">> rsync not found; using ssh + tar pipe"
  tex=(); for e in "${EXCLUDES[@]}"; do tex+=(--exclude="$e"); done
  # Mode letters need the leading dash here: with `-C DIR` first, GNU tar is in getopt mode and
  # a bare `czf` would be read as a filename ("must specify one of -Acdtrux").
  tar -C "$LOCAL" -czf - "${tex[@]}" "${PATHS[@]}" | ssh "$REMOTE" "mkdir -p '$DEST' && tar -C '$DEST' -xzf -"
fi

echo
echo "Pushed. Now on the VM:"
echo "  cd $DEST && make analysis && make stage-reports && make paper && python -m annstudy.paper --check-only"
echo "then pull back:  ./scripts/sync-from-remote.sh <IP>"
