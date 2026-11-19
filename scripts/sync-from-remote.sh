#!/usr/bin/env bash
# Pull results/reports/figures/logs/code from the remote VM into this local repo.
#
# Works with just `ssh` (tar-pipe fallback) and uses `rsync` automatically if present
# (faster, incremental). Excludes the huge/local-only dirs: data/, .venv/, .git/,
# per-run runner logs, and Python caches.
#
# Usage:
#   ./scripts/sync-from-remote.sh <IP|user@host> [--with-data]
#
# Examples:
#   ./scripts/sync-from-remote.sh 149.165.1.2                 # code + results (user defaults to exouser)
#   ./scripts/sync-from-remote.sh exouser@149.165.1.2         # explicit user@host
#   ./scripts/sync-from-remote.sh 149.165.1.2 --with-data     # also pull data/gt + samples
#   ./scripts/sync-from-remote.sh statwiseai-1                # an ssh-config alias also works
#
# The host may be a bare IP/hostname (user defaults to exouser) or user@host. You can still
# override via env (REMOTE=..., SRC=...) if you prefer. No --delete is used, so local-only
# files (e.g. paper/ drafts not yet on the remote) are preserved.

set -euo pipefail

DEFAULT_USER=exouser
WITH_DATA="${WITH_DATA:-0}"
HOST_ARG=""
for arg in "$@"; do
  case "$arg" in
    --with-data) WITH_DATA=1 ;;
    -h|--help)
      sed -n '3,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) HOST_ARG="$arg" ;;
  esac
done

if [ -n "$HOST_ARG" ]; then
  case "$HOST_ARG" in
    *@*) REMOTE="$HOST_ARG" ;;
    *)   REMOTE="$DEFAULT_USER@$HOST_ARG" ;;
  esac
else
  REMOTE="${REMOTE:-exouser@statwiseai-1}"   # fallback if no arg and no env override
fi

SRC="${SRC:-/home/exouser/Project/ann}"
# Local destination = the repo root (the parent of this script's dir), regardless of CWD.
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXCLUDES=(data .venv .git __pycache__ '*.pyc' logs/runner)

# Protected paths (.sync-protect): local-authoritative files a pull must never overwrite.
# Excluded from the transfer AND backed up/restored around it, so a pattern that fails to
# match still cannot clobber them.
PROTECT=()
if [ -f "$LOCAL/.sync-protect" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; line="$(echo "$line" | tr -d '\r' | xargs)"
    [ -n "$line" ] && PROTECT+=("${line%/}")
  done < "$LOCAL/.sync-protect"
fi
BACKUP="$(mktemp -d)"
for p in "${PROTECT[@]}"; do
  if [ -e "$LOCAL/$p" ]; then
    mkdir -p "$BACKUP/$(dirname "$p")" && cp -a "$LOCAL/$p" "$BACKUP/$p"
  fi
done

echo "Remote : $REMOTE:$SRC/"
echo "Local  : $LOCAL/"
echo "Mode   : $([ "$WITH_DATA" = 1 ] && echo 'code+results+data(gt/samples)' || echo 'code+results')"
[ ${#PROTECT[@]} -gt 0 ] && echo "Protect: ${PROTECT[*]}"
echo

if command -v rsync >/dev/null 2>&1; then
  echo ">> using rsync (incremental)"
  rsync_excludes=()
  for e in "${EXCLUDES[@]}"; do rsync_excludes+=(--exclude "$e"); done
  for p in "${PROTECT[@]}"; do rsync_excludes+=(--exclude "/$p"); done
  rsync -avzP "${rsync_excludes[@]}" "$REMOTE:$SRC/" "$LOCAL/"
  if [ "$WITH_DATA" = 1 ]; then
    # data/gt (ground truth) + samples/split, but never the multi-GB embeddings or built indexes.
    rsync -avzP --exclude 'emb/' --exclude 'indexes/' --exclude 'filters/' \
      "$REMOTE:$SRC/data/" "$LOCAL/data/"
  fi
else
  echo ">> rsync not found; using ssh + tar pipe (full pull of the selected paths, not incremental)"
  tar_excludes=()
  for e in "${EXCLUDES[@]}"; do tar_excludes+=(--exclude="./$e"); done
  for p in "${PROTECT[@]}"; do tar_excludes+=(--exclude="./$p"); done
  # shellcheck disable=SC2029
  ssh "$REMOTE" "cd '$SRC' && tar czf - ${tar_excludes[*]} ." | tar xzf - -C "$LOCAL"
  if [ "$WITH_DATA" = 1 ]; then
    ssh "$REMOTE" "cd '$SRC' && tar czf - --exclude=./data/emb --exclude=./data/indexes --exclude=./data/filters ./data" | tar xzf - -C "$LOCAL"
  fi
fi

# Restore protected paths from the pre-pull backup (belt and braces over the excludes).
for p in "${PROTECT[@]}"; do
  if [ -e "$BACKUP/$p" ]; then
    rm -rf "${LOCAL:?}/$p" && mkdir -p "$LOCAL/$(dirname "$p")" && cp -a "$BACKUP/$p" "$LOCAL/$p"
    echo "kept local: $p"
  fi
done
rm -rf "$BACKUP"

echo
echo "Done. Pulled into $LOCAL"

# ---------------------------------------------------------------------------
# One-time ssh setup so `statwiseai-1` resolves (put in ~/.ssh/config):
#
#   Host statwiseai-1
#       HostName <PUBLIC_IP_OF_THE_VM>
#       User exouser
#       IdentityFile ~/.ssh/<your_private_key>
#
# Then test:  ssh statwiseai-1 'hostname && pwd'
#
# Optional, for fast incremental syncs, install rsync on Windows (pick one):
#   scoop install rsync        # if you use scoop
#   choco install rsync        # if you use chocolatey (admin shell)
#   pacman -S rsync            # if you have MSYS2
# ---------------------------------------------------------------------------
