#!/usr/bin/env bash
# Overwrite the server's Python sources with this working copy: the top-level
# entrypoints and every .py under the source packages. Nothing else moves --
# no configs, no logs, no data, no venv, no non-.py file.
#
#   ./sync_code.sh irmers@myserver           # overwrite changed .py files
#   ./sync_code.sh irmers@myserver -n        # show what would change
#   ./sync_code.sh irmers@myserver --delete  # also remove remote .py files
#                                            # that no longer exist locally
#
# The host may also come from $METRIC_LEARNING_SERVER, the remote repository
# root from $METRIC_LEARNING_REMOTE_ROOT.
set -euo pipefail

SRV=""
DRY=()
DELETE=()
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY=(--dry-run) ;;
        --delete) DELETE=(--delete) ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) SRV="$arg" ;;
    esac
done
SRV="${SRV:-${METRIC_LEARNING_SERVER:-}}"
if [ -z "$SRV" ]; then
    echo "usage: $0 [user@]host [-n] [--delete]   (or set METRIC_LEARNING_SERVER)" >&2
    exit 2
fi

ROOT="${METRIC_LEARNING_REMOTE_ROOT:-/home/irmers/metric-learning}"
cd "$(dirname "$0")"

# Source roots holding .py files. server/ and data/ hold none, and
# react_trial_visualizer/ is the JS studio, so none of them are listed.
PACKAGES=(training utils models losses exporter scripts tests)
ENTRYPOINTS=(*.py)

# One password prompt: every later ssh/rsync rides this master connection.
CONTROL_DIR="$(mktemp -d)"
CONTROL="$CONTROL_DIR/cm"
cleanup() {
    ssh -o ControlPath="$CONTROL" -O exit "$SRV" 2>/dev/null || true
    rm -rf "$CONTROL_DIR"
}
trap cleanup EXIT

echo "==> opening SSH connection to $SRV"
ssh -o ControlPath="$CONTROL" -o ControlMaster=yes -o ControlPersist=600 -N -f "$SRV"

RSH="ssh -o ControlPath=$CONTROL"

# First match wins, so __pycache__ is pruned before '*/' can descend into it.
# --delete leaves everything the final --exclude skips untouched, so it can
# only ever remove a remote .py file that is gone locally.
FILTER=(
    --prune-empty-dirs
    --exclude='__pycache__/'
    --include='*/'
    --include='*.py'
    --exclude='*'
)

echo "==> entrypoints: ${ENTRYPOINTS[*]}"
rsync -az --info=stats1 "${DRY[@]}" -e "$RSH" \
    "${ENTRYPOINTS[@]}" "$SRV:$ROOT/"

for package in "${PACKAGES[@]}"; do
    echo "==> $package/"
    rsync -az --info=stats1 "${DRY[@]}" "${DELETE[@]}" -e "$RSH" "${FILTER[@]}" \
        "$package/" "$SRV:$ROOT/$package/"
done

echo
if [ ${#DRY[@]} -gt 0 ]; then
    echo "dry run only; nothing was written to $SRV"
else
    echo "synced to $SRV:$ROOT"
    [ ${#DELETE[@]} -gt 0 ] && echo "stale remote .py files were removed"
fi
