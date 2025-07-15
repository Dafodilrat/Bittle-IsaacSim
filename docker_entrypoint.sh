#!/bin/bash
set -e

cd "$ISAACSIM_PATH"/alpha
echo "[Docker Entrypoint] Pulling latest code..."
git pull || true  # allow fail if not a git repo

exec python3 exts/customView/customView/pyqt_interface.py "$@"
#  