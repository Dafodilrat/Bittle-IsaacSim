#!/bin/bash
set -e

cd "$ISAACSIM_PATH"/alpha
echo "[Docker Entrypoint] Pulling latest code..."
git pull || true  # allow fail if not a git repo
# # Run your Python script (e.g., test.py) using Isaac Sim's python
exec LIBGL_ALWAYS_INDIRECT=1 QT_QUICK_BACKEND=software python3 alpha/exts/customView/customView/pyqt_interface.py "$@"