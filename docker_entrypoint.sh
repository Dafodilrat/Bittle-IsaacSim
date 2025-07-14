#!/bin/bash
set -e

cd "$ISAACSIM_PATH"
echo "[Docker Entrypoint] Pulling latest code..."
git pull || true  # allow fail if not a git repo
ls -a

# Run your Python script (e.g., test.py) using Isaac Sim's python
exec python3 alpha/exts/customView/customView/pyqt_interface.py "$@"