#!/bin/bash
set -e

cd "$ISAACSIM_PATH"
echo "[Docker Entrypoint] Pulling latest code..."
git pull || true  # allow fail if not a git repo

# Run your Python script (e.g., test.py) using Isaac Sim's python
exec "$ISAACSIM_PATH/python.sh" "alpha/exts/customView/customView/pyqt_interface.py" "$@"