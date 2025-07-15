#!/bin/bash
set -e

cd "$ISAACSIM_PATH"/alpha
echo "[Docker Entrypoint] Pulling latest code..."
git pull || true  # allow fail if not a git repo
# # Run your Python script (e.g., test.py) using Isaac Sim's python
# export LIBGL_ALWAYS_INDIRECT=1 
# export QT_QUICK_BACKEND=software
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

exec python3 exts/customView/customView/pyqt_interface.py "$@"
#  