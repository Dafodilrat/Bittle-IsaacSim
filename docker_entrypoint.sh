#!/bin/bash
set -e

cd $ISAACSIM_PATH/alpha
echo "[Docker Entrypoint] Pulling latest code..."
git pull || true  # allow fail if not a git repo

# Run whatever script you want next
exec "$@" 
