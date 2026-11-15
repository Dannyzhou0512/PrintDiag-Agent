#!/usr/bin/env bash
# Stop a background-started PrintDiag cloud inference service.
# Foreground services are stopped with Ctrl+C in their terminal instead.
set -euo pipefail

APP_DIR="${PRINTDIAG_CLOUD_APP_DIR:-/data/projects/printdiag-cloud-inference}"
PID_FILE="$APP_DIR/server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "no PID file at $PID_FILE"
  echo "if the service runs in the foreground, press Ctrl+C in that terminal."
  exit 1
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "process $PID is not running; removing stale PID file"
  rm -f "$PID_FILE"
  exit 1
fi

kill "$PID"
for _ in $(seq 1 20); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "stopped pid $PID"
    exit 0
  fi
  sleep 1
done
echo "pid $PID did not exit within 20s; inspect logs before forcing anything" >&2
exit 1
