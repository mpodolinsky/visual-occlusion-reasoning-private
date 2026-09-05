#!/usr/bin/env bash
# One command: start the GR00T policy server if port 8000 is free, run
# collect.py, then stop only the server this script started.
#
#   ./scripts/groot/run.sh --replan-steps 8 --scene-variant both --num-trials 25
#   ./scripts/groot/run.sh --replan-steps 8 --task-id 0 --episode-index 0
#
# Env (optional):
#   GROOT_PORT (default 8000)  GROOT_SERVER_WAIT_SECS (default 900, model load is slow)
#   COLLECT_PYTHON (default .venv/bin/python)
#   STOP_PI05=1   pkill the pi0.5 feature server first (one GPU can't hold both)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PORT="${GROOT_PORT:-8000}"
WAIT_SECS="${GROOT_SERVER_WAIT_SECS:-900}"
COLLECT_PYTHON="${COLLECT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SERVE_SCRIPT="$REPO_ROOT/scripts/groot/serve.sh"
COLLECT_SCRIPT="$REPO_ROOT/scripts/groot/collect.py"

[[ -x "$COLLECT_PYTHON" ]] || { echo "COLLECT_PYTHON not executable: $COLLECT_PYTHON" >&2; exit 1; }

port_open() { "$COLLECT_PYTHON" - "$PORT" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(0.5)
try:
    sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    s.close()
PY
}

if [[ "${STOP_PI05:-0}" == "1" ]]; then
  echo "Stopping pi0.5 feature server (STOP_PI05=1)"
  pkill -f serve_pi05_with_features 2>/dev/null || true
  sleep 3
  pkill -9 -f serve_pi05_with_features 2>/dev/null || true
fi

mkdir -p "$REPO_ROOT/outputs/groot/logs"
SERVER_LOG="$REPO_ROOT/outputs/groot/logs/groot_server.log"
STARTED_SERVER=0
SERVER_PID=""
cleanup() {
  if [[ "$STARTED_SERVER" == "1" && -n "$SERVER_PID" ]]; then
    echo "Stopping GR00T server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_open; then
  echo "Policy server already on :$PORT -- reusing it (will not stop it)"
else
  echo "Starting GR00T server on :$PORT (log: $SERVER_LOG)"
  "$SERVE_SCRIPT" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  deadline=$((SECONDS + WAIT_SECS))
  while (( SECONDS < deadline )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited early:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
    port_open && { echo "GR00T server ready"; break; }
    sleep 3
  done
  port_open || { echo "timed out waiting for :$PORT" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
fi

# Not exec: keep the cleanup trap alive so a server we started gets stopped.
set +e
env MUJOCO_GL="${MUJOCO_GL:-egl}" "$COLLECT_PYTHON" "$COLLECT_SCRIPT" "$@"
STATUS=$?
set -e
exit "$STATUS"
