#!/usr/bin/env bash
# One command: start the pi0.5 feature server if port 8000 is free, run
# collect.py, then stop only the server this script started.
#
#   ./scripts/semantic_failure/run.sh --replan-steps 5 --scene-variant both
#   ./scripts/semantic_failure/run.sh --replan-steps 5 --task-id 0 --episode-index 0
#
# Env (optional):
#   FEATURE_PORT (default 8000)  FEATURE_SERVER_WAIT_SECS (default 600)
#   OPENPI_PYTHON (default submodules/openpi/.venv/bin/python)
#   COLLECT_PYTHON (default .venv/bin/python)
#   GEMINI_API_KEY (only needed with --label)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PORT="${FEATURE_PORT:-8000}"
WAIT_SECS="${FEATURE_SERVER_WAIT_SECS:-600}"
OPENPI_PYTHON="${OPENPI_PYTHON:-$REPO_ROOT/submodules/openpi/.venv/bin/python}"
COLLECT_PYTHON="${COLLECT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SERVER_SCRIPT="$REPO_ROOT/scripts/perception_probe/serve_pi05_with_features.py"
COLLECT_SCRIPT="$REPO_ROOT/scripts/semantic_failure/collect.py"

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

mkdir -p "$REPO_ROOT/outputs/semantic_failure/logs"
SERVER_LOG="$REPO_ROOT/outputs/semantic_failure/logs/feature_server.log"
STARTED_SERVER=0
SERVER_PID=""
cleanup() {
  if [[ "$STARTED_SERVER" == "1" && -n "$SERVER_PID" ]]; then
    echo "Stopping feature server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_open; then
  echo "Feature server already on :$PORT -- reusing it (will not stop it)"
else
  [[ -x "$OPENPI_PYTHON" ]] || { echo "OPENPI_PYTHON not executable: $OPENPI_PYTHON" >&2; exit 1; }
  echo "Starting feature server on :$PORT (log: $SERVER_LOG)"
  "$OPENPI_PYTHON" "$SERVER_SCRIPT" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  deadline=$((SECONDS + WAIT_SECS))
  while (( SECONDS < deadline )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited early:" >&2; tail -n 40 "$SERVER_LOG" >&2; exit 1; }
    port_open && { echo "Feature server ready"; break; }
    sleep 2
  done
  port_open || { echo "timed out waiting for :$PORT" >&2; exit 1; }
fi

# Not exec: keep the cleanup trap alive so a server we started gets stopped.
set +e
env MUJOCO_GL="${MUJOCO_GL:-egl}" "$COLLECT_PYTHON" "$COLLECT_SCRIPT" "$@"
STATUS=$?
set -e
exit "$STATUS"
