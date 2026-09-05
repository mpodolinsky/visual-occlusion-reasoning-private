#!/usr/bin/env bash
# One command: start the GR00T policy server (scripts/groot/serve.sh) if port
# 8000 is free, run a smoke episode, then the full LIBERO-X sample, then stop
# only the server this script started.
#
#   scripts/libero-x/run_all.sh                       # LEVEL1, 5 tasks x 10 rollouts
#   LEVEL=LEVEL2 N_TASKS=5 N_ROLLOUTS=10 scripts/libero-x/run_all.sh
#
# Env (optional):
#   GROOT_PORT (default 8000)  GROOT_SERVER_WAIT_SECS (default 900)
#   STOP_PI05=1   pkill the pi0.5 feature server first (one GPU can't hold both)
#   SKIP_SMOKE=1  skip the 1-task x 1-rollout smoke episode
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PORT="${GROOT_PORT:-8000}"
WAIT_SECS="${GROOT_SERVER_WAIT_SECS:-900}"
LEVEL="${LEVEL:-LEVEL1}"
N_TASKS="${N_TASKS:-5}"
N_ROLLOUTS="${N_ROLLOUTS:-10}"
PY="${REPO_ROOT}/.venv/bin/python"

[[ -x "$PY" ]] || { echo "top-level .venv missing: $PY" >&2; exit 1; }

port_open() { "$PY" - "$PORT" <<'PY'
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

mkdir -p "$REPO_ROOT/outputs/libero-x/logs"
SERVER_LOG="$REPO_ROOT/outputs/libero-x/logs/groot_server.log"
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
  "$REPO_ROOT/scripts/groot/serve.sh" >"$SERVER_LOG" 2>&1 &
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

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo ">> smoke: 1 task x 1 rollout"
  "$REPO_ROOT/scripts/libero-x/run_eval.sh" --smoke --level "$LEVEL" --port "$PORT"
fi

echo ">> full sample: $N_TASKS tasks x $N_ROLLOUTS rollouts ($LEVEL)"
"$REPO_ROOT/scripts/libero-x/run_eval.sh" --level "$LEVEL" --n-tasks "$N_TASKS" \
  --n-rollouts "$N_ROLLOUTS" --port "$PORT"

echo ">> done. results: $REPO_ROOT/outputs/libero-x/results.json"
