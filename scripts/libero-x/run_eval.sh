#!/usr/bin/env bash
# Run the LIBERO-X rollout eval in this repo's top-level venv.
#
#   scripts/libero-x/run_eval.sh --smoke
#   scripts/libero-x/run_eval.sh --level LEVEL1 --n-tasks 5 --n-rollouts 10
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
exec uv run python scripts/libero-x/run_eval.py "$@"
