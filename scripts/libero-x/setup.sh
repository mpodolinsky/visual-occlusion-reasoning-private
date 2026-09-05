#!/usr/bin/env bash
# One-time setup for the LIBERO-X side. Does NOT touch the GR00T policy side
# (submodules/Isaac-GR00T + checkpoints -- see scripts/groot/setup.sh, which
# must already have been run).
#
#   - init the submodules/LIBERO-X submodule (meituan/LIBERO-X, ~785MB)
#
# LIBERO-X's own `libero` package (needed for its new predicates/objects) is
# imported at runtime via a process-scoped sys.path insert (see
# libero_x_env.configure_libero_x) -- no pip install, no separate venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo ">> init submodules/LIBERO-X (meituan/LIBERO-X, large checkout)"
git submodule update --init submodules/LIBERO-X

test -d "${REPO_ROOT}/submodules/LIBERO-X/libero/libero_x/bddl/LEVEL1" || {
  echo "ERROR: submodules/LIBERO-X/libero/libero_x/bddl/LEVEL1 missing after init" >&2
  exit 1
}

echo
echo "Setup OK."
echo "  next: scripts/groot/serve.sh   (terminal 1 -- GR00T policy server)"
echo "        MUJOCO_GL=egl uv run python scripts/libero-x/run_eval.py --smoke"
