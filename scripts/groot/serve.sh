#!/usr/bin/env bash
# Launch the GR00T-N1.7 websocket policy server (openpi protocol, port 8000).
# Runs in submodules/Isaac-GR00T/.venv. Model load takes ~1-3 min.
#
#   scripts/groot/serve.sh
#   HOST=0.0.0.0 PORT=8000 GROOT_SUBCKPT=libero_10 scripts/groot/serve.sh --no-strict
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_DIR="${REPO_ROOT}/submodules/Isaac-GR00T"
CKPT="${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/${GROOT_SUBCKPT:-libero_10}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

[[ -d "${GROOT_DIR}/.git" || -f "${GROOT_DIR}/.git" ]] || {
  echo "submodules/Isaac-GR00T not initialised -- run scripts/groot/setup.sh first" >&2; exit 1; }
[[ -f "${CKPT}/config.json" ]] || {
  echo "checkpoint missing: ${CKPT}/config.json -- run scripts/groot/setup.sh first" >&2; exit 1; }

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# make the HF token visible to transformers/huggingface_hub in the uv venv
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

# GROOT_WITH_FEATURES=1 -> also serve the layer-16 backbone hidden states
[[ "${GROOT_WITH_FEATURES:-0}" == "1" ]] && set -- --with-features "$@"

cd "${GROOT_DIR}"
exec uv run python "${REPO_ROOT}/scripts/groot/server/serve_groot_ws.py" \
  --model-path "${CKPT}" --host "${HOST}" --port "${PORT}" "$@"
