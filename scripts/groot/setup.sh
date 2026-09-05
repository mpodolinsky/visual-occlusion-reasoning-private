#!/usr/bin/env bash
# One-time setup for the GR00T policy side. Does NOT touch 12's top-level venv or
# the pi0.5 / openpi submodule.
#
#   - init the Isaac-GR00T submodule (n1.7-release) + its recursive submodules
#   - uv venv (Python 3.10) + uv sync   (torch 2.7 cu128, flash-attn, gr00t)
#   - add websockets + msgpack for the openpi-style shim
#   - check gated backbone access (nvidia/Cosmos-Reason2-2B)
#   - hf download nvidia/GR00T-N1.7-LIBERO -> libero_10/ (~7 GB)
#
# Prereqs: accept the gated licenses first, logged in as your HF user:
#   https://huggingface.co/nvidia/GR00T-N1.7-LIBERO
#   https://huggingface.co/nvidia/Cosmos-Reason2-2B
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GROOT_DIR="${REPO_ROOT}/submodules/Isaac-GR00T"
CKPT_DIR="${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO"

command -v uv >/dev/null || { echo "uv not found -- install from https://astral.sh/uv" >&2; exit 1; }
command -v hf >/dev/null || { echo "hf (huggingface_hub) not found" >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "WARN: ffmpeg not on PATH (torchcodec needs it: sudo apt install -y ffmpeg)"

echo ">> init submodules/Isaac-GR00T"
git submodule update --init --recursive submodules/Isaac-GR00T

pushd "${GROOT_DIR}" >/dev/null
echo ">> uv sync (python 3.10) -- this is large (torch cu128 + flash-attn)"
uv sync --python 3.10
uv pip install "websockets>=13" "msgpack>=1.0"
uv run python -c "import gr00t; from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper; print('gr00t import ok')"
popd >/dev/null

echo ">> checking gated backbone access (nvidia/Cosmos-Reason2-2B)"
if ! hf download nvidia/Cosmos-Reason2-2B config.json --quiet >/dev/null 2>&1; then
  echo "ERROR: no access to nvidia/Cosmos-Reason2-2B (the GR00T VLM backbone)." >&2
  echo "  Request access (1-click, usually instant) while logged in as $(hf auth whoami 2>/dev/null | head -1):" >&2
  echo "    https://huggingface.co/nvidia/Cosmos-Reason2-2B" >&2
  exit 1
fi
hf download nvidia/Cosmos-Reason2-2B --quiet >/dev/null 2>&1 || true

mkdir -p "${CKPT_DIR}"
echo ">> downloading nvidia/GR00T-N1.7-LIBERO :: libero_10/ ..."
hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/*" --local-dir "${CKPT_DIR}" || {
  echo "hf download failed. If it was a 403, accept the model license:" >&2
  echo "  https://huggingface.co/nvidia/GR00T-N1.7-LIBERO" >&2
  exit 1
}
test -f "${CKPT_DIR}/libero_10/config.json" || {
  echo "ERROR: ${CKPT_DIR}/libero_10/config.json missing after download" >&2; exit 1; }

echo
echo "Setup OK."
echo "  checkpoint: ${CKPT_DIR}/libero_10"
echo "  next:  scripts/groot/serve.sh   (terminal 1)"
echo "         MUJOCO_GL=egl uv run python scripts/groot/collect.py --replan-steps 8 --scene-variant both --num-trials 25"
