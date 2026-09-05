"""Shared constants + per-episode config carrier for the GR00T-on-libero_10 eval.

Mirrors ``scripts/semantic_failure/constants.py``. This pipeline lives inside 12,
so the LIBERO env helpers are imported read-only from
``scripts/evaluation/eval_pi05_libero`` by :mod:`libero_env`. ``replan_steps`` is
always recorded, never silently assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# scripts/groot/ -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# This pipeline only ever touches the libero_10 suite (normal + occluded).
OCCLUDED_SUITE = "libero_10_occluded"

CONTROL_HZ = 20.0
DUMMY_ACTION = [0.0] * 6 + [-1.0]
ACTION_DIM = 7
# GR00T-N1.7 LIBERO_PANDA modality config emits a 16-step action horizon.
GROOT_ACTION_HORIZON = 16

# nvidia/GR00T-N1.7-LIBERO -> libero_10/ sub-checkpoint, fetched by setup.sh.
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "GR00T-N1.7-LIBERO" / "libero_10"
GROOT_EMBODIMENT = "LIBERO_PANDA"
DEFAULT_CHECKPOINT_LABEL = "nvidia/GR00T-N1.7-LIBERO/libero_10"
MODEL_ID = "gr00t-n1.7-libero"

SERVER_SCRIPT = "scripts/groot/server/serve_groot_ws.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "groot" / "libero_10"

# Backbone feature capture (--with-features). Layer-16 hidden states of the
# Cosmos-Reason2-2B (Qwen3-VL) backbone -- the pi0.5 SAVE-A analog.
FEATURE_SELECT_LAYER = 16
GROOT_HIDDEN = 2048            # Qwen3-VL text hidden size
GROOT_IMG_TOKENS = 64         # 256px / patch 16 -> 16x16 -> 2x2 merge -> 8x8 per camera
LANG_MAX_TOKENS = 200         # instruction tokens zero-padded to this (matches pi0.5)
STATE_FEATURE_DIM = 1536      # action head input_embedding_dim
FEATURE_SOURCE = "backbone"
FEATURE_SERVER_SCRIPT = SERVER_SCRIPT
FEATURE_MODULE = (
    "gr00t.model.modules.qwen3_backbone.Qwen3Backbone hidden_states[-1] "
    f"(select_layer={FEATURE_SELECT_LAYER}); state_features from Gr00tN1d7ActionHead"
)

# Aliases matching scripts/semantic_failure/constants.py names.
NUM_IMAGE_TOKENS = GROOT_IMG_TOKENS
HIDDEN_DIM = GROOT_HIDDEN


@dataclass
class EvalConfig:
    """Everything :func:`rollout.run_episode` needs for one episode."""

    host: str = "127.0.0.1"
    port: int = 8000
    occluded_suite: str = OCCLUDED_SUITE
    scene_variant: str = "normal"
    task_id: int = 0
    episode_index: int = 0
    seed: int = 7
    num_steps_wait: int = 10
    replan_steps: int = 8
    env_resolution: int = 256
    max_steps: int | None = None
    save_video: bool = True
    checkpoint_label: str = DEFAULT_CHECKPOINT_LABEL


def add_replan_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--replan-steps",
        type=int,
        required=required,
        default=None if required else 8,
        help=(
            "Actions executed per GR00T inference (the model horizon is 16). "
            "Recorded on every rollout; never assumed. NVIDIA's LIBERO README uses 8."
        ),
    )
