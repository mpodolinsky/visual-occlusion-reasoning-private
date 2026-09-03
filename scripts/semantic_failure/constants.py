"""Shared constants + per-episode config carrier.

Ported from 17-LIBERO-10-Semantic-Failure-Pipeline/src/config.py, trimmed:
this pipeline lives inside 12, so there is no "upstream" path -- the LIBERO
env helpers are imported directly from ``scripts/evaluation/eval_pi05_libero``
by :mod:`libero_env`. ``replan_steps`` is never silently assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# scripts/semantic_failure/ -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# This pipeline only ever touches the libero_10 suite (normal + occluded).
OCCLUDED_SUITE = "libero_10_occluded"

CONTROL_HZ = 20.0
SEGMENT_SECONDS = 3.0
DUMMY_ACTION = [0.0] * 6 + [-1.0]

NUM_IMAGE_TOKENS = 256
HIDDEN_DIM = 2048
ACTION_DIM = 7
FEATURE_SOURCE = "prefix"
FEATURE_MODULE = (
    "openpi.models.pi0.Pi0.PaliGemma.llm prefix_outputs[0] "
    "via sample_actions_with_features"
)
FEATURE_SERVER_SCRIPT = "scripts/perception_probe/serve_pi05_with_features.py"

DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG_NAME = "pi05_libero"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "semantic_failure" / "libero_10"


@dataclass
class PipelineConfig:
    """Everything :func:`rollout_runner.run_episode` needs for one episode."""

    host: str = "127.0.0.1"
    port: int = 8000
    occluded_suite: str = OCCLUDED_SUITE
    scene_variant: str = "normal"
    task_id: int = 0
    episode_index: int = 0
    seed: int = 7
    num_steps_wait: int = 10
    replan_steps: int = 5
    policy_image_size: int = 224
    env_resolution: int = 256
    max_steps: int | None = None
    save_video: bool = True
    model_id: str = DEFAULT_CONFIG_NAME
    checkpoint: str = DEFAULT_CHECKPOINT


def add_replan_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--replan-steps",
        type=int,
        required=required,
        default=None if required else 5,
        help="Actions executed per pi0.5 inference. Recorded on every rollout; never assumed.",
    )
