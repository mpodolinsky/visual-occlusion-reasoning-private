"""Aligned rollout records for the GR00T eval. Clocks are 0-based.

Lean version of ``scripts/semantic_failure/records.py`` -- no prefix-feature
tensors, no Gemini fields. GR00T backbone features can be added later without
touching the collection artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class AlignmentError(ValueError):
    """Raised by validation / feature-identity checks. Never silently truncate."""


@dataclass
class GrootFeatures:
    """Layer-16 backbone hidden states for one inference (fp16).

    Mirrors ``scripts/semantic_failure/records.PrefixFeatures``:
    base_image / wrist_image: (64, 2048).  language: (200, 2048) zero-padded,
    with language_mask (200,) bool and language_len (real token count).
    state_features: (1536,).
    """

    base_image: np.ndarray
    wrist_image: np.ndarray
    language: np.ndarray
    language_mask: np.ndarray
    language_len: int
    state_features: np.ndarray
    source: str = "backbone"
    module: str = ""
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class PolicyRecord:
    policy_step: int
    predicted_action_chunk: np.ndarray  # (H, 7) float32 -- raw model output
    executed_control_step_start: int
    executed_control_step_end: int
    features: GrootFeatures | None = None


@dataclass
class ControlRecord:
    control_step: int
    sim_step: int
    policy_step: int
    chunk_index: int
    executed_action: np.ndarray  # post-decode_action_step (gripper transformed)
    video_frame_id: int | None = None
    env_success: bool = False


@dataclass
class RolloutRecord:
    rollout_id: str
    suite: str
    scene_variant: str
    task_id: int
    task_file: str
    episode_index: int
    instruction: str
    replan_steps: int
    success: bool
    max_steps: int
    seed: int
    model_id: str
    checkpoint: str
    feature_source: str
    feature_server: str
    feature_server_version: str
    control_hz: float
    num_steps_wait: int
    sim_failure_category: str
    failing_predicate: str
    failure_detail: str
    elapsed_seconds: float
    policies: list[PolicyRecord]
    controls: list[ControlRecord]
    video_path: str = ""
    wrist_video_path: str = ""

    @property
    def n_policy(self) -> int:
        return len(self.policies)

    @property
    def n_control(self) -> int:
        return len(self.controls)
