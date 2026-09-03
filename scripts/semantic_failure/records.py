"""Aligned rollout records. Clocks are 0-based."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class AlignmentError(ValueError):
    pass


@dataclass
class FailureAnnotation:
    """VLM / physical failure timing. Null on success and before labeling."""

    failure_control_step: int | None = None
    failure_sim_step: int | None = None
    failure_policy_step: int | None = None
    failure_chunk_index: int | None = None
    first_post_failure_policy_step: int | None = None
    failure_type: str | None = None
    correction_action: str | None = None


@dataclass
class PrefixFeatures:
    base_image: np.ndarray
    wrist_image: np.ndarray
    language: np.ndarray
    language_mask: np.ndarray
    source: str = "prefix"
    module: str = ""
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class PolicyRecord:
    policy_step: int
    prefix_features: PrefixFeatures
    predicted_action_chunk: np.ndarray
    executed_control_step_start: int
    executed_control_step_end: int


@dataclass
class ControlRecord:
    control_step: int
    sim_step: int
    policy_step: int
    chunk_index: int
    executed_action: np.ndarray
    video_frame_id: int | None = None
    env_success: bool = False


@dataclass
class SemanticSegment:
    segment_index: int
    t_start_sec: float
    t_end_sec: float
    control_step_start: int
    control_step_end: int
    policy_step_start: int
    policy_step_end: int
    phrase: str = ""


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
    policies: list[PolicyRecord]
    controls: list[ControlRecord]
    failure: FailureAnnotation = field(default_factory=FailureAnnotation)
    video_path: str = ""
    wrist_video_path: str = ""
    semantic_timeline: list[SemanticSegment] = field(default_factory=list)
    vlm_failure: dict[str, Any] = field(default_factory=dict)

    @property
    def n_policy(self) -> int:
        return len(self.policies)

    @property
    def n_control(self) -> int:
        return len(self.controls)
