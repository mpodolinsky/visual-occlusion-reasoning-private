"""Alignment assertions for a GR00T rollout. Fail loudly. Never silently truncate.

Ported from ``scripts/semantic_failure/validation.py``. Two GR00T-specific
differences from the pi0.5 version:

- features are ``GrootFeatures`` (base_image / wrist_image / language), only
  checked when present (``--with-features``);
- the executed action is ``decode_action_step(predicted_chunk[chunk_index])``
  (GR00T's gripper dim is transformed before ``env.step``), so the
  executed <-> predicted check compares against the decoded chunk row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constants import CONTROL_HZ
from groot_obs import decode_action_step
from records import AlignmentError, RolloutRecord


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    header: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def format(self) -> str:
        h = self.header
        lines = [
            f"Rollout ID: {h.get('rollout_id', '')}",
            f"Suite / variant: {h.get('suite', '')} / {h.get('scene_variant', '')}",
            f"Task: {h.get('task', '')}",
            f"Episode: {h.get('episode_index', '')}",
            f"Success: {h.get('success', '')}",
            "",
            f"GR00T: {h.get('model_checkpoint', '')}",
            f"Feature source: {h.get('feature_source', '')}",
            f"Replan steps: {h.get('replan_steps', '')}",
            f"Control Hz: {h.get('control_hz', '')}",
            f"Policy inferences: {h.get('n_policy', '')}",
            f"Control steps: {h.get('n_control', '')}",
            f"Feature shapes: {h.get('feature_shapes', '')}",
            "",
            "Alignment:",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            extra = f"  ({c.detail})" if c.detail else ""
            lines.append(f"{c.name}:          {status}{extra}")
        lines.append("")
        lines.append("ALL REQUIRED CHECKS PASS." if self.all_passed else "DO NOT START BULK COLLECTION")
        return "\n".join(lines)


def validate_rollout(rollout: RolloutRecord) -> ValidationReport:
    report = ValidationReport()
    shapes: dict = {}
    if rollout.policies and rollout.policies[0].features is not None:
        f0 = rollout.policies[0].features
        shapes = {
            "base_image": tuple(f0.base_image.shape),
            "wrist_image": tuple(f0.wrist_image.shape),
            "language": tuple(f0.language.shape),
            "language_mask": tuple(f0.language_mask.shape),
            "state_features": tuple(f0.state_features.shape),
        }
    report.header = {
        "rollout_id": rollout.rollout_id,
        "suite": rollout.suite,
        "scene_variant": rollout.scene_variant,
        "task": rollout.task_file,
        "episode_index": str(rollout.episode_index),
        "success": str(rollout.success),
        "model_checkpoint": f"{rollout.model_id} / {rollout.checkpoint}",
        "feature_source": rollout.feature_source,
        "replan_steps": str(rollout.replan_steps),
        "control_hz": str(rollout.control_hz),
        "n_policy": str(rollout.n_policy),
        "n_control": str(rollout.n_control),
        "feature_shapes": str(shapes),
    }
    names = (
        "policy -> control",
        "control -> chunk",
        "executed action -> decoded chunk",
        "video frame == control_step",
        "sim_step monotonic + wait offset",
        "NaN/Inf features",
    )
    try:
        _assert_alignment(rollout)
        for name in names:
            report.add(name, True)
    except AlignmentError as exc:
        for name in names:
            report.add(name, False, str(exc))
    return report


def _assert_alignment(rollout: RolloutRecord) -> None:
    if rollout.n_policy < 1:
        raise AlignmentError("no policy inferences")
    if rollout.n_control < 1:
        raise AlignmentError("no control steps")
    if rollout.replan_steps < 1:
        raise AlignmentError("replan_steps must be recorded and >= 1")
    if abs(float(rollout.control_hz) - CONTROL_HZ) > 1e-6:
        raise AlignmentError(f"control_hz {rollout.control_hz} != {CONTROL_HZ}")

    by_policy: dict[int, list] = {}
    for c in rollout.controls:
        by_policy.setdefault(c.policy_step, []).append(c)

    # sim_step: strictly increasing, starts at num_steps_wait (wait steps are sim-only).
    sim_steps = [c.sim_step for c in rollout.controls]
    if sim_steps != sorted(sim_steps) or len(set(sim_steps)) != len(sim_steps):
        raise AlignmentError("sim_step not strictly increasing")
    if rollout.num_steps_wait and sim_steps[0] != rollout.num_steps_wait:
        raise AlignmentError(f"first sim_step {sim_steps[0]} != num_steps_wait {rollout.num_steps_wait}")

    cursor = 0
    for i, pol in enumerate(rollout.policies):
        if pol.policy_step != i:
            raise AlignmentError(f"policy_step ids not sequential: {pol.policy_step} != {i}")
        if pol.features is not None:
            for name, arr in (
                ("base_image", pol.features.base_image),
                ("wrist_image", pol.features.wrist_image),
                ("language", pol.features.language),
                ("state_features", pol.features.state_features),
            ):
                if not np.isfinite(np.asarray(arr, dtype=np.float32)).all():
                    raise AlignmentError(f"NaN/Inf in {name} at policy_step {i}")
            if np.allclose(
                np.asarray(pol.features.base_image, np.float32),
                np.asarray(pol.features.wrist_image, np.float32),
                atol=1e-4,
            ):
                raise AlignmentError(f"base_image == wrist_image at policy_step {i}")

        group = by_policy.get(i, [])
        if not group:
            raise AlignmentError(f"policy_step {i} has no ControlRecords")
        group.sort(key=lambda c: c.control_step)
        if pol.executed_control_step_start != cursor:
            raise AlignmentError(
                f"policy_step {i} start {pol.executed_control_step_start} != {cursor}"
            )
        chunk = np.asarray(pol.predicted_action_chunk, dtype=np.float32)
        for j, c in enumerate(group):
            if c.control_step != cursor + j:
                raise AlignmentError(f"control_step {c.control_step} expected {cursor + j}")
            if c.chunk_index != j:
                raise AlignmentError(f"control {c.control_step} chunk_index {c.chunk_index} != {j}")
            if c.video_frame_id is not None and c.video_frame_id != c.control_step:
                raise AlignmentError(
                    f"video_frame_id {c.video_frame_id} != control_step {c.control_step}"
                )
            if c.chunk_index >= chunk.shape[0]:
                raise AlignmentError(f"chunk_index {c.chunk_index} >= chunk len {chunk.shape[0]}")
            expected = np.asarray(decode_action_step(chunk[c.chunk_index]), dtype=np.float32)
            got = np.asarray(c.executed_action, dtype=np.float32)
            if expected.shape != got.shape or not np.allclose(expected, got, atol=1e-4, rtol=1e-4):
                raise AlignmentError(
                    f"executed action at control {c.control_step} != "
                    f"decode(predicted chunk[{c.chunk_index}]) of policy_step {i}"
                )
        if pol.executed_control_step_end != cursor + len(group) - 1:
            raise AlignmentError(f"policy_step {i} end mismatch")
        cursor += len(group)

    if cursor != rollout.n_control:
        raise AlignmentError(
            f"executed actions ({cursor}) != total control steps ({rollout.n_control})"
        )
