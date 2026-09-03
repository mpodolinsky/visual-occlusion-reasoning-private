"""Alignment assertions. Fail loudly. Never silently truncate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from records import AlignmentError, RolloutRecord
from constants import CONTROL_HZ


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
            f"π0.5: {h.get('model_checkpoint', '')}",
            f"Feature source: {h.get('feature_source', '')}",
            f"Replan steps: {h.get('replan_steps', '')}",
            f"Control Hz: {h.get('control_hz', '')}",
            f"Policy inferences: {h.get('n_policy', '')}",
            f"Control steps: {h.get('n_control', '')}",
            f"Prefix shapes: {h.get('prefix_shapes', '')}",
            "",
            "Alignment:",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            extra = f"  ({c.detail})" if c.detail else ""
            lines.append(f"{c.name}:          {status}{extra}")
        lines.append("")
        if not self.all_passed:
            lines.append("IF ANY REQUIRED ITEM = FAIL")
            lines.append("        ↓")
            lines.append("DO NOT START BULK COLLECTION")
        else:
            lines.append("ALL REQUIRED CHECKS PASS.")
        return "\n".join(lines)


def validate_rollout(rollout: RolloutRecord) -> ValidationReport:
    report = ValidationReport()
    shapes = {}
    if rollout.policies:
        p0 = rollout.policies[0].prefix_features
        shapes = {
            "base_image": tuple(p0.base_image.shape),
            "wrist_image": tuple(p0.wrist_image.shape),
            "language": tuple(p0.language.shape),
            "language_mask": tuple(p0.language_mask.shape),
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
        "prefix_shapes": str(shapes),
    }
    names = (
        "policy -> control",
        "control -> chunk",
        "feature -> policy",
        "executed action -> control",
        "video frame == control_step",
        "NaN/Inf features",
    )
    try:
        _assert_alignment(rollout)
        for name in names:
            report.add(name, True)
    except AlignmentError as exc:
        msg = str(exc)
        for name in names:
            report.add(name, False, msg)
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

    cursor = 0
    for i, pol in enumerate(rollout.policies):
        if pol.policy_step != i:
            raise AlignmentError(f"policy_step ids not sequential: {pol.policy_step} != {i}")
        feats = pol.prefix_features
        for name, arr in (
            ("base_image", feats.base_image),
            ("wrist_image", feats.wrist_image),
            ("language", feats.language),
        ):
            if not np.isfinite(np.asarray(arr, dtype=np.float32)).all():
                raise AlignmentError(f"NaN/Inf in {name} at policy_step {i}")
        chunk = np.asarray(pol.predicted_action_chunk)
        group = by_policy.get(i, [])
        if not group:
            raise AlignmentError(f"policy_step {i} has no ControlRecords")
        if pol.executed_control_step_start != cursor:
            raise AlignmentError(
                f"policy_step {i} start {pol.executed_control_step_start} != {cursor}"
            )
        for j, c in enumerate(group):
            if c.control_step != cursor + j:
                raise AlignmentError(f"control_step {c.control_step} expected {cursor + j}")
            if c.chunk_index != j:
                raise AlignmentError(
                    f"control {c.control_step} chunk_index {c.chunk_index} != {j}"
                )
            if c.video_frame_id is not None and c.video_frame_id != c.control_step:
                raise AlignmentError(
                    f"video_frame_id {c.video_frame_id} != control_step {c.control_step}"
                )
            pred = np.asarray(pol.predicted_action_chunk[c.chunk_index], dtype=np.float32)
            got = np.asarray(c.executed_action, dtype=np.float32)
            if pred.shape != got.shape or not np.allclose(pred, got, atol=1e-5, rtol=1e-5):
                raise AlignmentError(
                    f"executed action at control {c.control_step} != "
                    f"predicted chunk[{c.chunk_index}] of policy_step {i}"
                )
        if pol.executed_control_step_end != cursor + len(group) - 1:
            raise AlignmentError(f"policy_step {i} end mismatch")
        cursor += len(group)
        _ = chunk

    if cursor != rollout.n_control:
        raise AlignmentError(
            f"executed actions ({cursor}) != total control steps ({rollout.n_control})"
        )
