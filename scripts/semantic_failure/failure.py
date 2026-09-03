"""Run Dan's Gemini two-pass failure localizer; map onset frame onto clocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from records import FailureAnnotation, RolloutRecord
from timeline import map_onset_frame


def label_failure(rollout: RolloutRecord, backend, *, refine: bool = True) -> dict[str, Any]:
    from dan_label_with_vlm import label_one

    row = {
        "video_path": str(Path(rollout.video_path)),
        "task_desc": rollout.instruction,
        "failing_predicate": rollout.failing_predicate or rollout.instruction,
        "detail": rollout.failure_detail or rollout.sim_failure_category,
    }
    return label_one(backend, row, refine=refine)


def apply_failure_to_rollout(rollout: RolloutRecord, vlm: dict[str, Any]) -> None:
    rollout.vlm_failure = {
        k: v
        for k, v in vlm.items()
        if k
        not in {
            "vlm_raw_response",
            "vlm_refinement_raw_response",
        }
    }
    rollout.vlm_failure["vlm_raw_response"] = vlm.get("vlm_raw_response", "")
    onset = vlm.get("vlm_failure_onset_frame")
    if onset is None:
        return
    mapped = map_onset_frame(int(onset), rollout.n_control, rollout.replan_steps)
    ctrl = mapped["failure_control_step"]
    sim = rollout.controls[ctrl].sim_step if ctrl < rollout.n_control else None
    rollout.failure = FailureAnnotation(
        failure_control_step=ctrl,
        failure_sim_step=int(sim) if sim is not None else None,
        failure_policy_step=mapped["failure_policy_step"],
        failure_chunk_index=mapped["failure_chunk_index"],
        first_post_failure_policy_step=mapped["failure_policy_step"] + 1,
        failure_type=str(vlm.get("vlm_failure_mode") or "") or None,
        correction_action=str(vlm.get("vlm_recovery_action") or "") or None,
    )
