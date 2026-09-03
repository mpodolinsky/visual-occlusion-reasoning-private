"""Label one saved episode: Dan failure (if any), then 3s keyword phrases."""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from records import RolloutRecord
from dan_label_with_vlm import (
    COARSE_PROMPT_TEMPLATE,
    REFINE_PROMPT_TEMPLATE,
    TAXONOMY,
    compact_failure_for_context,
    label_one_on_session,
)
from failure import apply_failure_to_rollout
from semantic import KEYWORD_PROMPT, caption_timeline_on_session


def build_labeler_meta(backend, *, refine: bool) -> dict:
    """The provenance block written into ``labels.json``: which model produced
    the labels and the exact prompt templates used."""
    return {
        "backend": type(backend).__name__,
        "model": getattr(backend, "model", None),
        "refine": bool(refine),
        "labeled_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "pipeline": "scripts/semantic_failure",
        "prompts": {
            "keyword_phrases": KEYWORD_PROMPT,
            "failure_coarse": COARSE_PROMPT_TEMPLATE,
            "failure_refine": REFINE_PROMPT_TEMPLATE,
            "failure_taxonomy": TAXONOMY,
        },
    }


def failure_row(rollout: RolloutRecord) -> dict:
    return {
        "video_path": str(Path(rollout.video_path)),
        "task_desc": rollout.instruction,
        "failing_predicate": rollout.failing_predicate or rollout.instruction,
        "detail": rollout.failure_detail or rollout.sim_failure_category,
    }


def label_episode(rollout: RolloutRecord, backend, *, refine: bool = True) -> RolloutRecord:
    """Dan coarse (+ 1-FPS refine on a separate clip) then 3s phrases. No merge."""
    video_path = Path(rollout.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing episode video: {video_path}")

    failure_context = None
    with backend.open_video_session(video_path) as session:
        if not rollout.success:
            logging.info("Dan failure pass first (coarse on full video, then ±3s 1-FPS refine)")
            vlm = label_one_on_session(session, failure_row(rollout), refine=refine)
            apply_failure_to_rollout(rollout, vlm)
            failure_context = compact_failure_for_context(vlm)
        else:
            rollout.vlm_failure = {}
            logging.info("Success — skipping Dan failure VLM")
        rollout.semantic_timeline = caption_timeline_on_session(
            session, rollout, failure_context=failure_context
        )
    return rollout
