"""3-second keyword phrases on the full episode video (same Gemini session as Dan)."""

from __future__ import annotations

import json
import logging

from records import RolloutRecord, SemanticSegment
from timeline import build_timeline_windows

KEYWORD_PROMPT = """You are reviewing the SAME complete robot manipulation rollout.

The video is {hz:.0f} FPS. Each frame is exactly one executed robot action.
Episode duration is {duration:.1f} seconds ({n_control} control steps).

Task instruction:
"{instruction}"
{failure_context}
For EACH numbered 3-second window, write a SHORT KEYWORD PHRASE (2–6 words)
for what the manipulator is doing, like "reaching for mug" or "closing gripper"
or "idle / no progress". Not a full sentence. Only what is visible in THAT window.

Do not copy a failure mode, failure reason, or recovery action into these phrases.
Those are stored in a separate array. If a window happens to show a mistake,
describe the visible motion (e.g. "dropping can"), not the taxonomy name.

Windows:
{windows}

Return ONLY a JSON object:
{{"segments": [{{"segment_index": 0, "phrase": "..."}}, ...]}}
Include every segment_index listed above.
"""


def _window_lines(segments: list[SemanticSegment]) -> str:
    lines = []
    for seg in segments:
        lines.append(
            f"{seg.segment_index}. {seg.t_start_sec:.1f}–{seg.t_end_sec:.1f}s "
            f"(control steps {seg.control_step_start}–{seg.control_step_end}, "
            f"policy steps {seg.policy_step_start}–{seg.policy_step_end})"
        )
    return "\n".join(lines)


def _phrases_by_index(parsed, n: int) -> dict[int, str]:
    rows: list = []
    if isinstance(parsed, dict):
        rows = parsed.get("segments") or parsed.get("phrases") or parsed.get("descriptions") or []
    elif isinstance(parsed, list):
        rows = parsed
    out: dict[int, str] = {}
    if rows and all(isinstance(row, str) for row in rows) and len(rows) == n:
        return {i: row.strip() for i, row in enumerate(rows)}
    for i, row in enumerate(rows):
        if isinstance(row, str):
            out[i] = row.strip()
            continue
        if not isinstance(row, dict):
            continue
        idx = row.get("segment_index", i)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        phrase = str(
            row.get("phrase") or row.get("description") or row.get("text") or ""
        ).strip()
        out[idx] = phrase
    return out


def _parse_caption_payload(text: str):
    from dan_label_with_vlm import parse_json_response

    try:
        return parse_json_response(text)
    except ValueError:
        cleaned = text.strip()
        start = cleaned.find("[")
        if start == -1:
            raise
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(cleaned[start:])
        return parsed


def keyword_prompt(rollout: RolloutRecord, segments: list[SemanticSegment], failure_context: dict | None) -> str:
    if failure_context:
        ctx = (
            "\nA previous pass on this video produced this failure JSON "
            "(stored separately; do not merge it into the phrases):\n"
            f"{json.dumps(failure_context, indent=2)}\n"
        )
    else:
        ctx = "\n"
    return KEYWORD_PROMPT.format(
        hz=rollout.control_hz,
        duration=rollout.n_control / rollout.control_hz,
        n_control=rollout.n_control,
        instruction=rollout.instruction,
        failure_context=ctx,
        windows=_window_lines(segments),
    )


def apply_phrases(segments: list[SemanticSegment], parsed) -> list[SemanticSegment]:
    by_idx = _phrases_by_index(parsed, len(segments))
    for seg in segments:
        seg.phrase = by_idx.get(seg.segment_index, "")
        logging.info(
            "segment %d  %.1f–%.1fs  %s",
            seg.segment_index,
            seg.t_start_sec,
            seg.t_end_sec,
            seg.phrase or "(empty)",
        )
    return segments


def caption_timeline_on_session(session, rollout: RolloutRecord, failure_context: dict | None = None) -> list[SemanticSegment]:
    segments = build_timeline_windows(rollout.n_control, rollout.replan_steps, hz=rollout.control_hz)
    if not segments:
        return []
    logging.info(
        "3s keyword phrases (%d control steps, %d windows) on the same Gemini session",
        rollout.n_control,
        len(segments),
    )
    result = session.ask(keyword_prompt(rollout, segments, failure_context))
    parsed = _parse_caption_payload(result["text"])
    return apply_phrases(segments, parsed)
