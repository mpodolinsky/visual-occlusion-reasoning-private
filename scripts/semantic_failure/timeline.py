"""3-second control windows. VLM fills descriptions; clocks are computed here."""

from __future__ import annotations

from records import SemanticSegment
from constants import CONTROL_HZ, SEGMENT_SECONDS


def steps_per_segment(hz: float = CONTROL_HZ, segment_seconds: float = SEGMENT_SECONDS) -> int:
    n = int(round(segment_seconds * hz))
    if n < 1:
        raise ValueError("segment must cover at least one control step")
    return n


def build_timeline_windows(
    n_control: int,
    replan_steps: int,
    *,
    hz: float = CONTROL_HZ,
    segment_seconds: float = SEGMENT_SECONDS,
) -> list[SemanticSegment]:
    if n_control < 1:
        return []
    if replan_steps < 1:
        raise ValueError("replan_steps must be >= 1")
    width = steps_per_segment(hz, segment_seconds)
    out: list[SemanticSegment] = []
    start = 0
    idx = 0
    while start < n_control:
        end = min(n_control - 1, start + width - 1)
        out.append(
            SemanticSegment(
                segment_index=idx,
                t_start_sec=start / hz,
                t_end_sec=(end + 1) / hz,
                control_step_start=start,
                control_step_end=end,
                policy_step_start=start // replan_steps,
                policy_step_end=end // replan_steps,
            )
        )
        start = end + 1
        idx += 1
    return out


def map_onset_frame(
    onset_frame: int,
    n_control: int,
    replan_steps: int,
) -> dict[str, int]:
    """Dan onset_frame == control_step when video is 1:1 at CONTROL_HZ."""
    if onset_frame < 0 or onset_frame >= n_control:
        raise ValueError(
            f"onset_frame {onset_frame} out of range for n_control={n_control}"
        )
    return {
        "failure_control_step": int(onset_frame),
        "failure_policy_step": int(onset_frame) // int(replan_steps),
        "failure_chunk_index": int(onset_frame) % int(replan_steps),
    }
