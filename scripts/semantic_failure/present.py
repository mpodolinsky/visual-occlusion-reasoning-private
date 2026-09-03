"""Human-readable example cards for Michal."""

from __future__ import annotations

from pathlib import Path

from records import RolloutRecord


def format_example(rollout: RolloutRecord) -> str:
    lines = [
        f"# {rollout.rollout_id}",
        "",
        f"- **suite / variant:** `{rollout.suite}` / `{rollout.scene_variant}`",
        f"- **instruction:** {rollout.instruction}",
        f"- **success:** {rollout.success}",
        f"- **control steps:** {rollout.n_control}  ({rollout.n_control / rollout.control_hz:.1f}s at {rollout.control_hz:.0f} Hz)",
        f"- **policy inferences:** {rollout.n_policy}  (replan_steps={rollout.replan_steps})",
        f"- **video:** `{rollout.video_path}`",
        f"- **features:** `{rollout.n_policy}` SAVE-A prefix tensors (`base_image`/`wrist_image` 256×2048)",
        "",
        "## Semantic timeline",
        "",
        "| seconds | control_step | policy_step | phrase |",
        "|---|---|---|---|",
    ]
    if not rollout.semantic_timeline:
        lines.append("| — | — | — | *(not labeled yet — needs GEMINI_API_KEY)* |")
    for s in rollout.semantic_timeline:
        phrase = (s.phrase or "").replace("|", "/")
        lines.append(
            f"| {s.t_start_sec:.1f}–{s.t_end_sec:.1f} | "
            f"{s.control_step_start}–{s.control_step_end} | "
            f"{s.policy_step_start}–{s.policy_step_end} | {phrase} |"
        )
    lines += ["", "## Failure", ""]
    if rollout.success:
        lines.append("No. Episode succeeded; Dan failure VLM was not run.")
    elif not rollout.vlm_failure:
        lines.append("Episode failed, but VLM failure labels are missing.")
    else:
        v = rollout.vlm_failure
        f = rollout.failure
        lines += [
            f"- **yes**",
            f"- **onset type:** {v.get('vlm_failure_onset_type')}",
            f"- **onset:** {v.get('vlm_failure_onset_seconds')} s "
            f"({v.get('vlm_failure_onset_timestamp')})",
            f"- **frame / control_step:** {v.get('vlm_failure_onset_frame')} "
            f"(policy_step={f.failure_policy_step}, chunk_index={f.failure_chunk_index})",
            f"- **coarse onset:** {v.get('vlm_coarse_failure_onset_seconds')} s "
            f"(refined={v.get('vlm_temporal_refined')})",
            f"- **mode:** {v.get('vlm_failure_mode')}",
            f"- **reason:** {v.get('vlm_failure_reason')}",
            f"- **recovery:** {v.get('vlm_recovery_action')}",
            f"- **justification:** {v.get('vlm_justification')}",
        ]
    lines.append("")
    return "\n".join(lines)


def write_examples_index(run_dir: Path, rollouts: list[RolloutRecord]) -> Path:
    parts = [
        "# LIBERO-10 semantic + failure examples",
        "",
        "Suite: `libero_10` (original LIBERO + LIBERO-Occ). "
        "Not LIBERO-X. Features are π0.5 SAVE-A. "
        "Failure labels use Dan’s Gemini two-pass localizer (own npz keys). "
        "3-second keyword phrases are a separate VLM turn on the same video session.",
        "",
        "**Do not bulk-collect until Michal signs off on these labels.**",
        "",
    ]
    for i, r in enumerate(rollouts, start=1):
        parts.append(f"## Example {i}")
        parts.append("")
        parts.append(format_example(r).split("\n", 1)[-1])
        parts.append("")
    path = run_dir / "EXAMPLES.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
