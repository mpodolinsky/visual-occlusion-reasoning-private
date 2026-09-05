"""Human-readable example cards. Mirrors ``scripts/semantic_failure/present.py``
minus the Gemini timeline / failure sections (GR00T has no labeling pass).
"""

from __future__ import annotations

from pathlib import Path

from records import RolloutRecord


def format_example(rollout: RolloutRecord) -> str:
    has_feats = bool(rollout.policies) and all(p.features is not None for p in rollout.policies)
    feat_line = (
        f"`{rollout.n_policy}` layer-16 backbone tensors "
        f"(`base_image`/`wrist_image` 64x2048, `language` 200x2048, `state_features` 1536)"
        if has_feats
        else "not captured (run with `--with-features`)"
    )
    lines = [
        f"# {rollout.rollout_id}",
        "",
        f"- **suite / variant:** `{rollout.suite}` / `{rollout.scene_variant}`",
        f"- **instruction:** {rollout.instruction}",
        f"- **success:** {rollout.success}  "
        f"({rollout.sim_failure_category}"
        + (f"; {rollout.failing_predicate}" if rollout.failing_predicate else "")
        + ")",
        f"- **control steps:** {rollout.n_control}  "
        f"({rollout.n_control / rollout.control_hz:.1f}s at {rollout.control_hz:.0f} Hz)",
        f"- **policy inferences:** {rollout.n_policy}  (replan_steps={rollout.replan_steps})",
        f"- **checkpoint:** `{rollout.checkpoint}`",
        f"- **video:** `{rollout.video_path}`  /  `{rollout.wrist_video_path}`",
        f"- **features:** {feat_line}",
        "",
    ]
    return "\n".join(lines)


def write_examples_index(run_dir: Path, rollouts: list[RolloutRecord]) -> Path:
    parts = [
        "# GR00T-N1.7 LIBERO-10 examples",
        "",
        "Suite: `libero_10` (original LIBERO + LIBERO-Occ). Policy: "
        "`nvidia/GR00T-N1.7-LIBERO` -> `libero_10`. Features (when captured) are the "
        "layer-16 Cosmos-Reason2-2B backbone hidden states -- the pi0.5 SAVE-A analog.",
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
