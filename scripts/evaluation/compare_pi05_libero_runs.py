#!/usr/bin/env python3
"""Compare index-matched normal and occluded LIBERO evaluation episodes."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCCLUDED_CSV = REPO_ROOT / "outputs" / "pi05_libero_occ" / "episodes.csv"
DEFAULT_NORMAL_CSV = (
    REPO_ROOT / "outputs" / "pi05_libero_matched_normal" / "episodes.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "pi05_libero_comparison"
FAILURE_FIELDS = (
    "suite",
    "task",
    "prompt",
    "init_state_index",
    "normal_control_frames",
    "occluded_control_frames",
    "additional_occluded_frames",
    "normal_agent_video",
    "normal_wrist_video",
    "occluded_agent_video",
    "occluded_wrist_video",
)


EpisodeKey = tuple[str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occluded-csv", type=Path, default=DEFAULT_OCCLUDED_CSV)
    parser.add_argument("--normal-csv", type=Path, default=DEFAULT_NORMAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compare the intersection if one run does not contain every paired episode.",
    )
    return parser.parse_args()


def canonical_suite(suite: str) -> str:
    return suite.removesuffix("_occluded")


def episode_key(row: dict[str, str]) -> EpisodeKey:
    index = row.get("init_state_index") or row.get("episode")
    if index is None:
        raise ValueError("Episode row has neither init_state_index nor episode")
    return canonical_suite(row["suite"]), row["task"], int(index)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected boolean value, got {value!r}")
    return normalized == "true"


def read_episodes(path: Path) -> dict[EpisodeKey, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode CSV not found: {path}")
    rows: dict[EpisodeKey, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = episode_key(row)
            if key in rows:
                raise ValueError(f"Duplicate episode key {key} in {path}")
            if row.get("status") != "completed":
                raise ValueError(f"Episode {key} is not completed in {path}")
            rows[key] = row
    if not rows:
        raise ValueError(f"No episode rows found in {path}")
    return rows


def video_path(csv_path: Path, relative_path: str) -> Path:
    return (csv_path.parent / relative_path).resolve()


def failure_row(
    key: EpisodeKey,
    normal: dict[str, str],
    occluded: dict[str, str],
    normal_csv: Path,
    occluded_csv: Path,
) -> dict[str, Any]:
    normal_frames = int(normal["control_frames"])
    occluded_frames = int(occluded["control_frames"])
    return {
        "suite": key[0],
        "task": key[1],
        "prompt": normal.get("prompt") or occluded.get("prompt", ""),
        "init_state_index": key[2],
        "normal_control_frames": normal_frames,
        "occluded_control_frames": occluded_frames,
        "additional_occluded_frames": occluded_frames - normal_frames,
        "normal_agent_video": str(video_path(normal_csv, normal["video"])),
        "normal_wrist_video": str(video_path(normal_csv, normal["wrist_video"])),
        "occluded_agent_video": str(video_path(occluded_csv, occluded["video"])),
        "occluded_wrist_video": str(video_path(occluded_csv, occluded["wrist_video"])),
    }


def outcome_summary(pairs: Iterable[tuple[dict[str, str], dict[str, str]]]) -> dict[str, Any]:
    pairs = list(pairs)
    normal_successes = sum(parse_bool(normal["success"]) for normal, _ in pairs)
    occluded_successes = sum(parse_bool(occluded["success"]) for _, occluded in pairs)
    normal_only = sum(
        parse_bool(normal["success"]) and not parse_bool(occluded["success"])
        for normal, occluded in pairs
    )
    occluded_only = sum(
        not parse_bool(normal["success"]) and parse_bool(occluded["success"])
        for normal, occluded in pairs
    )
    both_success = sum(
        parse_bool(normal["success"]) and parse_bool(occluded["success"])
        for normal, occluded in pairs
    )
    neither = len(pairs) - normal_only - occluded_only - both_success
    return {
        "paired_episodes": len(pairs),
        "normal_successes": normal_successes,
        "normal_success_rate": normal_successes / len(pairs) if pairs else 0.0,
        "occluded_successes": occluded_successes,
        "occluded_success_rate": occluded_successes / len(pairs) if pairs else 0.0,
        "success_rate_drop": (
            (normal_successes - occluded_successes) / len(pairs) if pairs else 0.0
        ),
        "failed_with_occlusion_only": normal_only,
        "succeeded_with_occlusion_only": occluded_only,
        "succeeded_in_both": both_success,
        "failed_in_both": neither,
    }


def markdown_link(label: str, target: str, output_dir: Path) -> str:
    relative = os.path.relpath(target, output_dir)
    return f"[{label}]({quote(relative, safe='/._-')})"


def render_markdown(
    failures: list[dict[str, Any]],
    summary: dict[str, Any],
    per_suite: dict[str, dict[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# Episodes that failed only with occlusion",
        "",
        (
            f"The normal run succeeded and the occluded run failed in "
            f"**{summary['failed_with_occlusion_only']} of "
            f"{summary['paired_episodes']} paired episodes**."
        ),
        "",
        (
            "This is an observed outcome comparison at matching task and initial-state "
            "indices. The policy samples actions, so separate rollouts can also differ "
            "because they consume different RNG keys; this list should not be read as a "
            "causal estimate of occlusion by itself."
        ),
        "",
        "| Suite | Paired | Normal success | Occluded success | Failed only with occlusion |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            (
                f"| {suite} | {values['paired_episodes']} | "
                f"{values['normal_success_rate']:.2%} | "
                f"{values['occluded_success_rate']:.2%} | "
                f"{values['failed_with_occlusion_only']} |"
            )
            for suite, values in per_suite.items()
        ),
        "",
        "| Metric | Normal | Occluded |",
        "| --- | ---: | ---: |",
        (
            f"| Successes | {summary['normal_successes']} "
            f"({summary['normal_success_rate']:.2%}) | "
            f"{summary['occluded_successes']} "
            f"({summary['occluded_success_rate']:.2%}) |"
        ),
        "",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for failure in failures:
        grouped[(failure["suite"], failure["task"])].append(failure)

    current_suite = None
    for (suite, task), task_failures in sorted(grouped.items()):
        if suite != current_suite:
            lines.extend((f"## {suite}", ""))
            current_suite = suite
        readable_task = task.replace("_", " ")
        lines.extend((f"### {readable_task}", ""))
        for failure in task_failures:
            normal_links = " · ".join(
                (
                    markdown_link("agent", failure["normal_agent_video"], output_dir),
                    markdown_link("wrist", failure["normal_wrist_video"], output_dir),
                )
            )
            occluded_links = " · ".join(
                (
                    markdown_link("agent", failure["occluded_agent_video"], output_dir),
                    markdown_link("wrist", failure["occluded_wrist_video"], output_dir),
                )
            )
            lines.extend(
                (
                    (
                        f"- Initial state **{failure['init_state_index']:02d}**: normal "
                        f"succeeded in {failure['normal_control_frames']} frames; occluded "
                        f"failed after {failure['occluded_control_frames']} frames."
                    ),
                    f"  - Normal video: {normal_links}",
                    f"  - Occluded video: {occluded_links}",
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    normal_csv = args.normal_csv.resolve()
    occluded_csv = args.occluded_csv.resolve()
    normal_rows = read_episodes(normal_csv)
    occluded_rows = read_episodes(occluded_csv)
    normal_keys = set(normal_rows)
    occluded_keys = set(occluded_rows)
    if normal_keys != occluded_keys and not args.allow_partial:
        raise ValueError(
            "Runs are not exactly paired: "
            f"{len(normal_keys - occluded_keys)} normal-only keys and "
            f"{len(occluded_keys - normal_keys)} occluded-only keys. "
            "Use --allow-partial to compare their intersection."
        )

    keys = sorted(normal_keys & occluded_keys)
    pairs = [(normal_rows[key], occluded_rows[key]) for key in keys]
    failures = [
        failure_row(key, normal_rows[key], occluded_rows[key], normal_csv, occluded_csv)
        for key in keys
        if parse_bool(normal_rows[key]["success"])
        and not parse_bool(occluded_rows[key]["success"])
    ]
    summary = outcome_summary(pairs)
    per_suite = {
        suite: outcome_summary(
            (normal_rows[key], occluded_rows[key]) for key in keys if key[0] == suite
        )
        for suite in sorted({key[0] for key in keys})
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "occlusion_failures.csv", failures)
    (output_dir / "occlusion_failures.md").write_text(
        render_markdown(failures, summary, per_suite, output_dir), encoding="utf-8"
    )
    (output_dir / "comparison_summary.json").write_text(
        json.dumps({"overall": summary, "suites": per_suite}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Found {summary['failed_with_occlusion_only']} episodes that succeeded normally "
        f"and failed with occlusion among {summary['paired_episodes']} pairs."
    )
    print(f"Human-readable report: {output_dir / 'occlusion_failures.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
