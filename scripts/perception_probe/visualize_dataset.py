#!/usr/bin/env python3
"""Descriptive statistics + plots for the cached feature dataset (manifest.csv
produced by collect_features.py), before/independent of any probe training.

Answers:
  - How many episodes total, and what's the success/failure split?
  - Same at the STEP level (steps inherit their episode's label) -- failure
    episodes run systematically longer, so failures are overrepresented at
    the step level relative to the episode level; this script quantifies
    that gap directly.
  - Success/failure split by scene_variant (occluded vs. normal) -- the core
    comparison the whole project is testing.
  - Success/failure split by task and by suite -- some tasks/suites may be
    much harder than others, which (given the imbalance in how many steps
    each task/suite contributes) is a plausible route to the probe learning
    task-identity shortcuts instead of a generalizable notion of perceptual
    uncertainty.

No torch/model dependency -- this only reads manifest.csv, doesn't touch the
cached feature arrays themselves, and runs in seconds regardless of dataset
size.

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/visualize_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "submodules" / "Libero-Occ" / "benchmark_assets"
SUITES = (
    "libero_spatial_occluded",
    "libero_goal_occluded",
    "libero_object_occluded",
    "libero_10_occluded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "dataset_stats"
    )
    return parser.parse_args()


def read_manifest(features_dir: Path) -> list[dict]:
    with (features_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def task_to_suite_map() -> dict[str, str]:
    """manifest.csv's "suite" column is never populated by collect_features.py's
    rebuild_manifest -- rebuilt here instead from the BDDL filenames each suite
    actually contains (task stems are unique across all 4 suites in this
    benchmark, and shared between the occluded/normal scene variants of the
    same task, so one pass over the occluded suites' BDDL files covers both)."""
    mapping: dict[str, str] = {}
    for suite in SUITES:
        bddl_dir = BENCHMARK_ROOT / "bddl_files" / suite
        for bddl_path in sorted(bddl_dir.glob("*.bddl")):
            mapping[bddl_path.stem] = suite
    return mapping


def is_success(row: dict) -> bool:
    return row["success"] == "True"


def episode_level_stats(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(1 for r in rows if is_success(r))
    return {
        "n_episodes": n,
        "n_success": successes,
        "n_failure": n - successes,
        "success_rate": successes / n if n else None,
    }


def step_level_stats(rows: list[dict]) -> dict:
    total_steps = sum(int(r["inference_calls"]) for r in rows)
    success_steps = sum(int(r["inference_calls"]) for r in rows if is_success(r))
    return {
        "n_steps": total_steps,
        "n_success_steps": success_steps,
        "n_failure_steps": total_steps - success_steps,
        "success_rate": success_steps / total_steps if total_steps else None,
    }


def grouped_stats(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    result = []
    for name, group_rows in groups.items():
        ep = episode_level_stats(group_rows)
        st = step_level_stats(group_rows)
        result.append(
            {
                key: name,
                "n_episodes": ep["n_episodes"],
                "n_success": ep["n_success"],
                "n_failure": ep["n_failure"],
                "episode_success_rate": ep["success_rate"],
                "n_steps": st["n_steps"],
                "step_success_rate": st["success_rate"],
                "mean_steps_per_episode": st["n_steps"] / ep["n_episodes"] if ep["n_episodes"] else None,
            }
        )
    return result


def plot_episode_vs_step_failure_rate(episode_stats: dict, step_stats: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["episode-level", "step-level"]
    failure_rates = [
        1 - episode_stats["success_rate"] if episode_stats["success_rate"] is not None else 0.0,
        1 - step_stats["success_rate"] if step_stats["success_rate"] is not None else 0.0,
    ]
    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(labels, failure_rates, color=["#4C72B0", "#C44E52"])
    for bar, rate in zip(bars, failure_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.01, f"{rate:.1%}", ha="center")
    ax.set_ylabel("failure rate")
    ax.set_ylim(0, max(failure_rates) * 1.3 + 0.05)
    ax.set_title("Failure rate: episode-level vs. step-level\n(labels broadcast episode -> step)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scene_variant(rows_by_variant: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(rows_by_variant.keys())
    success = [rows_by_variant[v]["n_success"] for v in variants]
    failure = [rows_by_variant[v]["n_failure"] for v in variants]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(variants, success, label="success", color="#55A868")
    ax.bar(variants, failure, bottom=success, label="failure", color="#C44E52")
    for i, v in enumerate(variants):
        rate = rows_by_variant[v]["success_rate"]
        total = success[i] + failure[i]
        ax.text(i, total + 1, f"{rate:.1%} success" if rate is not None else "n/a", ha="center")
    ax.set_ylabel("episodes")
    ax.set_title("Episode outcome by scene variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grouped_success_rate(
    rows: list[dict], name_key: str, title: str, path: Path, color_by: dict[str, str] | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda r: (r["episode_success_rate"] if r["episode_success_rate"] is not None else -1))
    names = [r[name_key] for r in ordered]
    rates = [r["episode_success_rate"] if r["episode_success_rate"] is not None else 0.0 for r in ordered]
    counts = [r["n_episodes"] for r in ordered]
    colors = [color_by.get(n, "#4C72B0") for n in names] if color_by else "#4C72B0"

    fig, ax = plt.subplots(figsize=(11, max(3, 0.35 * len(names))))
    bars = ax.barh(names, rates, color=colors)
    ax.tick_params(axis="y", labelsize=8)
    for bar, n_episodes in zip(bars, counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={n_episodes}", va="center", fontsize=8)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("episode success rate")
    ax.set_xlim(0, 1.15)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    # Timestamped subdirectory per run -- this unconditionally overwrites its output files on
    # every run (no resume/skip logic), so a fixed shared path would silently clobber a
    # previous run's stats. "latest" tracks the most recent.
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = read_manifest(args.features_dir)
    logging.info("Loaded %d episodes from %s", len(rows), args.features_dir / "manifest.csv")

    suite_map = task_to_suite_map()
    unmapped = {r["task"] for r in rows} - set(suite_map)
    if unmapped:
        logging.warning("Could not map %d task(s) to a suite (BDDL file missing?): %s", len(unmapped), sorted(unmapped))
    for row in rows:
        row["suite"] = suite_map.get(row["task"], "unknown")

    # 1. Overall episode-level and step-level stats.
    overall_episodes = episode_level_stats(rows)
    overall_steps = step_level_stats(rows)
    logging.info(
        "Episodes: %d total, %d success, %d failure (%.1f%% success)",
        overall_episodes["n_episodes"], overall_episodes["n_success"], overall_episodes["n_failure"],
        100 * overall_episodes["success_rate"],
    )
    logging.info(
        "Steps:    %d total, %d success, %d failure (%.1f%% success)",
        overall_steps["n_steps"], overall_steps["n_success_steps"], overall_steps["n_failure_steps"],
        100 * overall_steps["success_rate"],
    )
    logging.info(
        "Failure rate amplification at step level: %.1f%% of episodes are failures, "
        "but %.1f%% of steps are (failure episodes run longer on average).",
        100 * (1 - overall_episodes["success_rate"]), 100 * (1 - overall_steps["success_rate"]),
    )

    # 2. Scene variant (occluded vs. normal) split.
    variant_stats = {name: episode_level_stats(part) for name, part in
                      {v: [r for r in rows if r["scene_variant"] == v] for v in sorted({r["scene_variant"] for r in rows})}.items()}
    for variant, stats in variant_stats.items():
        logging.info(
            "Scene variant %-10s: %d episodes, %.1f%% success",
            variant, stats["n_episodes"], 100 * stats["success_rate"] if stats["success_rate"] is not None else float("nan"),
        )

    # 3. Per-task and per-suite breakdown.
    by_task = sorted(grouped_stats(rows, "task"), key=lambda r: r["episode_success_rate"] or 0)
    for r in by_task:
        r["suite"] = suite_map.get(r["task"], "unknown")
    by_suite = sorted(grouped_stats(rows, "suite"), key=lambda r: r["episode_success_rate"] or 0)
    logging.info("Per-suite success rate (lowest first):")
    for r in by_suite:
        logging.info(
            "  %-28s %3d episodes  %.1f%% success  (%.0f steps/episode avg)",
            r["suite"], r["n_episodes"], 100 * (r["episode_success_rate"] or 0), r["mean_steps_per_episode"] or 0,
        )
    logging.info("Hardest 5 tasks (lowest success rate):")
    for r in by_task[:5]:
        logging.info(
            "  %-70s %3d episodes  %.1f%% success",
            r["task"], r["n_episodes"], 100 * (r["episode_success_rate"] or 0),
        )

    # Persist everything.
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "overall_episodes": overall_episodes,
                "overall_steps": overall_steps,
                "by_scene_variant": variant_stats,
                "by_task": by_task,
                "by_suite": by_suite,
            },
            indent=2,
        )
    )
    for name, table in (("by_task", by_task), ("by_suite", by_suite)):
        path = run_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    # Plots.
    plot_episode_vs_step_failure_rate(overall_episodes, overall_steps, run_dir / "episode_vs_step_failure_rate.png")
    plot_scene_variant(variant_stats, run_dir / "scene_variant_outcome.png")
    plot_grouped_success_rate(by_suite, "suite", "Success rate by suite", run_dir / "success_rate_by_suite.png")
    suite_colors = {
        s["suite"]: color
        for s, color in zip(by_suite, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    }
    task_colors = {t["task"]: suite_colors.get(suite_map.get(t["task"], "unknown"), "#4C72B0") for t in by_task}
    plot_grouped_success_rate(
        by_task, "task", "Success rate by task (colored by suite)",
        run_dir / "success_rate_by_task.png", color_by=task_colors,
    )

    latest_link = args.output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name, target_is_directory=True)

    logging.info("Saved stats + plots to %s (latest -> %s)", run_dir, latest_link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
