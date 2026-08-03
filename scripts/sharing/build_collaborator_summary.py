#!/usr/bin/env python3
"""Build a concise PDF summary of the matched LIBERO evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "pi05_libero_comparison" / "comparison_summary.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "pi05_libero_comparison"
    / "LIBERO_Pi05_Evaluation_Summary.pdf"
)
BLUE = "#17365d"
MID_BLUE = "#275d8c"
LIGHT_BLUE = "#eef5fb"
GRAY = "#5f6368"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_wrapped_text(fig, x: float, y: float, text: str, *, width: int, **kwargs) -> None:
    fig.text(x, y, textwrap.fill(text, width), **kwargs)


def add_table(ax, rows: list[list[str]], columns: list[str], bbox: list[float]) -> None:
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c9d1d9")
        if row == 0:
            cell.set_facecolor("#f3f6f9")
            cell.set_text_props(weight="bold", color=BLUE)


def page_one(pdf: PdfPages, data: dict) -> None:
    overall = data["overall"]
    suites = data["suites"]
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    fig.text(0.08, 0.93, "Physical Intelligence Pi05 on LIBERO-Occ", fontsize=20, weight="bold", color=BLUE)
    fig.text(0.08, 0.895, "Matched evaluation with and without scene-induced occlusion", fontsize=11, color=GRAY)

    callout = (
        f"Across {overall['paired_episodes']:,} matched task and initial-state pairs, "
        f"success decreased from {overall['normal_success_rate']:.1%} without occlusion "
        f"to {overall['occluded_success_rate']:.1%} with occlusion—a "
        f"{overall['success_rate_drop']:.1%} absolute drop."
    )
    add_wrapped_text(
        fig,
        0.10,
        0.825,
        callout,
        width=88,
        fontsize=12,
        color=BLUE,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": LIGHT_BLUE, "edgecolor": MID_BLUE},
    )

    fig.text(0.08, 0.735, "Suite-level results", fontsize=14, weight="bold", color=MID_BLUE)
    labels = (("LIBERO-10", "libero_10"), ("Goal", "libero_goal"), ("Object", "libero_object"), ("Spatial", "libero_spatial"))
    suite_rows = [
        [
            label,
            f"{suites[key]['normal_success_rate']:.1%}",
            f"{suites[key]['occluded_success_rate']:.1%}",
            f"−{suites[key]['success_rate_drop'] * 100:.1f} pp",
            str(suites[key]["failed_with_occlusion_only"]),
        ]
        for label, key in labels
    ]
    add_table(
        ax,
        suite_rows,
        ["Suite", "No occlusion", "Occlusion", "Change", "Normal-only success"],
        [0.08, 0.48, 0.84, 0.22],
    )

    fig.text(0.08, 0.415, "Matched episode outcomes", fontsize=14, weight="bold", color=MID_BLUE)
    outcome_rows = [
        ["Succeeded normally; failed with occlusion", f"{overall['failed_with_occlusion_only']:,}"],
        ["Succeeded with occlusion; failed normally", f"{overall['succeeded_with_occlusion_only']:,}"],
        ["Succeeded in both", f"{overall['succeeded_in_both']:,}"],
        ["Failed in both", f"{overall['failed_in_both']:,}"],
    ]
    add_table(ax, outcome_rows, ["Outcome", "Episodes"], [0.08, 0.18, 0.84, 0.20])
    fig.text(
        0.08,
        0.105,
        "Observed regressions: 178 LIBERO-10 · 67 Goal · 30 Spatial · 23 Object",
        fontsize=10.5,
        weight="bold",
        color=BLUE,
    )
    fig.text(0.08, 0.055, "Evaluation date: July 2026", fontsize=8.5, color=GRAY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_two(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.08, 0.93, "Interpretation and representative cases", fontsize=19, weight="bold", color=BLUE)

    fig.text(0.08, 0.865, "Most affected task families", fontsize=14, weight="bold", color=MID_BLUE)
    tasks = [
        (33, "Put both the cream cheese box and butter in the basket"),
        (28, "Put both moka pots on the stove"),
        (26, "Place the white mug and chocolate pudding relative to the plate"),
        (24, "Turn on the stove and place the moka pot on it"),
        (20, "Put the yellow-and-white mug in the microwave and close it"),
    ]
    y = 0.82
    for count, task in tasks:
        fig.text(0.10, y, f"• {task}: {count} regressions", fontsize=10.5, color="#202124")
        y -= 0.042

    fig.text(0.08, 0.57, "What the results suggest", fontsize=14, weight="bold", color=MID_BLUE)
    add_wrapped_text(
        fig,
        0.08,
        0.525,
        (
            "Regressions are strongly concentrated in LIBERO-10. This supports focusing on "
            "view selection for longer-horizon, multi-stage tasks, where an occlusion can "
            "affect several dependent actions. Object and Spatial show smaller performance "
            "drops and therefore less apparent headroom."
        ),
        width=100,
        fontsize=10.5,
        color="#202124",
        va="top",
    )

    fig.text(0.08, 0.39, "Evaluation details", fontsize=14, weight="bold", color=MID_BLUE)
    add_wrapped_text(
        fig,
        0.08,
        0.345,
        (
            "Each suite contains 10 scenarios with 50 initial states per scenario. Normal "
            "and occluded runs were paired by canonical suite, task filename, and initial-state "
            "index. Agent-view and wrist-camera videos were saved for every rollout."
        ),
        width=100,
        fontsize=10.5,
        color="#202124",
        va="top",
    )

    fig.text(0.08, 0.22, "Interpretation caveat", fontsize=14, weight="bold", color=MID_BLUE)
    add_wrapped_text(
        fig,
        0.08,
        0.175,
        (
            "Pi05 samples actions. These are observed differences between separately sampled, "
            "index-matched rollouts; they do not by themselves constitute a controlled causal "
            "estimate of the effect of occlusion."
        ),
        width=100,
        fontsize=10.5,
        color=GRAY,
        va="top",
    )
    fig.text(0.08, 0.055, "See the linked Drive artifacts for the full regression list and paired videos.", fontsize=8.5, color=GRAY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        page_one(pdf, data)
        page_two(pdf)
        metadata = pdf.infodict()
        metadata["Title"] = "LIBERO Pi05 Evaluation Summary"
        metadata["Author"] = "Timothy Duggan"
        metadata["Subject"] = "Matched LIBERO evaluation with and without occlusion"
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
