#!/usr/bin/env python3
"""Normal vs occluded success-rate comparison for a GR00T libero_10 run.

Reads the single ``manifest.csv`` written by ``collect.py``, pairs episodes on
``(task, episode)``, and writes ``<run-dir>/comparison/``:

- ``summary.json``          -- per-task + overall normal SR, occluded SR, SR drop
- ``occlusion_failures.md`` -- episodes that succeed normal but fail occluded,
  with links to all four videos.

Mirrors ``scripts/evaluation/compare_pi05_libero_runs.py`` semantics on the
semantic_failure-style single-dir layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import DEFAULT_OUTPUT_DIR  # noqa: E402


def _read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _key(row: dict) -> tuple[str, int]:
    return row["task"], int(row["episode"])


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--allow-partial", action="store_true", help="Compare the paired intersection only.")
    args = p.parse_args(argv)

    manifest = args.run_dir / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"no manifest.csv under {args.run_dir}")

    by_variant: dict[str, dict[tuple[str, int], dict]] = {"normal": {}, "occluded": {}}
    for row in _read_manifest(manifest):
        variant = row["scene_variant"]
        if variant in by_variant:
            by_variant[variant][_key(row)] = row
    normal, occluded = by_variant["normal"], by_variant["occluded"]

    if not args.allow_partial and set(normal) != set(occluded):
        raise SystemExit(
            f"normal/occluded key sets differ ({len(normal)} vs {len(occluded)}); "
            "pass --allow-partial to compare the intersection"
        )
    keys = sorted(set(normal) & set(occluded))
    if not keys:
        raise SystemExit("no paired (task, episode) rows between the two variants")

    per_task: dict[str, dict] = {}
    for task, episode in keys:
        d = per_task.setdefault(
            task,
            {"n": 0, "normal_succ": 0, "occ_succ": 0, "occ_only_fail": [], "both_fail": 0, "both_succ": 0},
        )
        ns = _truthy(normal[(task, episode)]["success"])
        os_ = _truthy(occluded[(task, episode)]["success"])
        d["n"] += 1
        d["normal_succ"] += int(ns)
        d["occ_succ"] += int(os_)
        if ns and not os_:
            d["occ_only_fail"].append(episode)
        elif not ns and not os_:
            d["both_fail"] += 1
        elif ns and os_:
            d["both_succ"] += 1

    total_n = sum(d["n"] for d in per_task.values())
    total_ns = sum(d["normal_succ"] for d in per_task.values())
    total_os = sum(d["occ_succ"] for d in per_task.values())
    summary = {
        "run_dir": str(args.run_dir),
        "paired_episodes": total_n,
        "normal_success_rate": total_ns / total_n,
        "occluded_success_rate": total_os / total_n,
        "success_rate_drop": (total_ns - total_os) / total_n,
        "occlusion_only_failures": sum(len(d["occ_only_fail"]) for d in per_task.values()),
        "per_task": {
            task: {
                "paired_episodes": d["n"],
                "normal_success_rate": d["normal_succ"] / d["n"],
                "occluded_success_rate": d["occ_succ"] / d["n"],
                "occlusion_only_failures": sorted(d["occ_only_fail"]),
                "both_fail": d["both_fail"],
                "both_succeed": d["both_succ"],
            }
            for task, d in sorted(per_task.items())
        },
    }

    out_dir = args.run_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GR00T libero_10 -- occlusion-only failures",
        "",
        f"Normal SR {summary['normal_success_rate']:.1%} -> occluded SR "
        f"{summary['occluded_success_rate']:.1%} "
        f"(drop {summary['success_rate_drop']:.1%}, {total_n} paired episodes).",
        "",
    ]
    for task, d in summary["per_task"].items():
        if not d["occlusion_only_failures"]:
            continue
        lines.append(f"## {task}")
        for episode in d["occlusion_only_failures"]:
            nr = normal[(task, episode)]["dir"]
            orr = occluded[(task, episode)]["dir"]
            lines.append(
                f"- ep{episode:03d}: "
                f"[normal agent]({quote(nr)}/rollout.mp4) - "
                f"[normal wrist]({quote(nr)}/wrist.mp4) - "
                f"[occluded agent]({quote(orr)}/rollout.mp4) - "
                f"[occluded wrist]({quote(orr)}/wrist.mp4)"
            )
        lines.append("")
    (out_dir / "occlusion_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"normal SR {summary['normal_success_rate']:.1%}  "
        f"occluded SR {summary['occluded_success_rate']:.1%}  "
        f"drop {summary['success_rate_drop']:.1%}  ({total_n} paired episodes)"
    )
    print(f"wrote {out_dir}/summary.json + occlusion_failures.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
