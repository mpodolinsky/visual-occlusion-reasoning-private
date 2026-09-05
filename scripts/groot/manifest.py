"""Global manifest + auto-resume for the GR00T libero_10 eval.

Copied from ``scripts/semantic_failure/manifest.py`` with the Gemini-label
columns dropped. Layout: ``<variant>/<NN>_<task>/ep<NNN>/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SCENE_VARIANTS = ("normal", "occluded")

MANIFEST_FIELDS = (
    "scene_variant",
    "suite",
    "task_id",
    "task",
    "prompt",
    "episode",
    "rollout_id",
    "success",
    "n_policy",
    "n_control",
    "control_hz",
    "replan_steps",
    "sim_failure_category",
    "failing_predicate",
    "elapsed_seconds",
    "has_features",
    "dir",
)


def task_dirname(task_id: int, task_stem: str) -> str:
    return f"{task_id + 1:02d}_{task_stem}"


def episode_dir(output_dir: Path, scene_variant: str, task_id: int, task_stem: str, episode: int) -> Path:
    return output_dir / scene_variant / task_dirname(task_id, task_stem) / f"ep{episode:03d}"


def episode_is_complete(directory: Path, *, expect_video: bool = True) -> bool:
    meta = directory / "rollout.json"
    npz = directory / "rollout.npz"
    if not (meta.is_file() and npz.is_file()):
        return False
    if expect_video and not (
        (directory / "rollout.mp4").is_file() and (directory / "wrist.mp4").is_file()
    ):
        return False
    try:
        json.loads(meta.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return True


def _row_from_meta(meta: dict, directory: Path, output_dir: Path) -> dict:
    task_stem = Path(meta.get("task_file", "")).stem
    return {
        "scene_variant": meta.get("scene_variant", ""),
        "suite": meta.get("suite", ""),
        "task_id": meta.get("task_id", ""),
        "task": task_stem,
        "prompt": meta.get("instruction", ""),
        "episode": meta.get("episode_index", ""),
        "rollout_id": meta.get("rollout_id", ""),
        "success": bool(meta.get("success", False)),
        "n_policy": meta.get("n_policy", ""),
        "n_control": meta.get("n_control", ""),
        "control_hz": meta.get("control_hz", ""),
        "replan_steps": meta.get("replan_steps", ""),
        "sim_failure_category": meta.get("sim_failure_category", ""),
        "failing_predicate": meta.get("failing_predicate", ""),
        "elapsed_seconds": meta.get("elapsed_seconds", ""),
        "has_features": bool(meta.get("has_features", False)),
        "dir": str(directory.relative_to(output_dir)),
    }


def rebuild_manifest(output_dir: Path) -> list[dict]:
    """Scan every saved episode under ``output_dir`` and rewrite ``manifest.csv``
    atomically. Called after every episode so an interrupted run still leaves a
    complete manifest for whatever made it to disk."""
    output_dir = Path(output_dir)
    rows: list[dict] = []
    for meta_path in sorted(output_dir.glob("*/*/ep*/rollout.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        rows.append(_row_from_meta(meta, meta_path.parent, output_dir))

    manifest_path = output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(manifest_path)
    return rows


def find_next_incomplete(
    output_dir: Path,
    task_stems: dict[tuple[str, int], str],
    num_trials: int,
    *,
    variants: tuple[str, ...] = SCENE_VARIANTS,
    expect_video: bool = True,
) -> tuple[str, int, int] | None:
    """First ``(scene_variant, task_id, episode_index)`` under ``output_dir`` that
    is not yet complete, iterating task-major then variant then episode (the same
    order ``collect.py`` fills the grid). ``None`` when the whole
    ``variants x 10 tasks x num_trials`` grid is complete."""
    for task_id in range(10):
        for scene_variant in variants:
            stem = task_stems.get((scene_variant, task_id))
            if stem is None:
                return scene_variant, task_id, 0
            for episode in range(num_trials):
                directory = episode_dir(output_dir, scene_variant, task_id, stem, episode)
                if not episode_is_complete(directory, expect_video=expect_video):
                    return scene_variant, task_id, episode
    return None
