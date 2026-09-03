#!/usr/bin/env python3
"""Label one saved episode dir: Dan's two-pass failure localizer first (failed
episodes only), then 3-second keyword phrases -- both on one shared Gemini video
session (:mod:`episode.label_episode`). Labels are written back into
``rollout.json`` and ``example.md`` is rendered.

    GEMINI_API_KEY=... uv run python scripts/semantic_failure/label.py <episode_dir>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import DEFAULT_GEMINI_MODEL  # noqa: E402
from episode import build_labeler_meta, label_episode  # noqa: E402
from present import format_example  # noqa: E402
from serialization import load_rollout, save_labels  # noqa: E402


def _already_labeled(rollout) -> bool:
    captioned = bool(rollout.semantic_timeline) and any(
        s.phrase for s in rollout.semantic_timeline
    )
    failure_done = rollout.success or bool(rollout.vlm_failure)
    return captioned and failure_done


def label_episode_dir(directory: Path, *, model: str, refine: bool, force: bool) -> bool:
    rollout = load_rollout(directory)
    if not rollout.video_path or not Path(rollout.video_path).is_file():
        candidate = directory / "rollout.mp4"
        if candidate.is_file():
            rollout.video_path = str(candidate)
        else:
            logging.error("no video for %s", directory)
            return False
    if _already_labeled(rollout) and not force:
        logging.info("already labeled, skipping %s", directory)
        return False

    from dan_label_with_vlm import create_backend

    backend = create_backend("gemini", model)
    logging.info("labeling %s (%d control steps)", directory.name, rollout.n_control)
    label_episode(rollout, backend, refine=refine)

    save_labels(directory, rollout, build_labeler_meta(backend, refine=refine))
    (directory / "example.md").write_text(format_example(rollout), encoding="utf-8")
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode_dir", type=Path)
    p.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-label an already-labeled episode.")
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        logging.error("GEMINI_API_KEY is not set")
        return 2

    changed = label_episode_dir(
        args.episode_dir, model=args.model, refine=not args.no_refine, force=args.force
    )
    example = args.episode_dir / "example.md"
    if example.is_file():
        print(example.read_text(encoding="utf-8"))
    return 0 if changed or example.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
