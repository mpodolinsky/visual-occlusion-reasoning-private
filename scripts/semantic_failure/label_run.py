#!/usr/bin/env python3
"""Label every episode under a collection run dir, skipping episodes already
labeled (labeling auto-resume), then refresh ``manifest.csv`` and write
``EXAMPLES.md``.

    GEMINI_API_KEY=... uv run python scripts/semantic_failure/label_run.py \
        [outputs/semantic_failure/libero_10]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import DEFAULT_GEMINI_MODEL, DEFAULT_OUTPUT_DIR  # noqa: E402
from label import label_episode_dir  # noqa: E402
from manifest import rebuild_manifest  # noqa: E402
from present import write_examples_index  # noqa: E402
from serialization import load_rollout  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        logging.error("GEMINI_API_KEY is not set")
        return 2

    run_dir = args.run_dir.resolve()
    episode_dirs = sorted(p.parent for p in run_dir.glob("*/*/ep*/rollout.json"))
    if not episode_dirs:
        logging.error("no episodes under %s", run_dir)
        return 1

    labeled = 0
    for directory in episode_dirs:
        try:
            if label_episode_dir(directory, model=args.model, refine=not args.no_refine, force=args.force):
                labeled += 1
        except Exception:
            logging.exception("labeling failed for %s", directory)

    rebuild_manifest(run_dir)
    rollouts = [load_rollout(d) for d in episode_dirs]
    index = write_examples_index(run_dir, rollouts)
    logging.info("labeled %d/%d episodes; wrote %s", labeled, len(episode_dirs), index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
