#!/usr/bin/env python3
"""Runs a trained probe checkpoint over every episode in one or more of its
run's split.json splits (default: just `calibration`) and caches each
episode's FULL raw + accumulated per-timestep score trajectory to disk,
keyed by run folder + checkpoint + which splits were included.

Why this exists: Appendix B of FAIL-Detect (arXiv:2503.08558) constructs a
functional-conformal-prediction band from calibration-set score TRAJECTORIES
(mu_t, the per-timestep modulation function, per-episode max deviations,
etc. -- see 13-SAFE/failure_prob/utils/conformal/functional_predictor.py for
the matching implementation). Building and iterating on that CP-band code
means re-slicing/re-aggregating the same trajectories over and over, but
none of it needs a fresh probe forward pass each time -- so this script
does the (GPU) inference part ONCE per (run, checkpoint, split set) and
caches the result; the CP-band construction itself can then be pure numpy
against the cached .npz, no model/features_dir/torch involved at all.

Choosing --splits:
  - 'calibration' (default): the run's own dedicated calibration split.
  - 'calibration test': `test` episodes are on SEEN tasks but were never used
    for training or class-weight computation either (see
    train_probe_time_dependent.py's split_episodes -- train/val/test/
    calibration are disjoint), so they're just as valid a conformal-
    calibration pool as `calibration` itself, and pooling them in meaningfully
    sharpens the CP quantile given how small `calibration` alone is (10% of
    seen episodes -- see construct_cp_band.ipynb's own warning about how
    coarse that makes N1/N2).
  - 'unseen': the zero-shot held-out TASKS (never seen during training AT
    ALL, not just held out of the loss) -- pass this alone to see how a band/
    threshold calibrated on seen-task data would need to look to also cover
    genuinely novel tasks, as opposed to just novel episodes of seen tasks.
  - 'val' is deliberately not offered here even though it's also never
    trained on directly: it's used every epoch for checkpoint selection
    (best_val_auroc), so it's not independent of the final chosen model the
    way test/calibration/unseen are.

Output: <output-dir>/<run_dir.parent.name>_<run_dir.name>_<checkpoint stem>_<splits>.npz
containing, per pooled episode: raw_scores and scores (both ragged
float32 arrays via allow_pickle=True object arrays, since episodes have
different lengths), plus lengths/tasks/success/scene_variant/source_split
arrays and metadata (cumsum, rmean, checkpoint, task_min_step). Skips
recomputation if the cache file already exists -- pass --force to overwrite.

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/compute_calibration_scores.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_arch_recovery import load_probe_model  # noqa: E402
from train_probe_time_dependent import EpisodeSequenceDataset, score_sequence  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A train_probe_time_dependent.py output run folder.")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "calibration",
        help="Where to cache the computed calibration score trajectories.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--rmean", action="store_true", default=False,
        help="Should match how the run was trained (see split.json's directory / training log for "
        "the --cumsum/--rmean flags actually used) -- only affects the cached `scores` array, not "
        "`raw_scores`, which is always the pre-accumulation per-step sigmoid regardless of these flags.",
    )
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    parser.add_argument(
        "--splits", nargs="+", default=["calibration"], choices=["calibration", "test", "unseen"],
        help="Which split.json split(s) to pool and score (see module docstring for what each "
        "means and why 'val' is excluded). Default: just 'calibration'.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if a matching cache file exists.")
    return parser.parse_args(argv)


def cache_path_for(run_dir: Path, checkpoint: str, output_dir: Path, splits: list[str]) -> Path:
    checkpoint_stem = Path(checkpoint).stem
    splits_suffix = "+".join(sorted(splits))
    return output_dir / f"{run_dir.parent.name}_{run_dir.name}_{checkpoint_stem}_{splits_suffix}.npz"


def compute_episode_arrays(
    model: torch.nn.Module, item: dict, cumsum: bool, rmean: bool, device: str
) -> tuple[np.ndarray, np.ndarray]:
    base = item["base_image"].unsqueeze(0).to(device)
    wrist = item["wrist_image"].unsqueeze(0).to(device)
    lang = item["language"].unsqueeze(0).to(device)
    lang_mask = item["language_mask"].unsqueeze(0).to(device)
    batch = {"base_image": base, "wrist_image": wrist, "language": lang, "language_mask": lang_mask}
    with torch.no_grad():
        _, raw_scores, scores = score_sequence(model, batch, cumsum, rmean)
    T = item["length"]
    return raw_scores[0, :T].cpu().numpy().astype(np.float32), scores[0, :T].cpu().numpy().astype(np.float32)


def ensure_cached(args: argparse.Namespace) -> Path:
    """Does the actual work (or confirms a cache hit) and returns the cache_path -- factored
    out of main() so run_joint_threshold_sweep.py can call this directly with a
    programmatically-built args.Namespace instead of shelling out."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for(args.run_dir, args.checkpoint, args.output_dir, args.splits)

    if cache_path.is_file() and not args.force:
        logging.info("Cache already exists at %s (pass --force to recompute) -- loading it to confirm.", cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        n = len(cached["lengths"])
        n_fail = int((~cached["success"]).sum())
        logging.info(
            "Cached calibration scores: n=%d (%d success, %d failure), splits=%s, cumsum=%s, rmean=%s, checkpoint=%s",
            n, n - n_fail, n_fail, cached["splits"].item(),
            cached["cumsum"].item(), cached["rmean"].item(), cached["checkpoint"].item(),
        )
        return cache_path

    split = json.loads((args.run_dir / "split.json").read_text())
    splits_used = list(args.splits)
    pooled_rows, source_splits = [], []
    for split_name in splits_used:
        split_rows = split[split_name]
        if not split_rows:
            raise ValueError(f"{args.run_dir}/split.json has an empty '{split_name}' split.")
        pooled_rows.extend(split_rows)
        source_splits.extend([split_name] * len(split_rows))
    logging.info(
        "Scoring %d episodes from %s (splits: %s)",
        len(pooled_rows), args.run_dir, ", ".join(f"{name}={splits_used.count(name)}" for name in set(splits_used)),
    )
    for split_name in splits_used:
        n_this = sum(1 for s in source_splits if s == split_name)
        logging.info("  %s: %d episodes", split_name, n_this)

    model, arch_source = load_probe_model(args.run_dir, args.checkpoint, args.device)
    logging.info("Loaded model architecture via: %s", arch_source)

    ds = EpisodeSequenceDataset(pooled_rows, args.features_dir)

    raw_scores_list, scores_list = [], []
    lengths, tasks, success, scene_variant = [], [], [], []
    for i in tqdm(range(len(ds)), desc="calibration episodes", unit="ep"):
        item = ds[i]
        raw, scores = compute_episode_arrays(model, item, args.cumsum, args.rmean, args.device)
        raw_scores_list.append(raw)
        scores_list.append(scores)
        lengths.append(item["length"])
        tasks.append(item["task"])
        success.append(item["label"] == 1.0)
        scene_variant.append(pooled_rows[i]["scene_variant"])

    np.savez(
        cache_path,
        raw_scores=np.array(raw_scores_list, dtype=object),
        scores=np.array(scores_list, dtype=object),
        lengths=np.array(lengths, dtype=np.int64),
        tasks=np.array(tasks, dtype=object),
        success=np.array(success, dtype=bool),
        scene_variant=np.array(scene_variant, dtype=object),
        source_split=np.array(source_splits, dtype=object),
        splits="+".join(splits_used),
        cumsum=args.cumsum,
        rmean=args.rmean,
        checkpoint=args.checkpoint,
        run_dir=str(args.run_dir),
        task_min_step=json.dumps(split["task_min_step"]),
        arch_source=arch_source,
    )
    n_fail = sum(1 for s in success if not s)
    logging.info(
        "Wrote %d pooled calibration episodes' score trajectories (%d success, %d failure) to %s",
        len(pooled_rows), len(pooled_rows) - n_fail, n_fail, cache_path,
    )
    return cache_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_cached(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
