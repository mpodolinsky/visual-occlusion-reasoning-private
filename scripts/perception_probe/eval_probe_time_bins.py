#!/usr/bin/env python3
"""Evaluates a trained perception probe on the held-out TEST set, broken out by
inference-call time bin, to see whether later steps in an episode (closer to
success/failure) are easier for the probe to classify than earlier ones.

Reads split.json (written by train_probe.py) to know exactly which episodes
are test, so this never accidentally evaluates on training or val data (val
already picked the checkpoint via per-epoch monitoring; test is the split
that's never touched until a final evaluation pass like this one) -- if
split.json doesn't exist yet or is stale relative to manifest.csv, retrain
first (train_probe.py always overwrites split.json to match whatever
manifest.csv it was run against).

Bins are equally wide, spanning [0, max_frames_observed) where max_frames_observed
is the longest episode (by inference-call count) in the held-out test set --
NOT the full dataset, since that's what's actually being evaluated here.

Run with the top-level project venv (plain PyTorch, no JAX/openpi needed):
    .venv/bin/python scripts/perception_probe/eval_probe_time_bins.py
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_model import PerceptionSuccessProbe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features"
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "latest" / "split.json",
        help="Written by train_probe.py; defines exactly which episodes are held out.",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "latest" / "probe_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "time_bins",
    )
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Matches eval_pi05_libero_with_probe.py's default."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Steps per probe forward pass.")
    return parser.parse_args()


def load_test_episodes(split_json: Path) -> list[dict]:
    if not split_json.is_file():
        raise FileNotFoundError(
            f"{split_json} not found -- run train_probe.py first, it writes split.json alongside "
            "the checkpoint so this script knows exactly which episodes are test."
        )
    return json.loads(split_json.read_text())["test"]


def bin_confusion_matrix(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (
        0.0 if precision is not None and recall is not None else None
    )
    return {
        "n": len(labels),
        "n_success": int(labels.sum()),
        "n_failure": int(len(labels) - labels.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float((preds == labels).mean()),
    }


def empty_bin_row() -> dict:
    return {
        "n": 0,
        "n_success": 0,
        "n_failure": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
    }


def plot_bins(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = [(r["frame_start"] + r["frame_end"]) / 2 for r in rows]
    f1 = [r["f1"] if r["f1"] is not None else float("nan") for r in rows]
    accuracy = [r["accuracy"] if r["accuracy"] is not None else float("nan") for r in rows]
    n = [r["n"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=[3, 1])
    ax1.plot(centers, f1, marker="o", label="F1")
    ax1.plot(centers, accuracy, marker="o", label="accuracy")
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax1.set_ylabel("score")
    ax1.set_title("Probe performance vs. inference-call time bin (held-out test)")
    ax1.legend()

    ax2.bar(centers, n, width=(rows[0]["frame_end"] - rows[0]["frame_start"]) * 0.9)
    ax2.set_xlabel("inference-call index (frame bin)")
    ax2.set_ylabel("n steps")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    test_rows = load_test_episodes(args.split_json)
    logging.info("Loaded %d held-out test episodes from %s", len(test_rows), args.split_json)

    probe = PerceptionSuccessProbe().to(args.device)
    probe.load_state_dict(torch.load(args.probe_checkpoint, map_location=args.device))
    probe.eval()
    logging.info("Loaded probe from %s onto %s", args.probe_checkpoint, args.device)

    # max_frames from manifest.csv's inference_calls -- no .npz decompression needed for this,
    # so we don't pay a full read-through just to find episode lengths.
    inference_calls_by_path = {}
    with (args.features_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        for manifest_row in csv.DictReader(stream):
            inference_calls_by_path[manifest_row["npz_path"]] = int(manifest_row["inference_calls"])
    max_frames = max(inference_calls_by_path[row["npz_path"]] for row in test_rows)
    logging.info("Max frames observed (test set): %d", max_frames)

    bin_edges = np.linspace(0, max_frames, args.num_bins + 1)
    bin_width = max_frames / args.num_bins if max_frames > 0 else 1.0

    # One episode decompressed at a time -- holding the whole held-out set in RAM at once
    # is the same eager-loading mistake that OOM-killed train_probe.py at this dataset size.
    all_bins: list[int] = []
    all_labels: list[float] = []
    all_scores: list[float] = []
    with torch.no_grad():
        for row in test_rows:
            with np.load(args.features_dir / row["npz_path"]) as data:
                base = data["base_image"]
                wrist = data["wrist_image"]
                lang = data["language"]
                lang_mask = data["language_mask"]
                label = float(row["success"] == "True")
                num_steps = base.shape[0]
                for start in range(0, num_steps, args.batch_size):
                    end = min(start + args.batch_size, num_steps)
                    base_t = torch.from_numpy(base[start:end].astype(np.float32)).to(args.device)
                    wrist_t = torch.from_numpy(wrist[start:end].astype(np.float32)).to(args.device)
                    lang_t = torch.from_numpy(lang[start:end].astype(np.float32)).to(args.device)
                    mask_t = torch.from_numpy(lang_mask[start:end]).to(args.device)
                    logits = probe(base_t, wrist_t, lang_t, mask_t)
                    scores = torch.sigmoid(logits).cpu().numpy()
                    for offset, score in enumerate(scores):
                        t = start + offset
                        bin_idx = min(int(t // bin_width), args.num_bins - 1)
                        all_bins.append(bin_idx)
                        all_labels.append(label)
                        all_scores.append(float(score))

    bins_arr = np.asarray(all_bins)
    labels_arr = np.asarray(all_labels)
    scores_arr = np.asarray(all_scores)
    logging.info("Total steps evaluated: %d", len(labels_arr))

    rows = []
    for bin_idx in range(args.num_bins):
        mask = bins_arr == bin_idx
        n = int(mask.sum())
        row = empty_bin_row() if n == 0 else bin_confusion_matrix(labels_arr[mask], scores_arr[mask], args.threshold)
        row["bin"] = bin_idx
        row["frame_start"] = float(bin_edges[bin_idx])
        row["frame_end"] = float(bin_edges[bin_idx + 1])
        rows.append(row)

    # Timestamped subdirectory per run -- this unconditionally overwrites its output files
    # on every run (no resume/skip logic), so a fixed shared path would silently clobber a
    # previous analysis (possibly of a different checkpoint). "latest" tracks the most recent.
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "time_bin_confusion.json").write_text(json.dumps(rows, indent=2))
    fieldnames = [
        "bin", "frame_start", "frame_end", "n", "n_success", "n_failure",
        "tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy",
    ]
    with (run_dir / "time_bin_confusion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_path = run_dir / "time_bin_confusion.png"
    plot_bins(rows, plot_path)

    latest_link = args.output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name, target_is_directory=True)

    logging.info(
        "%-4s %-16s %5s %5s %5s %4s %4s %4s %4s %7s %7s %7s %7s",
        "bin", "frames", "n", "ok", "fail", "tp", "tn", "fp", "fn", "prec", "rec", "f1", "acc",
    )
    for r in rows:
        frame_range = f"[{r['frame_start']:.0f},{r['frame_end']:.0f})"
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "n/a"  # noqa: E731
        logging.info(
            "%-4d %-16s %5d %5d %5d %4d %4d %4d %4d %7s %7s %7s %7s",
            r["bin"], frame_range, r["n"], r["n_success"], r["n_failure"],
            r["tp"], r["tn"], r["fp"], r["fn"],
            fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]), fmt(r["accuracy"]),
        )
    logging.info("Saved results to %s (latest -> %s)", run_dir, latest_link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
