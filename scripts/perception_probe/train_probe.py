#!/usr/bin/env python3
"""Trains the perception-uncertainty probe on features cached by collect_features.py.

Offline supervised learning on frozen, pre-computed features -- no simulator,
no pi0.5 backbone, no gradient through anything but the small probe itself.
Splits by EPISODE (not by step) so every step of a held-out episode stays
out of training, avoiding leakage across steps of the same rollout.

Run with the top-level project venv (plain PyTorch, no JAX/openpi needed):
    .venv/bin/python scripts/perception_probe/train_probe.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import random
import signal
import sys

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_model import PerceptionSuccessProbe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "probe"
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.6, help="Used for gradient updates."
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Used only to pick the best checkpoint (per-epoch monitoring during training).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Untouched during training; used once at the end for the reported confusion "
        "matrix / ROC curve, so checkpoint selection (val) and final reporting (test) are "
        "never computed on the same episodes.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.10,
        help="Untouched by this script entirely -- carved out and persisted in split.json for "
        "post-hoc probability calibration (e.g. temperature/Platt scaling) later.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel .npz decompression (StepDataset loads lazily "
        "per-item, so with num_workers=0 all decompression happens serially on the main process). "
        "train and val loaders each get their own pool of this many persistent workers, and both "
        "run every epoch -- see --chunk-size for the memory implication of that.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Episodes decompressed together per shuffle-buffer chunk (per DataLoader worker). "
        "Each episode's .npz is decompressed exactly once per epoch this way -- steps from all "
        "episodes in the current chunk are pooled and shuffled together before batching, so "
        "batches still mix many different episodes, not just one. Peak memory per loader is "
        "roughly num_workers x chunk_size x ~80MB/episode -- and train + val loaders run "
        "concurrently every epoch (not just at the end), so the real budget to size against is "
        "2x that, not 1x. Defaults (4 x 16 x 2 loaders =~ 10GB) are deliberately conservative; "
        "the previous defaults (8 x 32) worked out to ~40GB combined and OOM-killed a worker.",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold for the final confusion matrix (matches "
        "eval_pi05_libero_with_probe.py's --probe-threshold default).",
    )
    parser.add_argument("--wandb-project", default="pi05-perception-probe")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    return parser.parse_args()


class EpisodeChunkDataset(IterableDataset):
    """Streams every inference-call step exactly once per epoch, grouped into
    shuffled chunks of episodes so each episode's .npz is decompressed exactly
    once per epoch (per worker), instead of once per individual step request.

    Naive step-level random access (the previous approach) scatters a given
    episode's ~20-50 steps randomly across the whole epoch, so by the time a
    later step of the same episode comes up its decompressed arrays have long
    been evicted from any bounded cache -- forcing the full episode to be
    decompressed again, up to once per step. That's what saturated disk I/O
    (workers were re-reading close to the whole dataset every few minutes)
    without ever reaching the GPU.

    Instead: shuffle episode order, take `chunk_size` episodes at a time,
    decompress all of them (bounded memory: chunk_size x ~80MB/episode), pool
    every step from that chunk, shuffle the pool, and yield from it before
    moving to the next chunk. Batches are drawn from a random mix of up to
    chunk_size different episodes -- not a single episode in a row -- so this
    does not reintroduce the per-step-label correlation problem; it only
    bounds how many episodes can co-occur in the same shuffle window.
    """

    def __init__(self, rows: list[dict], features_dir: Path, chunk_size: int = 32, shuffle: bool = True):
        self.rows = rows
        self.features_dir = features_dir
        self.chunk_size = chunk_size
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = get_worker_info()
        rows = list(self.rows)
        rng = random.Random()  # OS-seeded; fresh shuffle every __iter__ call (i.e. every epoch)
        if self.shuffle:
            rng.shuffle(rows)
        if worker_info is not None:
            rows = rows[worker_info.id :: worker_info.num_workers]

        for start in range(0, len(rows), self.chunk_size):
            chunk_rows = rows[start : start + self.chunk_size]
            pool = []
            for row in chunk_rows:
                with np.load(self.features_dir / row["npz_path"]) as data:
                    base_image = data["base_image"]
                    wrist_image = data["wrist_image"]
                    language = data["language"]
                    language_mask = data["language_mask"]
                label = float(row["success"] == "True")
                for t in range(base_image.shape[0]):
                    pool.append((base_image[t], wrist_image[t], language[t], language_mask[t], label))
            if self.shuffle:
                rng.shuffle(pool)
            for base, wrist, lang, lang_mask, label in pool:
                yield (
                    torch.from_numpy(base.astype(np.float32)),
                    torch.from_numpy(wrist.astype(np.float32)),
                    torch.from_numpy(lang.astype(np.float32)),
                    torch.from_numpy(lang_mask),
                    torch.tensor(label, dtype=torch.float32),
                )


def count_steps(rows: list[dict]) -> int:
    return sum(int(row["inference_calls"]) for row in rows)


def read_manifest(features_dir: Path) -> list[dict]:
    with (features_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _partition_by_fraction(
    rows: list[dict], fractions: list[tuple[str, float]], rng: np.random.Generator
) -> dict[str, list[dict]]:
    """Randomly partitions rows into named buckets sized by fractions. Every
    bucket's count is rounded independently except the last, which absorbs
    whatever remains -- guarantees an exact partition (no row lost or double-
    counted to independent rounding) regardless of how the fractions round.
    """
    n = len(rows)
    indices = rng.permutation(n)
    result: dict[str, list[dict]] = {}
    pos = 0
    for i, (name, fraction) in enumerate(fractions):
        if i == len(fractions) - 1:
            count = n - pos
        else:
            count = min(int(round(fraction * n)), n - pos)
        result[name] = [rows[j] for j in indices[pos : pos + count]]
        pos += count
    return result


def split_episodes(
    rows: list[dict],
    seed: int,
    train_fraction: float = 0.6,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    calibration_fraction: float = 0.10,
) -> dict[str, list[dict]]:
    """Splits into train/val/test/calibration by EPISODE (all steps of one
    episode land in the same split); stratified by success/failure AND by
    scene_variant so every split contains both classes and both scene variants
    regardless of overall class imbalance (~8.5% failures in this dataset).

    - train: used for gradient updates.
    - val: used only to pick the best checkpoint (per-epoch monitoring).
    - test: untouched during training; used once at the end for the reported
      confusion matrix / ROC curve, so checkpoint selection (val) and final
      reporting (test) are never the same episodes.
    - calibration: untouched by this script entirely; reserved for post-hoc
      probability calibration (e.g. temperature/Platt scaling) later.
    """
    fractions = [
        ("train", train_fraction),
        ("val", val_fraction),
        ("test", test_fraction),
        ("calibration", calibration_fraction),
    ]
    total_fraction = sum(fraction for _, fraction in fractions)
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-6):
        raise ValueError(f"train/val/test/calibration fractions must sum to 1.0, got {total_fraction}")

    rng = np.random.default_rng(seed)
    splits: dict[str, list[dict]] = {name: [] for name, _ in fractions}

    failure_rows = [row for row in rows if row["success"] != "True"]
    for name, part in _partition_by_fraction(failure_rows, fractions, rng).items():
        splits[name].extend(part)

    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        if row["success"] == "True":
            by_variant.setdefault(row["scene_variant"], []).append(row)
    for variant_rows in by_variant.values():
        for name, part in _partition_by_fraction(variant_rows, fractions, rng).items():
            splits[name].extend(part)

    return splits


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), no sklearn dependency."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: str, total_batches: int | None = None, desc: str = "val"
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (labels, scores, total_bce_loss_sum) over one full pass of loader."""
    model.eval()
    all_labels, all_scores = [], []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for base, wrist, lang, lang_mask, label in tqdm(
            loader, total=total_batches, desc=desc, unit="batch", leave=False
        ):
            base, wrist, lang, lang_mask, label = (
                base.to(device),
                wrist.to(device),
                lang.to(device),
                lang_mask.to(device),
                label.to(device),
            )
            logits = model(base, wrist, lang, lang_mask)
            total_loss += loss_fn(logits, label).item()
            all_labels.append(label.cpu().numpy())
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_scores), total_loss


def evaluate(model: nn.Module, loader: DataLoader, device: str, total_batches: int | None = None) -> dict:
    labels, scores, total_loss = collect_predictions(model, loader, device, total_batches)
    preds = (scores >= 0.5).astype(np.float32)
    return {
        "loss": total_loss / len(labels),
        "accuracy": float((preds == labels).mean()),
        "auroc": auroc(labels, scores),
        "base_rate": float(labels.mean()),
        "n": int(len(labels)),
    }


def confusion_matrix(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)
    return {
        "threshold": threshold,
        "tp": int(((preds == 1) & (labels == 1)).sum()),
        "tn": int(((preds == 0) & (labels == 0)).sum()),
        "fp": int(((preds == 1) & (labels == 0)).sum()),
        "fn": int(((preds == 0) & (labels == 1)).sum()),
    }


def roc_curve_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fpr, tpr) arrays tracing the ROC curve, via cumulative counts over
    scores sorted descending -- equivalent to sweeping the decision threshold from
    1 to 0, without needing sklearn."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_pos = float((labels == 1).sum())
    total_neg = float((labels == 0).sum())
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    tpr = np.concatenate([[0.0], tps / max(total_pos, 1.0)])
    fpr = np.concatenate([[0.0], fps / max(total_neg, 1.0)])
    return fpr, tpr


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auroc_value: float | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    label = f"probe (AUROC={auroc_value:.3f})" if auroc_value is not None else "probe (AUROC=n/a)"
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Perception-probe ROC curve (test, best checkpoint)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt()


def available_memory_gb() -> float | None:
    """Linux-only, no psutil dependency: reads MemAvailable straight from /proc/meminfo.
    Returns None if unavailable (e.g. non-Linux) rather than raising -- this check is a
    warning, not something worth failing a run over if it can't be read."""
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB -> GB
    except OSError:
        pass
    return None


def check_memory_budget(num_workers: int, chunk_size: int, bytes_per_episode: float = 80e6) -> None:
    """train and val DataLoaders each hold up to num_workers x chunk_size episodes'
    decompressed arrays resident at once, and BOTH run every epoch (not just at the end) --
    so the real peak to check against is 2x one loader's worst case, not 1x. Warns loudly
    instead of silently OOM-killing a worker 20 minutes into a run, which is what happened
    with this script's previous defaults (8 workers x 32 chunk_size => ~40GB combined)."""
    per_loader_gb = num_workers * chunk_size * bytes_per_episode / 1e9
    peak_gb = 2 * per_loader_gb  # train + val loaders concurrent, worst case
    available_gb = available_memory_gb()
    logging.info(
        "Estimated peak DataLoader memory: ~%.1fGB (train + val loaders, %d workers x %d chunk_size each)",
        peak_gb, num_workers, chunk_size,
    )
    if available_gb is not None:
        logging.info("Currently available system memory: %.1fGB", available_gb)
        if peak_gb > 0.7 * available_gb:
            logging.warning(
                "Estimated peak DataLoader memory (~%.1fGB) is close to or over available system "
                "memory (%.1fGB) -- a worker is likely to get OOM-killed mid-run. Lower "
                "--num-workers and/or --chunk-size (their product is what matters), or free up "
                "memory (check `ps aux --sort=-%%mem` for other heavy processes) before retrying.",
                peak_gb, available_gb,
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)

    # Ctrl-C (SIGINT) already raises KeyboardInterrupt; also map plain `kill <pid>`
    # (SIGTERM) to the same exception so stopping the run either way finishes the
    # current epoch's bookkeeping and still produces a confusion matrix + ROC curve
    # off whatever probe_best.pt exists so far. `kill -9` (SIGKILL) can't be caught
    # by any process, so that still leaves things as they were at the last save.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    check_memory_budget(args.num_workers, args.chunk_size)

    # Every run gets its own timestamped subdirectory under --output-dir, never writing
    # directly into a shared path -- a careless verification/smoke-test run pointed at the
    # bare --output-dir once overwrote a real, fully-trained probe_last.pt + history.json in
    # place with no way to recover them. A "latest" symlink is kept pointing at the most
    # recent run's directory so default --probe-checkpoint/--split-json paths in the eval
    # scripts keep working without needing to be updated by hand after every run.
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Run outputs going to %s", run_dir)

    rows = read_manifest(args.features_dir)
    splits = split_episodes(
        rows,
        args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    train_rows, val_rows, test_rows, calibration_rows = (
        splits["train"], splits["val"], splits["test"], splits["calibration"]
    )
    failures = {name: sum(1 for r in part if r["success"] != "True") for name, part in splits.items()}
    logging.info(
        "Episodes: %d train (%d failures), %d val (%d failures), %d test (%d failures), "
        "%d calibration (%d failures) (of %d total)",
        len(train_rows), failures["train"],
        len(val_rows), failures["val"],
        len(test_rows), failures["test"],
        len(calibration_rows), failures["calibration"],
        len(rows),
    )

    split_key = lambda r: {  # noqa: E731
        "npz_path": r["npz_path"],
        "scene_variant": r["scene_variant"],
        "task": r["task"],
        "episode": r["episode"],
        "success": r["success"],
    }
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
                "calibration_fraction": args.calibration_fraction,
                "manifest_rows_at_split_time": len(rows),
                "train": [split_key(r) for r in train_rows],
                "val": [split_key(r) for r in val_rows],
                "test": [split_key(r) for r in test_rows],
                "calibration": [split_key(r) for r in calibration_rows],
            },
            indent=2,
        )
    )
    logging.info("Saved train/val/test/calibration episode split to %s", run_dir / "split.json")

    train_steps = count_steps(train_rows)
    val_steps = count_steps(val_rows)
    test_steps = count_steps(test_rows)
    logging.info("Steps: %d train, %d val, %d test", train_steps, val_steps, test_steps)

    train_ds = EpisodeChunkDataset(train_rows, args.features_dir, chunk_size=args.chunk_size, shuffle=True)
    val_ds = EpisodeChunkDataset(val_rows, args.features_dir, chunk_size=args.chunk_size, shuffle=False)
    test_ds = EpisodeChunkDataset(test_rows, args.features_dir, chunk_size=args.chunk_size, shuffle=False)

    # persistent_workers keeps a loader's whole worker pool (and each worker's in-flight
    # chunk_size-episode buffer, ~num_workers x chunk_size x 80MB/episode) resident as OS
    # processes for as long as the loader object is alive -- not just while actively
    # iterating it. train is iterated every epoch, many epochs per run, so paying the
    # worker-spawn cost once and reusing the pool is worth it. val is also iterated every
    # epoch, but only for one short pass -- keeping its pool alive continuously in between
    # means its workers' resident memory sits alongside train's for the whole run, and a
    # subprocess's RSS reflects its high-water mark, not its current live objects (glibc
    # doesn't hand freed heap back to the OS eagerly) -- so train's workers can still look
    # "big" in RSS right as val's are ramping up for their own chunk, even though train's
    # generator already finished yielding. Non-persistent val workers are spun up fresh for
    # each validation pass and fully torn down afterward, so there's no window where both
    # pools are resident at once -- the small per-epoch respawn cost is worth it for that.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, drop_last=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    positives = sum(int(row["inference_calls"]) for row in train_rows if row["success"] == "True")
    negatives = train_steps - positives
    pos_weight = torch.tensor(negatives / max(positives, 1), dtype=torch.float32, device=args.device)
    logging.info("Train label balance: %d success / %d failure steps (pos_weight=%.3f)", positives, negatives, pos_weight.item())

    model = PerceptionSuccessProbe().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.config.update(
            {
                "train_episodes": len(train_rows),
                "train_failures": failures["train"],
                "val_episodes": len(val_rows),
                "val_failures": failures["val"],
                "test_episodes": len(test_rows),
                "test_failures": failures["test"],
                "calibration_episodes": len(calibration_rows),
                "calibration_failures": failures["calibration"],
                "train_steps": train_steps,
                "val_steps": val_steps,
                "test_steps": test_steps,
                "pos_weight": pos_weight.item(),
            }
        )

    best_val_loss = float("inf")
    best_epoch = None
    history = []

    train_batches = -(-train_steps // args.batch_size)  # ceil div; total is approximate since
    val_batches = -(-val_steps // args.batch_size)  # worker splitting can round chunk boundaries
    test_batches = -(-test_steps // args.batch_size)

    interrupted = False
    try:
        try:
            epoch_bar = tqdm(range(args.epochs), desc="epochs", unit="epoch")
            for epoch in epoch_bar:
                model.train()
                running_loss, running_n = 0.0, 0
                batch_bar = tqdm(
                    train_loader, total=train_batches, desc=f"epoch {epoch} train", unit="batch", leave=False
                )
                for base, wrist, lang, lang_mask, label in batch_bar:
                    base, wrist, lang, lang_mask, label = (
                        base.to(args.device),
                        wrist.to(args.device),
                        lang.to(args.device),
                        lang_mask.to(args.device),
                        label.to(args.device),
                    )
                    optimizer.zero_grad()
                    logits = model(base, wrist, lang, lang_mask)
                    loss = loss_fn(logits, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * label.shape[0]
                    running_n += label.shape[0]
                    batch_bar.set_postfix(loss=f"{running_loss / running_n:.4f}")

                train_metrics = {"loss": running_loss / running_n}
                val_metrics = evaluate(model, val_loader, args.device, total_batches=val_batches)
                history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
                epoch_bar.set_postfix(
                    train_loss=f"{train_metrics['loss']:.4f}",
                    val_loss=f"{val_metrics['loss']:.4f}",
                    val_auroc=f"{val_metrics['auroc']:.3f}" if val_metrics["auroc"] is not None else "n/a",
                )
                logging.info(
                    "epoch %2d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  val_auroc=%s",
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    f"{val_metrics['auroc']:.3f}" if val_metrics["auroc"] is not None else "n/a",
                )
                if not args.no_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss": train_metrics["loss"],
                            "val/loss": val_metrics["loss"],
                            "val/accuracy": val_metrics["accuracy"],
                            "val/auroc": val_metrics["auroc"],
                            "val/base_rate": val_metrics["base_rate"],
                        }
                    )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_epoch = epoch
                    torch.save(model.state_dict(), run_dir / "probe_best.pt")
        except KeyboardInterrupt:
            interrupted = True
            logging.warning(
                "Training interrupted (%d epoch%s completed) -- finalizing with the best checkpoint "
                "saved so far instead of crashing out.",
                len(history),
                "" if len(history) == 1 else "s",
            )

        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), run_dir / "probe_last.pt")
        logging.info("Saved probe weights + history to %s", run_dir)

        latest_link = args.output_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name, target_is_directory=True)
        logging.info("Updated %s -> %s", latest_link, run_dir.name)

        # Training is done with these -- release their persistent worker pools (each holding
        # up to num_workers x chunk_size x ~80MB/episode resident) before spinning up the test
        # pass, instead of running train/val/test worker pools all at once. Without this, a
        # DataLoader worker gets OOM-killed right as the test pass starts (observed in practice).
        del train_loader, val_loader

        if best_epoch is None:
            logging.warning(
                "No epoch finished before interruption -- no probe_best.pt to evaluate, "
                "skipping confusion matrix and ROC curve."
            )
        else:
            # Confusion matrix + ROC curve on TEST, using the BEST checkpoint (the one
            # eval_pi05_libero_with_probe.py actually loads by default), not the last epoch --
            # meaningful whether training ran to completion or was stopped early. Deliberately
            # test, not val: val already picked this checkpoint (lowest val loss across epochs),
            # so reporting on val again would evaluate the model on the same data that selected
            # it, inflating the reported numbers. Test is never touched until this one pass.
            model.load_state_dict(torch.load(run_dir / "probe_best.pt", map_location=args.device))
            test_labels, test_scores, _ = collect_predictions(
                model, test_loader, args.device, test_batches, desc="test (final)"
            )
            cm = confusion_matrix(test_labels, test_scores, args.eval_threshold)
            final_auroc = auroc(test_labels, test_scores)
            (run_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))
            logging.info(
                "Confusion matrix (best checkpoint, threshold=%.2f): TP=%d TN=%d FP=%d FN=%d",
                args.eval_threshold, cm["tp"], cm["tn"], cm["fp"], cm["fn"],
            )
            logging.info(
                "Final test AUROC (best checkpoint): %s", f"{final_auroc:.3f}" if final_auroc is not None else "n/a"
            )

            roc_path = run_dir / "roc_curve.png"
            fpr, tpr = roc_curve_points(test_labels, test_scores)
            plot_roc_curve(fpr, tpr, final_auroc, roc_path)
            logging.info("Saved ROC curve to %s", roc_path)

            if not args.no_wandb:
                wandb.summary["best_epoch"] = best_epoch
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["final_test_auroc"] = final_auroc
                wandb.summary["confusion_matrix"] = cm
                wandb.summary["interrupted"] = interrupted
                wandb.log({"test/roc_curve": wandb.Image(str(roc_path))})

                best_artifact = wandb.Artifact(
                    "probe_best",
                    type="model",
                    metadata={"epoch": best_epoch, "val_loss": best_val_loss, "confusion_matrix": cm, "test_auroc": final_auroc},
                )
                best_artifact.add_file(str(run_dir / "probe_best.pt"))
                wandb.log_artifact(best_artifact)

                last_artifact = wandb.Artifact(
                    "probe_last",
                    type="model",
                    metadata={"epoch": len(history) - 1, "val_loss": history[-1]["val"]["loss"], "interrupted": interrupted},
                )
                last_artifact.add_file(str(run_dir / "probe_last.pt"))
                wandb.log_artifact(last_artifact)
    finally:
        if not args.no_wandb:
            wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
