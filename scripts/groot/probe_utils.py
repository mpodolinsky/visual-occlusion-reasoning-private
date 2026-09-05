"""Vendored helpers for the GR00T perception-probe fork.

Copied verbatim (no behavioural change) from scripts/perception_probe/ so this
directory's probe training is self-contained and does not import from the pi0.5
probe package:

  - auroc, confusion_matrix, roc_curve_points, plot_roc_curve,
    available_memory_gb, split_episodes, _partition_by_fraction
        <- scripts/perception_probe/train_probe.py
  - plot_overlay, _group_mean_curve
        <- scripts/perception_probe/rollout_unseen_with_scores.py

Pure numpy / matplotlib -- no openpi, no JAX, no torch.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# metrics + ROC (train_probe.py)
# --------------------------------------------------------------------------- #


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), no sklearn dependency."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


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


# --------------------------------------------------------------------------- #
# episode splitting (train_probe.py)
# --------------------------------------------------------------------------- #


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
    regardless of overall class imbalance.

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


# --------------------------------------------------------------------------- #
# score-overlay plot (rollout_unseen_with_scores.py)
# --------------------------------------------------------------------------- #


def _group_mean_curve(
    group: list[list[tuple[int, float, float]]],
) -> tuple[list[float], list[float], list[float]] | None:
    """Averages raw/accumulated scores INDEX-by-INDEX (not by wall-clock time)
    across a group of same-outcome episodes, so a longer episode's tail just
    drops out of the average once shorter episodes in the group have ended,
    rather than the whole curve needing a common length. Index i's x
    position is taken from whichever episode reaches that index first,
    since every trace advances one inference call at a time regardless of
    episode length (same replan_steps throughout)."""
    if not group:
        return None
    max_len = max(len(trace) for trace in group)
    xs, mean_raw, mean_acc = [], [], []
    for i in range(max_len):
        at_i = [trace[i] for trace in group if i < len(trace)]
        xs.append(at_i[0][0] / 20.0)
        mean_raw.append(sum(p[1] for p in at_i) / len(at_i))
        mean_acc.append(sum(p[2] for p in at_i) / len(at_i))
    return xs, mean_raw, mean_acc


def plot_overlay(
    traces: list[tuple[list[tuple[int, float, float]], bool]], args, path: Path, title: str | None = None,
) -> None:
    """All rollouts' score-over-time curves on one pair of axes -- failures in
    red, successes in blue -- so patterns that only show up across many
    episodes (e.g. failures separating from successes earlier/later, or not
    at all) are visible at a glance instead of comparing plots one at a time.
    Each group's index-wise mean is overlaid as a bold line on top of the
    thin individual ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    individual_success, individual_fail = "cornflowerblue", "lightcoral"
    mean_success, mean_fail = "navy", "darkred"

    fig, (ax_raw, ax_acc) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    success_traces = [trace for trace, success in traces if success]
    fail_traces = [trace for trace, success in traces if not success]
    for trace, success in traces:
        seconds = [t[0] / 20.0 for t in trace]  # LIBERO runs at 20Hz
        raw_scores = [t[1] for t in trace]
        accumulated = [t[2] for t in trace]
        color = individual_success if success else individual_fail
        ax_raw.plot(seconds, raw_scores, color=color, alpha=0.35, linewidth=1)
        ax_acc.plot(seconds, accumulated, color=color, alpha=0.35, linewidth=1)

    for group, color in ((success_traces, mean_success), (fail_traces, mean_fail)):
        mean_curve = _group_mean_curve(group)
        if mean_curve is None:
            continue
        xs, mean_raw, mean_acc = mean_curve
        ax_raw.plot(xs, mean_raw, color=color, alpha=1.0, linewidth=3, zorder=5)
        ax_acc.plot(xs, mean_acc, color=color, alpha=1.0, linewidth=3, zorder=5)

    ax_raw.set_ylabel("raw p(failure)")
    ax_raw.set_ylim(-0.05, 1.05)
    ax_raw.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    acc_label = "running mean" if args.rmean else ("cumsum" if args.cumsum else "raw")
    ax_acc.set_ylabel(f"accumulated score ({acc_label})")
    ax_acc.set_xlabel("time (s)")
    if args.rmean:
        ax_acc.set_ylim(-0.05, 1.05)

    handles = [
        plt.Line2D([0], [0], color=individual_success, linewidth=1, alpha=0.6, label=f"success (n={len(success_traces)})"),
        plt.Line2D([0], [0], color=individual_fail, linewidth=1, alpha=0.6, label=f"failure (n={len(fail_traces)})"),
        plt.Line2D([0], [0], color=mean_success, linewidth=3, label="success mean"),
        plt.Line2D([0], [0], color=mean_fail, linewidth=3, label="failure mean"),
    ]
    ax_raw.legend(handles=handles, loc="upper right")
    fig.suptitle(title or f"All rollouts overlaid (n={len(traces)})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
