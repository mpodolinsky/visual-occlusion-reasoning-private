#!/usr/bin/env python3
"""Computes empirical SELECTIVE risk across a sequence of abstention
thresholds on the held-out CALIBRATION split, as the first step towards a
Learn-Then-Test (LTT) style selective-abstention calibration: choosing an
abstention cutoff with a PAC-style guarantee on the model's error rate,
conditional on not abstaining.

The classification rule itself is fixed, not swept: the probe's raw score is
treated as P(success), so P(failure) = 1 - score, and the predicted class is
whichever is larger -- i.e. predict success iff score > 0.5. What IS swept is
lambda, an abstention cutoff on the winning class's probability,
confidence = max(score, 1 - score) in [0.5, 1]: for a given lambda, we only
make a prediction on steps where confidence >= lambda, and abstain
otherwise. The empirical (selective) risk R_hat(lambda) is the
misclassification rate restricted to the covered (non-abstained) subset --
NOT computed over all steps, since abstained steps make no prediction to be
right or wrong about.

This script only computes and plots the empirical selective risk curve (and
coverage, and its Clopper-Pearson upper confidence bound) -- the later LTT
step (turning the per-lambda UCB into a PAC-valid, multiplicity-corrected
test to select a lambda with risk control at some target level) is not
implemented here.

Uses the CALIBRATION split specifically (not test): split.json's
"calibration" episodes are never touched by train_probe.py itself (not used
for gradient updates, checkpoint selection, or final test-set reporting), so
they're the right, still-unseen-by-any-decision data to calibrate a
threshold against.

Run with the top-level project venv (plain PyTorch, no JAX/openpi needed):
    .venv/bin/python scripts/perception_probe/calibrate_probe.py
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
from scipy.stats import beta
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
        help="Written by train_probe.py; defines exactly which episodes are calibration.",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "latest" / "probe_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "calibration",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=100,
        help="Abstention cutoffs lambda swept over [0.5, 1.0], inclusive. Below 0.5 every step "
        "would be covered (confidence = max(score, 1-score) is never < 0.5), so that half of "
        "[0, 1] is redundant and excluded.",
    )
    parser.add_argument(
        "--min-covered-samples",
        type=int,
        default=25,
        help="Thresholds whose covered (non-abstained) subset has fewer than this many samples "
        "are dropped from risk/risk_ucb (set to None/NaN) rather than reported -- a selective "
        "risk computed from a handful of covered steps is too noisy to be meaningful, and the "
        "Clopper-Pearson UCB in particular gets very wide at small n.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Per-threshold miscoverage level for the Clopper-Pearson upper confidence bound on "
        "risk: P(true risk R(lambda) <= R_ucb(lambda)) >= 1 - delta. This is the pointwise bound "
        "LTT combines (e.g. via a union bound / Bonferroni correction over all num_thresholds "
        "lambdas) into a family-wise PAC guarantee -- that combination step is not done here.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Steps per probe forward pass.")
    return parser.parse_args()


def load_calibration_episodes(split_json: Path) -> list[dict]:
    if not split_json.is_file():
        raise FileNotFoundError(
            f"{split_json} not found -- run train_probe.py first, it writes split.json alongside "
            "the checkpoint so this script knows exactly which episodes are calibration."
        )
    return json.loads(split_json.read_text())["calibration"]


def collect_calibration_predictions(
    probe: torch.nn.Module,
    features_dir: Path,
    rows: list[dict],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, scores) over every step of every calibration episode.

    Decompresses one episode's .npz at a time -- holding the whole calibration
    set in RAM at once is the same eager-loading mistake that OOM-killed
    train_probe.py at this dataset size.
    """
    all_labels: list[float] = []
    all_scores: list[float] = []
    with torch.no_grad():
        for row in rows:
            with np.load(features_dir / row["npz_path"]) as data:
                base = data["base_image"]
                wrist = data["wrist_image"]
                lang = data["language"]
                lang_mask = data["language_mask"]
                label = float(row["success"] == "True")
                num_steps = base.shape[0]
                for start in range(0, num_steps, batch_size):
                    end = min(start + batch_size, num_steps)
                    base_t = torch.from_numpy(base[start:end].astype(np.float32)).to(device)
                    wrist_t = torch.from_numpy(wrist[start:end].astype(np.float32)).to(device)
                    lang_t = torch.from_numpy(lang[start:end].astype(np.float32)).to(device)
                    mask_t = torch.from_numpy(lang_mask[start:end]).to(device)
                    logits = probe(base_t, wrist_t, lang_t, mask_t)
                    scores = torch.sigmoid(logits).cpu().numpy()
                    all_scores.extend(float(s) for s in scores)
                    all_labels.extend([label] * (end - start))
    return np.asarray(all_labels), np.asarray(all_scores)


def clopper_pearson_upper(n_incorrect: int, n: int, delta: float) -> float:
    """Exact one-sided (1 - delta) Clopper-Pearson upper confidence bound on a
    binomial proportion, i.e. the supremum over true risk r such that the
    binomial CDF P(Binom(n, r) <= n_incorrect) is still >= delta:

        R_ucb = sup { r in [0, 1] : Binom_CDF(n_incorrect; n, r) >= delta }

    Computed via the standard Beta-quantile inversion of that CDF (exact, no
    normal/Wald approximation, valid at any n including small calibration
    sets): R_ucb = Beta.ppf(1 - delta, n_incorrect + 1, n - n_incorrect),
    with the usual edge case that R_ucb = 1 when every step is misclassified.
    """
    if n_incorrect == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, n_incorrect + 1, n - n_incorrect))


def empirical_selective_risk_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int,
    delta: float,
    min_covered_samples: int,
) -> list[dict]:
    """Selective risk R_hat(lambda) for an abstention cutoff lambda swept evenly
    over [0.5, 1.0].

    The predicted class is fixed (not swept): pred = 1{score > 0.5} (predict
    success iff P(success) is the larger of the two probabilities). What
    varies with lambda is which steps we bother to predict on at all:
    confidence = max(score, 1 - score) is the winning class's probability,
    and we only predict (rather than abstain) where confidence >= lambda.

    R_hat(lambda) = (# misclassified among covered steps) / (# covered steps)
    -- abstained steps are excluded from both numerator and denominator, so
    this is the risk conditional on the model choosing to answer, not the
    risk over the whole calibration set. Coverage = (# covered) / n is
    reported alongside since a threshold's risk is only meaningful together
    with how much of the data it actually covers (raising lambda can drive
    risk to 0 by abstaining on nearly everything).

    Also reports, per threshold, the Clopper-Pearson upper confidence bound on
    the TRUE selective risk at level (1 - delta), computed only from the
    covered subset's (n_incorrect, n_covered) -- the quantity Learn-Then-Test
    would select thresholds against for a PAC-style guarantee, rather than
    the (optimistic, finite-sample-noisy) empirical risk alone.

    Thresholds whose covered subset has fewer than min_covered_samples steps
    have risk/risk_ucb set to None ("dropped") rather than reported: a
    selective risk computed from a handful of covered steps is too noisy to
    be meaningful, and the Clopper-Pearson UCB in particular gets very wide
    at small n (visible as the erratic swings near lambda -> 1 in the plot
    before this fallback was added). coverage/n_covered are still reported
    for every threshold regardless, so the drop is visible rather than silent.
    """
    thresholds = np.linspace(0.5, 1.0, num_thresholds)
    n = len(labels)
    preds = (scores > 0.5).astype(int)
    confidence = np.maximum(scores, 1.0 - scores)
    correct = (preds == labels.astype(int))

    rows = []
    for lam in thresholds:
        covered = confidence >= lam
        n_covered = int(covered.sum())
        has_enough = n_covered >= min_covered_samples
        n_incorrect = int(n_covered - correct[covered].sum()) if n_covered > 0 else 0
        n_failure_covered = int((labels[covered] != 1).sum()) if n_covered > 0 else 0
        rows.append(
            {
                "threshold": float(lam),
                "risk": (n_incorrect / n_covered) if has_enough else None,
                "risk_ucb": (
                    clopper_pearson_upper(n_incorrect, n_covered, delta) if has_enough else None
                ),
                "coverage": n_covered / n,
                # Class mixture of the covered subset -- abstaining is not necessarily neutral
                # across classes (the probe may be more confident, and so more likely to answer,
                # on one class than the other), so the covered subset's failure rate can drift
                # away from the overall calibration set's ~16.5% base rate as lambda rises.
                "failure_pct_covered": (n_failure_covered / n_covered) if has_enough else None,
                "n_incorrect": n_incorrect,
                "n_failure_covered": n_failure_covered,
                "n_covered": n_covered,
                "n": n,
            }
        )
    return rows


def plot_selective_risk_curve(
    rows: list[dict], delta: float, num_episodes: int, num_steps: int, path: Path
) -> None:
    """Plots R_hat(lambda) and its Clopper-Pearson UCB against the left axis,
    and coverage against a twin right axis -- risk alone is misleading here
    since raising lambda trivially drives it towards 0 by abstaining on
    almost everything, so coverage needs to be read alongside it. No "best
    threshold" marker: unlike the earlier (non-selective) risk curve, the
    minimum here is degenerate (the highest lambda with any coverage at
    all), so choosing a threshold is a real decision, not a curve minimum.

    A second subplot below shows the covered subset's failure percentage per
    threshold: abstention isn't necessarily class-neutral (the probe may be
    more confident, and so more likely to answer, on one class than the
    other), so the class mixture actually being scored can drift away from
    the calibration set's overall base rate as lambda rises -- risk alone
    doesn't show whether a low-risk, high-lambda subset is low-risk because
    the probe is good there, or because it's mostly/only the easy majority
    class left standing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in rows]
    risks = [r["risk"] if r["risk"] is not None else float("nan") for r in rows]
    risk_ucbs = [r["risk_ucb"] if r["risk_ucb"] is not None else float("nan") for r in rows]
    coverage = [r["coverage"] for r in rows]
    failure_pct = [
        r["failure_pct_covered"] if r["failure_pct_covered"] is not None else float("nan")
        for r in rows
    ]
    overall_failure_rate = rows[0]["n_failure_covered"] / rows[0]["n_covered"]  # lambda=0.5: full coverage

    fig, (ax, ax3) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, height_ratios=[3, 1.3]
    )
    l1, = ax.plot(thresholds, risks, marker=".", linewidth=1, color="tab:blue", label="selective risk R_hat(λ)")
    l2, = ax.plot(
        thresholds, risk_ucbs, marker=".", linewidth=1, linestyle="--", color="tab:orange",
        label=f"Clopper-Pearson UCB, 1-δ={1 - delta:.2f}",
    )
    ax.set_ylabel("selective risk (misclassification rate on covered subset)")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    l3, = ax2.plot(thresholds, coverage, marker=".", linewidth=1, color="tab:green", label="coverage")
    ax2.set_ylabel("coverage (fraction not abstained)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(0, 1.02)

    ax.set_title(
        "Selective risk, Clopper-Pearson upper bound & coverage\nvs. abstention cutoff (calibration split)"
        f"\n({num_episodes} episodes, {num_steps} steps)"
    )

    l4, = ax3.plot(thresholds, failure_pct, marker=".", linewidth=1, color="tab:red", label="failure % of covered subset")
    l5 = ax3.axhline(
        overall_failure_rate, linestyle="--", color="gray", linewidth=1,
        label=f"overall calibration-set failure rate ({overall_failure_rate:.1%})",
    )
    ax3.set_xlabel("abstention cutoff λ (predict iff max(p_success, p_failure) ≥ λ)")
    ax3.set_ylabel("failure %\nof covered subset")
    ax3.set_ylim(0, 1.02)
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    # Legend placed above the axes (outside the plotting area) rather than inside it --
    # inside placement overlapped the coverage curve, which runs high near lambda=0.5.
    fig.legend(
        handles=[l1, l2, l3, l4, l5], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    calibration_rows = load_calibration_episodes(args.split_json)
    logging.info(
        "Loaded %d held-out calibration episodes from %s", len(calibration_rows), args.split_json
    )

    probe = PerceptionSuccessProbe().to(args.device)
    probe.load_state_dict(torch.load(args.probe_checkpoint, map_location=args.device))
    probe.eval()
    logging.info("Loaded probe from %s onto %s", args.probe_checkpoint, args.device)

    labels, scores = collect_calibration_predictions(
        probe, args.features_dir, calibration_rows, args.device, args.batch_size
    )
    logging.info(
        "Total calibration steps: %d (%d success, %d failure)",
        len(labels), int(labels.sum()), int(len(labels) - labels.sum()),
    )

    rows = empirical_selective_risk_curve(
        labels, scores, args.num_thresholds, args.delta, args.min_covered_samples
    )
    n_dropped = sum(1 for r in rows if r["risk"] is None)
    if n_dropped:
        logging.info(
            "Dropped %d/%d thresholds with fewer than --min-covered-samples=%d covered steps",
            n_dropped, len(rows), args.min_covered_samples,
        )
    for r in rows[::10]:
        logging.info(
            "λ=%.3f  R_hat=%s  UCB(1-δ=%.2f)=%s  coverage=%.3f  n_covered=%d/%d",
            r["threshold"],
            f"{r['risk']:.4f}" if r["risk"] is not None else "n/a",
            1 - args.delta,
            f"{r['risk_ucb']:.4f}" if r["risk_ucb"] is not None else "n/a",
            r["coverage"], r["n_covered"], r["n"],
        )

    # Timestamped subdirectory per run, "latest" symlink kept pointing at the most recent --
    # same pattern as train_probe.py / eval_probe_time_bins.py, so a verification run can
    # never silently clobber a previous calibration analysis.
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "empirical_risk.json").write_text(
        json.dumps(
            {
                "split_json": str(args.split_json),
                "probe_checkpoint": str(args.probe_checkpoint),
                "num_calibration_episodes": len(calibration_rows),
                "num_calibration_steps": int(len(labels)),
                "num_thresholds": args.num_thresholds,
                "delta": args.delta,
                "min_covered_samples": args.min_covered_samples,
                "curve": rows,
            },
            indent=2,
        )
    )
    with (run_dir / "empirical_risk.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "threshold", "risk", "risk_ucb", "coverage", "failure_pct_covered",
                "n_incorrect", "n_failure_covered", "n_covered", "n",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_path = run_dir / "empirical_risk.png"
    plot_selective_risk_curve(
        rows, args.delta, len(calibration_rows), len(labels), plot_path
    )

    latest_link = args.output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name, target_is_directory=True)

    logging.info("Saved results to %s (latest -> %s)", run_dir, latest_link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
