#!/usr/bin/env python3
"""Hyperparameter / architecture sweep driver for the time-dependent perception probe.

Each "cell" is a named set of CLI overrides for train_probe_time_dependent.py.
Every cell is run once per --seed (the same value is used for both the
train/val/test split and the unseen-task draw, so a cell's seeds are genuinely
independent replicas). Results are aggregated into a CSV + comparison plots
under outputs/perception_probe/sweeps/<phase>/.

Runs are ranked by UNSEEN-split `separation` -- the standardized gap between
the failure-episode and success-episode score distributions at the task-min-
step cutoff (added to train_probe_time_dependent.py's *_metrics.json). Unseen,
not test, because with only ~112 failure episodes in the whole suite the
test-vs-unseen gap is the overfitting readout.

Phases:
  1  loss / objective     (raw-target BCE vs SAFE hinge, accumulation, time
                           weighting, class weight)
  2  regularization       (lambda_reg, dropout, lr, fixed input projection,
                           weight decay)  -- uses --base-args as the loss regime
  3  architecture         (pool type, key_dim, hidden dims, n_queries,
                           modality ablation)              -- also --base-args

Typical use:
    # see the plan + ETA without running anything
    .venv/bin/python scripts/perception_probe/sweep.py --phase 1 2 3 --dry-run

    # run phase 1 (3 seeds), then look at outputs/perception_probe/sweeps/phase1/
    .venv/bin/python scripts/perception_probe/sweep.py --phase 1

    # re-aggregate an already-run phase without re-training
    .venv/bin/python scripts/perception_probe/sweep.py --phase 1 --aggregate-only

Resumable: a cell/seed whose output dir already holds latest/test_metrics.json
is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TRAIN = Path(__file__).resolve().parent / "train_probe_time_dependent.py"
SWEEPS_ROOT = REPO_ROOT / "outputs" / "perception_probe" / "sweeps"

# Loss regime phases 2 & 3 build on. Phase 1 (sweeps_maxsteps80/) showed
# raw-target BCE + cumsum >> + rmean on unseen separation (2.14 vs 0.76), so
# cumsum (--raw-target-loss with no --rmean) is the base.
DEFAULT_BASE_ARGS = ["--raw-target-loss"]


def _cell(name: str, *args: str) -> dict:
    return {"name": name, "args": list(args)}


# --- Phase 1: loss / objective -------------------------------------------------
# raw-target BCE trains the RAW per-step score toward 1 (fail) / 0 (success) --
# the only loss with "1" as an actual target. The hinge losses push an
# unbounded accumulator. --use-time-weighting / --threshold only affect the
# hinge path (time_dependent_raw_target_loss ignores them).
PHASE1 = [
    _cell("raw_bce_rmean", "--raw-target-loss", "--rmean"),
    _cell("raw_bce_cumsum", "--raw-target-loss"),
    _cell("raw_bce_rmean_failx2", "--raw-target-loss", "--rmean", "--lambda-fail", "2"),
    _cell("raw_bce_rmean_failx4", "--raw-target-loss", "--rmean", "--lambda-fail", "4"),
    _cell("hinge_cumsum"),
    _cell("hinge_rmean", "--rmean"),
    _cell("hinge_timeweight", "--use-time-weighting"),
    _cell("hinge_timeweight_failx4", "--use-time-weighting", "--lambda-fail", "4"),
    _cell("hinge_thresh25", "--use-threshold", "--threshold", "25"),
    _cell("hinge_thresh50", "--use-threshold", "--threshold", "50"),
    _cell("hinge_rmean_timeweight", "--rmean", "--use-time-weighting"),
]

# --- Phase 2: regularization (on top of --base-args) --------------------------
PHASE2 = [
    _cell("baseline"),
    _cell("reg_1e-3", "--lambda-reg", "1e-3"),
    _cell("reg_1e-1", "--lambda-reg", "1e-1"),
    _cell("reg_1e0", "--lambda-reg", "1"),
    _cell("dropout_0.0", "--dropout", "0.0"),
    _cell("dropout_0.3", "--dropout", "0.3"),
    _cell("dropout_0.5", "--dropout", "0.5"),
    _cell("lr_3e-4", "--lr", "3e-4"),
    _cell("lr_3e-3", "--lr", "3e-3"),
    _cell("adamw_wd_1e-2", "--optimizer", "adamw", "--weight-decay", "1e-2"),
    _cell("input_proj_256", "--input-proj-dim", "256"),
    _cell("input_proj_512", "--input-proj-dim", "512"),
    _cell("gradclip_1.0", "--grad-max-norm", "1.0"),
]

# --- Phase 3: architecture (on top of --base-args) ---------------------------
PHASE3 = [
    _cell("attn_default"),
    _cell("pool_mean", "--pool-type", "mean"),
    _cell("pool_max", "--pool-type", "max"),
    _cell("pool_gated", "--pool-type", "gated"),
    _cell("pool_gated_q4", "--pool-type", "gated", "--n-queries", "4"),
    _cell("pool_topk8", "--pool-type", "topk", "--topk", "8"),
    _cell("pool_topk32", "--pool-type", "topk", "--topk", "32"),
    _cell("attn_q4", "--n-queries", "4"),
    _cell("attn_q8", "--n-queries", "8"),
    _cell("key_dim_32", "--key-dim", "32"),
    _cell("key_dim_64", "--key-dim", "64"),
    _cell("key_dim_256", "--key-dim", "256"),
    _cell("hidden_64", "--hidden-dim", "64"),
    _cell("hidden_128", "--hidden-dim", "128"),
    _cell("mlp_2layer", "--n-hidden-layers", "2"),
    _cell("pool_temp_0.5", "--pool-temperature", "0.5"),
    _cell("modalities_base", "--modalities", "base"),
    _cell("modalities_base_wrist", "--modalities", "base", "wrist"),
    _cell("share_image_pool", "--share-image-pool"),
]

# --- Phase 4: alternative loss functions (on top of --base-args = cumsum BCE) ---
# The concern: per-step BCE labels every failure-episode timestep as failure=1,
# but early failure frames look like successes -> label noise -> weak per-frame
# signal. MIL / focal / ranking each attack that differently.
PHASE4 = [
    _cell("bce_cumsum"),                                              # control = the current/prod loss (raw-target BCE, cumsum)
    _cell("bce_cumsum_fp32", "--no-amp"),                             # same, fp32 -- closest to the pre-sweep prod run
    _cell("focal_g1", "--focal-gamma", "1"),
    _cell("focal_g2", "--focal-gamma", "2"),
    _cell("focal_g3", "--focal-gamma", "3"),
    _cell("mil_max", "--mil-pool", "max"),
    _cell("mil_lse", "--mil-pool", "lse"),
    _cell("mil_topk4", "--mil-pool", "topk", "--mil-topk", "4"),
    _cell("mil_topk16", "--mil-pool", "topk", "--mil-topk", "16"),
    _cell("rank_0.5", "--ranking-weight", "0.5"),
    _cell("rank_2.0", "--ranking-weight", "2.0"),
    _cell("mil_lse_rank1", "--mil-pool", "lse", "--ranking-weight", "1.0"),
]

# --- Phase 5: best-combo confirmation (on top of --base-args = cumsum BCE) -----
# Stacks the phases 1-3 winners that had a reliable (epoch <= 8) checkpoint:
# lr 3e-4, n_queries 4, share_image_pool. Run at --epochs 8 (kills the late-
# checkpoint overfit) x 3 seeds (each seed = a different unseen-task draw, so
# mean +/- std is finally trustworthy). See sweeps/FINDINGS.md.
PHASE5 = [
    _cell("default"),                                                 # control: all defaults, lr 1e-3
    _cell("combo", "--lr", "3e-4", "--n-queries", "4", "--share-image-pool"),
    _cell("combo_plus", "--lr", "3e-4", "--n-queries", "4", "--share-image-pool",
          "--pool-temperature", "0.5", "--dropout", "0.3"),
]

# --- Phase 6: training dynamics -- can we beat plain default? ------------------
# Phase 5 finding: the probe reaches its generalizable ceiling in ~1 epoch, then
# overfits; plain default only wins because its slow convergence lands the
# best-val checkpoint before the rot. These cells attack that directly:
# fewer epochs, LR decay, warmup, checkpoint-by-separation. All on default arch,
# raw-target BCE + cumsum, batch 8.
PHASE6 = [
    _cell("default"),                                   # epochs 10, control
    _cell("e5", "--epochs", "5"),
    _cell("e7", "--epochs", "7"),
    _cell("e16", "--epochs", "16"),
    _cell("cosine", "--lr-schedule", "cosine"),
    _cell("cosine_e16", "--lr-schedule", "cosine", "--epochs", "16"),
    _cell("step_half", "--lr-gamma", "0.5", "--lr-step-size", "3"),
    _cell("warmup", "--warmup-steps", "300"),
    _cell("select_sep", "--select-by", "separation"),
    _cell("lr_5e-4", "--lr", "5e-4"),
    _cell("lr_2e-3", "--lr", "2e-3"),
]

# --- Phase 7: regularization on the default arch ------------------------------
PHASE7 = [
    _cell("default"),                                   # control
    _cell("ls_0.05", "--label-smoothing", "0.05"),
    _cell("ls_0.1", "--label-smoothing", "0.1"),
    _cell("do_0.2", "--dropout", "0.2"),
    _cell("do_0.3", "--dropout", "0.3"),
    _cell("reg_3e-2", "--lambda-reg", "3e-2"),
    _cell("reg_1e-1", "--lambda-reg", "1e-1"),
    _cell("adamw", "--optimizer", "adamw", "--weight-decay", "1e-2"),
    _cell("gradclip", "--grad-max-norm", "1.0"),
    _cell("ls_0.05_cosine", "--label-smoothing", "0.05", "--lr-schedule", "cosine"),
]

PHASE_GRIDS = {1: PHASE1, 2: PHASE2, 3: PHASE3, 4: PHASE4, 5: PHASE5, 6: PHASE6, 7: PHASE7}
PHASE_USES_BASE_ARGS = {1: False, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6, 7], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=8)
    # With --max-steps 80 + bf16 autocast, batch 8 / workers 2 / prefetch 1 keeps
    # each run near ~4GB GPU, ~4GB RAM, ~3.5GB /dev/shm -- so three phases fit
    # concurrently on the 24GB card and 16GB shm. Raise batch only when running
    # phases sequentially.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    p.add_argument(
        "--base-args", nargs="*", default=None,
        help=f"Loss-regime args prepended to every phase-2/3 cell. Default: {' '.join(DEFAULT_BASE_ARGS)}",
    )
    p.add_argument("--extra-args", nargs="*", default=[], help="Appended to every cell in every phase.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan + ETA, run nothing.")
    p.add_argument("--aggregate-only", action="store_true", help="Skip training, just (re)build CSVs + plots.")
    p.add_argument("--minutes-per-run", type=float, default=17.0, help="ETA estimate only.")
    p.add_argument("--only", nargs="+", default=None, help="Restrict to these cell names.")
    return p.parse_args()


def cell_out_dir(phase: int, cell_name: str, seed: int) -> Path:
    return SWEEPS_ROOT / f"phase{phase}" / "runs" / f"{cell_name}__seed{seed}"


def is_done(out_dir: Path) -> bool:
    return (out_dir / "latest" / "test_metrics.json").is_file()


def build_command(phase: int, cell: dict, seed: int, args: argparse.Namespace) -> list[str]:
    base = list(args.base_args if args.base_args is not None else DEFAULT_BASE_ARGS)
    cmd = [
        str(PYTHON), str(TRAIN),
        "--no-wandb",
        "--features-dir", str(args.features_dir),
        "--output-dir", str(cell_out_dir(phase, cell["name"], seed)),
        "--seed", str(seed),
        "--unseen-task-seed", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--max-steps", str(args.max_steps),
    ]
    if PHASE_USES_BASE_ARGS[phase]:
        cmd += base
    cmd += cell["args"] + list(args.extra_args)
    return cmd


def run_cell(phase: int, cell: dict, seed: int, args: argparse.Namespace) -> dict:
    out_dir = cell_out_dir(phase, cell["name"], seed)
    if is_done(out_dir):
        logging.info("[phase %d] %s seed %d -- already done, skipping", phase, cell["name"], seed)
        return {"status": "skipped", "seconds": 0.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(phase, cell, seed, args)
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    logging.info("[phase %d] %s seed %d -- starting", phase, cell["name"], seed)
    start = time.monotonic()
    with (out_dir / "run.log").open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
    seconds = time.monotonic() - start
    status = "ok" if proc.returncode == 0 and is_done(out_dir) else f"FAILED(rc={proc.returncode})"
    logging.info("[phase %d] %s seed %d -- %s (%.1f min)", phase, cell["name"], seed, status, seconds / 60)
    return {"status": status, "seconds": seconds}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_row(phase: int, cell: dict, seed: int) -> dict:
    out_dir = cell_out_dir(phase, cell["name"], seed)
    row = {
        "phase": phase, "cell": cell["name"], "seed": seed,
        "args": " ".join(cell["args"]),
    }
    test = _load_json(out_dir / "latest" / "test_metrics.json")
    unseen = _load_json(out_dir / "latest" / "unseen_metrics.json")
    if test is None:
        row["status"] = "missing"
        return row
    row["status"] = "ok"
    row["best_epoch"] = test.get("best_epoch")
    row["best_val_auroc"] = test.get("best_val_auroc")
    row["test_auroc"] = test.get("auroc")
    row["test_separation"] = test.get("separation")
    row["test_acc"] = test.get("accuracy")
    if unseen is not None:
        row["unseen_auroc"] = unseen.get("auroc")
        row["unseen_separation"] = unseen.get("separation")
        row["unseen_acc"] = unseen.get("accuracy")
        row["unseen_mean_fail"] = unseen.get("mean_fail_score")
        row["unseen_mean_success"] = unseen.get("mean_success_score")
        if row.get("test_auroc") is not None and unseen.get("auroc") is not None:
            row["overfit_gap_auroc"] = row["test_auroc"] - unseen["auroc"]
    return row


def _agg(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def aggregate(phase: int, cells: list[dict], seeds: list[int]) -> None:
    phase_dir = SWEEPS_ROOT / f"phase{phase}"
    rows = [collect_row(phase, cell, seed) for cell in cells for seed in seeds]

    per_seed_path = phase_dir / "results.csv"
    fields = sorted({k for r in rows for k in r})
    with per_seed_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = []
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell["name"] and r.get("status") == "ok"]
        if not cr:
            summary.append({"cell": cell["name"], "args": " ".join(cell["args"]), "n_seeds": 0})
            continue
        entry = {"cell": cell["name"], "args": " ".join(cell["args"]), "n_seeds": len(cr)}
        for metric in ("unseen_separation", "unseen_auroc", "test_separation", "test_auroc",
                       "overfit_gap_auroc", "best_val_auroc", "best_epoch"):
            mean, std = _agg([r.get(metric) for r in cr])
            entry[f"{metric}_mean"] = None if mean is None else round(mean, 4)
            entry[f"{metric}_std"] = None if std is None else round(std, 4)
        summary.append(entry)

    summary.sort(key=lambda e: (e.get("unseen_separation_mean") is None, -(e.get("unseen_separation_mean") or 0)))
    summary_path = phase_dir / "summary.csv"
    sfields = sorted({k for e in summary for k in e})
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)

    logging.info("phase %d ranking by unseen separation (mean over seeds):", phase)
    for e in summary:
        logging.info(
            "  %-26s  unseen_sep=%s  unseen_auroc=%s  overfit_gap=%s  args=%s",
            e["cell"],
            f"{e.get('unseen_separation_mean')}" if e.get("unseen_separation_mean") is not None else "  n/a",
            f"{e.get('unseen_auroc_mean')}" if e.get("unseen_auroc_mean") is not None else "n/a",
            f"{e.get('overfit_gap_auroc_mean')}" if e.get("overfit_gap_auroc_mean") is not None else "n/a",
            e["args"] or "(base)",
        )

    _plots(phase, summary, rows)
    _copy_winner_overlays(phase, summary, seeds)
    logging.info("phase %d: wrote %s, %s, plots/", phase, per_seed_path, summary_path)


def _plots(phase: int, summary: list[dict], rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = SWEEPS_ROOT / f"phase{phase}" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ranked = [e for e in summary if e.get("unseen_separation_mean") is not None]
    if ranked:
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(ranked) + 1)))
        names = [e["cell"] for e in ranked][::-1]
        means = [e["unseen_separation_mean"] for e in ranked][::-1]
        errs = [e.get("unseen_separation_std") or 0 for e in ranked][::-1]
        ax.barh(names, means, xerr=errs, color="tab:blue")
        ax.set_xlabel("unseen-split separation  (mean ± std over seeds)")
        ax.set_title(f"phase {phase}: failure/success score gap on unseen tasks")
        fig.tight_layout()
        fig.savefig(plots_dir / "unseen_separation.png", dpi=150)
        plt.close(fig)

    pts = [(r["test_auroc"], r["unseen_auroc"], r["cell"])
           for r in rows if r.get("test_auroc") is not None and r.get("unseen_auroc") is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 6))
        for ta, ua, _ in pts:
            ax.scatter(ta, ua, color="tab:purple", alpha=0.7)
        lo = min(min(p[0] for p in pts), min(p[1] for p in pts)) - 0.02
        ax.plot([lo, 1.0], [lo, 1.0], "--", color="gray", label="no overfitting")
        ax.set_xlabel("test AUROC (seen tasks)")
        ax.set_ylabel("unseen AUROC (zero-shot)")
        ax.set_title(f"phase {phase}: generalization gap")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "test_vs_unseen_auroc.png", dpi=150)
        plt.close(fig)


def _copy_winner_overlays(phase: int, summary: list[dict], seeds: list[int]) -> None:
    """Copy the score-overlay / ROC plots of the top-ranked cell (from one seed
    that produced them) into the phase's plots/ dir for a quick eyeball."""
    ranked = [e for e in summary if e.get("unseen_separation_mean") is not None]
    if not ranked:
        return
    winner = ranked[0]["cell"]
    plots_dir = SWEEPS_ROOT / f"phase{phase}" / "plots"
    for seed in seeds:
        src_dir = cell_out_dir(phase, winner, seed) / "latest"
        found = False
        for fname in ("test_scores_overlay.png", "unseen_scores_overlay.png", "roc_curve_unseen.png"):
            src = src_dir / fname
            if src.is_file():
                shutil.copy(src, plots_dir / f"winner_{winner}_{fname}")
                found = True
        if found:
            return


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    plans = []
    for phase in args.phase:
        cells = PHASE_GRIDS[phase]
        if args.only:
            cells = [c for c in cells if c["name"] in set(args.only)]
        for cell in cells:
            for seed in args.seeds:
                done = is_done(cell_out_dir(phase, cell["name"], seed))
                plans.append((phase, cell, seed, done))

    todo = [p for p in plans if not p[3]]
    logging.info(
        "Sweep plan: %d phase(s), %d cell-seed runs total, %d already done, %d to run.",
        len(args.phase), len(plans), len(plans) - len(todo), len(todo),
    )
    for phase in args.phase:
        (SWEEPS_ROOT / f"phase{phase}").mkdir(parents=True, exist_ok=True)
        cells = [c for c in PHASE_GRIDS[phase] if not args.only or c["name"] in set(args.only)]
        (SWEEPS_ROOT / f"phase{phase}" / "grid.json").write_text(
            json.dumps({"seeds": args.seeds, "epochs": args.epochs,
                        "base_args": args.base_args if args.base_args is not None else DEFAULT_BASE_ARGS,
                        "cells": cells}, indent=2)
        )

    if args.dry_run:
        eta_h = len(todo) * args.minutes_per_run / 60
        for phase, cell, seed, _ in todo:
            logging.info("  would run: %s", " ".join(build_command(phase, cell, seed, args)))
        logging.info("ETA: %d runs x ~%.0f min = ~%.1f h serial", len(todo), args.minutes_per_run, eta_h)
        return 0

    if not args.aggregate_only:
        t0 = time.monotonic()
        for i, (phase, cell, seed, _) in enumerate(todo, 1):
            run_cell(phase, cell, seed, args)
            elapsed = (time.monotonic() - t0) / 60
            logging.info("progress: %d/%d runs, %.0f min elapsed, ~%.0f min left",
                         i, len(todo), elapsed, elapsed / i * (len(todo) - i))
            # re-aggregate after every run so partial results are always usable
            for phase_n in args.phase:
                cells_n = [c for c in PHASE_GRIDS[phase_n] if not args.only or c["name"] in set(args.only)]
                try:
                    aggregate(phase_n, cells_n, args.seeds)
                except Exception:
                    logging.exception("aggregation failed (continuing sweep)")

    for phase in args.phase:
        cells = [c for c in PHASE_GRIDS[phase] if not args.only or c["name"] in set(args.only)]
        aggregate(phase, cells, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
