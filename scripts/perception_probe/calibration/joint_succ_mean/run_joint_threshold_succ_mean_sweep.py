#!/usr/bin/env python3
"""Runs construct_joint_threshold_succ_mean.py's algorithm -- the same joint FPR/FNR
threshold calibration as run_joint_threshold_sweep.py, EXCEPT mu_t (the reference
trajectory eta_h(t) is built around) is fit from successful episodes only, not pooled
over both classes (see construct_joint_threshold_succ_mean.py's module docstring for the
rationale) -- over a batch of trained probe run directories.

For each model in --models-json, scores both dataset pools (seen = calibration+test,
unseen), reusing/caching each via compute_calibration_scores.py (so re-running this sweep
after only tweaking alpha/delta/h-sweep doesn't re-run the model). The joint threshold h* is
calibrated ONLY on seen data (mu_t [successes only], the h-sweep, Learn-then-Test
p-values, Bonferroni selection); unseen is NOT recalibrated independently -- that fixed
rule is instead applied as-is to unseen episodes, to check empirically how far the
guarantee established on seen data actually holds (or how far it's violated) on genuinely
novel tasks. Writes a fresh timestamped output folder per model with all the plots and a
summary.json holding the selected h, its detection time, the Bayes'-rule guarantee (seen)
/ empirical FPR-FNR-P(F|E) and whether they still meet alpha_fpr/alpha_fnr (unseen).

models.json shape:
    {"models": [{"name": "base", "run_dir": "outputs/perception_probe/.../20260825_122919"},
                {"name": "sweep_cell_a", "run_dir": "outputs/perception_probe/sweeps/.../latest"},
                ...]}

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/calibration/joint_succ_mean/run_joint_threshold_succ_mean_sweep.py \\
        --models-json scripts/perception_probe/calibration/libero-10-occ-models/models.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
CALIBRATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CALIBRATION_DIR))
sys.path.insert(0, str(CALIBRATION_DIR.parent))
from construct_joint_threshold_succ_mean import evaluate_threshold_on_dataset, run_joint_threshold  # noqa: E402
import compute_calibration_scores as ccs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-json", type=Path, required=True, help="See module docstring for the schema.")
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rmean", action="store_true", default=False)
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    parser.add_argument(
        "--scores-cache-dir", type=Path, default=CALIBRATION_DIR,
        help="Where compute_calibration_scores.py caches score trajectories -- defaults to the "
        "shared cache dir also used by run_joint_threshold_sweep.py, so this doesn't rescore "
        "models already scored there.",
    )
    parser.add_argument("--force-rescore", action="store_true", help="Rescore even if a matching cache exists.")

    # Joint-threshold algorithm knobs -- see construct_joint_threshold_succ_mean.py.
    parser.add_argument("--alpha-fpr", type=float, default=0.25)
    parser.add_argument("--alpha-fnr", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.1, help="Bonferroni family-wise error rate over the h-grid.")
    parser.add_argument("--h-min", type=float, default=0.0)
    parser.add_argument("--h-max", type=float, default=2000.0)
    parser.add_argument("--num-h", type=int, default=20)
    parser.add_argument("--t-max-zoom", type=int, default=60)

    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parent / "runs",
        help="Each model gets its own output_root/<name>_<timestamp>/ folder.",
    )
    return parser.parse_args()


def resolve_run_dir(raw_path: str) -> Path:
    """A models.json entry's run_dir should be the directory containing probe_best.pt and
    split.json -- but tolerate a path that points at the checkpoint file itself (its parent
    is then the run_dir)."""
    p = Path(raw_path)
    if p.is_file():
        logging.warning("run_dir %s is a file, not a directory -- using its parent %s instead.", p, p.parent)
        return p.parent
    return p


def score_dataset(model_cfg: dict, args: argparse.Namespace, run_dir: Path, splits: list[str]) -> Path:
    ccs_args = ccs.parse_args([
        str(run_dir),
        "--features-dir", str(args.features_dir),
        "--checkpoint", args.checkpoint,
        "--output-dir", str(args.scores_cache_dir),
        "--device", args.device,
        "--splits", *splits,
        *(["--rmean"] if args.rmean else []),
        *([] if args.cumsum else ["--no-cumsum"]),
        *(["--force"] if args.force_rescore else []),
    ])
    return ccs.ensure_cached(ccs_args)


def run_one_model(model_cfg: dict, args: argparse.Namespace) -> dict:
    name = model_cfg["name"]
    run_dir = resolve_run_dir(model_cfg["run_dir"])
    if not (run_dir / args.checkpoint).is_file():
        raise FileNotFoundError(f"{run_dir / args.checkpoint} not found for model '{name}'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = args.output_root / f"{name}_{timestamp}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Traceability: a symlink back to where the model was actually trained, so a run folder
    # here is never orphaned from its source.
    run_dir_link = model_output_dir / "model_run_dir"
    if run_dir_link.is_symlink() or run_dir_link.exists():
        run_dir_link.unlink()
    run_dir_link.symlink_to(run_dir.resolve(), target_is_directory=True)

    seen_npz = score_dataset(model_cfg, args, run_dir, ["calibration", "test"])
    unseen_npz = score_dataset(model_cfg, args, run_dir, ["unseen"])

    seen_cache = ccs.np.load(seen_npz, allow_pickle=True)
    arch_source = str(seen_cache["arch_source"].item()) if "arch_source" in seen_cache else "unknown (stale cache)"

    # Calibrate (mu_t, the h-sweep, Learn-then-Test p-values, Bonferroni selection) on seen
    # data only. unseen is deliberately NOT recalibrated independently -- the point is to
    # check how far the guarantee established on seen data actually holds (or how far it's
    # violated) on genuinely novel tasks, by applying that exact fixed rule as-is.
    try:
        seen_result = run_joint_threshold(
            seen_npz, model_output_dir / "seen", dataset_label="seen",
            alpha_fpr=args.alpha_fpr, alpha_fnr=args.alpha_fnr, delta=args.delta,
            h_min=args.h_min, h_max=args.h_max, num_h=args.num_h, t_max_zoom=args.t_max_zoom,
        )
    except Exception as exc:
        logging.warning("[%s/seen] %s", name, exc)
        seen_result = {"dataset_label": "seen", "error": str(exc)}

    unseen_result: dict
    if "error" in seen_result:
        unseen_result = {"dataset_label": "unseen", "error": "skipped -- seen calibration failed"}
    else:
        try:
            unseen_result = evaluate_threshold_on_dataset(
                unseen_npz, model_output_dir / "unseen",
                threshold_curve_path=Path(seen_result["threshold_curve_path"]),
                alpha_fpr=args.alpha_fpr, alpha_fnr=args.alpha_fnr, t_max_zoom=args.t_max_zoom,
                dataset_label="unseen",
            )
        except Exception as exc:
            logging.warning("[%s/unseen] %s", name, exc)
            unseen_result = {"dataset_label": "unseen", "error": str(exc)}

    summary = {
        "model_name": name,
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "architecture_source": arch_source,
        "seen": seen_result,
        "unseen": unseen_result,
    }
    (model_output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("[%s] Wrote summary to %s", name, model_output_dir / "summary.json")
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    models = json.loads(args.models_json.read_text())["models"]
    logging.info("Loaded %d model(s) from %s", len(models), args.models_json)

    sweep_summary = []
    for model_cfg in models:
        name = model_cfg.get("name") or model_cfg.get("run_dir")
        try:
            sweep_summary.append(run_one_model(model_cfg, args))
        except Exception:
            logging.exception("[%s] Failed -- skipping, continuing with the rest of the sweep.", name)

    sweep_summary_path = args.output_root / f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    sweep_summary_path.write_text(json.dumps(sweep_summary, indent=2))
    logging.info(
        "Sweep done: %d/%d model(s) succeeded. Combined summary at %s",
        len(sweep_summary), len(models), sweep_summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
