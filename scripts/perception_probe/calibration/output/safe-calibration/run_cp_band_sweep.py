#!/usr/bin/env python3
"""Runs construct_cp_band.py's functional-conformal-band algorithm (FAIL-Detect
Appendix B) over a batch of trained probe run directories -- the same models.json-driven
batch pattern as run_joint_threshold_sweep.py, but for the CP band instead of the joint
FPR/FNR threshold. For each model in --models-json: scores both dataset pools (seen =
calibration+test, unseen), reusing/caching via compute_calibration_scores.py, fits the
band mu_t/s_A/h/eta_t on seen successful episodes ONLY, then applies that fixed band
as-is to unseen episodes (no refitting) to check how far the whole-trajectory coverage
guarantee established on seen data actually holds on genuinely novel tasks. Writes a
fresh timestamped output folder per model with the plots and a summary.json.

models.json shape (same file used by run_joint_threshold_sweep.py / run_alpha_sweep.py):
    {"models": [{"name": "base", "run_dir": "outputs/perception_probe/.../20260825_122919"},
                ...]}

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/calibration/output/safe-calibration/run_cp_band_sweep.py \\
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

REPO_ROOT = Path(__file__).resolve().parents[5]
CALIBRATION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CALIBRATION_DIR))
sys.path.insert(0, str(CALIBRATION_DIR.parent))
from construct_cp_band import evaluate_band_on_dataset, run_cp_band  # noqa: E402
import compute_calibration_scores as ccs  # noqa: E402
from run_joint_threshold_sweep import resolve_run_dir, score_dataset  # noqa: E402


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
        help="Where compute_calibration_scores.py caches score trajectories (shared with "
        "run_joint_threshold_sweep.py's cache, so a repeat run doesn't rescore).",
    )
    parser.add_argument("--force-rescore", action="store_true", help="Rescore even if a matching cache exists.")

    # CP-band algorithm knobs -- see construct_cp_band.py.
    parser.add_argument("--alpha", type=float, default=0.15, help="1 - target whole-trajectory coverage.")
    parser.add_argument("--seed", type=int, default=0, help="D_cal_A / D_cal_B split seed.")
    parser.add_argument("--ab-split-fraction", type=float, default=0.3, help="Fraction of D_cal used as D_cal_A.")

    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parent / "runs",
        help="Each model gets its own output_root/<name>_<timestamp>/ folder.",
    )
    return parser.parse_args()


def run_one_model(model_cfg: dict, args: argparse.Namespace) -> dict:
    name = model_cfg["name"]
    run_dir = resolve_run_dir(model_cfg["run_dir"])
    if not (run_dir / args.checkpoint).is_file():
        raise FileNotFoundError(f"{run_dir / args.checkpoint} not found for model '{name}'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = args.output_root / f"{name}_{timestamp}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Traceability: a symlink back to where the model was actually trained.
    run_dir_link = model_output_dir / "model_run_dir"
    if run_dir_link.is_symlink() or run_dir_link.exists():
        run_dir_link.unlink()
    run_dir_link.symlink_to(run_dir.resolve(), target_is_directory=True)

    seen_npz = score_dataset(model_cfg, args, run_dir, ["calibration", "test"])
    unseen_npz = score_dataset(model_cfg, args, run_dir, ["unseen"])

    seen_cache = ccs.np.load(seen_npz, allow_pickle=True)
    arch_source = str(seen_cache["arch_source"].item()) if "arch_source" in seen_cache else "unknown (stale cache)"

    # Fit the band on seen successful episodes only. unseen is NOT refit -- the fixed band
    # is applied as-is, to check how far the coverage guarantee established on seen data
    # actually holds (or how far it's violated) on genuinely novel tasks.
    try:
        seen_result = run_cp_band(
            seen_npz, model_output_dir, alpha=args.alpha, seed=args.seed,
            ab_split_fraction=args.ab_split_fraction, dataset_label="seen",
        )
    except Exception as exc:
        logging.warning("[%s/seen] %s", name, exc)
        seen_result = {"dataset_label": "seen", "error": str(exc)}

    unseen_result: dict
    if "error" in seen_result:
        unseen_result = {"dataset_label": "unseen", "error": "skipped -- seen band fit failed"}
    else:
        try:
            unseen_result = evaluate_band_on_dataset(
                unseen_npz, model_output_dir, band_path=Path(seen_result["band_path"]),
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
        "alpha": args.alpha,
        "seed": args.seed,
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

    print(f"\n{'model':>16s} {'arch':>10s} {'h':>8s} {'calB_cov':>9s} {'fail_flag':>10s} {'seen_det':>9s} "
          f"{'uns_cov':>8s} {'holds':>6s} {'uns_flag':>9s} {'uns_det':>8s}")
    for s in sweep_summary:
        seen, unseen = s.get("seen", {}), s.get("unseen", {})
        if "error" in seen:
            print(f"{s['model_name']:>16s}  seen ERROR: {seen['error']}")
            continue
        holds = "yes" if unseen.get("guarantee_holds") else ("NO" if "error" not in unseen else "?")
        uns_cov = f"{unseen['coverage']:.3f}" if "error" not in unseen and unseen.get("coverage") is not None else "n/a"
        uns_flag = f"{unseen['failure_flag_rate']:.3f}" if "error" not in unseen and unseen.get("failure_flag_rate") is not None else "n/a"
        uns_det = f"{unseen['avg_detection_time']:.3f}" if "error" not in unseen and unseen.get("avg_detection_time") is not None else "n/a"
        seen_det = f"{seen['avg_detection_time']:.3f}" if seen.get("avg_detection_time") is not None else "n/a"
        print(f"{s['model_name']:>16s} {s['architecture_source']:>10s} {seen['h']:>8.3f} "
              f"{seen['cal_B_coverage']:>9.3f} {seen['failure_flag_rate']:>10.3f} {seen_det:>9s} "
              f"{uns_cov:>8s} {holds:>6s} {uns_flag:>9s} {uns_det:>8s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
