#!/usr/bin/env python3
"""Runs construct_joint_threshold.py's calibration + transfer-check pipeline for ONE model
across a grid of (alpha_fpr, alpha_fnr) combinations, at a fixed h-sweep resolution --
unlike run_joint_threshold_sweep.py (one model per set of fixed alphas), this holds the
model fixed and varies the risk targets, to see how the certified threshold/guarantee moves
as alpha is relaxed or tightened. Reuses the already-cached score trajectories (via
compute_calibration_scores.py) -- no rescoring, just re-running the pure-numpy calibration
for each alpha combo.

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/calibration/run_alpha_sweep.py \\
        --models-json scripts/perception_probe/calibration/libero-10-occ-models/models.json \\
        --model-name best-sweep-3 --alphas 0.10 0.15 0.20 --num-h 10
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from construct_joint_threshold import evaluate_threshold_on_dataset, run_joint_threshold  # noqa: E402
import compute_calibration_scores as ccs  # noqa: E402
from run_joint_threshold_sweep import resolve_run_dir, score_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="Must match a \"name\" entry in --models-json.")
    parser.add_argument(
        "--alphas", type=float, nargs="+", required=True,
        help="Grid of alpha values, applied to BOTH alpha_fpr and alpha_fnr (full cross product).",
    )
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rmean", action="store_true", default=False)
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    parser.add_argument("--scores-cache-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force-rescore", action="store_true", help="Rescore even if a matching cache exists.")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--h-min", type=float, default=0.0)
    parser.add_argument("--h-max", type=float, default=2000.0)
    parser.add_argument("--num-h", type=int, default=20)
    parser.add_argument("--t-max-zoom", type=int, default=60)
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Defaults to calibration/runs/<model-name>_alpha_sweep_<timestamp>/.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    models = json.loads(args.models_json.read_text())["models"]
    model_cfg = next((m for m in models if m["name"] == args.model_name), None)
    if model_cfg is None:
        raise ValueError(f"'{args.model_name}' not found in {args.models_json} (have: {[m['name'] for m in models]})")
    run_dir = resolve_run_dir(model_cfg["run_dir"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        Path(__file__).resolve().parent / "runs" / f"{args.model_name}_alpha_sweep_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    seen_npz = score_dataset(model_cfg, args, run_dir, ["calibration", "test"])
    unseen_npz = score_dataset(model_cfg, args, run_dir, ["unseen"])

    results = []
    for alpha_fpr in args.alphas:
        for alpha_fnr in args.alphas:
            label = f"fpr{alpha_fpr:g}_fnr{alpha_fnr:g}"
            combo_dir = output_root / label
            entry = {"alpha_fpr": alpha_fpr, "alpha_fnr": alpha_fnr}
            try:
                seen_result = run_joint_threshold(
                    seen_npz, combo_dir / "seen", dataset_label="seen",
                    alpha_fpr=alpha_fpr, alpha_fnr=alpha_fnr, delta=args.delta,
                    h_min=args.h_min, h_max=args.h_max, num_h=args.num_h, t_max_zoom=args.t_max_zoom,
                )
                entry["seen"] = seen_result
                entry["unseen"] = evaluate_threshold_on_dataset(
                    unseen_npz, combo_dir / "unseen",
                    threshold_curve_path=Path(seen_result["threshold_curve_path"]),
                    alpha_fpr=alpha_fpr, alpha_fnr=alpha_fnr, t_max_zoom=args.t_max_zoom, dataset_label="unseen",
                )
            except Exception as exc:
                logging.warning("[%s] %s", label, exc)
                entry["error"] = str(exc)
            results.append(entry)

    summary_path = output_root / "alpha_sweep_summary.json"
    summary_path.write_text(json.dumps({"model_name": args.model_name, "run_dir": str(run_dir), "results": results}, indent=2))
    logging.info("Wrote %d combo(s) to %s", len(results), summary_path)

    print(f"\n{'alpha_fpr':>9s} {'alpha_fnr':>9s} {'h*':>8s} {'seenFPR':>8s} {'seenFNR':>8s} {'seenDet':>8s} "
          f"{'unsFPR':>8s} {'unsFNR':>8s} {'holds':>6s} {'unsDet':>8s} {'unsP(F|E)':>10s}")
    for e in results:
        if "error" in e:
            print(f"{e['alpha_fpr']:>9.2f} {e['alpha_fnr']:>9.2f}  ERROR: {e['error']}")
            continue
        s, u = e["seen"], e["unseen"]
        holds = "yes" if (u["guarantee_holds_fpr"] and u["guarantee_holds_fnr"]) else "NO"
        print(f"{e['alpha_fpr']:>9.2f} {e['alpha_fnr']:>9.2f} {s['selected_h']:>8.2f} {s['fpr']:>8.3f} "
              f"{s['fnr']:>8.3f} {s['detection_time']:>8.3f} {u['fpr']:>8.3f} {u['fnr']:>8.3f} {holds:>6s} "
              f"{u['detection_time']:>8.3f} {u['empirical_p_f_given_e']:>10.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
