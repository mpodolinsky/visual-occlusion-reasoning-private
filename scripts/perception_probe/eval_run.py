#!/usr/bin/env python3
"""Given a train_probe_time_dependent.py run folder: load its checkpoint,
evaluate it on that run's unseen (zero-shot) tasks (from split.json), write
the result to <run_dir>/unseen_metrics.json, and export the model to
<run_dir>/probe.onnx.

Useful for runs that crashed before reaching their own final zero-shot eval
(e.g. a DataLoader worker crash mid-training) but still left behind a
checkpoint + split.json. Reuses train_probe_time_dependent.py's own
dataset/eval code so this is exactly the eval the script would have run
itself.

The ONNX export is a plain forward pass over one timestep's features
(base_image, wrist_image, language, language_mask -> failure-probability
logit) -- PerceptionSuccessProbe itself, not the time-dependent score
accumulation (cumsum/rmean) wrapped around it in score_sequence(), since
that's a training-time loop construct, not part of the model graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_features import build_task_suite_map  # noqa: E402
from probe_model import FEATURE_DIM, PerceptionSuccessProbe  # noqa: E402
from train_probe_time_dependent import (  # noqa: E402
    EpisodeSequenceDataset,
    collate_pad_episodes,
    compute_task_min_steps,
    run_eval_pass,
)
from train_probe import auroc, confusion_matrix, read_manifest  # noqa: E402

NUM_IMAGE_TOKENS = 256
NUM_LANGUAGE_TOKENS = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="A train_probe_time_dependent.py output run folder.")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # score_sequence() hyperparameters -- not persisted anywhere on disk from the
    # original run (only wandb's config carries them), so they must be repeated
    # here to match how the checkpoint was actually trained if the eval numbers
    # are to mean anything.
    parser.add_argument("--rmean", action="store_true", default=False)
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--use-time-weighting", action="store_true", default=False)
    parser.add_argument("--use-threshold", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--raw-target-loss", action="store_true", default=False)
    parser.add_argument("--lambda-success", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument("--eval-threshold", type=float, default=0.0)
    parser.add_argument(
        "--no-task-min-step-eval", dest="task_min_step_eval", action="store_false", default=True,
        help="See train_probe_time_dependent.py's flag of the same name.",
    )
    parser.add_argument("--skip-eval", action="store_true", default=False, help="Only export ONNX.")
    parser.add_argument("--skip-onnx", action="store_true", default=False, help="Only run the eval.")
    args = parser.parse_args()

    model = PerceptionSuccessProbe().to(args.device)
    state = torch.load(args.run_dir / args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    if not args.skip_eval:
        split = json.loads((args.run_dir / "split.json").read_text())
        unseen_rows = split["unseen"]
        print(f"Unseen tasks ({len(split['unseen_tasks'])}): {split['unseen_tasks']}")
        print(f"Unseen episodes: {len(unseen_rows)}")

        if "task_min_step" in split:
            task_min_step = split["task_min_step"]
        else:
            # Older runs' split.json predates this field -- recompute it exactly the way
            # train_probe_time_dependent.py's main() does, from the full suite (not just
            # the unseen rows), so results are consistent with a fresh run.
            all_rows = read_manifest(args.features_dir)
            task_suite_map = build_task_suite_map()
            suite_rows = [r for r in all_rows if task_suite_map.get(r["task"]) == split["suite"]]
            task_min_step = compute_task_min_steps(suite_rows)
        print(f"Task min steps: {task_min_step}")

        ds = EpisodeSequenceDataset(unseen_rows, args.features_dir)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_pad_episodes,
        )

        metrics = run_eval_pass(model, loader, args.device, args, task_min_step, desc="unseen (zero-shot)")
        labels, final_scores = metrics["_labels"], metrics["_final_scores"]
        cm = confusion_matrix(1.0 - labels, final_scores, args.eval_threshold)
        final_auroc = auroc(labels, -final_scores)

        summary = {
            "checkpoint": args.checkpoint, "rmean": args.rmean, "cumsum": args.cumsum,
            "success_loss": metrics["success_loss"], "fail_loss": metrics["fail_loss"],
            "accuracy": metrics["accuracy"], "auroc": final_auroc, "n": metrics["n"],
            "confusion_matrix": cm,
        }
        out_path = args.run_dir / "unseen_metrics.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"Wrote {out_path}")

    if not args.skip_onnx:
        onnx_path = args.run_dir / "probe.onnx"
        dummy = (
            torch.randn(1, NUM_IMAGE_TOKENS, FEATURE_DIM, device=args.device),
            torch.randn(1, NUM_IMAGE_TOKENS, FEATURE_DIM, device=args.device),
            torch.randn(1, NUM_LANGUAGE_TOKENS, FEATURE_DIM, device=args.device),
            torch.ones(1, NUM_LANGUAGE_TOKENS, dtype=torch.bool, device=args.device),
        )
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["base_image", "wrist_image", "language", "language_mask"],
            output_names=["failure_logit"],
            dynamic_axes={
                "base_image": {0: "batch"}, "wrist_image": {0: "batch"},
                "language": {0: "batch"}, "language_mask": {0: "batch"},
                "failure_logit": {0: "batch"},
            },
        )
        print(f"Wrote {onnx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
