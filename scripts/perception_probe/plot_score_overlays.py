#!/usr/bin/env python3
"""Given a train_probe_time_dependent.py run folder: load its checkpoint,
run it over every cached episode in the TEST split (seen tasks, held out
from training) and the UNSEEN split (zero-shot tasks), and produce one
overlay plot per split -- raw per-step score and accumulated score over
time, every episode's curve on the same axes, failures in red and successes
in blue. Same plot style as rollout_unseen_with_scores.py's plot_overlay,
just computed from the cached .npz features instead of a live rollout (so
it covers the FULL test/unseen split -- 53/150 episodes here -- not just
however many were live-rolled-out).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_model import PerceptionSuccessProbe  # noqa: E402
from rollout_unseen_with_scores import plot_overlay  # noqa: E402
from train_probe_time_dependent import EpisodeSequenceDataset, score_sequence  # noqa: E402

REPLAN_STEPS = 5  # matches collect_features.py's --replan-steps default: each inference call -> 5 control frames


def compute_episode_trace(
    model: torch.nn.Module, item: dict, cumsum: bool, rmean: bool, device: str
) -> tuple[list[tuple[int, float, float]], bool]:
    base = item["base_image"].unsqueeze(0).to(device)
    wrist = item["wrist_image"].unsqueeze(0).to(device)
    lang = item["language"].unsqueeze(0).to(device)
    lang_mask = item["language_mask"].unsqueeze(0).to(device)
    batch = {"base_image": base, "wrist_image": wrist, "language": lang, "language_mask": lang_mask}
    with torch.no_grad():
        raw_scores, scores = score_sequence(model, batch, cumsum, rmean)
    T = item["length"]
    trace = [
        (round(t * REPLAN_STEPS), raw_scores[0, t].item(), scores[0, t].item()) for t in range(T)
    ]
    return trace, item["label"] == 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A train_probe_time_dependent.py output run folder.")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rmean", action="store_true", default=False)
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    args = parser.parse_args()

    split = json.loads((args.run_dir / "split.json").read_text())

    model = PerceptionSuccessProbe().to(args.device)
    model.load_state_dict(torch.load(args.run_dir / args.checkpoint, map_location=args.device))
    model.eval()

    for split_name in ("test", "unseen"):
        rows = split[split_name]
        ds = EpisodeSequenceDataset(rows, args.features_dir)
        traces = []
        for i in range(len(ds)):
            item = ds[i]
            traces.append(compute_episode_trace(model, item, args.cumsum, args.rmean, args.device))
        n_success = sum(1 for _, success in traces if success)
        print(f"{split_name}: n={len(traces)} ({n_success} success, {len(traces) - n_success} failure)")

        out_path = args.run_dir / f"{split_name}_scores_overlay.png"
        title = f"{split_name} split -- {n_success} success / {len(traces) - n_success} failure (n={len(traces)})"
        plot_overlay(traces, args, out_path, title=title)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
