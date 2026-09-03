#!/usr/bin/env python3
"""Post-hoc: how well does a trained probe's latent z retrieve the right task
from the 10 CLIP text anchors?  (train-time align_acc is logged to wandb only,
never persisted -- this recovers it from probe_best.pt + split.json.)

v2: handles --decouple-z checkpoints (head + separate embed head) and matches
the training-time anchor treatment with --center-anchors.

Retrieval = argmax_i cos(mean-pooled z, anchor_i) over all 10 task
instructions; "correct" == the episode's own task.

    .venv/bin/python scripts/perception_probe/clip_align_v2/eval_retrieval.py \
        [--center-anchors] <run_dir> [more run dirs...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))  # our patched probe_model
from probe_model import PerceptionSuccessProbe  # noqa: E402

CLIP = REPO / "outputs" / "perception_probe" / "clip"
FEATURES = REPO / "outputs" / "perception_probe" / "features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def infer_arch(state: dict) -> dict:
    if not any(k.startswith("embed.") for k in state):
        return {}
    lin = sorted(k for k in state if k.startswith("embed.") and k.endswith(".weight")
                 and state[k].ndim == 2)
    shapes = [tuple(state[k].shape) for k in lin]
    kw = {"embed_dim": shapes[0][0]} if len(shapes) == 1 else \
         {"embed_hidden": shapes[0][0], "embed_dim": shapes[-1][0]}
    q = state.get("pool_base.query")
    if q is not None and q.ndim == 2:
        kw["n_queries"] = q.shape[0]
    # decoupled: has BOTH a detection head.* AND embed.* but no classifier.*
    if any(k.startswith("head.") for k in state) and not any(k.startswith("classifier.") for k in state):
        kw["decouple_z"] = True
    return kw


@torch.no_grad()
def episode_z(model: PerceptionSuccessProbe, key: dict) -> torch.Tensor:
    with np.load(FEATURES / key["npz_path"]) as d:
        b = torch.from_numpy(d["base_image"]).to(DEVICE)
        w = torch.from_numpy(d["wrist_image"]).to(DEVICE)
        lang = torch.from_numpy(d["language"]).to(DEVICE)
        lm = torch.from_numpy(d["language_mask"]).to(DEVICE)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        _, z = model(b.to(torch.bfloat16), w.to(torch.bfloat16), lang.to(torch.bfloat16), lm,
                     return_embedding=True)
    return z.float().mean(dim=0)


def main() -> None:
    args = sys.argv[1:]
    center = "--center-anchors" in args
    run_dirs = [Path(p) for p in args if not p.startswith("--")]
    if not run_dirs:
        sys.exit(__doc__)

    anchors = torch.from_numpy(np.load(CLIP / "task_instruction_embeddings.npy")).float().to(DEVICE)
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    if center:
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        anchors = torch.nn.functional.normalize(anchors, dim=-1)
    task_idx = json.loads((CLIP / "task_index.json").read_text())
    print(f"anchors: {'mean-centered' if center else 'raw'}  "
          f"(off-diag cos mean {((anchors @ anchors.T)[~torch.eye(10, dtype=bool)]).mean():+.3f})\n")

    for rd in run_dirs:
        rd = rd.resolve()
        split = json.loads((rd / "split.json").read_text())
        state = torch.load(rd / "probe_best.pt", map_location=DEVICE)
        state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
        arch = infer_arch(state)
        if not arch:
            print(f"{rd.name}: no z -- skipping\n")
            continue
        model = PerceptionSuccessProbe(**arch).to(DEVICE).eval()
        model.load_state_dict(state)

        print(f"=== {rd.parent.name}/{rd.name}   arch={arch} ===")
        for name in ("train", "test", "unseen"):
            keys = split[name]
            zs = torch.nn.functional.normalize(torch.stack([episode_z(model, k) for k in keys]), dim=-1)
            sim = zs @ anchors.T
            pred = sim.argmax(dim=-1).cpu().numpy()
            true = np.array([task_idx[k["task"]] for k in keys])
            top1 = (pred == true).mean()
            top3 = np.mean([t in row for t, row in zip(true, sim.topk(3, dim=-1).indices.cpu().numpy())])
            corr_cos = sim[torch.arange(len(true)), torch.from_numpy(true)].mean().item()
            print(f"  {name:6s} n={len(keys):3d} over {len(set(true)):2d} tasks | "
                  f"top1={top1:.3f}  top3={top3:.3f}  cos(z,true)={corr_cos:+.3f}  (chance=0.10)")
        print()


if __name__ == "__main__":
    main()
