#!/usr/bin/env python3
"""Post-hoc: how well does a trained probe's latent z retrieve the right task
from the 10 CLIP text anchors?  (train-time align_acc is logged to wandb only,
never persisted -- this recovers it from probe_best.pt + split.json.)

Retrieval = argmax_i cos(mean-pooled z, CLIP_anchor_i) over all 10 task
instructions; "correct" == the episode's own task. Reported per split.

    .venv/bin/python scripts/perception_probe/eval_clip_retrieval.py \
        outputs/perception_probe/align_sweep/align_0.1/latest [more run dirs...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "perception_probe"))
from probe_model import PerceptionSuccessProbe  # noqa: E402

CLIP = REPO / "outputs" / "perception_probe" / "clip"
FEATURES = REPO / "outputs" / "perception_probe" / "features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def infer_arch(state: dict) -> dict:
    """Reconstruct the embed_dim / embed_hidden kwargs from checkpoint shapes."""
    if not any(k.startswith("embed.") for k in state):
        return {}  # no z -- plain probe
    lin = sorted(k for k in state if k.startswith("embed.") and k.endswith(".weight")
                 and state[k].ndim == 2)  # Linear weights only (skip the LayerNorm)
    # embed.1 is either Linear(6144->embed_dim) or Linear(6144->embed_hidden)
    shapes = [tuple(state[k].shape) for k in lin]
    if len(shapes) == 1:
        return {"embed_dim": shapes[0][0]}
    # [ (embed_hidden, 6144), (embed_dim, embed_hidden) ]
    kw = {"embed_hidden": shapes[0][0], "embed_dim": shapes[-1][0]}
    q = state.get("pool_base.query")
    if q is not None and q.ndim == 2:
        kw["n_queries"] = q.shape[0]
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
    return z.float().mean(dim=0)  # (embed_dim,) mean over timesteps


def main() -> None:
    run_dirs = [Path(p) for p in sys.argv[1:]] or sys.exit(__doc__)
    anchors = torch.from_numpy(np.load(CLIP / "task_instruction_embeddings.npy")).float().to(DEVICE)
    anchors = torch.nn.functional.normalize(anchors, dim=-1)  # (10, 512)
    task_idx = json.loads((CLIP / "task_index.json").read_text())

    for rd in run_dirs:
        rd = rd.resolve()
        split = json.loads((rd / "split.json").read_text())
        state = torch.load(rd / "probe_best.pt", map_location=DEVICE)
        state = state.get("model", state) if isinstance(state, dict) and "model" in state else state
        arch = infer_arch(state)
        if not arch:
            print(f"{rd.name}: no z (plain probe) -- skipping\n")
            continue
        model = PerceptionSuccessProbe(**arch).to(DEVICE).eval()
        model.load_state_dict(state)

        print(f"=== {rd.parent.name}/{rd.name}   arch={arch} ===")
        for name in ("train", "test", "unseen"):
            keys = split[name]
            zs = torch.stack([episode_z(model, k) for k in keys])  # (N, d)
            zs = torch.nn.functional.normalize(zs, dim=-1)
            sim = zs @ anchors.T  # (N, 10)
            pred = sim.argmax(dim=-1).cpu().numpy()
            true = np.array([task_idx[k["task"]] for k in keys])
            top1 = (pred == true).mean()
            # top-3
            top3 = np.mean([t in row for t, row in zip(true, sim.topk(3, dim=-1).indices.cpu().numpy())])
            # mean cos to the correct anchor
            corr_cos = sim[torch.arange(len(true)), torch.from_numpy(true)].mean().item()
            n_tasks = len(set(true))
            print(f"  {name:6s} n={len(keys):3d} over {n_tasks:2d} tasks | "
                  f"top1={top1:.3f}  top3={top3:.3f}  cos(z,true)={corr_cos:+.3f}  (chance={1/10:.2f})")
        print()


if __name__ == "__main__":
    main()
