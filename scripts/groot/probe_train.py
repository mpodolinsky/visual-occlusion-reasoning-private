#!/usr/bin/env python3
"""Time-dependent perception-probe training on frozen GR00T-N1.7 backbone features.

This is a fork of scripts/perception_probe/train_probe_time_dependent.py -- the
SAFE-style IndepModel loss / score-accumulation design is unchanged; only the
feature source is swapped from pi0.5 to GR00T-N1.7:

  - features come from outputs/groot/libero_10/<variant>/<NN>_<task>/ep<NNN>/
    rollout.npz (collected by scripts/groot/), read via probe_data.read_probe_manifest
    -- NOT scripts/perception_probe/'s collect_features.py layout.
  - GR00T taps its Cosmos-Reason2-2B backbone at layer 16 (mid-stack) rather
    than pi0.5's final prefix layer; 64 image tokens per camera (2x2 spatial
    merge) rather than 256; the language block is zero-padded to a fixed 200.
    None of that touches the probe -- PerceptionSuccessProbe pools over the
    token axis, so it is agnostic to token count.
  - GR00T's rollout.npz also carries a state_features vector; it is deliberately
    NOT wired into the probe here, for strict 3-modality parity (base/wrist/lang)
    with the pi0.5 sweep so the two VLAs' results compare cleanly.
  - target is episode `success` (True/False), same as the pi0.5 sweep. No
    Gemini / VLM labels.

Everything else -- the per-timestep sigmoid + cumsum accumulation, the
raw-target BCE / SAFE hinge / MIL / ranking losses, the seen/unseen task split,
the task-min-step-truncated eval, the `separation` metric -- is identical to the
pi0.5 trainer. See that file's module docstring for the full SAFE lineage.

Run with the top-level project venv (plain PyTorch, no JAX/openpi needed):
    .venv/bin/python scripts/groot/probe_train.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import signal
import sys

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_data import (  # noqa: E402
    EpisodeSequenceDataset,
    collate_pad_episodes,
    read_probe_manifest,
    suite_of,
)
from probe_model import FEATURE_DIM, PerceptionSuccessProbe  # noqa: E402
from probe_utils import (  # noqa: E402
    auroc,
    available_memory_gb,
    confusion_matrix,
    plot_overlay,
    plot_roc_curve,
    roc_curve_points,
    split_episodes,
)

# GR00T-N1.7 / libero_10 checkpoint: 2x2 spatial merge -> 64 image tokens per
# camera; instruction tokens zero-padded to 200; replan_steps=8 in the collected
# data (each inference call -> 8 control frames). NUM_*_TOKENS only shape the
# ONNX export dummy; REPLAN_STEPS only labels the score-overlay x-axis.
NUM_IMAGE_TOKENS = 64
NUM_LANGUAGE_TOKENS = 200
REPLAN_STEPS = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """argv defaults to sys.argv[1:]; pass an explicit list to re-parse a saved
    invocation (command.txt-based architecture recovery)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features-dir", type=Path, default=REPO_ROOT / "outputs" / "groot" / "libero_10"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "groot" / "probe_time_dependent"
    )
    parser.add_argument(
        "--suite", default="libero_10_occluded",
        help="Only episodes whose manifest 'suite' column matches are used (both scene variants -- "
        "occluded AND normal -- since scene_variant is a separate axis from suite).",
    )
    parser.add_argument(
        "--num-unseen-tasks", type=int, default=3,
        help="How many of --suite's tasks to hold out entirely from train/val/test/calibration, "
        "chosen randomly (seeded by --unseen-task-seed) from the tasks NOT already named by "
        "--unseen-tasks.",
    )
    parser.add_argument(
        "--unseen-task-seed", type=int, default=None,
        help="Seed for the random selection of unseen tasks. Defaults to --seed if not given.",
    )
    parser.add_argument(
        "--unseen-tasks", nargs="+", default=None,
        help="Task stems (manifest.csv 'task' column) to force into the unseen set. Remainder of "
        "--num-unseen-tasks filled in randomly.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=15, help="SAFE's ModelConfig default is 1000.")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Episodes per batch (NOT steps -- each item is a whole variable-length episode, "
        "padded to the batch's max length).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)

    # --- Probe architecture (all default to the original PerceptionSuccessProbe) ---
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--key-dim", type=int, default=128, help="TokenPool key/query width.")
    parser.add_argument(
        "--pool-type", choices=["attention", "gated", "topk", "mean", "max"], default="attention"
    )
    parser.add_argument("--n-queries", type=int, default=1, help="Parallel pooling queries (attention/gated/topk).")
    parser.add_argument("--topk", type=int, default=None, help="Only used by --pool-type topk.")
    parser.add_argument("--pool-temperature", type=float, default=1.0)
    parser.add_argument(
        "--modalities", nargs="+", default=["base", "wrist", "lang"],
        choices=["base", "wrist", "lang"], help="Which token groups the probe consumes.",
    )
    parser.add_argument("--share-image-pool", action="store_true", help="base and wrist share one pool module.")
    parser.add_argument(
        "--input-proj-dim", type=int, default=None,
        help="Fixed (non-trained) random projection applied to every token before pooling.",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None,
        help="Insert an embedding layer: pool -> Linear(embed_dim) -> GELU -> Dropout = z (CLIP-align target).",
    )
    parser.add_argument(
        "--embed-hidden", type=int, default=None,
        help="Compression bottleneck before the embed layer: head_in -> embed_hidden -> embed_dim.",
    )
    parser.add_argument(
        "--align-weight", type=float, default=0.0,
        help="If >0, add align_weight * InfoNCE(z, CLIP_text[task]). Needs --embed-dim.",
    )
    parser.add_argument("--align-temp", type=float, default=0.07, help="InfoNCE temperature for --align-weight.")
    parser.add_argument(
        "--clip-embeddings", type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "clip" / "task_instruction_embeddings.npy",
        help="(N_tasks, D) L2-normalised CLIP text embeddings; task_index.json alongside maps task->row. "
        "Only loaded when --align-weight > 0 (unused by the phase 1-7 sweep).",
    )
    parser.add_argument("--patience", type=int, default=None, help="Early stop after N epochs with no val-metric improvement.")
    parser.add_argument(
        "--max-steps", type=int, default=80,
        help="Clip every episode (train AND eval loaders) to this many inference calls at load "
        "time (0 = no cap). For GR00T (replan 8, up to ~65 inferences/episode) this rarely binds.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch per worker.")
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable bfloat16 autocast for the probe forward (kept fp32).",
    )

    # --- Loss / model hyperparameters (match SAFE IndepModelConfig / ModelConfig) ---
    parser.add_argument(
        "--cumsum", dest="cumsum", action="store_true", default=True,
        help="Accumulate per-step sigmoid scores over time via cumsum (IndepModelConfig default: True).",
    )
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    parser.add_argument(
        "--rmean", action="store_true", default=False,
        help="Divide the cumsum by elapsed timesteps (running mean instead of running sum).",
    )
    parser.add_argument(
        "--use-time-weighting", action="store_true", default=False,
        help="Front-load the failure-sequence loss on early timesteps (hinge path only).",
    )
    parser.add_argument(
        "--use-threshold", action="store_true", default=False,
        help="Failure-sequence loss becomes relu(threshold - score) instead of unbounded -score. Hinge path only.",
    )
    parser.add_argument("--threshold", type=float, default=50.0, help="Only used if --use-threshold.")
    parser.add_argument(
        "--raw-target-loss", action="store_true", default=False,
        help="Symmetric per-timestep BCE pushing the RAW sigmoid score toward 1 (failure) / 0 (success), "
        "summed over each episode's valid timesteps. --cumsum/--rmean then only affect the eval-time "
        "accumulated score, not the loss.",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=0.0,
        help="Focal-loss exponent on the raw-target BCE (0 = plain BCE). Only used with --raw-target-loss.",
    )
    parser.add_argument(
        "--mil-pool", choices=["max", "lse", "topk"], default=None,
        help="Multiple-instance-learning loss: pool per-step logits to one episode logit, then BCE "
        "against the episode label. Overrides --raw-target-loss / the hinge loss when set.",
    )
    parser.add_argument("--mil-topk", type=int, default=8, help="k for --mil-pool topk.")
    parser.add_argument(
        "--ranking-weight", type=float, default=0.0,
        help="If >0, add ranking_weight * pairwise_ranking_loss (softplus AUROC surrogate) on top.",
    )
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--lambda-success", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument(
        "--lambda-reg", type=float, default=1e-2,
        help="L2 weight-decay-style regularization coefficient on the loss (weight params only).",
    )
    parser.add_argument("--grad-max-norm", type=float, default=None)

    # --- Optimizer / scheduler, matching BaseModel.get_optimizer() ---
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd", "sgdm"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Only used by --optimizer adamw.")
    parser.add_argument("--lr-step-size", type=int, default=300, help="StepLR period in EPOCHS (--lr-schedule step).")
    parser.add_argument("--lr-gamma", type=float, default=1.0, help="1.0 = no decay (ModelConfig default).")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--lr-schedule", choices=["step", "cosine"], default="step",
        help="'step' = StepLR(--lr-step-size epochs, --lr-gamma); 'cosine' = cosine anneal to 0 over --epochs.",
    )
    parser.add_argument(
        "--select-by", choices=["auroc", "separation"], default="auroc",
        help="Which val metric picks probe_best.pt (and drives --patience).",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Soften the raw-target BCE targets: failure -> 1-eps, success -> eps. Only used with --raw-target-loss.",
    )

    parser.add_argument(
        "--eval-threshold", type=float, default=0.0,
        help="Decision boundary on the (task-min-step-truncated) accumulated score (predict failure iff "
        "score > threshold) -- NOT a probability threshold.",
    )
    parser.add_argument(
        "--no-task-min-step-eval", dest="task_min_step_eval", action="store_false", default=True,
        help="Score each episode at its own full length instead of its task's shortest observed rollout "
        "length. NOT recommended -- kept only for comparison.",
    )
    parser.add_argument("--wandb-project", default="gr00t-perception-probe")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def score_sequence(
    model: nn.Module, batch: dict, cumsum: bool, rmean: bool, amp: bool = True,
    chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs PerceptionSuccessProbe independently at every (padded) timestep,
    then sigmoids and optionally accumulates over time. Returns (step_logits,
    raw_scores, accumulated_scores)."""
    base, wrist, lang, lang_mask = (
        batch["base_image"], batch["wrist_image"], batch["language"], batch["language_mask"],
    )
    B, T = base.shape[0], base.shape[1]
    bf = base.reshape(B * T, *base.shape[2:])
    wf = wrist.reshape(B * T, *wrist.shape[2:])
    lf = lang.reshape(B * T, *lang.shape[2:])
    mf = lang_mask.reshape(B * T, *lang_mask.shape[2:])
    step = chunk if chunk else B * T
    with torch.autocast(device_type=base.device.type, dtype=torch.bfloat16, enabled=amp):
        logits = torch.cat([
            model(bf[i:i + step].to(torch.bfloat16), wf[i:i + step].to(torch.bfloat16),
                  lf[i:i + step].to(torch.bfloat16), mf[i:i + step])
            for i in range(0, B * T, step)
        ], dim=0)  # (B*T,)
    step_logits = logits.float().view(B, T)
    raw_scores = torch.sigmoid(step_logits)

    scores = raw_scores
    if cumsum or rmean:
        scores = torch.cumsum(scores, dim=1)
        if rmean:
            scores = scores / torch.arange(1, T + 1, device=scores.device, dtype=scores.dtype).unsqueeze(0)

    return step_logits, raw_scores, scores


def get_time_weight(use_weighting: bool, valid_masks: torch.Tensor) -> torch.Tensor:
    """Verbatim port of 13-SAFE/failure_prob/model/utils.py's get_time_weight."""
    B, T = valid_masks.shape
    seq_lengths = valid_masks.sum(dim=1)
    if use_weighting:
        t = torch.arange(T, device=valid_masks.device, dtype=valid_masks.dtype).unsqueeze(0).expand(B, -1)
        w = t / seq_lengths.unsqueeze(1)
        w = 5 * torch.exp(-3 * w) + 1
        w = w * valid_masks
        normalizer = w.sum(-1) / seq_lengths
        w = w / normalizer.unsqueeze(1)
    else:
        w = torch.ones(B, T, device=valid_masks.device, dtype=valid_masks.dtype) * valid_masks
    return w


def time_dependent_mlp_loss(
    scores: torch.Tensor,
    valid_masks: torch.Tensor,
    success_labels: torch.Tensor,
    class_weights: tuple[float, float],
    use_time_weighting: bool = False,
    use_threshold: bool = False,
    threshold: float = 50.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Verbatim port of IndepModel.forward_compute_loss's loss computation."""
    B = scores.shape[0]
    time_weights = get_time_weight(use_time_weighting, valid_masks).to(scores)

    loss_success = torch.relu(scores)
    if use_threshold:
        loss_fail = time_weights * torch.relu(threshold - scores)
    else:
        loss_fail = time_weights * (-scores)

    is_success = (success_labels == 1).float().unsqueeze(1)
    losses = is_success * loss_success + (1.0 - is_success) * loss_fail

    seq_loss = (losses * valid_masks).sum(-1) / valid_masks.sum(-1)

    fail_mask = (success_labels == 0).float()
    success_mask = (success_labels == 1).float()
    fail_loss = (fail_mask * seq_loss).sum()
    success_loss = (success_mask * seq_loss).sum()

    fail_weight, success_weight = class_weights
    monitor_loss = (fail_weight * fail_loss + success_weight * success_loss) / B

    logs = {
        "monitor_loss": monitor_loss.item(),
        "success_loss": (success_loss / success_mask.sum().clamp(min=1)).item(),
        "fail_loss": (fail_loss / fail_mask.sum().clamp(min=1)).item(),
    }
    return monitor_loss, logs


def time_dependent_raw_target_loss(
    raw_scores: torch.Tensor,
    valid_masks: torch.Tensor,
    success_labels: torch.Tensor,
    class_weights: tuple[float, float],
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric per-timestep BCE: push the RAW sigmoid score toward 1 at every
    timestep of a failure episode and toward 0 at every timestep of a success
    episode. Aggregated per-episode via SUM over valid timesteps."""
    is_success = (success_labels == 1).float().unsqueeze(1)

    eps = 1e-7
    clamped = raw_scores.clamp(eps, 1.0 - eps)
    if label_smoothing > 0:
        ls = label_smoothing
        loss_fail = -((1 - ls) * torch.log(clamped) + ls * torch.log(1.0 - clamped))
        loss_success = -((1 - ls) * torch.log(1.0 - clamped) + ls * torch.log(clamped))
    else:
        loss_success = -torch.log(1.0 - clamped)
        loss_fail = -torch.log(clamped)
    if focal_gamma > 0:
        loss_success = (clamped**focal_gamma) * loss_success
        loss_fail = ((1.0 - clamped) ** focal_gamma) * loss_fail
    losses = is_success * loss_success + (1.0 - is_success) * loss_fail

    seq_loss = (losses * valid_masks).sum(-1)

    fail_mask = (success_labels == 0).float()
    success_mask = (success_labels == 1).float()
    fail_loss = (fail_mask * seq_loss).sum()
    success_loss = (success_mask * seq_loss).sum()

    fail_weight, success_weight = class_weights
    B = raw_scores.shape[0]
    monitor_loss = (fail_weight * fail_loss + success_weight * success_loss) / B

    logs = {
        "monitor_loss": monitor_loss.item(),
        "success_loss": (success_loss / success_mask.sum().clamp(min=1)).item(),
        "fail_loss": (fail_loss / fail_mask.sum().clamp(min=1)).item(),
    }
    return monitor_loss, logs


def time_dependent_mil_loss(
    step_logits: torch.Tensor,
    valid_masks: torch.Tensor,
    success_labels: torch.Tensor,
    class_weights: tuple[float, float],
    pool: str = "lse",
    topk: int = 8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multiple-instance-learning loss: pool the per-timestep logits into ONE
    episode logit, then BCE that against the episode label."""
    neg_inf = torch.finfo(step_logits.dtype).min
    masked = step_logits.masked_fill(~valid_masks.bool(), neg_inf)
    lengths = valid_masks.sum(dim=1).clamp(min=1)

    if pool == "max":
        bag_logit = masked.max(dim=1).values
    elif pool == "lse":
        bag_logit = torch.logsumexp(masked, dim=1) - torch.log(lengths)
    elif pool == "topk":
        k = min(topk, step_logits.shape[1])
        bag_logit = masked.topk(k, dim=1).values
        bag_logit = torch.where(bag_logit <= neg_inf / 2, torch.zeros_like(bag_logit), bag_logit)
        bag_logit = bag_logit.sum(dim=1) / valid_masks.sum(dim=1).clamp(min=1).clamp(max=k)
    else:
        raise ValueError(f"unknown MIL pool {pool!r}")

    target = (success_labels == 0).float()
    per_ep = nn.functional.binary_cross_entropy_with_logits(bag_logit, target, reduction="none")

    fail_mask = target
    success_mask = 1.0 - target
    fail_loss = (fail_mask * per_ep).sum()
    success_loss = (success_mask * per_ep).sum()
    fail_weight, success_weight = class_weights
    B = step_logits.shape[0]
    monitor_loss = (fail_weight * fail_loss + success_weight * success_loss) / B
    logs = {
        "monitor_loss": monitor_loss.item(),
        "success_loss": (success_loss / success_mask.sum().clamp(min=1)).item(),
        "fail_loss": (fail_loss / fail_mask.sum().clamp(min=1)).item(),
    }
    return monitor_loss, logs


def pairwise_ranking_loss(
    episode_scores: torch.Tensor,
    success_labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """softplus(s_success - s_failure + margin) averaged over every
    failure/success pair in the batch -- a smooth AUROC surrogate."""
    fail = episode_scores[success_labels == 0]
    succ = episode_scores[success_labels == 1]
    if fail.numel() == 0 or succ.numel() == 0:
        return episode_scores.sum() * 0.0
    diff = succ[None, :] - fail[:, None] + margin
    return nn.functional.softplus(diff).mean()


def episode_embedding(model: nn.Module, batch: dict, amp: bool, max_steps_per_ep: int = 12) -> torch.Tensor:
    """Mean-pool the probe's --embed-dim latent z over a subsample of each
    episode's valid timesteps -> (B, embed_dim)."""
    cast = torch.bfloat16 if amp else torch.float32
    valid = batch["valid_masks"]
    device = batch["base_image"].device
    B = valid.shape[0]
    idx_b, idx_t = [], []
    for b in range(B):
        v = valid[b].nonzero(as_tuple=False).flatten()
        if v.numel() == 0:
            continue
        if v.numel() > max_steps_per_ep:
            sel = torch.linspace(0, v.numel() - 1, max_steps_per_ep, device=v.device).round().long()
            v = v[sel]
        idx_t.append(v.to(device))
        idx_b.append(torch.full((v.numel(),), b, device=device, dtype=torch.long))
    idx_b = torch.cat(idx_b)
    idx_t = torch.cat(idx_t)

    def gather(key: str) -> torch.Tensor:
        return batch[key][idx_b, idx_t].to(cast)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        _, z = model(
            gather("base_image"), gather("wrist_image"),
            gather("language"), batch["language_mask"][idx_b, idx_t],
            return_embedding=True,
        )
    z = z.float()
    out = torch.zeros(B, z.shape[-1], device=device, dtype=z.dtype).index_add(0, idx_b, z)
    cnt = torch.zeros(B, 1, device=device, dtype=z.dtype).index_add(
        0, idx_b, torch.ones(idx_b.shape[0], 1, device=device, dtype=z.dtype)
    )
    return out / cnt.clamp(min=1.0)


def clip_alignment_loss(z_ep, task_idx, clip_embeds, temp):
    """InfoNCE: normalised z_ep closest to its own task's CLIP embedding."""
    zc = torch.nn.functional.normalize(z_ep, dim=-1)
    sim = zc @ clip_embeds.T / temp
    loss = torch.nn.functional.cross_entropy(sim, task_idx)
    acc = (sim.argmax(dim=-1) == task_idx).float().mean().item()
    return loss, acc


def compute_primary_loss(
    args: argparse.Namespace,
    step_logits: torch.Tensor,
    raw_scores: torch.Tensor,
    scores: torch.Tensor,
    batch: dict,
    class_weights: tuple[float, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dispatches to whichever primary loss the flags select, then optionally
    adds a pairwise-ranking auxiliary term (--ranking-weight)."""
    vm, labels = batch["valid_masks"], batch["success_labels"]
    if args.mil_pool:
        loss, logs = time_dependent_mil_loss(
            step_logits, vm, labels, class_weights, pool=args.mil_pool, topk=args.mil_topk
        )
    elif args.raw_target_loss:
        loss, logs = time_dependent_raw_target_loss(
            raw_scores, vm, labels, class_weights,
            focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        )
    else:
        loss, logs = time_dependent_mlp_loss(
            scores, vm, labels, class_weights,
            args.use_time_weighting, args.use_threshold, args.threshold,
        )
    if args.ranking_weight > 0:
        ep = final_episode_scores(scores, batch["lengths"])
        rank = pairwise_ranking_loss(ep, labels, margin=args.ranking_margin)
        loss = loss + args.ranking_weight * rank
        logs["ranking_loss"] = rank.item()
    return loss, logs


def l2_regularization_loss(model: nn.Module, lambda_reg: float) -> torch.Tensor:
    """Verbatim port of BaseModel.compute_regularization_loss."""
    if lambda_reg == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg_loss = sum(
        torch.sum(param**2) for name, param in model.named_parameters() if "bias" not in name
    )
    return lambda_reg * reg_loss


def compute_class_weights(rows: list[dict], lambda_fail: float, lambda_success: float) -> tuple[float, float]:
    """Verbatim port of RolloutDataset.__init__'s class-weight computation."""
    n = len(rows)
    n_fail = sum(1 for r in rows if r["success"] != "True")
    n_success = n - n_fail
    freq_fail = (n_fail + 1) / n
    freq_success = (n_success + 1) / n
    return (1.0 / freq_fail) * lambda_fail, (1.0 / freq_success) * lambda_success


def get_optimizer(model: nn.Module, args: argparse.Namespace):
    """Verbatim port of BaseModel.get_optimizer()."""
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgdm":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.lr_schedule == "cosine":
        step_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        step_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    if args.warmup_steps > 0:
        lr_lambda = lambda step: min((step + 1) / args.warmup_steps, 1.0)  # noqa: E731
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[args.warmup_steps]
        )
    else:
        scheduler = step_scheduler
    return optimizer, scheduler


def final_episode_scores(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Gathers each episode's score at its OWN last valid timestep (index length-1)."""
    idx = (lengths - 1).clamp(min=0).view(-1, 1)
    return scores.gather(1, idx).squeeze(1)


def compute_task_min_steps(rows: list[dict]) -> dict[str, int]:
    """Per-task minimum observed episode length (in inference calls), over EVERY
    episode of that task in `rows`. manifest.csv's "inference_calls" column
    already IS each episode's T."""
    mins: dict[str, int] = {}
    for row in rows:
        length = int(row["inference_calls"])
        task = row["task"]
        mins[task] = min(mins[task], length) if task in mins else length
    return mins


def truncated_episode_scores(
    scores: torch.Tensor, lengths: torch.Tensor, tasks: list[str], task_min_step: dict[str, int]
) -> torch.Tensor:
    """Gathers each episode's score at index (task_min_step[task] - 1) -- the
    SAME, fixed cutoff for every episode of a given task."""
    idx = torch.tensor(
        [task_min_step[task] - 1 for task in tasks], device=scores.device, dtype=torch.long
    )
    idx = idx.clamp(min=0, max=scores.shape[1] - 1).view(-1, 1)
    return scores.gather(1, idx).squeeze(1)


def export_probe_onnx(model: nn.Module, device: str, path: Path) -> None:
    """Exports PerceptionSuccessProbe to ONNX for inspection (e.g. Netron) -- a
    plain single-timestep forward pass, NOT the score accumulation."""
    dummy = (
        torch.randn(1, NUM_IMAGE_TOKENS, FEATURE_DIM, device=device),
        torch.randn(1, NUM_IMAGE_TOKENS, FEATURE_DIM, device=device),
        torch.randn(1, NUM_LANGUAGE_TOKENS, FEATURE_DIM, device=device),
        torch.ones(1, NUM_LANGUAGE_TOKENS, dtype=torch.bool, device=device),
    )
    torch.onnx.export(
        model, dummy, path,
        input_names=["base_image", "wrist_image", "language", "language_mask"],
        output_names=["failure_logit"],
        dynamic_axes={
            "base_image": {0: "batch"}, "wrist_image": {0: "batch"},
            "language": {0: "batch"}, "language_mask": {0: "batch"},
            "failure_logit": {0: "batch"},
        },
    )


def compute_episode_trace_for_overlay(
    model: nn.Module, item: dict, cumsum: bool, rmean: bool, device: str
) -> tuple[list[tuple[int, float, float]], bool]:
    """Raw + accumulated score at every timestep of ONE cached episode (full
    length, no task-min-step truncation), for plot_overlay()."""
    base = item["base_image"].unsqueeze(0).to(device)
    wrist = item["wrist_image"].unsqueeze(0).to(device)
    lang = item["language"].unsqueeze(0).to(device)
    lang_mask = item["language_mask"].unsqueeze(0).to(device)
    batch = {"base_image": base, "wrist_image": wrist, "language": lang, "language_mask": lang_mask}
    with torch.no_grad():
        _, raw_scores, scores = score_sequence(model, batch, cumsum, rmean, chunk=128)
    T = item["length"]
    trace = [
        (round(t * REPLAN_STEPS), raw_scores[0, t].item(), scores[0, t].item()) for t in range(T)
    ]
    return trace, item["label"] == 1.0


def plot_split_score_overlay(
    model: nn.Module, rows: list[dict], features_dir: Path, args: argparse.Namespace, path: Path, title: str,
) -> None:
    """Runs compute_episode_trace_for_overlay over every episode in `rows` and
    writes one overlay plot."""
    ds = EpisodeSequenceDataset(rows, features_dir)
    traces = [
        compute_episode_trace_for_overlay(model, ds[i], args.cumsum, args.rmean, args.device)
        for i in range(len(ds))
    ]
    plot_overlay(traces, args, path, title=title)


def run_eval_pass(
    model: nn.Module, loader: DataLoader, device: str, args: argparse.Namespace,
    task_min_step: dict[str, int] | None = None, desc: str = "val",
) -> dict:
    """Deliberately does NOT report a combined monitor_loss on held-out data --
    matching 13-SAFE/failure_prob/train.py. Reports per-class success_loss /
    fail_loss alongside AUROC / accuracy / separation."""
    model.eval()
    all_labels, all_final_scores = [], []
    total_success_loss, total_success_n = 0.0, 0
    total_fail_loss, total_fail_n = 0.0, 0
    class_weights = (1.0, 1.0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            tasks = batch["tasks"]
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            step_logits, raw_scores, scores = score_sequence(
                model, batch, args.cumsum, args.rmean, amp=not args.no_amp, chunk=128
            )
            _, logs = compute_primary_loss(
                args, step_logits, raw_scores, scores, batch, class_weights
            )
            B = batch["success_labels"].shape[0]
            n_success = int((batch["success_labels"] == 1).sum().item())
            n_fail = B - n_success
            total_success_loss += logs["success_loss"] * n_success
            total_success_n += n_success
            total_fail_loss += logs["fail_loss"] * n_fail
            total_fail_n += n_fail
            all_labels.append(batch["success_labels"].cpu().numpy())
            if args.task_min_step_eval:
                final = truncated_episode_scores(scores, batch["lengths"], tasks, task_min_step)
            else:
                final = final_episode_scores(scores, batch["lengths"])
            all_final_scores.append(final.cpu().numpy())

    labels = np.concatenate(all_labels)
    final_scores = np.concatenate(all_final_scores)

    fail_scores = final_scores[labels == 0]
    success_scores = final_scores[labels == 1]
    if len(fail_scores) and len(success_scores) and final_scores.std() > 1e-9:
        separation = float((fail_scores.mean() - success_scores.mean()) / final_scores.std())
    else:
        separation = None

    preds = (final_scores > args.eval_threshold).astype(np.float32)
    predicted_success = 1.0 - preds
    return {
        "success_loss": total_success_loss / max(total_success_n, 1),
        "fail_loss": total_fail_loss / max(total_fail_n, 1),
        "accuracy": float((predicted_success == labels).mean()),
        "auroc": auroc(labels, -final_scores),
        "separation": separation,
        "mean_fail_score": float(fail_scores.mean()) if len(fail_scores) else None,
        "mean_success_score": float(success_scores.mean()) if len(success_scores) else None,
        "n": int(total_success_n + total_fail_n),
        "_labels": labels,
        "_final_scores": final_scores,
    }


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt()


def build_model(args: argparse.Namespace) -> PerceptionSuccessProbe:
    """The one place args -> PerceptionSuccessProbe kwargs is spelled out."""
    return PerceptionSuccessProbe(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_hidden_layers=args.n_hidden_layers,
        key_dim=args.key_dim,
        pool_type=args.pool_type,
        n_queries=args.n_queries,
        topk=args.topk,
        pool_temperature=args.pool_temperature,
        modalities=tuple(args.modalities),
        share_image_pool=args.share_image_pool,
        input_proj_dim=args.input_proj_dim,
        embed_dim=args.embed_dim,
        embed_hidden=args.embed_hidden,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Run outputs going to %s", run_dir)

    available_gb = available_memory_gb()
    if available_gb is not None:
        logging.info("Currently available system memory: %.1fGB", available_gb)

    all_rows = read_probe_manifest(args.features_dir)
    suite_rows = [r for r in all_rows if suite_of(r) == args.suite]
    if not suite_rows:
        raise ValueError(
            f"No episodes found for --suite {args.suite!r} in {args.features_dir}/manifest.csv "
            f"(known suites: {sorted(set(suite_of(r) for r in all_rows))})"
        )
    suite_tasks = sorted({r["task"] for r in suite_rows})
    forced_unseen = set(args.unseen_tasks or [])
    missing_forced = forced_unseen - set(suite_tasks)
    if missing_forced:
        raise ValueError(
            f"--unseen-tasks {sorted(missing_forced)} not found among --suite {args.suite!r}'s "
            f"tasks: {suite_tasks}"
        )
    if len(forced_unseen) > args.num_unseen_tasks:
        raise ValueError(
            f"--unseen-tasks names {len(forced_unseen)} tasks, more than --num-unseen-tasks "
            f"({args.num_unseen_tasks})"
        )
    unseen_task_seed = args.unseen_task_seed if args.unseen_task_seed is not None else args.seed
    remaining_needed = args.num_unseen_tasks - len(forced_unseen)
    candidates = sorted(set(suite_tasks) - forced_unseen)
    rng = np.random.default_rng(unseen_task_seed)
    randomly_chosen = list(rng.choice(candidates, size=remaining_needed, replace=False)) if remaining_needed else []
    unseen_task_set = forced_unseen | set(randomly_chosen)
    logging.info(
        "Unseen-task selection: seed=%d, %d forced (%s), %d randomly chosen (%s)",
        unseen_task_seed, len(forced_unseen), sorted(forced_unseen), len(randomly_chosen), sorted(randomly_chosen),
    )

    rows = [r for r in suite_rows if r["task"] not in unseen_task_set]
    unseen_rows = [r for r in suite_rows if r["task"] in unseen_task_set]
    logging.info(
        "Suite %s: %d episodes over %d tasks; %d tasks held out as unseen (%d episodes), "
        "%d tasks seen (%d episodes) for train/val/test/calibration",
        args.suite, len(suite_rows), len(suite_tasks),
        len(unseen_task_set), len(unseen_rows), len(suite_tasks) - len(unseen_task_set), len(rows),
    )
    logging.info("Unseen (zero-shot only) tasks: %s", sorted(unseen_task_set))

    task_min_step = compute_task_min_steps(suite_rows)
    logging.info("Per-task min observed episode length (eval truncation cutoff): %s", task_min_step)

    splits = split_episodes(
        rows, args.seed,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, calibration_fraction=args.calibration_fraction,
    )
    train_rows, val_rows, test_rows, calibration_rows = (
        splits["train"], splits["val"], splits["test"], splits["calibration"]
    )
    failures = {name: sum(1 for r in part if r["success"] != "True") for name, part in splits.items()}
    logging.info(
        "Episodes: %d train (%d failures), %d val (%d failures), %d test (%d failures), "
        "%d calibration (%d failures) (of %d seen)",
        len(train_rows), failures["train"], len(val_rows), failures["val"],
        len(test_rows), failures["test"], len(calibration_rows), failures["calibration"], len(rows),
    )

    split_key = lambda r: {  # noqa: E731
        "npz_path": r["npz_path"], "scene_variant": r["scene_variant"],
        "task": r["task"], "episode": r["episode"], "success": r["success"],
    }
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "suite": args.suite,
                "unseen_task_seed": unseen_task_seed,
                "unseen_tasks": sorted(unseen_task_set),
                "train_fraction": args.train_fraction, "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction, "calibration_fraction": args.calibration_fraction,
                "manifest_rows_at_split_time": len(all_rows),
                "suite_rows_at_split_time": len(suite_rows),
                "task_min_step": task_min_step,
                "train": [split_key(r) for r in train_rows],
                "val": [split_key(r) for r in val_rows],
                "test": [split_key(r) for r in test_rows],
                "calibration": [split_key(r) for r in calibration_rows],
                "unseen": [split_key(r) for r in unseen_rows],
            },
            indent=2,
        )
    )

    class_weights = compute_class_weights(train_rows, args.lambda_fail, args.lambda_success)
    logging.info(
        "Train class weights (fail_weight=%.3f, success_weight=%.3f) from %d failures / %d successes",
        class_weights[0], class_weights[1], failures["train"], len(train_rows) - failures["train"],
    )

    cap = args.max_steps or None
    train_ds = EpisodeSequenceDataset(train_rows, args.features_dir, max_steps=cap)
    val_ds = EpisodeSequenceDataset(val_rows, args.features_dir, max_steps=cap)
    test_ds = EpisodeSequenceDataset(test_rows, args.features_dir, max_steps=cap)

    loader_kw = dict(collate_fn=collate_pad_episodes, num_workers=args.num_workers)
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = build_model(args).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Probe: %s  (%d trainable params)", model, n_params)
    optimizer, scheduler = get_optimizer(model, args)

    clip_embeds = None
    task_to_clip_idx = {}
    if args.align_weight > 0:
        if args.embed_dim is None:
            raise ValueError("--align-weight requires --embed-dim (nowhere to align without a z latent)")
        emb = np.load(args.clip_embeddings)
        task_to_clip_idx = json.loads((args.clip_embeddings.parent / "task_index.json").read_text())
        clip_embeds = torch.tensor(emb, dtype=torch.float32, device=args.device)
        clip_embeds = torch.nn.functional.normalize(clip_embeds, dim=-1)
        logging.info("CLIP alignment: %d task embeddings (dim %d), weight=%.3g temp=%.3g",
                     clip_embeds.shape[0], clip_embeds.shape[1], args.align_weight, args.align_temp)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.config.update(
            {
                "train_episodes": len(train_rows), "train_failures": failures["train"],
                "val_episodes": len(val_rows), "val_failures": failures["val"],
                "test_episodes": len(test_rows), "test_failures": failures["test"],
                "class_weight_fail": class_weights[0], "class_weight_success": class_weights[1],
            }
        )

    best_val_auroc = -float("inf")
    best_val_metric = -float("inf")
    best_epoch = None
    epochs_since_improvement = 0
    history = []
    interrupted = False

    try:
        try:
            epoch_bar = tqdm(range(args.epochs), desc="epochs", unit="epoch")
            for epoch in epoch_bar:
                model.train()
                running_loss, running_n = 0.0, 0
                batch_bar = tqdm(train_loader, desc=f"epoch {epoch} train", unit="batch", leave=False)
                for batch in batch_bar:
                    batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    optimizer.zero_grad()

                    step_logits, raw_scores, scores = score_sequence(
                        model, batch, args.cumsum, args.rmean, amp=not args.no_amp
                    )
                    loss, logs = compute_primary_loss(
                        args, step_logits, raw_scores, scores, batch, class_weights
                    )
                    reg_loss = l2_regularization_loss(model, args.lambda_reg)
                    total_loss = loss + reg_loss
                    if args.align_weight > 0 and clip_embeds is not None:
                        z_ep = episode_embedding(model, batch, amp=not args.no_amp)
                        tix = torch.tensor([task_to_clip_idx[t] for t in batch["tasks"]],
                                           device=args.device, dtype=torch.long)
                        al, al_acc = clip_alignment_loss(z_ep, tix, clip_embeds, args.align_temp)
                        total_loss = total_loss + args.align_weight * al
                        logs["align_loss"] = al.item(); logs["align_acc"] = al_acc

                    total_loss.backward()
                    if args.grad_max_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm)
                    optimizer.step()

                    B = batch["success_labels"].shape[0]
                    running_loss += total_loss.item() * B
                    running_n += B
                    batch_bar.set_postfix(loss=f"{running_loss / running_n:.4f}")
                    if not args.no_wandb:
                        wandb.log(
                            {
                                "train_loss/monitor_loss": logs["monitor_loss"],
                                "train_loss/success_loss": logs["success_loss"],
                                "train_loss/fail_loss": logs["fail_loss"],
                                "train_loss/reg_loss": reg_loss.item(),
                                "train_loss/total_loss": total_loss.item(),
                            }
                        )

                scheduler.step()
                train_metrics = {"loss": running_loss / running_n}
                val_metrics = run_eval_pass(model, val_loader, args.device, args, task_min_step, desc="val")
                history.append({
                    "epoch": epoch, "train": train_metrics,
                    "val": {k: v for k, v in val_metrics.items() if not k.startswith("_")},
                })
                epoch_bar.set_postfix(
                    train_loss=f"{train_metrics['loss']:.4f}",
                    val_acc=f"{val_metrics['accuracy']:.3f}",
                    val_auroc=f"{val_metrics['auroc']:.3f}" if val_metrics["auroc"] is not None else "n/a",
                )
                sep_str = f"{val_metrics['separation']:.3f}" if val_metrics["separation"] is not None else "n/a"
                logging.info(
                    "epoch %2d  train_loss=%.4f  val_acc=%.3f  val_auroc=%s  val_sep=%s  "
                    "val_success_loss=%.4f  val_fail_loss=%.4f  lr=%.2e",
                    epoch, train_metrics["loss"], val_metrics["accuracy"],
                    f"{val_metrics['auroc']:.3f}" if val_metrics["auroc"] is not None else "n/a",
                    sep_str, val_metrics["success_loss"], val_metrics["fail_loss"],
                    optimizer.param_groups[0]["lr"],
                )
                if not args.no_wandb:
                    wandb.log(
                        {
                            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                            "val/accuracy": val_metrics["accuracy"],
                            "val/auroc": val_metrics["auroc"],
                            "val/separation": val_metrics["separation"],
                            "val/success_loss": val_metrics["success_loss"],
                            "val/fail_loss": val_metrics["fail_loss"],
                        }
                    )
                sel = val_metrics.get(args.select_by)
                if sel is not None and sel > best_val_metric:
                    best_val_metric = sel
                    best_val_auroc = val_metrics["auroc"]
                    best_epoch = epoch
                    epochs_since_improvement = 0
                    torch.save(model.state_dict(), run_dir / "probe_best.pt")
                else:
                    epochs_since_improvement += 1
                    if args.patience is not None and epochs_since_improvement >= args.patience:
                        logging.info(
                            "Early stop: no val-%s improvement in %d epochs (best epoch %s).",
                            args.select_by, args.patience, best_epoch,
                        )
                        break
        except KeyboardInterrupt:
            interrupted = True
            logging.warning(
                "Training interrupted (%d epoch%s completed) -- finalizing with the best checkpoint "
                "saved so far.", len(history), "" if len(history) == 1 else "s",
            )

        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), run_dir / "probe_last.pt")
        logging.info("Saved probe weights + history to %s", run_dir)

        latest_link = args.output_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name, target_is_directory=True)
        logging.info("Updated %s -> %s", latest_link, run_dir.name)

        if best_epoch is None:
            logging.warning("No epoch finished -- skipping final test evaluation.")
        else:
            model.load_state_dict(torch.load(run_dir / "probe_best.pt", map_location=args.device))
            test_metrics = run_eval_pass(model, test_loader, args.device, args, task_min_step, desc="test (final)")
            test_labels, test_final_scores = test_metrics["_labels"], test_metrics["_final_scores"]

            cm = confusion_matrix(1.0 - test_labels, test_final_scores, args.eval_threshold)
            final_auroc = auroc(test_labels, -test_final_scores)
            test_summary = {
                "success_loss": test_metrics["success_loss"],
                "fail_loss": test_metrics["fail_loss"], "accuracy": test_metrics["accuracy"],
                "auroc": final_auroc, "separation": test_metrics["separation"],
                "mean_fail_score": test_metrics["mean_fail_score"],
                "mean_success_score": test_metrics["mean_success_score"],
                "best_epoch": best_epoch, "best_val_auroc": best_val_auroc,
                "n": test_metrics["n"], "confusion_matrix": cm,
            }
            (run_dir / "test_metrics.json").write_text(json.dumps(test_summary, indent=2))
            logging.info(
                "Confusion matrix (best checkpoint, failure-score threshold=%.2f, positive=FAILURE): "
                "TP=%d TN=%d FP=%d FN=%d", args.eval_threshold, cm["tp"], cm["tn"], cm["fp"], cm["fn"],
            )
            logging.info(
                "Final test metrics (best checkpoint): success_loss=%.4f fail_loss=%.4f "
                "accuracy=%.3f auroc=%s n=%d",
                test_summary["success_loss"], test_summary["fail_loss"],
                test_summary["accuracy"], f"{final_auroc:.3f}" if final_auroc is not None else "n/a",
                test_summary["n"],
            )

            roc_path = run_dir / "roc_curve.png"
            fpr, tpr = roc_curve_points(test_labels, -test_final_scores)
            plot_roc_curve(fpr, tpr, final_auroc, roc_path)
            logging.info("Saved ROC curve to %s", roc_path)

            unseen_ds = EpisodeSequenceDataset(unseen_rows, args.features_dir, max_steps=cap)
            unseen_loader = DataLoader(
                unseen_ds, batch_size=args.batch_size, shuffle=False, **loader_kw
            )
            unseen_metrics = run_eval_pass(
                model, unseen_loader, args.device, args, task_min_step, desc="unseen (zero-shot)"
            )
            unseen_labels, unseen_final_scores = unseen_metrics["_labels"], unseen_metrics["_final_scores"]
            unseen_cm = confusion_matrix(1.0 - unseen_labels, unseen_final_scores, args.eval_threshold)
            unseen_auroc = auroc(unseen_labels, -unseen_final_scores)
            unseen_summary = {
                "success_loss": unseen_metrics["success_loss"],
                "fail_loss": unseen_metrics["fail_loss"], "accuracy": unseen_metrics["accuracy"],
                "auroc": unseen_auroc, "separation": unseen_metrics["separation"],
                "mean_fail_score": unseen_metrics["mean_fail_score"],
                "mean_success_score": unseen_metrics["mean_success_score"],
                "n": unseen_metrics["n"], "confusion_matrix": unseen_cm,
            }
            (run_dir / "unseen_metrics.json").write_text(json.dumps(unseen_summary, indent=2))
            logging.info(
                "Zero-shot UNSEEN-task confusion matrix (failure-score threshold=%.2f, positive=FAILURE): "
                "TP=%d TN=%d FP=%d FN=%d", args.eval_threshold,
                unseen_cm["tp"], unseen_cm["tn"], unseen_cm["fp"], unseen_cm["fn"],
            )
            logging.info(
                "Zero-shot UNSEEN-task metrics: success_loss=%.4f fail_loss=%.4f "
                "accuracy=%.3f auroc=%s n=%d",
                unseen_summary["success_loss"], unseen_summary["fail_loss"],
                unseen_summary["accuracy"], f"{unseen_auroc:.3f}" if unseen_auroc is not None else "n/a",
                unseen_summary["n"],
            )
            unseen_roc_path = run_dir / "roc_curve_unseen.png"
            unseen_fpr, unseen_tpr = roc_curve_points(unseen_labels, -unseen_final_scores)
            plot_roc_curve(unseen_fpr, unseen_tpr, unseen_auroc, unseen_roc_path)
            logging.info("Saved zero-shot unseen-task ROC curve to %s", unseen_roc_path)

            onnx_path = run_dir / "probe.onnx"
            try:
                export_probe_onnx(model, args.device, onnx_path)
                logging.info("Exported ONNX model to %s", onnx_path)
            except Exception:
                logging.exception("ONNX export failed -- skipping (training results are unaffected)")
                onnx_path = None

            test_overlay_path = run_dir / "test_scores_overlay.png"
            plot_split_score_overlay(
                model, test_rows, args.features_dir, args, test_overlay_path,
                title=f"test split -- {test_summary['n'] - failures['test']} success / "
                f"{failures['test']} failure (n={test_summary['n']})",
            )
            logging.info("Saved test-split score overlay to %s", test_overlay_path)

            unseen_overlay_path = run_dir / "unseen_scores_overlay.png"
            n_unseen_fail = sum(1 for r in unseen_rows if r["success"] != "True")
            plot_split_score_overlay(
                model, unseen_rows, args.features_dir, args, unseen_overlay_path,
                title=f"unseen split -- {len(unseen_rows) - n_unseen_fail} success / "
                f"{n_unseen_fail} failure (n={len(unseen_rows)})",
            )
            logging.info("Saved unseen-split score overlay to %s", unseen_overlay_path)

            if not args.no_wandb:
                wandb.summary["best_epoch"] = best_epoch
                wandb.summary["best_val_auroc"] = best_val_auroc
                wandb.summary["interrupted"] = interrupted

                wandb.summary["test/success_loss"] = test_summary["success_loss"]
                wandb.summary["test/fail_loss"] = test_summary["fail_loss"]
                wandb.summary["test/accuracy"] = test_summary["accuracy"]
                wandb.summary["test/auroc"] = test_summary["auroc"]
                wandb.summary["test/n"] = test_summary["n"]
                wandb.summary["test/confusion_matrix"] = cm

                wandb.summary["unseen/success_loss"] = unseen_summary["success_loss"]
                wandb.summary["unseen/fail_loss"] = unseen_summary["fail_loss"]
                wandb.summary["unseen/accuracy"] = unseen_summary["accuracy"]
                wandb.summary["unseen/auroc"] = unseen_summary["auroc"]
                wandb.summary["unseen/n"] = unseen_summary["n"]
                wandb.summary["unseen/confusion_matrix"] = unseen_cm

                wandb.log({"test/roc_curve": wandb.Image(str(roc_path))})
                wandb.log({"unseen/roc_curve": wandb.Image(str(unseen_roc_path))})
                wandb.log({"test/scores_overlay": wandb.Image(str(test_overlay_path))})
                wandb.log({"unseen/scores_overlay": wandb.Image(str(unseen_overlay_path))})

                best_artifact = wandb.Artifact(
                    "probe_time_dependent_best", type="model",
                    metadata={
                        "epoch": best_epoch, "val_auroc": best_val_auroc,
                        "test_metrics": test_summary, "unseen_metrics": unseen_summary,
                    },
                )
                best_artifact.add_file(str(run_dir / "probe_best.pt"))
                if onnx_path is not None:
                    best_artifact.add_file(str(onnx_path))
                wandb.log_artifact(best_artifact)

                last_artifact = wandb.Artifact(
                    "probe_time_dependent_last", type="model",
                    metadata={
                        "epoch": len(history) - 1,
                        "val_auroc": history[-1]["val"]["auroc"] if history else None,
                        "interrupted": interrupted,
                    },
                )
                last_artifact.add_file(str(run_dir / "probe_last.pt"))
                wandb.log_artifact(last_artifact)
    finally:
        if not args.no_wandb:
            wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
