#!/usr/bin/env python3
"""Time-dependent training loop for the perception-uncertainty probe, matching
SAFE's IndepModel loss and score-accumulation design as closely as our feature
representation allows (../13-SAFE/failure_prob/model/indep.py, defaults from
../13-SAFE/failure_prob/conf/__init__.py's IndepModelConfig/ModelConfig).

What's the same as SAFE:
  - The per-timestep "projector" score is passed through sigmoid, then
    accumulated over time with a plain (gradient-flowing) cumsum -- so the
    score a rollout carries at timestep t is the running sum of every
    per-step score up to and including t, not just the current step's score.
  - The loss (see `time_dependent_mlp_loss` below) is copied verbatim from
    IndepModel.forward_compute_loss's default configuration: a one-sided
    hinge `relu(score)` on success sequences (push the accumulated score down
    to <= 0), an unbounded `-score` term on failure sequences (keep pushing
    the accumulated score up, optionally front-loaded by `--use-time-
    weighting`), aggregated per-sequence (mean over valid timesteps) and then
    combined with inverse-class-frequency weights exactly like SAFE's
    RolloutDataset.get_class_weights().
  - L2 weight regularization (`--lambda-reg`), scaled the same way
    (compute_regularization_loss: sum of squared weight -- not bias --
    params, times lambda_reg).
  - Optimizer/scheduler shape (Adam by default, StepLR with optional linear
    warmup) from BaseModel.get_optimizer().

What's deliberately different, and why:
  - SAFE's "projector" is a small standalone MLP on a single pooled hidden-
    state vector per timestep. Our features are per-TOKEN (256 image tokens
    x 2048-dim per camera, 200 language tokens x 2048-dim, per timestep) --
    there is no single vector to project. PerceptionSuccessProbe (unchanged,
    imported from probe_model.py) already plays exactly the projector's role
    (reduce raw per-token features -> one scalar per timestep via learned
    attention pooling + an MLP head), so it's reused as-is instead of adding
    a second MLP on top of it.
  - SAFE's RolloutDataset pads every rollout in a split to that split's
    single max length ONCE, up front, and keeps the whole padded tensor
    resident (feasible because their per-timestep feature is one small
    vector). Our per-timestep tokens are a few hundred times larger, so
    materializing an entire split that way risks the OOM problems already
    hit once in train_probe.py. This script pads PER BATCH instead, via
    EpisodeSequenceDataset + collate_pad_episodes below, lazily
    decompressing one full episode's .npz per __getitem__ call (still just
    one decompression per episode per epoch -- the natural granularity here,
    since each dataset item already IS one whole episode, not a single step
    like train_probe.py's EpisodeChunkDataset had to work around).
  - Padded timesteps are filled by repeating the last valid timestep (edge
    padding), not zeros -- an all-zero language_mask row would make
    AttentionPool's softmax divide by nothing and emit NaN even though those
    steps are excluded from the loss (NaN * 0 == NaN, so a masked-out step
    can still poison a batch's gradient if its raw score isn't finite).
  - --batch-size defaults far below SAFE's (64 for their MLP sweeps): a
    batch of whole per-token-feature episodes is much heavier than a batch
    of pooled vectors. See the memory note on --batch-size below.
  - --epochs defaults to 15 (matching train_probe.py), not SAFE's
    n_epochs=1000 default -- tune with --epochs if you want to match that.

Scope: only --suite's episodes (default libero_10_occluded, both scene
variants) are used at all. Within that suite, --num-unseen-tasks tasks
(default 3, chosen randomly and reproducibly via --unseen-task-seed, or
--seed if that's not given -- --unseen-tasks can force specific tasks into
the set instead/as well) are held out entirely -- never appear in
train/val/test/calibration, not even in the class-weight computation -- and
only get a final zero-shot evaluation pass on the trained checkpoint. This
mirrors 13-SAFE/failure_prob/data/utils.py's split_rollouts_by_seen_unseen
exactly. With the defaults that's 7 seen tasks x 2 scene variants x 25
episodes = 350 episodes for train/val/test/calibration, and 3 unseen tasks x
2 variants x 25 = 150 episodes for zero-shot eval only. Which 3 tasks varies
by seed; the actual chosen set is always logged and saved to split.json.

Run with the top-level project venv (plain PyTorch, no JAX/openpi needed):
    .venv/bin/python scripts/perception_probe/train_probe_time_dependent.py
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/perception_probe (sibling modules)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # clip_align_v2 -- our patched probe_model wins
from collect_features import build_task_suite_map  # noqa: E402
from probe_model import FEATURE_DIM, PerceptionSuccessProbe  # noqa: E402
from rollout_unseen_with_scores import plot_overlay  # noqa: E402
from train_probe import (  # noqa: E402
    auroc,
    available_memory_gb,
    confusion_matrix,
    plot_roc_curve,
    read_manifest,
    roc_curve_points,
    split_episodes,
)

NUM_IMAGE_TOKENS = 256
NUM_LANGUAGE_TOKENS = 200
REPLAN_STEPS = 5  # matches collect_features.py's --replan-steps default: each inference call -> 5 control frames



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """argv defaults to sys.argv[1:] (via parser.parse_args(None)); pass an explicit list to
    re-parse a saved invocation (see run_joint_threshold_sweep.py's command.txt-based
    architecture recovery)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "probe_time_dependent"
    )
    parser.add_argument(
        "--suite", default="libero_10_occluded",
        help="Only episodes whose task belongs to this occluded-suite name are used (both scene "
        "variants -- occluded AND normal -- of every task in it, since scene_variant is a "
        "separate axis from suite). Task-to-suite membership is derived from the LIBERO BDDL "
        "files the same way collect_features.py does, since manifest.csv itself has no suite column.",
    )
    parser.add_argument(
        "--num-unseen-tasks", type=int, default=3,
        help="How many of --suite's tasks to hold out entirely from train/val/test/calibration, "
        "chosen randomly (seeded by --unseen-task-seed) from the tasks NOT already named by "
        "--unseen-tasks. Ignored for any task explicitly named via --unseen-tasks.",
    )
    parser.add_argument(
        "--unseen-task-seed", type=int, default=None,
        help="Seed for the random selection of unseen tasks (see --num-unseen-tasks). Defaults to "
        "--seed if not given, so the whole run is reproducible off one seed by default, but can "
        "be varied independently (e.g. to try several random unseen-task splits at a fixed "
        "--seed for the train/val/test/calibration partition).",
    )
    parser.add_argument(
        "--unseen-tasks", nargs="+", default=None,
        help="Task stems (as they appear in manifest.csv's 'task' column) to force into the "
        "unseen set regardless of random selection. Held out entirely from "
        "train/val/test/calibration -- only used for a final zero-shot evaluation pass on the "
        "trained checkpoint. If this covers fewer than --num-unseen-tasks tasks, the remainder "
        "is filled in randomly (see --unseen-task-seed); if omitted entirely, all "
        "--num-unseen-tasks are chosen randomly.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=15, help="SAFE's ModelConfig default is 1000.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Episodes per batch (NOT steps -- each item is a whole variable-length episode, "
        "padded to the batch's max length). Memory is roughly "
        "batch_size x max_episode_len x ~2.8MB/timestep (256+256+200 fp16 tokens x 2048-dim) "
        "for the padded batch tensor alone, plus autograd's saved activations through "
        "PerceptionSuccessProbe applied batch_size x max_len times. SAFE's own MLP sweeps use "
        "batch_size=64, but that's over pooled single-vector-per-timestep features, not "
        "per-token ones -- 64 here would likely OOM. Raise cautiously.",
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
        help="Fixed (non-trained) random projection applied to every token before pooling -- "
        "shrinks the head's parameter count. None = off (full 2048-dim).",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None,
        help="Insert an embedding layer: pool(6144) -> Linear(embed_dim) -> GELU -> Dropout = z, "
        "then z -> Linear(hidden_dim) -> ... -> 1. z is the target for CLIP text alignment "
        "(set 512 for CLIP ViT-B). None = original single-stage head.",
    )
    parser.add_argument(
        "--embed-hidden", type=int, default=None,
        help="Compression bottleneck before the embed layer: head_in -> embed_hidden -> embed_dim. "
        "A bare head_in -> 512 doubles the head params and memorises the train set; "
        "--embed-hidden 256 keeps capacity ~= the default head.",
    )
    parser.add_argument(
        "--align-weight", type=float, default=0.0,
        help="If >0, add align_weight * InfoNCE(z, CLIP_text[task]) -- pulls the --embed-dim "
        "latent z toward the CLIP text embedding of the episode's task instruction. Needs --embed-dim.",
    )
    parser.add_argument("--align-temp", type=float, default=0.07, help="InfoNCE temperature for --align-weight.")
    parser.add_argument("--align-center-anchors", action="store_true",
                        help="v2 edit 1: mean-center the CLIP anchors (remove the shared 'manipulation "
                        "instruction' direction) before the InfoNCE / retrieval -- off-diag cos 0.78 -> -0.11.")
    parser.add_argument("--align-cos-weight", type=float, default=0.0,
                        help="v2 edit 2: extra term align_cos_weight * mean(1 - cos(z, c_true)) so z lands "
                        "ON its anchor, not just ranks it first.")
    parser.add_argument("--decouple-z", action="store_true",
                        help="v2 edit 3: z is a parallel projection off `pooled`; the detection head is the "
                        "original single-stage head, untouched by align_weight.")
    parser.add_argument("--reseed-after-build", action="store_true",
                        help="reset torch.manual_seed(seed) after build_model so decoupled and plain "
                        "runs get identical dropout streams -- isolates arch effect from RNG-shift noise.")
    parser.add_argument("--z-stop-grad", action="store_true",
                        help="v2 edit 3b: with --decouple-z, feed pooled.detach() into the z head so the "
                        "align loss trains ONLY z_head -- pools + detection head fully insulated.")
    parser.add_argument(
        "--clip-embeddings", type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "clip" / "task_instruction_embeddings.npy",
        help="(N_tasks, D) L2-normalised CLIP text embeddings; task_index.json alongside maps task->row.",
    )
    parser.add_argument("--patience", type=int, default=None, help="Early stop after N epochs with no val-AUROC improvement.")
    parser.add_argument(
        "--max-steps", type=int, default=80,
        help="Clip every episode (train AND eval loaders) to this many inference calls at load "
        "time (0 = no cap). Trims the low-signal tail of long failure episodes and is the "
        "biggest single lever on DataLoader shared-memory and GPU footprint. Eval metrics are "
        "unaffected -- they truncate to each task's min observed length (<=70) anyway. The "
        "score-overlay plots always use the full, uncapped trajectory.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch per worker.")
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable bfloat16 autocast for the probe forward (kept fp32). AMP roughly halves "
        "GPU memory and speeds up the per-token matmuls on the 4090.",
    )

    # --- Loss / model hyperparameters, named and defaulted to match
    # IndepModelConfig / ModelConfig in 13-SAFE/failure_prob/conf/__init__.py ---
    parser.add_argument(
        "--cumsum", dest="cumsum", action="store_true", default=True,
        help="Accumulate per-step sigmoid scores over time via cumsum (IndepModelConfig default: True).",
    )
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    parser.add_argument(
        "--rmean", action="store_true", default=False,
        help="If set, divide the cumsum by elapsed timesteps (running mean instead of running "
        "sum). IndepModelConfig default: False.",
    )
    parser.add_argument(
        "--use-time-weighting", action="store_true", default=False,
        help="Front-load the failure-sequence loss on early timesteps (ModelConfig default: False "
        "-- not enabled in any of SAFE's own batch_training submit scripts either).",
    )
    parser.add_argument(
        "--use-threshold", action="store_true", default=False,
        help="If set, the failure-sequence loss becomes relu(threshold - score) (a margin hinge) "
        "instead of the unbounded -score term. IndepModelConfig default: False. Ignored if "
        "--raw-target-loss is set.",
    )
    parser.add_argument("--threshold", type=float, default=50.0, help="Only used if --use-threshold.")
    parser.add_argument(
        "--raw-target-loss", action="store_true", default=False,
        help="Use time_dependent_raw_target_loss instead of the SAFE-ported hinge loss: a symmetric "
        "per-timestep regression pushing the RAW (pre-accumulation) sigmoid score toward 1 at every "
        "timestep of a failure episode and toward 0 at every timestep of a success episode, summed "
        "(not averaged) over each episode's valid timesteps. --cumsum/--rmean still control the "
        "SEPARATE accumulated score used for eval-time AUROC (see --task-min-step-eval); they no "
        "longer affect what the loss itself is computed on.",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=0.0,
        help="Focal-loss exponent on the raw-target BCE (0 = plain BCE). Down-weights already-"
        "confident timesteps so training focuses on ambiguous frames. Only used with --raw-target-loss.",
    )
    parser.add_argument(
        "--mil-pool", choices=["max", "lse", "topk"], default=None,
        help="Multiple-instance-learning loss: pool per-step logits to one episode logit "
        "(max / log-mean-exp / top-k mean), then BCE against the episode label -- drops the "
        "assumption that every failure-episode timestep is itself a failure frame. Overrides "
        "--raw-target-loss / the hinge loss when set.",
    )
    parser.add_argument("--mil-topk", type=int, default=8, help="k for --mil-pool topk.")
    parser.add_argument(
        "--ranking-weight", type=float, default=0.0,
        help="If >0, add ranking_weight * pairwise_ranking_loss (softplus AUROC surrogate on the "
        "final accumulated episode score) on top of the primary loss.",
    )
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--lambda-success", type=float, default=1.0)
    parser.add_argument("--lambda-fail", type=float, default=1.0)
    parser.add_argument(
        "--lambda-reg", type=float, default=1e-2,
        help="L2 weight-decay-style regularization coefficient on the loss (compute_regularization_loss). "
        "SAFE sweeps this over {1e-3, 1e-2, 1e-1, 1} for the MLP baseline.",
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
        help="'step' = StepLR(--lr-step-size epochs, --lr-gamma); 'cosine' = cosine anneal to 0 "
        "over --epochs. Late-epoch LR decay counters the fast-convergence-then-overfit pattern.",
    )
    parser.add_argument(
        "--select-by", choices=["auroc", "separation"], default="auroc",
        help="Which val metric picks probe_best.pt (and drives --patience).",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Soften the raw-target BCE targets: failure -> 1-eps, success -> eps. Curbs the "
        "overconfidence that comes with overfitting. Only used with --raw-target-loss.",
    )

    parser.add_argument(
        "--eval-threshold", type=float, default=0.0,
        help="Decision boundary on the (task-min-step-truncated) accumulated score: the hinge loss "
        "trains success scores towards <= 0 and failure scores upward from there, so 0 is the "
        "natural boundary (predict failure iff score > threshold) -- NOT a probability threshold "
        "like train_probe.py's --eval-threshold=0.5. With --raw-target-loss the accumulated score "
        "is still whatever --cumsum/--rmean produce, so this threshold still needs separate tuning "
        "for that scale.",
    )
    parser.add_argument(
        "--no-task-min-step-eval", dest="task_min_step_eval", action="store_false", default=True,
        help="By default, AUROC/accuracy/confusion-matrix on val/test/unseen are computed from each "
        "episode's accumulated score truncated to its TASK's shortest observed rollout length "
        "(min inference_calls across every episode of that task in --features-dir/manifest.csv), "
        "matching 13-SAFE/failure_prob/utils/metrics.py's task_min_step truncation -- this keeps "
        "episode length (which LIBERO ties directly to success/failure: episodes stop the instant "
        "they succeed, but always run to the suite's step cap on failure) from leaking into the "
        "eval metric. Pass this flag to instead score each episode at its own full length (the old "
        "behavior) -- NOT recommended, kept only for comparison.",
    )
    parser.add_argument("--wandb-project", default="pi05-perception-probe")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


class EpisodeSequenceDataset(Dataset):
    """One item == one whole episode's full (T, ...) feature sequence, lazily
    decompressed from its .npz on access. Unlike train_probe.py's
    EpisodeChunkDataset, there's no need for a chunked shuffle-buffer here --
    each episode is already the unit of iteration (not exploded into T
    separate steps), so plain map-style random access decompresses every
    episode's .npz exactly once per epoch, same as the chunked version did,
    without needing the extra bookkeeping."""

    def __init__(self, rows: list[dict], features_dir: Path, max_steps: int | None = None):
        self.rows = rows
        self.features_dir = features_dir
        # Clip each episode to its first `max_steps` timesteps at load time (not
        # in the training loop) so a padded batch is never larger than it needs
        # to be -- the single biggest lever on both DataLoader shared-memory use
        # and GPU footprint. Safe for eval too: task_min_step (the eval cutoff)
        # is <= 70 for every task here, well under any sane cap. Pass None for
        # the score-overlay plots, which want the full trajectory.
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        # Kept as float16 (their on-disk dtype) all the way to the GPU -- a padded
        # batch of these per-token tensors is multi-GB, and float32 here would
        # double every DataLoader worker's shared-memory footprint (16GB /dev/shm
        # is not enough for that with several workers). score_sequence() casts to
        # float32 on-device right before the probe.
        cap = self.max_steps
        with np.load(self.features_dir / row["npz_path"]) as data:
            base_image = data["base_image"][:cap]
            wrist_image = data["wrist_image"][:cap]
            language = data["language"][:cap]
            language_mask = data["language_mask"][:cap]
        return {
            "base_image": torch.from_numpy(base_image),
            "wrist_image": torch.from_numpy(wrist_image),
            "language": torch.from_numpy(language),
            "language_mask": torch.from_numpy(language_mask),
            "length": base_image.shape[0],
            "label": float(row["success"] == "True"),
            "task": row["task"],
        }


def _edge_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pads x (T, ...) up to target_len along dim 0 by repeating the last
    timestep -- NOT zero padding. A zero-padded language_mask row would be
    all-False, and AttentionPool's masked_fill(-inf) + softmax over an
    all-(-inf) row emits NaN; repeating the last real (valid) timestep keeps
    every padded step numerically finite. Padded steps never influence the
    loss (they're excluded via valid_masks), but NaN * 0 == NaN, so an
    unmasked NaN score would silently poison the whole batch's gradient."""
    pad_len = target_len - x.shape[0]
    if pad_len == 0:
        return x
    last = x[-1:].expand(pad_len, *x.shape[1:])
    return torch.cat([x, last], dim=0)


def collate_pad_episodes(batch: list[dict]) -> dict:
    max_len = max(item["length"] for item in batch)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    valid_masks = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T)
    return {
        "base_image": torch.stack([_edge_pad(item["base_image"], max_len) for item in batch]),
        "wrist_image": torch.stack([_edge_pad(item["wrist_image"], max_len) for item in batch]),
        "language": torch.stack([_edge_pad(item["language"], max_len) for item in batch]),
        "language_mask": torch.stack([_edge_pad(item["language_mask"], max_len) for item in batch]),
        "valid_masks": valid_masks,
        "lengths": lengths,
        "success_labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "tasks": [item["task"] for item in batch],  # plain list of strings -- not a tensor
    }


def score_sequence(
    model: nn.Module, batch: dict, cumsum: bool, rmean: bool, amp: bool = True,
    chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs PerceptionSuccessProbe independently at every (padded) timestep,
    then sigmoids and optionally accumulates over time -- mirrors IndepModel
    .forward with final_act_layer="sigmoid" (IndepModelConfig's default).
    Returns (raw_scores, accumulated_scores), both (B, T). raw_scores is the
    per-timestep sigmoid output before any accumulation -- used by
    --raw-target-loss. accumulated_scores applies cumsum/rmean on top --
    used by the SAFE-ported hinge loss and always used for eval-time AUROC
    (truncated to each task's shortest observed rollout length; see
    --no-task-min-step-eval)."""
    # inputs stay bfloat16 (not float32) -- a padded batch of per-token features is
    # multi-GB and the fp32 copy was the bulk of the probe's GPU footprint. The
    # matmul-heavy pooling runs under bf16 autocast; LayerNorm / softmax / the
    # sigmoid below are auto-promoted back to fp32 by autocast.
    base, wrist, lang, lang_mask = (
        batch["base_image"], batch["wrist_image"], batch["language"], batch["language_mask"],
    )
    B, T = base.shape[0], base.shape[1]
    bf = base.reshape(B * T, *base.shape[2:])
    wf = wrist.reshape(B * T, *wrist.shape[2:])
    lf = lang.reshape(B * T, *lang.shape[2:])
    mf = lang_mask.reshape(B * T, *lang_mask.shape[2:])
    step = chunk if chunk else B * T  # chunk the flattened forward to cap peak
    # cast to bf16 per chunk -- a bf16 copy of the whole padded batch is multi-GB
    # and OOMs when the pi0.5 server is holding most of the card.
    with torch.autocast(device_type=base.device.type, dtype=torch.bfloat16, enabled=amp):
        logits = torch.cat([
            model(bf[i:i + step].to(torch.bfloat16), wf[i:i + step].to(torch.bfloat16),
                  lf[i:i + step].to(torch.bfloat16), mf[i:i + step])
            for i in range(0, B * T, step)
        ], dim=0)  # (B*T,)
    step_logits = logits.float().view(B, T)  # (B, T) pre-sigmoid, for the MIL loss
    raw_scores = torch.sigmoid(step_logits)  # (B, T), in [0, 1] per timestep

    scores = raw_scores
    if cumsum or rmean:
        scores = torch.cumsum(scores, dim=1)  # (B, T), running sum -- gradient flows normally
        if rmean:
            scores = scores / torch.arange(1, T + 1, device=scores.device, dtype=scores.dtype).unsqueeze(0)

    return step_logits, raw_scores, scores


def get_time_weight(use_weighting: bool, valid_masks: torch.Tensor) -> torch.Tensor:
    """Verbatim port of 13-SAFE/failure_prob/model/utils.py's get_time_weight."""
    B, T = valid_masks.shape
    seq_lengths = valid_masks.sum(dim=1)  # (B,)
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
    scores: torch.Tensor,  # (B, T) accumulated scores, from score_sequence()
    valid_masks: torch.Tensor,  # (B, T)
    success_labels: torch.Tensor,  # (B,) 1 = success, 0 = failure
    class_weights: tuple[float, float],  # (fail_weight, success_weight)
    use_time_weighting: bool = False,
    use_threshold: bool = False,
    threshold: float = 50.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Verbatim port of IndepModel.forward_compute_loss's loss computation
    (13-SAFE/failure_prob/model/indep.py), operating on scores already
    produced by score_sequence() instead of on a projector MLP's own output --
    see the module docstring for why that split is where our implementation
    diverges from SAFE's, and why the loss itself does not."""
    B = scores.shape[0]
    time_weights = get_time_weight(use_time_weighting, valid_masks).to(scores)

    loss_success = torch.relu(scores)  # success: push accumulated score <= 0
    if use_threshold:
        loss_fail = time_weights * torch.relu(threshold - scores)  # failure: margin hinge
    else:
        loss_fail = time_weights * (-scores)  # failure: push accumulated score up, unbounded

    is_success = (success_labels == 1).float().unsqueeze(1)  # (B, 1)
    losses = is_success * loss_success + (1.0 - is_success) * loss_fail  # (B, T)

    seq_loss = (losses * valid_masks).sum(-1) / valid_masks.sum(-1)  # (B,) mean over valid timesteps

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
    raw_scores: torch.Tensor,  # (B, T) RAW per-timestep sigmoid scores, from score_sequence()
    valid_masks: torch.Tensor,  # (B, T)
    success_labels: torch.Tensor,  # (B,) 1 = success, 0 = failure
    class_weights: tuple[float, float],  # (fail_weight, success_weight)
    focal_gamma: float = 0.0,  # >0 -> focal loss: down-weight already-confident timesteps
    label_smoothing: float = 0.0,  # soften targets to 1-eps / eps
) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric per-timestep target-classification loss: pushes the RAW
    (pre-accumulation) sigmoid score toward 1 at every timestep of a failure
    episode and toward 0 at every timestep of a success episode -- i.e.
    y_i*(-log(s_t)) + (1 - y_i)*(-log(1 - s_t)) with y_i = 1 for failure
    (note the sign flip from success_labels, which use 1 = success). This is
    plain per-timestep binary cross-entropy toward that same 0/1 target.

    An earlier version of this loss used a LINEAR penalty (y_i*(1 - s_t) +
    (1 - y_i)*s_t) instead of this log one -- mathematically that's
    gradient-equivalent (for theta) to the literal y_i*(t - s_t) + (1-y_i)*
    s_t formulation with t the timestep index, since t doesn't depend on any
    model parameter. But d(loss)/d(logit_t) = d(loss)/d(s_t) * s_t*(1-s_t)
    (the sigmoid's own derivative), and a LINEAR loss has constant d(loss)/
    d(s_t) -- so once s_t saturates near 0 or 1 for ANY reason, the s_t*(1-
    s_t) factor collapses toward 0 and the gradient vanishes even when the
    prediction is flat-out wrong (e.g. s_t~1 on a success timestep, target
    0). Confirmed empirically: a smoke test with the linear version got
    stuck with bit-identical val metrics across all 3 epochs. BCE's -log(.)
    curvature is specifically shaped to cancel that factor:
    d(-log(s_t))/d(logit_t) = s_t - 1 and d(-log(1-s_t))/d(logit_t) = s_t --
    both reduce to (s_t - target), which only vanishes once the prediction
    is actually CORRECT, not merely saturated.

    Aggregated per-episode via SUM (not mean) over valid timesteps -- the
    "cumsum version" of this loss, as opposed to a "rmean version" that
    would instead average over valid timesteps (left as a follow-up: swap
    the .sum(-1) below for the same mean-over-valid-timesteps normalization
    time_dependent_mlp_loss uses, if this turns out to over-weight longer
    episodes' gradients within a batch).

    Both terms are non-negative (BCE's -log(.) of a [0,1] input), so
    success_loss and fail_loss can't partially cancel if ever combined into
    one number -- see run_eval_pass's docstring for why the hinge loss's
    combined value can."""
    is_success = (success_labels == 1).float().unsqueeze(1)  # (B, 1)

    eps = 1e-7
    clamped = raw_scores.clamp(eps, 1.0 - eps)
    if label_smoothing > 0:
        # target for a failure timestep is (1 - ls), for a success it is ls
        ls = label_smoothing
        loss_fail = -((1 - ls) * torch.log(clamped) + ls * torch.log(1.0 - clamped))
        loss_success = -((1 - ls) * torch.log(1.0 - clamped) + ls * torch.log(clamped))
    else:
        loss_success = -torch.log(1.0 - clamped)  # -> 0 as s_t -> 0
        loss_fail = -torch.log(clamped)  # -> 0 as s_t -> 1
    if focal_gamma > 0:
        # Focal modulation (Lin et al. 2017): scale each timestep's BCE by
        # (1 - p_correct)^gamma so easy, already-confident frames contribute
        # little gradient and the optimizer concentrates on the ambiguous ones
        # (early failure frames that still look like a success, etc.).
        loss_success = (clamped**focal_gamma) * loss_success  # p_correct = 1 - s_t
        loss_fail = ((1.0 - clamped) ** focal_gamma) * loss_fail  # p_correct = s_t
    losses = is_success * loss_success + (1.0 - is_success) * loss_fail  # (B, T)

    seq_loss = (losses * valid_masks).sum(-1)  # (B,) SUM over valid timesteps, not mean

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
    step_logits: torch.Tensor,  # (B, T) per-timestep RAW logits, from score_sequence()
    valid_masks: torch.Tensor,  # (B, T)
    success_labels: torch.Tensor,  # (B,) 1 = success, 0 = failure
    class_weights: tuple[float, float],  # (fail_weight, success_weight)
    pool: str = "lse",  # {"max", "lse", "topk"}
    topk: int = 8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multiple-instance-learning loss: pool the per-timestep logits into ONE
    episode logit, then BCE that against the episode label. This drops the
    (noisy) assumption that every timestep of a failure episode is itself a
    failure frame -- a failure bag only needs one failure frame; a success bag
    needs none -- so the probe is free to fire on the few frames that actually
    show the occlusion's consequence instead of being forced to call "failure"
    on ambiguous early frames.

      max   : hardest frame wins (classic MIL). Sparse gradient.
      lse   : log-sum-exp, length-normalised (log mean exp) -> a soft, smooth
              max; every frame gets some gradient, the confident ones more.
      topk  : mean of the k highest-logit valid frames -> between max and mean.
    """
    neg_inf = torch.finfo(step_logits.dtype).min
    masked = step_logits.masked_fill(~valid_masks.bool(), neg_inf)
    lengths = valid_masks.sum(dim=1).clamp(min=1)

    if pool == "max":
        bag_logit = masked.max(dim=1).values
    elif pool == "lse":
        bag_logit = torch.logsumexp(masked, dim=1) - torch.log(lengths)  # log-mean-exp
    elif pool == "topk":
        k = min(topk, step_logits.shape[1])
        bag_logit = masked.topk(k, dim=1).values
        bag_logit = torch.where(bag_logit <= neg_inf / 2, torch.zeros_like(bag_logit), bag_logit)
        bag_logit = bag_logit.sum(dim=1) / valid_masks.sum(dim=1).clamp(min=1).clamp(max=k)
    else:
        raise ValueError(f"unknown MIL pool {pool!r}")

    target = (success_labels == 0).float()  # 1 = failure
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
    episode_scores: torch.Tensor,  # (B,) one accumulated score per episode
    success_labels: torch.Tensor,  # (B,) 1 = success, 0 = failure
    margin: float = 1.0,
) -> torch.Tensor:
    """softplus(s_success - s_failure + margin) averaged over every
    failure/success pair in the batch -- a smooth AUROC surrogate: it only
    cares that failures outrank successes, not about calibrated values.
    Returns 0 when the batch has no failure or no success episode."""
    fail = episode_scores[success_labels == 0]
    succ = episode_scores[success_labels == 1]
    if fail.numel() == 0 or succ.numel() == 0:
        return episode_scores.sum() * 0.0
    diff = succ[None, :] - fail[:, None] + margin  # (n_fail, n_succ)
    return nn.functional.softplus(diff).mean()


def episode_embedding(model: nn.Module, batch: dict, amp: bool, max_steps_per_ep: int = 12) -> torch.Tensor:
    """Mean-pool the probe's --embed-dim latent z over each episode's valid timesteps -> (B, embed_dim).

    Only a subsample of up to `max_steps_per_ep` evenly-spaced valid timesteps per
    episode is pushed through the model. The align loss just needs an episode-level
    z, and a second full B*T forward here OOMs when the pi0.5 server holds most of
    the GPU; gradients still flow through the subsampled steps.
    """
    cast = torch.bfloat16 if amp else torch.float32
    valid = batch["valid_masks"]  # (B, T)
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
    z = z.float()  # (N, D)
    out = torch.zeros(B, z.shape[-1], device=device, dtype=z.dtype).index_add(0, idx_b, z)
    cnt = torch.zeros(B, 1, device=device, dtype=z.dtype).index_add(
        0, idx_b, torch.ones(idx_b.shape[0], 1, device=device, dtype=z.dtype)
    )
    return out / cnt.clamp(min=1.0)


def clip_alignment_loss(z_ep, task_idx, clip_embeds, temp, cos_weight: float = 0.0):
    """InfoNCE (+ optional direct-cosine term): normalised z_ep closest to its own
    task's CLIP embedding. Returns (loss, retrieval_acc)."""
    zc = torch.nn.functional.normalize(z_ep, dim=-1)
    sim = zc @ clip_embeds.T / temp
    loss = torch.nn.functional.cross_entropy(sim, task_idx)
    acc = (sim.argmax(dim=-1) == task_idx).float().mean().item()
    if cos_weight > 0:
        c_true = clip_embeds[task_idx]  # (B, D), already unit-norm
        loss = loss + cos_weight * (1.0 - (zc * c_true).sum(dim=-1)).mean()
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
    adds a pairwise-ranking auxiliary term (--ranking-weight). Used by both the
    training loop and the eval pass so their reported numbers line up."""
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
    """Verbatim port of BaseModel.compute_regularization_loss: L2 over every
    weight (not bias) parameter, scaled by lambda_reg."""
    if lambda_reg == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg_loss = sum(
        torch.sum(param**2) for name, param in model.named_parameters() if "bias" not in name
    )
    return lambda_reg * reg_loss


def compute_class_weights(rows: list[dict], lambda_fail: float, lambda_success: float) -> tuple[float, float]:
    """Verbatim port of RolloutDataset.__init__'s class-weight computation
    (Laplace-smoothed inverse class frequency), returned as (fail_weight,
    success_weight)."""
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
    """Gathers each episode's score at its OWN last valid timestep (index
    length-1), not column -1 of the padded tensor -- edge-padding means a
    cumsum over the repeated last frame keeps climbing past the real episode
    end, so column -1 would overstate the terminal score for every episode
    shorter than the batch max. NOT length-safe on its own -- see
    truncated_episode_scores below for why LIBERO episode length leaks into
    this directly (failures always run to the suite's step cap; successes
    stop the instant they succeed)."""
    idx = (lengths - 1).clamp(min=0).view(-1, 1)
    return scores.gather(1, idx).squeeze(1)


def compute_task_min_steps(rows: list[dict]) -> dict[str, int]:
    """Per-task minimum observed episode length (in inference calls), over
    EVERY episode of that task in `rows` (seen and unseen both -- compute
    this from the full --suite row set, before any train/val/test/
    calibration/unseen split), matching 13-SAFE/failure_prob/data/utils.py's
    set_task_min_step. manifest.csv's "inference_calls" column already IS
    each episode's T (written at collection time as base_image.shape[0]),
    so this is a single pass over the CSV rows, no .npz re-reading needed."""
    mins: dict[str, int] = {}
    for row in rows:
        length = int(row["inference_calls"])
        task = row["task"]
        mins[task] = min(mins[task], length) if task in mins else length
    return mins


def truncated_episode_scores(
    scores: torch.Tensor, lengths: torch.Tensor, tasks: list[str], task_min_step: dict[str, int]
) -> torch.Tensor:
    """Gathers each episode's score at index (task_min_step[task] - 1) --
    the SAME, fixed cutoff for every episode of a given task, regardless of
    that episode's own (possibly much longer) actual length. Matches SAFE's
    eval-time task_min_step truncation (utils/metrics.py's
    compute_roc_by_quantiles at q=1.0): every rollout of a task is scored
    only up to the shortest rollout ever observed for that task, so a
    failure that happened to run long (LIBERO runs every failure to the
    suite's step cap) can't be told apart from a success purely by having
    had more timesteps to accumulate evidence in. Every episode of a task is
    guaranteed length >= task_min_step[task] by construction, so the index
    is always valid (the clamp is a safety net, not expected to bind)."""
    idx = torch.tensor(
        [task_min_step[task] - 1 for task in tasks], device=scores.device, dtype=torch.long
    )
    idx = idx.clamp(min=0, max=scores.shape[1] - 1).view(-1, 1)
    return scores.gather(1, idx).squeeze(1)


def export_probe_onnx(model: nn.Module, device: str, path: Path) -> None:
    """Exports PerceptionSuccessProbe to ONNX for inspection (e.g. Netron) --
    a plain single-timestep forward pass (features -> failure logit), NOT
    the time-dependent score accumulation (cumsum/rmean) wrapped around it
    in score_sequence(), since that's a training/eval-time loop construct,
    not part of the model graph."""
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
    length, no task-min-step truncation), for plot_overlay(). Mirrors
    rollout_unseen_with_scores.py's live per-step scoring, but reading
    cached .npz features instead of an active rollout."""
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
    """Runs compute_episode_trace_for_overlay over every episode in `rows`
    and writes one overlay plot (all episodes' score curves, group means
    bolded) -- see plot_overlay() in rollout_unseen_with_scores.py."""
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
    matching 13-SAFE/failure_prob/train.py, which never calls
    forward_compute_loss on val/test/unseen at all and judges the model
    purely by rank/threshold-based classification metrics (eval_scores_roc_prc,
    conformal-prediction coverage, etc. in failure_prob/utils/routines.py).
    Combining the success term (relu(score), >= 0) and the failure term
    (-score, <= 0 once scores are correctly positive) into one number via
    addition lets them partially cancel -- a model that's equally, badly
    wrong on both classes can read as "~0 loss" even though accuracy is poor.
    success_loss/fail_loss (each averaged only within its own class, never
    summed against the other) don't have that failure mode and are reported
    instead, alongside AUROC/accuracy."""
    model.eval()
    all_labels, all_final_scores = [], []
    total_success_loss, total_success_n = 0.0, 0
    total_fail_loss, total_fail_n = 0.0, 0
    class_weights = (1.0, 1.0)  # unweighted for reporting -- weighting only shapes the training gradient
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release the training epoch's reserved blocks before the eval forwards
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

    # Standardized gap between the failure-episode and success-episode score
    # distributions at the task-min-step cutoff -- a scale-free version of "make
    # the success/failure means as far apart as possible" (d'/Cohen's d style),
    # so it can't be gamed by just inflating the logit scale. This is the
    # primary quantity sweep.py ranks configs by.
    fail_scores = final_scores[labels == 0]
    success_scores = final_scores[labels == 1]
    if len(fail_scores) and len(success_scores) and final_scores.std() > 1e-9:
        separation = float((fail_scores.mean() - success_scores.mean()) / final_scores.std())
    else:
        separation = None

    preds = (final_scores > args.eval_threshold).astype(np.float32)  # predict FAILURE (0) iff score > threshold
    # preds==1 means "predicted failure" here since higher score => more failure evidence;
    # translate to success-label space (1 = success) for the accuracy comparison below.
    predicted_success = 1.0 - preds
    return {
        "success_loss": total_success_loss / max(total_success_n, 1),
        "fail_loss": total_fail_loss / max(total_fail_n, 1),
        "accuracy": float((predicted_success == labels).mean()),
        "auroc": auroc(labels, -final_scores),  # rank by -score: higher score => more likely failure (label 0)
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
    """The one place args -> PerceptionSuccessProbe kwargs is spelled out -- pulled out of
    main() so probe_arch_recovery.py's command.txt-based reconstruction (re-parsing a saved
    invocation with parse_args(argv)) builds models the exact same way training did, instead
    of maintaining a second copy of this mapping. Caller does .to(device)."""
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
        decouple_z=args.decouple_z,
        z_stop_grad=args.z_stop_grad,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    # Keep DataLoader batch tensors small enough for the default sharing
    # strategy's /dev/shm budget (16GB, shared across concurrent runs): inputs
    # stay float16, episodes are clipped to --max-steps at load time, and
    # --prefetch-factor is low. The "file_system" strategy was tried and its
    # torch_shm_manager helper errors out ("Invalid argument") on this box.

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.info("Run outputs going to %s", run_dir)

    available_gb = available_memory_gb()
    if available_gb is not None:
        logging.info("Currently available system memory: %.1fGB", available_gb)

    all_rows = read_manifest(args.features_dir)
    task_suite_map = build_task_suite_map()
    suite_rows = [r for r in all_rows if task_suite_map.get(r["task"]) == args.suite]
    if not suite_rows:
        raise ValueError(
            f"No episodes found for --suite {args.suite!r} in {args.features_dir}/manifest.csv "
            f"(known suites: {sorted(set(task_suite_map.values()))})"
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
    candidates = sorted(set(suite_tasks) - forced_unseen)  # sorted first for a deterministic RNG draw order
    rng = np.random.default_rng(unseen_task_seed)
    randomly_chosen = list(rng.choice(candidates, size=remaining_needed, replace=False)) if remaining_needed else []
    unseen_task_set = forced_unseen | set(randomly_chosen)
    logging.info(
        "Unseen-task selection: seed=%d, %d forced (%s), %d randomly chosen (%s)",
        unseen_task_seed, len(forced_unseen), sorted(forced_unseen), len(randomly_chosen), sorted(randomly_chosen),
    )

    rows = [r for r in suite_rows if r["task"] not in unseen_task_set]  # "seen" -- used for train/val/test/calibration
    unseen_rows = [r for r in suite_rows if r["task"] in unseen_task_set]  # zero-shot eval only, never trained on
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
    if args.reseed_after_build:
        torch.manual_seed(args.seed)  # RNG-parity probe: undo the stream shift from building extra params (z_head)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Probe: %s  (%d trainable params)", model, n_params)
    optimizer, scheduler = get_optimizer(model, args)

    clip_embeds = None
    task_to_clip_idx = {}
    if args.align_weight > 0:
        if args.embed_dim is None:
            raise ValueError("--align-weight requires --embed-dim (nowhere to align without a z latent)")  # decouple-z also needs --embed-dim
        emb = np.load(args.clip_embeddings)
        task_to_clip_idx = json.loads((args.clip_embeddings.parent / "task_index.json").read_text())
        clip_embeds = torch.tensor(emb, dtype=torch.float32, device=args.device)
        clip_embeds = torch.nn.functional.normalize(clip_embeds, dim=-1)
        if args.align_center_anchors:
            clip_embeds = clip_embeds - clip_embeds.mean(dim=0, keepdim=True)
            clip_embeds = torch.nn.functional.normalize(clip_embeds, dim=-1)
        logging.info("CLIP alignment: %d task embeddings (dim %d), weight=%.3g temp=%.3g center=%s cos_w=%.3g decouple=%s",
                     clip_embeds.shape[0], clip_embeds.shape[1], args.align_weight, args.align_temp,
                     args.align_center_anchors, args.align_cos_weight, args.decouple_z)

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
    best_val_metric = -float("inf")  # value of args.select_by at the best epoch
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
                        al, al_acc = clip_alignment_loss(z_ep, tix, clip_embeds, args.align_temp, args.align_cos_weight)
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
                # Checkpoint selection: val AUROC by default (matches SAFE's eval philosophy),
                # or val separation with --select-by separation (val AUROC keeps rising during
                # seen-task overfit; separation can flatten sooner -- see sweeps/FINDINGS.md).
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

            # Computed in "failure score" space directly: label 1 = failure, and final_score is
            # already higher when there's more accumulated failure evidence, so tp/fp read
            # naturally as failure-detection counts with no sign-flipping needed.
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

            # Zero-shot evaluation on the held-out UNSEEN tasks -- same best checkpoint, same
            # loss/scoring code, but these episodes never appeared in train/val/test/calibration
            # at all (not even the class-weight computation). This is the number that actually
            # answers "does the probe generalize to a new task", not the in-distribution test AUROC.
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
