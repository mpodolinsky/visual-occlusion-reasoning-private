"""Data layer for the GR00T perception-probe fork.

The pi0.5 probe (scripts/perception_probe/) reads a manifest.csv written by
collect_features.py with columns ``npz_path`` / ``inference_calls`` /
``control_frames`` and per-episode ``episode_<NNN>.npz`` files.

GR00T rollouts are already on disk under ``outputs/groot/libero_10/`` in the
scripts/groot/manifest.py format (columns ``dir`` / ``n_policy`` / ``n_control``,
one ``rollout.npz`` per episode dir). ``rollout.npz`` already carries the four
arrays the probe dataset consumes -- ``base_image (n_policy, 64, 2048)``,
``wrist_image`` same, ``language (n_policy, 200, 2048)``, ``language_mask
(n_policy, 200)`` -- so only the manifest plumbing needs a shim.

``read_probe_manifest`` re-emits the GR00T rows in the exact dict shape the
forked trainer expects. ``EpisodeSequenceDataset`` / ``_edge_pad`` /
``collate_pad_episodes`` are copied verbatim from
scripts/perception_probe/train_probe_time_dependent.py (they read exactly the
four npz keys, unchanged).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Keys the trainer's row-consuming code touches: EpisodeSequenceDataset
# (npz_path, success, task), compute_task_min_steps (inference_calls, task),
# split_episodes (success, scene_variant), main()'s suite filter + logging
# (suite, prompt, episode).
_REQUIRED_ROW_KEYS = (
    "npz_path",
    "inference_calls",
    "control_frames",
    "success",
    "task",
    "prompt",
    "episode",
    "scene_variant",
    "suite",
)


def read_probe_manifest(features_dir: Path) -> list[dict]:
    """Read ``features_dir/manifest.csv`` (scripts/groot/manifest.py format) and
    return rows shaped like scripts/perception_probe/train_probe.py's
    ``read_manifest`` output -- i.e. with ``npz_path`` / ``inference_calls`` /
    ``control_frames`` keys the forked trainer consumes."""
    features_dir = Path(features_dir)
    with (features_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows: list[dict] = []
    for r in raw_rows:
        row = dict(r)  # keep every original column too (harmless, aids debugging)
        row["npz_path"] = f"{r['dir']}/rollout.npz"
        row["inference_calls"] = r["n_policy"]
        row["control_frames"] = r["n_control"]
        rows.append(row)
    return rows


def suite_of(row: dict) -> str:
    """Task -> occluded-suite name. GR00T's manifest already records it per row
    (always ``libero_10_occluded`` here), so unlike the pi0.5 trainer's
    ``collect_features.build_task_suite_map`` there is no BDDL lookup to do."""
    return row["suite"]


class EpisodeSequenceDataset(Dataset):
    """One item == one whole episode's full (T, ...) feature sequence, lazily
    decompressed from its .npz on access. Copied verbatim from
    scripts/perception_probe/train_probe_time_dependent.py."""

    def __init__(self, rows: list[dict], features_dir: Path, max_steps: int | None = None):
        self.rows = rows
        self.features_dir = Path(features_dir)
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
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
    timestep -- NOT zero padding (an all-False language_mask row would make
    AttentionPool's softmax emit NaN). Copied verbatim."""
    pad_len = target_len - x.shape[0]
    if pad_len == 0:
        return x
    last = x[-1:].expand(pad_len, *x.shape[1:])
    return torch.cat([x, last], dim=0)


def collate_pad_episodes(batch: list[dict]) -> dict:
    """Copied verbatim from train_probe_time_dependent.py."""
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
        "tasks": [item["task"] for item in batch],
    }
