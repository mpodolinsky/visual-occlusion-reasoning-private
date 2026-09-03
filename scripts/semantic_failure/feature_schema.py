"""SAVE-A identity checks. Fail if this is not the fused prefix from the feature server."""

from __future__ import annotations

from typing import Any

import numpy as np

from records import AlignmentError, PrefixFeatures
from constants import FEATURE_MODULE, FEATURE_SOURCE, HIDDEN_DIM, NUM_IMAGE_TOKENS

REQUIRED_KEYS = ("base_image", "wrist_image", "language", "language_mask")


def identify_save_a(prefix_features: dict[str, Any], actions: np.ndarray) -> PrefixFeatures:
    missing = [k for k in REQUIRED_KEYS if k not in prefix_features]
    if missing:
        raise AlignmentError(
            f"prefix_features missing {missing}. "
            "Start serve_pi05_with_features.py, not stock serve_policy.py."
        )
    base = np.asarray(prefix_features["base_image"])
    wrist = np.asarray(prefix_features["wrist_image"])
    language = np.asarray(prefix_features["language"])
    mask = np.asarray(prefix_features["language_mask"])
    if base.ndim != 2 or base.shape != (NUM_IMAGE_TOKENS, HIDDEN_DIM):
        raise AlignmentError(f"base_image shape {base.shape} != ({NUM_IMAGE_TOKENS}, {HIDDEN_DIM})")
    if wrist.shape != (NUM_IMAGE_TOKENS, HIDDEN_DIM):
        raise AlignmentError(f"wrist_image shape {wrist.shape} != ({NUM_IMAGE_TOKENS}, {HIDDEN_DIM})")
    if language.ndim != 2 or language.shape[-1] != HIDDEN_DIM:
        raise AlignmentError(f"language shape {language.shape} is not (*, {HIDDEN_DIM})")
    if mask.shape != language.shape[:1]:
        raise AlignmentError(f"language_mask {mask.shape} != language length {language.shape[0]}")
    if not np.isfinite(base.astype(np.float32)).all():
        raise AlignmentError("base_image has NaN/Inf")
    if not np.isfinite(wrist.astype(np.float32)).all():
        raise AlignmentError("wrist_image has NaN/Inf")
    if not np.isfinite(language.astype(np.float32)).all():
        raise AlignmentError("language has NaN/Inf")
    if not bool(mask.any()):
        raise AlignmentError("language_mask is all-False")
    if np.allclose(base.astype(np.float32), wrist.astype(np.float32), atol=1e-4):
        raise AlignmentError("base_image and wrist_image are identical — wrong prefix slice")
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise AlignmentError(f"actions must be (horizon, dim), got {actions.shape}")
    if not np.isfinite(actions.astype(np.float32)).all():
        raise AlignmentError("actions have NaN/Inf")
    return PrefixFeatures(
        base_image=base,
        wrist_image=wrist,
        language=language,
        language_mask=mask.astype(bool),
        source=FEATURE_SOURCE,
        module=FEATURE_MODULE,
        shapes={
            "base_image": tuple(base.shape),
            "wrist_image": tuple(wrist.shape),
            "language": tuple(language.shape),
            "language_mask": tuple(mask.shape),
            "actions": tuple(actions.shape),
        },
    )
