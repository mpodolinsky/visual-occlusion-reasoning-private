"""Backbone-feature identity checks. Fail if this is not the split layer-16
hidden state block from ``serve_groot_ws.py --with-features``.

Mirrors ``scripts/semantic_failure/feature_schema.identify_save_a``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from constants import (
    FEATURE_MODULE,
    FEATURE_SOURCE,
    HIDDEN_DIM,
    LANG_MAX_TOKENS,
    NUM_IMAGE_TOKENS,
    STATE_FEATURE_DIM,
)
from records import AlignmentError, GrootFeatures

REQUIRED_KEYS = ("base_image", "wrist_image", "language", "language_mask", "state_features")


def identify_groot_features(features: dict[str, Any], actions: np.ndarray) -> GrootFeatures:
    missing = [k for k in REQUIRED_KEYS if k not in features]
    if missing:
        raise AlignmentError(
            f"backbone features missing {missing}. "
            "Start the server with --with-features (GROOT_WITH_FEATURES=1)."
        )
    base = np.asarray(features["base_image"])
    wrist = np.asarray(features["wrist_image"])
    language = np.asarray(features["language"])
    mask = np.asarray(features["language_mask"])
    state = np.asarray(features["state_features"])

    if base.shape != (NUM_IMAGE_TOKENS, HIDDEN_DIM):
        raise AlignmentError(f"base_image shape {base.shape} != ({NUM_IMAGE_TOKENS}, {HIDDEN_DIM})")
    if wrist.shape != (NUM_IMAGE_TOKENS, HIDDEN_DIM):
        raise AlignmentError(f"wrist_image shape {wrist.shape} != ({NUM_IMAGE_TOKENS}, {HIDDEN_DIM})")
    if language.shape != (LANG_MAX_TOKENS, HIDDEN_DIM):
        raise AlignmentError(f"language shape {language.shape} != ({LANG_MAX_TOKENS}, {HIDDEN_DIM})")
    if mask.shape != (LANG_MAX_TOKENS,):
        raise AlignmentError(f"language_mask {mask.shape} != ({LANG_MAX_TOKENS},)")
    if state.shape != (STATE_FEATURE_DIM,):
        raise AlignmentError(f"state_features {state.shape} != ({STATE_FEATURE_DIM},)")

    for name, arr in (("base_image", base), ("wrist_image", wrist), ("language", language),
                      ("state_features", state)):
        if not np.isfinite(arr.astype(np.float32)).all():
            raise AlignmentError(f"{name} has NaN/Inf")
    if not bool(mask.any()):
        raise AlignmentError("language_mask is all-False (no instruction tokens)")
    real = language[mask.astype(bool)]
    if real.size and not np.any(real.astype(np.float32)):
        raise AlignmentError("language real region is all-zero")
    if np.allclose(base.astype(np.float32), wrist.astype(np.float32), atol=1e-4):
        raise AlignmentError("base_image and wrist_image are identical -- wrong token split")

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise AlignmentError(f"actions must be (horizon, dim), got {actions.shape}")
    if not np.isfinite(actions.astype(np.float32)).all():
        raise AlignmentError("actions have NaN/Inf")

    lang_len = features.get("language_len")
    lang_len = int(mask.sum()) if lang_len is None else int(lang_len)

    return GrootFeatures(
        base_image=base.astype(np.float16),
        wrist_image=wrist.astype(np.float16),
        language=language.astype(np.float16),
        language_mask=mask.astype(bool),
        language_len=lang_len,
        state_features=state.astype(np.float16),
        source=FEATURE_SOURCE,
        module=FEATURE_MODULE,
        shapes={
            "base_image": tuple(base.shape),
            "wrist_image": tuple(wrist.shape),
            "language": tuple(language.shape),
            "language_mask": tuple(mask.shape),
            "state_features": tuple(state.shape),
            "actions": tuple(actions.shape),
        },
    )
