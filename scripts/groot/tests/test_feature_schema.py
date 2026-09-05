"""Pure tests for the backbone-feature identity check. No torch / GR00T venv."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_schema import identify_groot_features  # noqa: E402
from records import AlignmentError  # noqa: E402


def _good(n_lang: int = 9) -> dict:
    rng = np.random.default_rng(1)
    lang = np.zeros((200, 2048), np.float32)
    lang[:n_lang] = rng.standard_normal((n_lang, 2048))
    mask = np.zeros((200,), np.bool_)
    mask[:n_lang] = True
    return {
        "base_image": rng.standard_normal((64, 2048)).astype(np.float32),
        "wrist_image": (rng.standard_normal((64, 2048)) + 5).astype(np.float32),
        "language": lang,
        "language_mask": mask,
        "language_len": np.int32(n_lang),
        "state_features": rng.standard_normal((1536,)).astype(np.float32),
    }


ACTIONS = np.zeros((16, 7), np.float32)


class FeatureSchemaTest(unittest.TestCase):
    def test_good(self) -> None:
        f = identify_groot_features(_good(), ACTIONS)
        self.assertEqual(f.base_image.shape, (64, 2048))
        self.assertEqual(f.language.shape, (200, 2048))
        self.assertEqual(f.language_len, 9)
        self.assertEqual(f.source, "backbone")
        self.assertEqual(f.shapes["actions"], (16, 7))
        self.assertEqual(f.base_image.dtype, np.float16)

    def test_missing_key(self) -> None:
        d = _good()
        del d["state_features"]
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, ACTIONS)

    def test_wrong_shape(self) -> None:
        d = _good()
        d["base_image"] = d["base_image"][:32]
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, ACTIONS)

    def test_base_equals_wrist(self) -> None:
        d = _good()
        d["wrist_image"] = d["base_image"].copy()
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, ACTIONS)

    def test_nonfinite(self) -> None:
        d = _good()
        d["language"][0, 0] = np.nan
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, ACTIONS)

    def test_empty_mask(self) -> None:
        d = _good()
        d["language_mask"] = np.zeros((200,), np.bool_)
        with self.assertRaises(AlignmentError):
            identify_groot_features(d, ACTIONS)


if __name__ == "__main__":
    unittest.main()
