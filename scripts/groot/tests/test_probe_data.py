"""Pure tests for the GR00T perception-probe data layer. No GPU / network / sim.

Covers probe_data.read_probe_manifest (scripts/groot manifest -> trainer row
shape), probe_data.collate_pad_episodes / _edge_pad, and probe_utils.split_episodes
stratification.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe_data import _edge_pad, collate_pad_episodes, read_probe_manifest  # noqa: E402
from probe_utils import split_episodes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
GROOT_FEATURES_DIR = REPO_ROOT / "outputs" / "groot" / "libero_10"

# What the forked trainer's row-consuming code touches.
_TRAINER_ROW_KEYS = {
    "npz_path", "inference_calls", "control_frames", "success",
    "task", "prompt", "episode", "scene_variant", "suite",
}


@unittest.skipUnless(
    (GROOT_FEATURES_DIR / "manifest.csv").is_file(),
    f"no collected GR00T dataset at {GROOT_FEATURES_DIR}",
)
class ReadProbeManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = read_probe_manifest(GROOT_FEATURES_DIR)

    def test_row_count(self) -> None:
        self.assertEqual(len(self.rows), 500)

    def test_row_keys_superset(self) -> None:
        self.assertTrue(_TRAINER_ROW_KEYS.issubset(self.rows[0]))

    def test_success_is_python_bool_string(self) -> None:
        self.assertEqual({r["success"] for r in self.rows}, {"True", "False"})

    def test_npz_path_resolves(self) -> None:
        for r in self.rows[:5]:
            self.assertTrue((GROOT_FEATURES_DIR / r["npz_path"]).is_file(), r["npz_path"])
            self.assertTrue(r["npz_path"].endswith("/rollout.npz"))

    def test_inference_calls_are_positive_ints(self) -> None:
        for r in self.rows:
            self.assertGreater(int(r["inference_calls"]), 0)
            self.assertGreaterEqual(int(r["control_frames"]), int(r["inference_calls"]))

    def test_single_suite(self) -> None:
        self.assertEqual({r["suite"] for r in self.rows}, {"libero_10_occluded"})

    def test_npz_has_the_four_probe_arrays(self) -> None:
        with np.load(GROOT_FEATURES_DIR / self.rows[0]["npz_path"]) as d:
            for k, tokdim in (("base_image", 64), ("wrist_image", 64), ("language", 200)):
                self.assertEqual(d[k].ndim, 3)
                self.assertEqual(d[k].shape[1], tokdim)
                self.assertEqual(d[k].shape[2], 2048)
            self.assertEqual(d["language_mask"].shape[1], 200)


class EdgePadTest(unittest.TestCase):
    def test_repeats_last_row_not_zeros(self) -> None:
        x = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
        padded = _edge_pad(x, 5)
        self.assertEqual(padded.shape, (5, 3))
        self.assertTrue(torch.equal(padded[2], x[-1]))
        self.assertTrue(torch.equal(padded[4], x[-1]))

    def test_noop_when_already_full(self) -> None:
        x = torch.zeros(4, 2)
        self.assertTrue(torch.equal(_edge_pad(x, 4), x))


class CollateTest(unittest.TestCase):
    def _item(self, length: int, label: float) -> dict:
        return {
            "base_image": torch.randn(length, 64, 2048, dtype=torch.float16),
            "wrist_image": torch.randn(length, 64, 2048, dtype=torch.float16),
            "language": torch.randn(length, 200, 2048, dtype=torch.float16),
            "language_mask": torch.ones(length, 200, dtype=torch.bool),
            "length": length,
            "label": label,
            "task": "TASK_A",
        }

    def test_pads_to_batch_max_and_marks_valid(self) -> None:
        batch = collate_pad_episodes([self._item(3, 1.0), self._item(7, 0.0)])
        self.assertEqual(batch["base_image"].shape, (2, 7, 64, 2048))
        self.assertEqual(batch["language_mask"].shape, (2, 7, 200))
        self.assertTrue(torch.equal(batch["lengths"], torch.tensor([3, 7])))
        self.assertTrue(torch.equal(batch["valid_masks"][0], torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)))
        self.assertTrue(torch.equal(batch["success_labels"], torch.tensor([1.0, 0.0])))
        self.assertEqual(batch["tasks"], ["TASK_A", "TASK_A"])


class SplitEpisodesTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        for i in range(200):
            rows.append({
                "success": "True" if i % 4 else "False",
                "scene_variant": "normal" if i % 2 else "occluded",
                "task": f"T{i % 10}",
                "episode": str(i),
                "npz_path": f"x/ep{i:03d}/rollout.npz",
            })
        return rows

    def test_every_split_has_both_classes_and_variants(self) -> None:
        splits = split_episodes(self._rows(), seed=0)
        total = sum(len(v) for v in splits.values())
        self.assertEqual(total, 200)  # exact partition, nothing lost
        for name, part in splits.items():
            with self.subTest(split=name):
                self.assertTrue(any(r["success"] == "True" for r in part))
                self.assertTrue(any(r["success"] == "False" for r in part))
                self.assertEqual({r["scene_variant"] for r in part}, {"normal", "occluded"})


if __name__ == "__main__":
    unittest.main()
