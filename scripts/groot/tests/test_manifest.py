"""Pure tests for manifest + serialization round-trip. No sim / network / GPU."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import manifest as M  # noqa: E402
from serialization import load_rollout, save_rollout  # noqa: E402

from _fakes import build_rollout  # noqa: E402


class SerializationRoundTripTest(unittest.TestCase):
    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            rec = build_rollout("normal", 0, 0, success=True, n_policy=2, replan=4)
            save_rollout(d, rec)
            self.assertTrue((d / "rollout.json").is_file())
            self.assertTrue((d / "rollout.npz").is_file())

            back = load_rollout(d)
            self.assertEqual(back.rollout_id, rec.rollout_id)
            self.assertTrue(back.success)
            self.assertEqual(back.n_control, 8)
            self.assertEqual(back.n_policy, 2)
            self.assertEqual(back.replan_steps, 4)
            self.assertEqual([c.policy_step for c in back.controls], [0, 0, 0, 0, 1, 1, 1, 1])
            self.assertEqual(back.feature_source, "backbone")
            with np.load(d / "rollout.npz") as data:
                self.assertEqual(int(data["img_tokens"]), 64)
                self.assertEqual(int(data["hidden"]), 2048)
                self.assertEqual(data["video_frame_id"].tolist(), list(range(8)))

    def test_features_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            save_rollout(d, build_rollout("normal", 0, 0, success=False, with_features=True,
                                          n_policy=2, replan=3))
            with np.load(d / "rollout.npz") as data:
                self.assertTrue(bool(data["has_features"]))
                self.assertEqual(data["base_image"].shape, (2, 64, 2048))
                self.assertEqual(data["language"].shape, (2, 200, 2048))
                self.assertEqual(data["language_mask"].shape, (2, 200))
                self.assertEqual(data["state_features"].shape, (2, 1536))
                self.assertEqual(data["base_image"].dtype, np.float16)
            back = load_rollout(d)
            self.assertIsNotNone(back.policies[0].features)
            self.assertEqual(back.policies[0].features.language_len, 7)
            self.assertEqual(int(back.policies[0].features.language_mask.sum()), 7)
            self.assertEqual(back.policies[0].features.source, "backbone")
            meta = json.loads((d / "rollout.json").read_text())
            self.assertTrue(meta["has_features"])
            self.assertEqual(meta["policies"][0]["language_len"], 7)
            self.assertEqual(meta["policies"][0]["feature_source"], "backbone")

    def test_no_features_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "ep000"
            save_rollout(d, build_rollout("normal", 0, 0, success=True, n_policy=2, replan=3))
            with np.load(d / "rollout.npz") as data:
                self.assertFalse(bool(data["has_features"]))
                self.assertNotIn("base_image", data.files)
            self.assertIsNone(load_rollout(d).policies[0].features)


class ManifestTest(unittest.TestCase):
    def _populate(self, root: Path, variants, tasks, episodes, *, with_video: bool) -> None:
        for v in variants:
            for t in tasks:
                stem = "KITCHEN_SCENE_demo"
                for e in episodes:
                    d = M.episode_dir(root, v, t, stem, e)
                    save_rollout(d, build_rollout(v, t, e, success=(e % 2 == 0),
                                                  n_policy=2, replan=3))
                    if with_video:
                        (d / "rollout.mp4").write_bytes(b"x")
                        (d / "wrist.mp4").write_bytes(b"x")

    def test_complete_and_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._populate(root, ["normal", "occluded"], [0], [0, 1], with_video=True)
            d0 = M.episode_dir(root, "normal", 0, "KITCHEN_SCENE_demo", 0)
            self.assertTrue(M.episode_is_complete(d0))
            rows = M.rebuild_manifest(root)
            self.assertEqual(len(rows), 4)
            self.assertEqual(set(M.MANIFEST_FIELDS), set(rows[0]))
            self.assertTrue((root / "manifest.csv").is_file())

    def test_incomplete_without_video(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._populate(root, ["normal"], [0], [0], with_video=False)
            d0 = M.episode_dir(root, "normal", 0, "KITCHEN_SCENE_demo", 0)
            self.assertFalse(M.episode_is_complete(d0, expect_video=True))
            self.assertTrue(M.episode_is_complete(d0, expect_video=False))

    def test_find_next_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stems = {(v, t): "KITCHEN_SCENE_demo" for v in ("normal", "occluded") for t in range(10)}
            self._populate(root, ["normal", "occluded"], [0], [0, 1], with_video=True)
            nxt = M.find_next_incomplete(root, stems, 2, variants=("normal", "occluded"))
            self.assertEqual(nxt, ("normal", 1, 0))


if __name__ == "__main__":
    unittest.main()
