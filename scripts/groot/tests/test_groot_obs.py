"""Pure tests for the GR00T obs/action bridge. No sim / network / GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import groot_obs  # noqa: E402


def _fake_obs(res: int = 256) -> dict:
    rng = np.random.default_rng(0)
    return {
        "agentview_image": rng.integers(0, 256, (res, res, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 256, (res, res, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, -0.2, 0.9], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }


class BuildFlatObsTest(unittest.TestCase):
    def test_shapes_and_keys(self) -> None:
        flat = groot_obs.build_flat_obs(_fake_obs(), "pick up the black bowl")
        # exactly the keys server/smoke_server.py sends
        expected = {
            "video.image", "video.wrist_image",
            "state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw",
            "state.gripper", groot_obs.LANG_KEY, "task",
        }
        self.assertEqual(set(flat), expected)
        self.assertEqual(flat["video.image"].shape, (1, 1, 256, 256, 3))
        self.assertEqual(flat["video.image"].dtype, np.uint8)
        self.assertEqual(flat["video.wrist_image"].shape, (1, 1, 256, 256, 3))
        for k in ("state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw"):
            self.assertEqual(flat[k].shape, (1, 1, 1))
            self.assertEqual(flat[k].dtype, np.float32)
        self.assertEqual(flat["state.gripper"].shape, (1, 1, 2))
        self.assertEqual(flat[groot_obs.LANG_KEY], ["pick up the black bowl"])

    def test_180_rotation_default(self) -> None:
        obs = _fake_obs()
        flat = groot_obs.build_flat_obs(obs, "x")
        np.testing.assert_array_equal(
            flat["video.image"][0, 0], obs["agentview_image"][::-1, ::-1]
        )

    def test_state_values(self) -> None:
        flat = groot_obs.build_flat_obs(_fake_obs(), "x")
        # identity quaternion -> zero axis-angle
        np.testing.assert_allclose(
            [flat["state.roll"], flat["state.pitch"], flat["state.yaw"]], 0.0, atol=1e-6
        )
        self.assertAlmostEqual(float(flat["state.x"][0, 0, 0]), 0.1, places=5)
        np.testing.assert_allclose(flat["state.gripper"][0, 0], [0.02, -0.02], atol=1e-6)


class DecodeActionStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self._native = groot_obs._LIBEROX_NATIVE

    def tearDown(self) -> None:
        groot_obs._LIBEROX_NATIVE = self._native

    def test_nvidia_convention_gripper(self) -> None:
        groot_obs._LIBEROX_NATIVE = False
        # model gripper ~1 (closed in [0,1]) -> 2*1-1=1 -> sign=1 -> invert=-1
        out = groot_obs.decode_action_step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        self.assertEqual(out[-1], -1.0)
        # model gripper ~0 (open) -> -1 -> sign=-1 -> invert=+1
        out = groot_obs.decode_action_step(np.array([0, 0, 0, 0, 0, 0, 0.0]))
        self.assertEqual(out[-1], 1.0)
        self.assertEqual(len(out), 7)

    def test_native_convention_gripper(self) -> None:
        groot_obs._LIBEROX_NATIVE = True
        out = groot_obs.decode_action_step(np.array([0, 0, 0, 0, 0, 0, 0.7]))
        self.assertEqual(out[-1], 1.0)
        out = groot_obs.decode_action_step(np.array([0, 0, 0, 0, 0, 0, -0.4]))
        self.assertEqual(out[-1], -1.0)

    def test_pose_dims_passthrough(self) -> None:
        groot_obs._LIBEROX_NATIVE = False
        row = np.array([0.01, -0.02, 0.03, 0.1, -0.1, 0.2, 0.0])
        out = groot_obs.decode_action_step(row)
        np.testing.assert_allclose(out[:6], row[:6], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
