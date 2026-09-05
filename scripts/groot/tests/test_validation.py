"""Pure tests for the GR00T rollout alignment gate. No sim / network / GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import validate_rollout  # noqa: E402

from _fakes import build_rollout  # noqa: E402


class ValidationTest(unittest.TestCase):
    def test_aligned_rollout_passes(self) -> None:
        r = validate_rollout(build_rollout(success=True, n_policy=3, replan=8))
        self.assertTrue(r.all_passed, r.format())

    def test_aligned_with_features_passes(self) -> None:
        r = validate_rollout(build_rollout(with_features=True, n_policy=2, replan=4))
        self.assertTrue(r.all_passed, r.format())

    def test_broken_executed_action_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.controls[3].executed_action = rec.controls[3].executed_action + 0.5
        self.assertFalse(validate_rollout(rec).all_passed)

    def test_non_sequential_policy_step_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.policies[1].policy_step = 5
        self.assertFalse(validate_rollout(rec).all_passed)

    def test_video_frame_mismatch_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.controls[2].video_frame_id = 99
        self.assertFalse(validate_rollout(rec).all_passed)

    def test_sim_step_wait_offset_checked(self) -> None:
        rec = build_rollout(n_policy=2, replan=4, num_steps_wait=10)
        for c in rec.controls:
            c.sim_step = c.control_step  # drop the wait offset
        self.assertFalse(validate_rollout(rec).all_passed)

    def test_identical_base_wrist_fails(self) -> None:
        rec = build_rollout(with_features=True, n_policy=2, replan=3)
        rec.policies[0].features.wrist_image = rec.policies[0].features.base_image.copy()
        self.assertFalse(validate_rollout(rec).all_passed)

    def test_nonfinite_feature_fails(self) -> None:
        rec = build_rollout(with_features=True, n_policy=2, replan=3)
        rec.policies[0].features.language[0, 0] = np.float16("inf")
        self.assertFalse(validate_rollout(rec).all_passed)

    def test_total_mismatch_fails(self) -> None:
        rec = build_rollout(n_policy=2, replan=4)
        rec.controls.pop()
        self.assertFalse(validate_rollout(rec).all_passed)


if __name__ == "__main__":
    unittest.main()
