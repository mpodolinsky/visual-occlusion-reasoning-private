from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from timeline import build_timeline_windows, map_onset_frame, steps_per_segment


class TimelineTests(unittest.TestCase):
    def test_sixty_steps_per_three_seconds(self) -> None:
        self.assertEqual(steps_per_segment(20.0, 3.0), 60)

    def test_windows(self) -> None:
        segs = build_timeline_windows(130, replan_steps=5)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0].control_step_start, 0)
        self.assertEqual(segs[0].control_step_end, 59)
        self.assertEqual(segs[0].policy_step_start, 0)
        self.assertEqual(segs[0].policy_step_end, 11)
        self.assertAlmostEqual(segs[0].t_end_sec, 3.0)
        self.assertEqual(segs[2].control_step_start, 120)
        self.assertEqual(segs[2].control_step_end, 129)
        self.assertAlmostEqual(segs[2].t_start_sec, 6.0)

    def test_onset_maps_to_policy_and_chunk(self) -> None:
        mapped = map_onset_frame(248, n_control=500, replan_steps=5)
        self.assertEqual(mapped["failure_control_step"], 248)
        self.assertEqual(mapped["failure_policy_step"], 49)
        self.assertEqual(mapped["failure_chunk_index"], 3)


if __name__ == "__main__":
    unittest.main()
