from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from records import (
    ControlRecord,
    FailureAnnotation,
    PolicyRecord,
    PrefixFeatures,
    RolloutRecord,
    SemanticSegment,
)
from rollout_runner import finalize_executed_ranges
from validation import validate_rollout
from feature_schema import identify_save_a
from serialization import (
    load_label_document,
    load_labels,
    load_rollout,
    save_labels,
    save_rollout,
)


def _feats(seed: int) -> PrefixFeatures:
    rng = np.random.default_rng(seed)
    lang = 12
    return PrefixFeatures(
        base_image=rng.normal(size=(256, 2048)).astype(np.float32),
        wrist_image=(rng.normal(size=(256, 2048)) + 2.0).astype(np.float32),
        language=rng.normal(size=(lang, 2048)).astype(np.float32),
        language_mask=np.ones(lang, dtype=bool),
        source="prefix",
        module="test",
        shapes={"base_image": (256, 2048), "wrist_image": (256, 2048), "language": (lang, 2048)},
    )


def make_synthetic(*, replan: int = 5, n_control: int = 7) -> RolloutRecord:
    rng = np.random.default_rng(0)
    policies: list[PolicyRecord] = []
    controls: list[ControlRecord] = []
    t = 0
    p = 0
    horizon = 10
    while t < n_control:
        chunk = rng.normal(size=(horizon, 7)).astype(np.float32)
        n_take = min(replan, n_control - t)
        policies.append(
            PolicyRecord(
                policy_step=p,
                prefix_features=_feats(p + 1),
                predicted_action_chunk=chunk,
                executed_control_step_start=t,
                executed_control_step_end=t + n_take - 1,
            )
        )
        for j in range(n_take):
            controls.append(
                ControlRecord(
                    control_step=t + j,
                    sim_step=10 + t + j,
                    policy_step=p,
                    chunk_index=j,
                    executed_action=chunk[j],
                    video_frame_id=t + j,
                )
            )
        t += n_take
        p += 1
    return RolloutRecord(
        rollout_id="normal__t004__ep000__synth",
        suite="libero_10_occluded",
        scene_variant="normal",
        task_id=4,
        task_file="synth.bddl",
        episode_index=0,
        instruction="put the soup in the basket",
        replan_steps=replan,
        success=False,
        max_steps=520,
        seed=7,
        model_id="pi05_libero",
        checkpoint="test",
        feature_source="prefix",
        feature_server="serve_pi05_with_features.py",
        feature_server_version="{}",
        control_hz=20.0,
        num_steps_wait=10,
        sim_failure_category="unsatisfied_goal",
        failing_predicate="in soup basket",
        failure_detail="never satisfied",
        policies=policies,
        controls=controls,
        failure=FailureAnnotation(),
    )


class RecordsTests(unittest.TestCase):
    def test_validate_synthetic(self) -> None:
        report = validate_rollout(make_synthetic())
        self.assertTrue(report.all_passed, report.format())

    def test_roundtrip(self) -> None:
        import tempfile

        rollout = make_synthetic(n_control=12)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_rollout(directory, rollout)
            loaded = load_rollout(directory)
        self.assertEqual(loaded.n_control, 12)
        self.assertEqual(loaded.n_policy, 3)
        self.assertTrue(validate_rollout(loaded).all_passed)

    def test_stock_server_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefix_features missing"):
            identify_save_a({"actions": np.zeros((10, 7))}, np.zeros((10, 7)))

    def test_labels_in_separate_files(self) -> None:
        import json
        import tempfile

        rollout = make_synthetic(n_control=7)
        rollout.semantic_timeline = [
            SemanticSegment(0, 0.0, 3.0, 0, 5, 0, 1, phrase="reaching for can"),
            SemanticSegment(1, 3.0, 3.5, 6, 6, 1, 1, phrase="closing gripper"),
        ]
        rollout.vlm_failure = {
            "vlm_failure_onset_frame": 6,
            "vlm_failure_onset_seconds": 0.3,
            "vlm_failure_mode": "grasp_failure",
            "vlm_failure_reason": "slipped off the rim",
            "vlm_recovery_action": "regrasp lower",
        }
        from records import FailureAnnotation

        rollout.failure = FailureAnnotation(failure_control_step=6, failure_policy_step=1, failure_chunk_index=1)
        meta = {"backend": "GeminiBackend", "model": "gemini-x", "refine": True,
                "prompts": {"keyword_phrases": "PROMPT"}}

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_rollout(directory, rollout)
            # rollout.json / rollout.npz carry no Gemini fields
            rj = json.loads((directory / "rollout.json").read_text())
            self.assertNotIn("semantic_timeline", rj)
            self.assertNotIn("vlm_failure", rj)
            self.assertNotIn("failure_annotation", rj)
            self.assertFalse((directory / "labels.json").exists())
            with np.load(directory / "rollout.npz", allow_pickle=False) as feats:
                self.assertNotIn("sem_phrase", feats.files)

            save_labels(directory, rollout, meta)
            self.assertTrue((directory / "labels.json").exists())
            self.assertTrue((directory / "labels.npz").exists())

            doc = load_label_document(directory)
            self.assertEqual(doc["labeler"]["model"], "gemini-x")
            self.assertEqual(doc["labeler"]["prompts"]["keyword_phrases"], "PROMPT")
            self.assertEqual(doc["semantic_timeline"][0]["phrase"], "reaching for can")
            self.assertEqual(doc["failure_annotation"]["failure_control_step"], 6)

            arr = load_labels(directory)
            self.assertEqual(str(arr["sem_phrase"][0]), "reaching for can")
            self.assertEqual(int(arr["fail_onset_frame"]), 6)

            # load_rollout repopulates the record from labels.json
            loaded = load_rollout(directory)
            self.assertEqual(loaded.semantic_timeline[1].phrase, "closing gripper")
            self.assertEqual(loaded.vlm_failure["vlm_failure_mode"], "grasp_failure")
            self.assertEqual(loaded.failure.failure_control_step, 6)

        # unlabeled dir -> None / empty
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_rollout(directory, make_synthetic(n_control=7))
            self.assertIsNone(load_labels(directory))
            self.assertIsNone(load_label_document(directory))
            self.assertEqual(load_rollout(directory).semantic_timeline, [])

    def test_finalize_ranges(self) -> None:
        rollout = make_synthetic(n_control=7)
        for p in rollout.policies:
            p.executed_control_step_end = -1
        finalize_executed_ranges(rollout.policies, rollout.controls)
        self.assertEqual(rollout.policies[0].executed_control_step_end, 4)
        self.assertEqual(rollout.policies[1].executed_control_step_end, 6)


if __name__ == "__main__":
    unittest.main()
