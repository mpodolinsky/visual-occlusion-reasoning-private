"""Shared builders for a correctly-aligned fake GR00T rollout (no sim / GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

from groot_obs import decode_action_step  # noqa: E402
from records import ControlRecord, GrootFeatures, PolicyRecord, RolloutRecord  # noqa: E402


def fake_features(n_lang: int = 7, *, seed: int = 0) -> GrootFeatures:
    rng = np.random.default_rng(seed)
    lang = np.zeros((200, 2048), np.float16)
    lang[:n_lang] = rng.standard_normal((n_lang, 2048)).astype(np.float16)
    mask = np.zeros((200,), np.bool_)
    mask[:n_lang] = True
    return GrootFeatures(
        base_image=rng.standard_normal((64, 2048)).astype(np.float16),
        wrist_image=(rng.standard_normal((64, 2048)) + 3.0).astype(np.float16),
        language=lang,
        language_mask=mask,
        language_len=n_lang,
        state_features=rng.standard_normal((1536,)).astype(np.float16),
        module="test",
        shapes={"base_image": (64, 2048)},
    )


def build_rollout(
    scene_variant: str = "normal",
    task_id: int = 0,
    episode: int = 0,
    *,
    success: bool = True,
    with_features: bool = False,
    n_policy: int = 3,
    replan: int = 8,
    horizon: int = 16,
    num_steps_wait: int = 10,
) -> RolloutRecord:
    rng = np.random.default_rng(42)
    policies: list[PolicyRecord] = []
    controls: list[ControlRecord] = []
    cs = 0
    for ps in range(n_policy):
        chunk = rng.uniform(-1, 1, (horizon, 7)).astype(np.float32)
        start = cs
        for j in range(replan):
            executed = np.asarray(decode_action_step(chunk[j]), dtype=np.float32)
            controls.append(
                ControlRecord(
                    control_step=cs, sim_step=cs + num_steps_wait, policy_step=ps,
                    chunk_index=j, executed_action=executed, video_frame_id=cs,
                )
            )
            cs += 1
        policies.append(
            PolicyRecord(
                policy_step=ps, predicted_action_chunk=chunk,
                executed_control_step_start=start, executed_control_step_end=cs - 1,
                features=fake_features(seed=ps) if with_features else None,
            )
        )
    return RolloutRecord(
        rollout_id=f"{scene_variant}__t{task_id:03d}__ep{episode:03d}__demo",
        suite="libero_10_occluded", scene_variant=scene_variant, task_id=task_id,
        task_file="/x/KITCHEN_SCENE_demo.bddl", episode_index=episode,
        instruction="do the thing", replan_steps=replan, success=success, max_steps=520,
        seed=7, model_id="gr00t-n1.7-libero", checkpoint="nvidia/GR00T-N1.7-LIBERO/libero_10",
        feature_source="backbone", feature_server="scripts/groot/server/serve_groot_ws.py",
        feature_server_version="{'policy': 'gr00t-n1.7-libero_10'}",
        control_hz=20.0, num_steps_wait=num_steps_wait,
        sim_failure_category="success" if success else "unsatisfied_goal",
        failing_predicate="" if success else "(on obj table)",
        failure_detail="" if success else "(on obj table)",
        elapsed_seconds=12.3, policies=policies, controls=controls,
    )
