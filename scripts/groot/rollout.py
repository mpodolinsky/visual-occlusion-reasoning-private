"""GR00T control loop on a libero_10 task.

Modelled on ``scripts/semantic_failure/rollout_runner.py`` + repo 16's
``sim/run_eval.py::run_episode``, but uses **12's init protocol**
(``env.set_init_state`` + ``num_steps_wait`` DUMMY_ACTION settle steps, matching
``eval_pi05_libero``) so the numbers stay comparable to the pi0.5 runs.

Talks to the GR00T policy over the openpi websocket protocol (see
``server/serve_groot_ws.py``); only ``result["actions"]`` is required.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any

import numpy as np

from constants import CONTROL_HZ, DUMMY_ACTION, FEATURE_SERVER_SCRIPT, FEATURE_SOURCE, MODEL_ID, EvalConfig
from feature_schema import identify_groot_features
from groot_obs import build_flat_obs, decode_action_step
from libero_env import goal_predicate_strings, load_eval_module
from records import ControlRecord, GrootFeatures, PolicyRecord, RolloutRecord

_FEATURE_KEYS = ("base_image", "wrist_image", "language", "language_mask", "state_features")


def _features_from_result(result: dict, raw_actions: np.ndarray) -> GrootFeatures | None:
    if not all(k in result for k in _FEATURE_KEYS):
        return None
    return identify_groot_features(result, raw_actions)


def _rollout_id(scene_variant: str, task_id: int, episode_index: int, bddl_stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in bddl_stem)[:80].rstrip("_")
    return f"{scene_variant}__t{task_id:03d}__ep{episode_index:03d}__{safe}"


def _finalize_ranges(policies: list[PolicyRecord], controls: list[ControlRecord]) -> None:
    by_policy: dict[int, list[ControlRecord]] = {}
    for control in controls:
        by_policy.setdefault(control.policy_step, []).append(control)
    for pol in policies:
        group = sorted(by_policy.get(pol.policy_step, []), key=lambda c: c.control_step)
        if group:
            pol.executed_control_step_start = group[0].control_step
            pol.executed_control_step_end = group[-1].control_step


def run_episode(
    *,
    env: Any,
    client: Any,
    initial_state: Any,
    instruction: str,
    cfg: EvalConfig,
    task_file: str,
    max_steps: int,
    server_metadata: str,
) -> RolloutRecord:
    eval_mod = getattr(env, "_pipeline_eval", None) or load_eval_module()
    t0 = time.time()

    client.reset()
    env.reset()
    observation = env.set_init_state(initial_state)

    sim_step_next = 0
    for _ in range(cfg.num_steps_wait):
        observation, _, _, _ = env.step(DUMMY_ACTION)
        sim_step_next += 1

    policies: list[PolicyRecord] = []
    controls: list[ControlRecord] = []
    video_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    action_plan: collections.deque[tuple[int, int, np.ndarray]] = collections.deque()
    success = False
    control_step_next = 0

    while not success and control_step_next < max_steps:
        if not action_plan:
            result = client.infer(build_flat_obs(observation, instruction))
            if "actions" not in result:
                raise RuntimeError(f"policy response missing 'actions': keys={list(result)}")
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise RuntimeError(f"bad action chunk shape {actions.shape}, expected (H, 7)")
            policy_step = len(policies)
            n_take = min(cfg.replan_steps, max_steps - control_step_next, int(actions.shape[0]))
            if n_take < 1:
                raise RuntimeError(f"empty action chunk {actions.shape}")
            policies.append(
                PolicyRecord(
                    policy_step=policy_step,
                    predicted_action_chunk=actions,
                    executed_control_step_start=control_step_next,
                    executed_control_step_end=control_step_next + n_take - 1,
                    features=_features_from_result(result, actions),
                )
            )
            for j in range(n_take):
                action_plan.append((policy_step, j, actions[j]))

        policy_step, chunk_index, raw = action_plan.popleft()
        action = decode_action_step(raw)
        observation, _, done, _ = env.step(action)
        success = bool(done)
        if cfg.save_video:
            video_frames.append(
                np.asarray(eval_mod.rotate_camera_image(observation, "agentview_image"))
            )
            wrist_frames.append(
                np.asarray(eval_mod.rotate_camera_image(observation, "robot0_eye_in_hand_image"))
            )
        controls.append(
            ControlRecord(
                control_step=control_step_next,
                sim_step=sim_step_next,
                policy_step=policy_step,
                chunk_index=chunk_index,
                executed_action=np.asarray(action, dtype=np.float32),
                video_frame_id=control_step_next if cfg.save_video else None,
                env_success=success,
            )
        )
        control_step_next += 1
        sim_step_next += 1

    _finalize_ranges(policies, controls)
    goals = goal_predicate_strings(env)
    if success:
        sim_cat, fail_pred, detail = "success", "", ""
    else:
        sim_cat = "unsatisfied_goal"
        fail_pred = goals[0] if goals else instruction
        detail = "; ".join(goals) if goals else "episode ended without env success"

    rollout = RolloutRecord(
        rollout_id=_rollout_id(cfg.scene_variant, cfg.task_id, cfg.episode_index, Path(task_file).stem),
        suite=cfg.occluded_suite,
        scene_variant=cfg.scene_variant,
        task_id=cfg.task_id,
        task_file=str(task_file),
        episode_index=cfg.episode_index,
        instruction=instruction,
        replan_steps=cfg.replan_steps,
        success=success,
        max_steps=max_steps,
        seed=cfg.seed,
        model_id=MODEL_ID,
        checkpoint=cfg.checkpoint_label,
        feature_source=FEATURE_SOURCE,
        feature_server=FEATURE_SERVER_SCRIPT,
        feature_server_version=server_metadata,
        control_hz=CONTROL_HZ,
        num_steps_wait=cfg.num_steps_wait,
        sim_failure_category=sim_cat,
        failing_predicate=str(fail_pred),
        failure_detail=detail,
        elapsed_seconds=round(time.time() - t0, 2),
        policies=policies,
        controls=controls,
    )
    rollout._video_frames = video_frames  # type: ignore[attr-defined]
    rollout._wrist_frames = wrist_frames  # type: ignore[attr-defined]
    return rollout
