"""12 control loop + feature infer + explicit clocks. Wait steps are sim-only."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

import numpy as np

from records import ControlRecord, FailureAnnotation, PolicyRecord, RolloutRecord
from constants import CONTROL_HZ, DUMMY_ACTION, FEATURE_SERVER_SCRIPT, FEATURE_SOURCE, PipelineConfig
from libero_env import goal_predicate_strings, load_eval_module
from pi05_client import FeatureClient
from serialization import rollout_id


def finalize_executed_ranges(policies: list[PolicyRecord], controls: list[ControlRecord]) -> None:
    by_policy: dict[int, list[ControlRecord]] = {}
    for control in controls:
        by_policy.setdefault(control.policy_step, []).append(control)
    for pol in policies:
        group = by_policy.get(pol.policy_step, [])
        if not group:
            continue
        group.sort(key=lambda c: c.control_step)
        pol.executed_control_step_start = group[0].control_step
        pol.executed_control_step_end = group[-1].control_step


def run_episode(
    *,
    env: Any,
    client: FeatureClient,
    initial_state: Any,
    instruction: str,
    cfg: PipelineConfig,
    task_file: str,
    max_steps: int,
    feature_server_version: str,
) -> RolloutRecord:
    eval_mod = getattr(env, "_pipeline_eval", None) or load_eval_module()
    import cv2

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
            obs_payload = eval_mod.make_policy_observation(
                observation, instruction, cfg.policy_image_size, cv2
            )
            result = client.infer(obs_payload)
            actions = result["actions"]
            feats = result["prefix_features"]
            policy_step = len(policies)
            n_take = min(cfg.replan_steps, max_steps - control_step_next, int(actions.shape[0]))
            if n_take < 1:
                raise RuntimeError(f"empty action chunk {actions.shape}")
            start = control_step_next
            end = control_step_next + n_take - 1
            policies.append(
                PolicyRecord(
                    policy_step=policy_step,
                    prefix_features=feats,
                    predicted_action_chunk=np.asarray(actions, dtype=np.float32),
                    executed_control_step_start=start,
                    executed_control_step_end=end,
                )
            )
            for j in range(n_take):
                action_plan.append((policy_step, j, np.asarray(actions[j], dtype=np.float32)))

        policy_step, chunk_index, action = action_plan.popleft()
        observation, _, done, _ = env.step(action.tolist())
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
                executed_action=action,
                video_frame_id=control_step_next if cfg.save_video else None,
                env_success=success,
            )
        )
        control_step_next += 1
        sim_step_next += 1

    finalize_executed_ranges(policies, controls)
    goals = goal_predicate_strings(env)
    if success:
        sim_cat, fail_pred, detail = "success", "", ""
    else:
        sim_cat = "unsatisfied_goal"
        fail_pred = goals[0] if goals else instruction
        detail = "; ".join(goals) if goals else "episode ended without env success"

    rollout = RolloutRecord(
        rollout_id=rollout_id(
            cfg.scene_variant, cfg.task_id, cfg.episode_index, Path(task_file).stem
        ),
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
        model_id=cfg.model_id,
        checkpoint=cfg.checkpoint,
        feature_source=FEATURE_SOURCE,
        feature_server=FEATURE_SERVER_SCRIPT,
        feature_server_version=feature_server_version,
        control_hz=CONTROL_HZ,
        num_steps_wait=cfg.num_steps_wait,
        sim_failure_category=sim_cat,
        failing_predicate=str(fail_pred),
        failure_detail=detail,
        policies=policies,
        controls=controls,
        failure=FailureAnnotation(),
        video_path="",
        semantic_timeline=[],
    )
    rollout._video_frames = video_frames  # type: ignore[attr-defined]
    rollout._wrist_frames = wrist_frames  # type: ignore[attr-defined]
    return rollout
