"""Save / load one GR00T rollout. No silent truncation.

Two files per episode, written once by collection and never rewritten:

- ``rollout.json``  -- flat metadata + per-policy clock summary.
- ``rollout.npz``   -- executed actions, predicted action chunks, and the four
  0-based clocks (control / sim / policy / chunk). This is intentionally kept
  even though the eval only needs success/steps: it lets GR00T backbone
  features be joined in later, row by row, without recollecting.

Videos (``rollout.mp4`` / ``wrist.mp4``) are written by :mod:`recorder`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from constants import (
    CONTROL_HZ,
    FEATURE_MODULE,
    FEATURE_SOURCE,
    GROOT_HIDDEN,
    GROOT_IMG_TOKENS,
)
from records import ControlRecord, GrootFeatures, PolicyRecord, RolloutRecord


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _save_npz_atomic(npz_path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = npz_path.with_name(npz_path.stem + ".writing.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(npz_path)


def rollout_meta(rollout: RolloutRecord) -> dict:
    return {
        "rollout_id": rollout.rollout_id,
        "suite": rollout.suite,
        "scene_variant": rollout.scene_variant,
        "task_id": rollout.task_id,
        "task_file": rollout.task_file,
        "episode_index": rollout.episode_index,
        "instruction": rollout.instruction,
        "replan_steps": rollout.replan_steps,
        "success": rollout.success,
        "max_steps": rollout.max_steps,
        "seed": rollout.seed,
        "model_id": rollout.model_id,
        "checkpoint": rollout.checkpoint,
        "feature_source": rollout.feature_source,
        "feature_server": rollout.feature_server,
        "feature_server_version": rollout.feature_server_version,
        "control_hz": rollout.control_hz,
        "num_steps_wait": rollout.num_steps_wait,
        "sim_failure_category": rollout.sim_failure_category,
        "failing_predicate": rollout.failing_predicate,
        "failure_detail": rollout.failure_detail,
        "elapsed_seconds": rollout.elapsed_seconds,
        "video_path": rollout.video_path,
        "wrist_video_path": rollout.wrist_video_path,
        "n_policy": rollout.n_policy,
        "n_control": rollout.n_control,
        "has_features": bool(rollout.policies) and all(p.features is not None for p in rollout.policies),
        "policies": [_policy_meta(p) for p in rollout.policies],
    }


def _policy_meta(p: PolicyRecord) -> dict:
    d = {
        "policy_step": p.policy_step,
        "executed_control_step_start": p.executed_control_step_start,
        "executed_control_step_end": p.executed_control_step_end,
        "predicted_chunk_length": int(p.predicted_action_chunk.shape[0]),
    }
    if p.features is None:
        d.update(feature_source=None, feature_module=None, feature_shapes={}, language_len=None)
    else:
        f = p.features
        d.update(
            feature_source=f.source or FEATURE_SOURCE,
            feature_module=f.module or FEATURE_MODULE,
            feature_shapes={k: list(v) for k, v in f.shapes.items()},
            language_len=int(f.language_len),
        )
    return d


def save_rollout(
    directory: Path,
    rollout: RolloutRecord,
    *,
    json_name: str = "rollout.json",
    npz_name: str = "rollout.npz",
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / json_name
    npz_path = directory / npz_name
    _write_json_atomic(meta_path, rollout_meta(rollout))

    n_pol = rollout.n_policy
    horizon = max((p.predicted_action_chunk.shape[0] for p in rollout.policies), default=0)
    act_dim = int(rollout.controls[0].executed_action.shape[0]) if rollout.controls else 7

    chunks = np.zeros((n_pol, horizon, act_dim), dtype=np.float32)
    chunk_len = np.zeros((n_pol,), dtype=np.int32)
    for i, p in enumerate(rollout.policies):
        h = p.predicted_action_chunk.shape[0]
        chunks[i, :h] = p.predicted_action_chunk.astype(np.float32)
        chunk_len[i] = h

    executed = (
        np.stack([c.executed_action for c in rollout.controls]).astype(np.float32)
        if rollout.controls
        else np.zeros((0, act_dim), dtype=np.float32)
    )

    has_features = bool(rollout.policies) and all(p.features is not None for p in rollout.policies)
    feature_payload: dict[str, np.ndarray] = {}
    if has_features:
        feats = [p.features for p in rollout.policies]
        feature_payload = {
            "base_image": np.stack([f.base_image for f in feats]).astype(np.float16),
            "wrist_image": np.stack([f.wrist_image for f in feats]).astype(np.float16),
            "language": np.stack([f.language for f in feats]).astype(np.float16),
            "language_mask": np.stack([f.language_mask for f in feats]).astype(np.bool_),
            "language_len": np.asarray([f.language_len for f in feats], dtype=np.int32),
            "state_features": np.stack([f.state_features for f in feats]).astype(np.float16),
        }

    payload = {
        "predicted_action_chunks": chunks,
        "predicted_chunk_len": chunk_len,
        "executed_actions": executed,
        "control_step": np.asarray([c.control_step for c in rollout.controls], dtype=np.int32),
        "sim_step": np.asarray([c.sim_step for c in rollout.controls], dtype=np.int32),
        "policy_step": np.asarray([c.policy_step for c in rollout.controls], dtype=np.int32),
        "chunk_index": np.asarray([c.chunk_index for c in rollout.controls], dtype=np.int32),
        "video_frame_id": np.asarray(
            [-1 if c.video_frame_id is None else c.video_frame_id for c in rollout.controls],
            dtype=np.int32,
        ),
        "success": np.bool_(rollout.success),
        "replan_steps": np.int32(rollout.replan_steps),
        "n_policy": np.int32(n_pol),
        "n_control": np.int32(rollout.n_control),
        "img_tokens": np.int32(GROOT_IMG_TOKENS),
        "hidden": np.int32(GROOT_HIDDEN),
        "control_hz": np.float32(rollout.control_hz or CONTROL_HZ),
        "has_features": np.bool_(has_features),
    }
    payload.update(feature_payload)
    _save_npz_atomic(npz_path, payload)
    return meta_path, npz_path


def load_rollout(
    directory: Path,
    *,
    json_name: str = "rollout.json",
    npz_name: str = "rollout.npz",
) -> RolloutRecord:
    meta = json.loads((directory / json_name).read_text(encoding="utf-8"))
    with np.load(directory / npz_name, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    n_ctrl = int(arrays["n_control"])
    if arrays["executed_actions"].shape[0] != n_ctrl:
        raise ValueError("npz executed_actions length != n_control (refusing to truncate)")

    has_features = bool(arrays.get("has_features", np.bool_(False))) and "base_image" in arrays

    policies: list[PolicyRecord] = []
    for pmeta in meta["policies"]:
        i = int(pmeta["policy_step"])
        clen = int(arrays["predicted_chunk_len"][i])
        features = None
        if has_features:
            features = GrootFeatures(
                base_image=np.asarray(arrays["base_image"][i]),
                wrist_image=np.asarray(arrays["wrist_image"][i]),
                language=np.asarray(arrays["language"][i]),
                language_mask=np.asarray(arrays["language_mask"][i]),
                language_len=int(arrays["language_len"][i]),
                state_features=np.asarray(arrays["state_features"][i]),
                source=pmeta.get("feature_source") or FEATURE_SOURCE,
                module=pmeta.get("feature_module") or FEATURE_MODULE,
                shapes={k: tuple(v) for k, v in (pmeta.get("feature_shapes") or {}).items()},
            )
        policies.append(
            PolicyRecord(
                policy_step=i,
                predicted_action_chunk=np.asarray(arrays["predicted_action_chunks"][i, :clen]),
                executed_control_step_start=int(pmeta["executed_control_step_start"]),
                executed_control_step_end=int(pmeta["executed_control_step_end"]),
                features=features,
            )
        )

    controls: list[ControlRecord] = []
    for i in range(n_ctrl):
        vid = int(arrays["video_frame_id"][i])
        controls.append(
            ControlRecord(
                control_step=int(arrays["control_step"][i]),
                sim_step=int(arrays["sim_step"][i]),
                policy_step=int(arrays["policy_step"][i]),
                chunk_index=int(arrays["chunk_index"][i]),
                executed_action=np.asarray(arrays["executed_actions"][i]),
                video_frame_id=None if vid < 0 else vid,
            )
        )

    return RolloutRecord(
        rollout_id=meta["rollout_id"],
        suite=meta.get("suite", ""),
        scene_variant=meta.get("scene_variant", ""),
        task_id=int(meta["task_id"]),
        task_file=meta["task_file"],
        episode_index=int(meta["episode_index"]),
        instruction=meta["instruction"],
        replan_steps=int(meta["replan_steps"]),
        success=bool(meta["success"]),
        max_steps=int(meta["max_steps"]),
        seed=int(meta["seed"]),
        model_id=meta.get("model_id", ""),
        checkpoint=meta.get("checkpoint", ""),
        feature_source=meta.get("feature_source") or FEATURE_SOURCE,
        feature_server=meta.get("feature_server") or "",
        feature_server_version=meta.get("feature_server_version") or meta.get("server_metadata") or "",
        control_hz=float(meta.get("control_hz") or CONTROL_HZ),
        num_steps_wait=int(meta.get("num_steps_wait") or 0),
        sim_failure_category=meta.get("sim_failure_category") or "",
        failing_predicate=meta.get("failing_predicate") or "",
        failure_detail=meta.get("failure_detail") or "",
        elapsed_seconds=float(meta.get("elapsed_seconds") or 0.0),
        policies=policies,
        controls=controls,
        video_path=meta.get("video_path") or "",
        wrist_video_path=meta.get("wrist_video_path") or "",
    )
