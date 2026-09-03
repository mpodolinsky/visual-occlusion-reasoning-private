"""Save/load rollout. No silent truncation.

Collection writes three frozen files, never rewritten afterwards:

- ``rollout.json``  -- metadata + per-policy/per-control clock records.
- ``rollout.npz``   -- the heavy feature / action / clock arrays.
- ``rollout.mp4``   -- 20 fps, one frame per control step.

The labeler writes two small files and touches nothing above:

- ``labels.json``   -- the Gemini backend + model + prompt templates used, plus
  ``semantic_timeline`` (3-second phrases), ``vlm_failure`` (Dan's dict), and
  ``failure_annotation`` (onset mapped onto control/policy/chunk).
- ``labels.npz``    -- the same phrases / failure fields as arrays, for
  array-aligned downstream loading.

So a labeled dataset differs from an unlabeled one by only a few KB per episode
-- cheap to diff, sync, and re-upload.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from records import (
    ControlRecord,
    FailureAnnotation,
    PolicyRecord,
    PrefixFeatures,
    RolloutRecord,
    SemanticSegment,
)
from constants import CONTROL_HZ, FEATURE_MODULE, FEATURE_SOURCE

_FAILURE_ANNOTATION_FIELDS = (
    "failure_control_step",
    "failure_sim_step",
    "failure_policy_step",
    "failure_chunk_index",
    "first_post_failure_policy_step",
    "failure_type",
    "correction_action",
)


def rollout_id(scene_variant: str, task_id: int, episode_index: int, bddl_stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in bddl_stem)
    safe = safe[:80].rstrip("_")
    return f"{scene_variant}__t{task_id:03d}__ep{episode_index:03d}__{safe}"


def _segment_dict(s: SemanticSegment) -> dict:
    phrase = s.phrase or ""
    return {
        "segment_index": s.segment_index,
        "t_start_sec": s.t_start_sec,
        "t_end_sec": s.t_end_sec,
        "control_step_start": s.control_step_start,
        "control_step_end": s.control_step_end,
        "policy_step_start": s.policy_step_start,
        "policy_step_end": s.policy_step_end,
        "phrase": phrase,
        "description": phrase,
    }


def _failure_annotation_dict(fa: FailureAnnotation) -> dict:
    return {k: getattr(fa, k) for k in _FAILURE_ANNOTATION_FIELDS}


def _u(text: str | None, dtype: str) -> np.ndarray:
    return np.asarray(str(text or ""), dtype=dtype)


def _label_npz_arrays(rollout: RolloutRecord) -> dict[str, np.ndarray]:
    """Dan failure and 3s phrases as arrays. Their own file; never merged with features."""
    segs = rollout.semantic_timeline
    n = len(segs)
    sem = {
        "sem_t_start": np.asarray([s.t_start_sec for s in segs], dtype=np.float32),
        "sem_t_end": np.asarray([s.t_end_sec for s in segs], dtype=np.float32),
        "sem_control_start": np.asarray([s.control_step_start for s in segs], dtype=np.int32),
        "sem_control_end": np.asarray([s.control_step_end for s in segs], dtype=np.int32),
        "sem_phrase": np.asarray([s.phrase for s in segs], dtype="U256")
        if n
        else np.zeros((0,), dtype="U256"),
    }
    v = rollout.vlm_failure or {}
    onset = v.get("vlm_failure_onset_frame")
    seconds = v.get("vlm_failure_onset_seconds")
    fail = {
        "fail_onset_frame": np.int32(-1 if onset is None else int(onset)),
        "fail_onset_seconds": np.float32(np.nan if seconds is None else float(seconds)),
        "fail_mode": _u(v.get("vlm_failure_mode"), "U128"),
        "fail_reason": _u(v.get("vlm_failure_reason"), "U512"),
        "fail_recovery": _u(v.get("vlm_recovery_action"), "U512"),
    }
    return {**sem, **fail}


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _save_npz_atomic(npz_path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = npz_path.with_name(npz_path.stem + ".writing.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(npz_path)


# --------------------------------------------------------------------------- #
# rollout.json / rollout.npz  (collection; frozen)
# --------------------------------------------------------------------------- #

def rollout_meta(rollout: RolloutRecord) -> dict:
    """Everything the simulator + policy produced. No Gemini fields -- those
    live in ``labels.json`` so this file is written once and never touched."""
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
        "video_path": rollout.video_path,
        "wrist_video_path": rollout.wrist_video_path,
        "n_policy": rollout.n_policy,
        "n_control": rollout.n_control,
        "policies": [
            {
                "policy_step": p.policy_step,
                "executed_control_step_start": p.executed_control_step_start,
                "executed_control_step_end": p.executed_control_step_end,
                "predicted_chunk_length": int(p.predicted_action_chunk.shape[0]),
                "feature_source": p.prefix_features.source,
                "feature_module": p.prefix_features.module,
                "feature_shapes": {k: list(v) for k, v in p.prefix_features.shapes.items()},
            }
            for p in rollout.policies
        ],
    }


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
    n_ctrl = rollout.n_control
    lang_len = max(p.prefix_features.language.shape[0] for p in rollout.policies)
    hidden = rollout.policies[0].prefix_features.base_image.shape[-1]
    img_tok = rollout.policies[0].prefix_features.base_image.shape[0]
    act_dim = int(rollout.controls[0].executed_action.shape[0])
    horizon = max(p.predicted_action_chunk.shape[0] for p in rollout.policies)

    base = np.stack([p.prefix_features.base_image for p in rollout.policies]).astype(np.float16)
    wrist = np.stack([p.prefix_features.wrist_image for p in rollout.policies]).astype(np.float16)
    language = np.zeros((n_pol, lang_len, hidden), dtype=np.float16)
    language_mask = np.zeros((n_pol, lang_len), dtype=np.bool_)
    chunks = np.zeros((n_pol, horizon, act_dim), dtype=np.float32)
    chunk_len = np.zeros((n_pol,), dtype=np.int32)
    for i, p in enumerate(rollout.policies):
        L = p.prefix_features.language.shape[0]
        language[i, :L] = p.prefix_features.language.astype(np.float16)
        language_mask[i, :L] = p.prefix_features.language_mask
        h = p.predicted_action_chunk.shape[0]
        chunks[i, :h] = p.predicted_action_chunk.astype(np.float32)
        chunk_len[i] = h

    executed = np.stack([c.executed_action for c in rollout.controls]).astype(np.float32)
    payload = {
        "base_image": base,
        "wrist_image": wrist,
        "language": language,
        "language_mask": language_mask,
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
        "n_control": np.int32(n_ctrl),
        "img_tokens": np.int32(img_tok),
        "hidden": np.int32(hidden),
        "control_hz": np.float32(rollout.control_hz or CONTROL_HZ),
    }
    _save_npz_atomic(npz_path, payload)
    return meta_path, npz_path


# --------------------------------------------------------------------------- #
# labels.json / labels.npz  (labeling; additive)
# --------------------------------------------------------------------------- #

def label_document(rollout: RolloutRecord, labeler: dict) -> dict:
    """The standalone Gemini-labeling record. ``labeler`` is the backend / model
    / prompt-template block from :func:`episode.build_labeler_meta`."""
    return {
        "rollout_id": rollout.rollout_id,
        "scene_variant": rollout.scene_variant,
        "task_id": rollout.task_id,
        "task_file": rollout.task_file,
        "instruction": rollout.instruction,
        "success": rollout.success,
        "labeler": labeler,
        "semantic_timeline": [_segment_dict(s) for s in rollout.semantic_timeline],
        "vlm_failure": rollout.vlm_failure,
        "failure_annotation": _failure_annotation_dict(rollout.failure),
    }


def save_labels(
    directory: Path,
    rollout: RolloutRecord,
    labeler: dict,
    *,
    labels_json: str = "labels.json",
    labels_npz: str = "labels.npz",
) -> tuple[Path, Path]:
    """Write ``labels.json`` + ``labels.npz``. Does not touch ``rollout.*``."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / labels_json
    npz_path = directory / labels_npz
    _write_json_atomic(json_path, label_document(rollout, labeler))
    _save_npz_atomic(npz_path, _label_npz_arrays(rollout))
    return json_path, npz_path


def load_label_document(directory: Path, *, labels_json: str = "labels.json") -> dict | None:
    """Parsed ``labels.json`` (incl. the ``labeler`` block), or ``None``."""
    path = directory / labels_json
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(directory: Path, *, labels_npz: str = "labels.npz") -> dict[str, np.ndarray] | None:
    """The ``sem_*`` / ``fail_*`` arrays, or ``None`` if the episode is unlabeled."""
    path = directory / labels_npz
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _timeline_from_dicts(rows: list[dict]) -> list[SemanticSegment]:
    return [
        SemanticSegment(
            segment_index=int(s["segment_index"]),
            t_start_sec=float(s["t_start_sec"]),
            t_end_sec=float(s["t_end_sec"]),
            control_step_start=int(s["control_step_start"]),
            control_step_end=int(s["control_step_end"]),
            policy_step_start=int(s["policy_step_start"]),
            policy_step_end=int(s["policy_step_end"]),
            phrase=str(s.get("phrase") or s.get("description") or ""),
        )
        for s in rows or []
    ]


def _apply_label_document(record: RolloutRecord, doc: dict) -> None:
    record.semantic_timeline = _timeline_from_dicts(doc.get("semantic_timeline") or [])
    record.vlm_failure = dict(doc.get("vlm_failure") or {})
    fa = doc.get("failure_annotation") or {}
    record.failure = FailureAnnotation(**{k: fa.get(k) for k in _FAILURE_ANNOTATION_FIELDS})


# --------------------------------------------------------------------------- #

def load_rollout(
    directory: Path,
    *,
    json_name: str = "rollout.json",
    npz_name: str = "rollout.npz",
    labels_json: str = "labels.json",
) -> RolloutRecord:
    meta = json.loads((directory / json_name).read_text(encoding="utf-8"))
    with np.load(directory / npz_name, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    n_pol = int(arrays["n_policy"])
    n_ctrl = int(arrays["n_control"])
    if arrays["executed_actions"].shape[0] != n_ctrl:
        raise ValueError("npz executed_actions length != n_control (refusing to truncate)")
    if arrays["base_image"].shape[0] != n_pol:
        raise ValueError("npz base_image length != n_policy (refusing to truncate)")

    policies: list[PolicyRecord] = []
    for i, pmeta in enumerate(meta["policies"]):
        feats = PrefixFeatures(
            base_image=np.asarray(arrays["base_image"][i]),
            wrist_image=np.asarray(arrays["wrist_image"][i]),
            language=np.asarray(arrays["language"][i]),
            language_mask=np.asarray(arrays["language_mask"][i]),
            source=pmeta.get("feature_source", FEATURE_SOURCE),
            module=pmeta.get("feature_module", FEATURE_MODULE),
            shapes={k: tuple(v) for k, v in pmeta.get("feature_shapes", {}).items()},
        )
        clen = int(arrays["predicted_chunk_len"][i])
        chunk = np.asarray(arrays["predicted_action_chunks"][i, :clen])
        policies.append(
            PolicyRecord(
                policy_step=int(pmeta["policy_step"]),
                prefix_features=feats,
                predicted_action_chunk=chunk,
                executed_control_step_start=int(pmeta["executed_control_step_start"]),
                executed_control_step_end=int(pmeta["executed_control_step_end"]),
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

    record = RolloutRecord(
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
        model_id=meta["model_id"],
        checkpoint=meta["checkpoint"],
        feature_source=meta["feature_source"],
        feature_server=meta["feature_server"],
        feature_server_version=meta.get("feature_server_version", ""),
        control_hz=float(meta.get("control_hz") or CONTROL_HZ),
        num_steps_wait=int(meta.get("num_steps_wait") or 0),
        sim_failure_category=meta.get("sim_failure_category") or "",
        failing_predicate=meta.get("failing_predicate") or "",
        failure_detail=meta.get("failure_detail") or "",
        policies=policies,
        controls=controls,
        failure=FailureAnnotation(),
        video_path=meta.get("video_path") or "",
        wrist_video_path=meta.get("wrist_video_path") or "",
        semantic_timeline=[],
        vlm_failure={},
    )

    doc = load_label_document(directory, labels_json=labels_json)
    if doc is not None:
        _apply_label_document(record, doc)
    return record
