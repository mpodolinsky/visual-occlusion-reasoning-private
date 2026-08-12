#!/usr/bin/env python3
"""Evaluate pi0.5 on LIBERO-Occ, restricted to the held-out CALIBRATION
episodes (per split.json), overlaying BOTH the probe's learned attention as a
spatial heatmap AND its confidence-gated prediction border on the review
videos.

Forked from eval_pi05_libero_with_probe_calibration.py (calibration-episode
restriction + abstain-gated border) and eval_pi05_libero_with_probe_heatmap.py
(learned attention heatmap) -- combines both rather than picking one:

  1. Only runs episodes in the CALIBRATION split (split.json's "calibration"
     list, written by train_probe.py) -- untouched by training, checkpoint
     selection, or test-set reporting, so they're the right episodes to look
     at when sanity-checking the probe against real rollout footage.

  2. The border follows the same selective-classification rule
     calibrate_probe.py computes risk against: predicted class is fixed
     (success iff P(success) > 0.5), but the border is only colored
     green/red when the winning class's probability is at least
     --probe-abstain-threshold (default 0.6); below that it's gray/ABSTAIN.

  3. The probe's AttentionPool modules (pool_base, pool_wrist in
     probe_model.py) compute a softmax weight per image token as part of
     pooling 196/256 SigLIP patch tokens down to one vector. Those per-token
     weights are recovered here, reshaped back into their 16x16 spatial patch
     grid (SigLIP So400m/14 over a 224px image -> 224/14 = 16 patches/side),
     resized to frame resolution, and alpha-blended as a heatmap -- showing
     WHERE the probe is attending, alongside WHAT it predicted (border) and
     HOW confident it is (whether the border commits to a color at all).

Requires the FEATURE-SERVING server (serve_pi05_with_features.py), not the
stock serve_policy.py -- only the feature server's infer() response includes
"prefix_features", which is what the probe consumes.

Run with the top-level project venv (has robosuite/torch, not JAX/openpi --
this talks to the model over the websocket, it doesn't load it):
    .venv/bin/python scripts/perception_probe/eval_pi05_libero_with_probe_calibration_heatmap.py \
        --suites libero_spatial_occluded
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "submodules" / "Libero-Occ" / "benchmark_assets"
LIBERO_ROOT = REPO_ROOT / "submodules" / "openpi" / "third_party" / "libero"
STANDARD_BENCHMARK_ROOT = LIBERO_ROOT / "libero" / "libero"
SUITES = (
    "libero_spatial_occluded",
    "libero_goal_occluded",
    "libero_object_occluded",
    "libero_10_occluded",
)
MAX_STEPS = {
    "libero_spatial_occluded": 220,
    "libero_object_occluded": 280,
    "libero_goal_occluded": 300,
    "libero_10_occluded": 520,
}
DUMMY_ACTION = [0.0] * 6 + [-1.0]
EPISODE_FIELDS = (
    "suite",
    "scene_variant",
    "task_id",
    "task",
    "prompt",
    "episode",
    "init_state_index",
    "seed",
    "status",
    "success",
    "control_frames",
    "sim_frames",
    "video_frames",
    "inference_calls",
    "elapsed_seconds",
    "video",
    "wrist_video",
    "error",
)
PATCH_GRID_SIZE = 16  # SigLIP So400m/14 over a 224px image -> 224/14 = 16 patches/side

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_model import PerceptionSuccessProbe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Feature-serving policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Feature-serving policy server port.")
    parser.add_argument(
        "--scene-variant",
        choices=("occluded", "normal"),
        default="occluded",
        help="Restricts to this scene variant's calibration episodes (the calibration split "
        "contains both, but one env/benchmark-root setup is used per run).",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITES,
        default=list(SUITES),
        help="Suites to evaluate (defaults to all four); only tasks with at least one "
        "calibration episode are actually run.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "latest" / "split.json",
        help="Written by train_probe.py; defines exactly which episodes are calibration.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument(
        "--policy-image-size",
        type=int,
        default=224,
        help="Square image size sent to the policy.",
    )
    parser.add_argument(
        "--env-resolution",
        type=int,
        default=256,
        help="Simulator render size; 256 matches OpenPI's LIBERO evaluation.",
    )
    parser.add_argument(
        "--video-resolution",
        type=int,
        default=256,
        help="Square resolution of saved review videos. Kept at env-resolution by default "
        "so the overlay text and heatmap detail stay legible after encoding.",
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (defaults according to --scene-variant).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to skip episode keys already present in episodes.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun and overwrite already-recorded episodes instead of skipping them (e.g. after "
        "retraining the probe or changing --probe-checkpoint/--probe-abstain-threshold/"
        "--heatmap-alpha). Also bypasses the --no-resume safety check, since overwriting is "
        "itself an explicit opt-in.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after an episode error instead of stopping for a safe resume.",
    )
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="GPU passed to robosuite; -1 uses its default selection.",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "probe" / "latest" / "probe_best.pt",
        help="Trained PerceptionSuccessProbe weights (see train_probe.py).",
    )
    parser.add_argument(
        "--probe-abstain-threshold",
        type=float,
        default=0.6,
        help="Selective-classification abstention cutoff lambda, matching calibrate_probe.py: "
        "the predicted class is always whichever of P(success), P(failure) is larger, but the "
        "overlay only commits to that color when max(P(success), P(failure)) is at least this "
        "value -- otherwise it shows as an abstention (gray) rather than a colored prediction.",
    )
    parser.add_argument(
        "--probe-device",
        default=None,
        help="Device for the probe (defaults to cuda if available, else cpu).",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.45,
        help="Blend strength of the attention heatmap over the raw frame (0=invisible, 1=opaque).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap the number of calibration episodes actually run (e.g. for a quick smoke test); "
        "unset runs every matching calibration episode. Truncates in split.json's calibration "
        "list order, not randomly.",
    )
    return parser.parse_args()


def configure_libero(benchmark_root: Path = BENCHMARK_ROOT) -> None:
    """Point the bundled LIBERO package at the selected benchmark assets."""
    sys.path.insert(0, str(LIBERO_ROOT))
    config_dir = REPO_ROOT / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": REPO_ROOT / "outputs" / "datasets",
        "assets": benchmark_root / "assets"
        if (benchmark_root / "assets").is_dir()
        else benchmark_root,
    }
    (config_dir / "config.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in config.items()),
        encoding="utf-8",
    )


def make_optional_matplotlib_stub() -> None:
    """Satisfy LIBERO's unused segmentation-only matplotlib import."""
    try:
        import matplotlib.cm  # noqa: F401
    except ModuleNotFoundError:
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []  # type: ignore[attr-defined]
        matplotlib.cm = ModuleType("matplotlib.cm")  # type: ignore[attr-defined]
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.cm"] = matplotlib.cm


def load_initial_states(path: Path, torch: Any) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert LIBERO's xyzw quaternion to a three-vector axis angle."""
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


def resize_rgb(image: np.ndarray, size: int, cv2: Any) -> np.ndarray:
    interpolation = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


class VideoRecorder:
    """Incrementally encode a browser-compatible H.264 MP4 via FFmpeg."""

    def __init__(self, path: Path, resolution: int, fps: float, cv2: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required to save H.264 review videos")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{resolution}x{resolution}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._resolution = resolution
        self._cv2 = cv2
        self.frames = 0

    def add(self, rgb: np.ndarray) -> None:
        frame = resize_rgb(rgb, self._resolution, self._cv2)
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg video input is closed")
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        error = self._process.stderr.read().decode("utf-8", errors="replace")
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}: {error.strip()}")


def rotate_camera_image(observation: dict[str, Any], key: str) -> np.ndarray:
    """Match the 180-degree orientation correction in OpenPI's evaluator."""
    return np.ascontiguousarray(observation[key][::-1, ::-1])


def make_policy_observation(
    observation: dict[str, Any], prompt: str, image_size: int, cv2: Any
) -> dict[str, Any]:
    base_image = resize_rgb(rotate_camera_image(observation, "agentview_image"), image_size, cv2)
    wrist_image = resize_rgb(
        rotate_camera_image(observation, "robot0_eye_in_hand_image"), image_size, cv2
    )
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                observation["robot0_eef_pos"],
                quat_to_axis_angle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
    }


def attention_pool_weights(pool: Any, torch: Any, tokens: Any) -> np.ndarray:
    """Recomputes AttentionPool's per-token softmax weights (same math as
    AttentionPool.forward in probe_model.py) without discarding them -- the
    module itself only returns the pooled vector, so this mirrors its two
    lines (key projection + softmax over the learned query) to recover the
    intermediate attention map instead of modifying probe_model.py."""
    with torch.no_grad():
        keys = pool.key_proj(tokens)  # (1, num_tokens, key_dim)
        scores = keys @ pool.query  # (1, num_tokens)
        weights = torch.softmax(scores, dim=-1)
    return weights.squeeze(0).cpu().numpy()


def run_probe_with_attention(
    probe: Any, torch: Any, device: str, features: dict[str, Any]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Runs the trained probe on one inference call's cached prefix features,
    returning P(success) plus the base/wrist image AttentionPool weights
    (256,) used to build it -- shows not just WHAT the probe predicted but
    WHERE in each image it was looking when it did.

    features has the same {base_image, wrist_image, language, language_mask}
    shape as what collect_features.py caches per step -- a single (unbatched)
    step's worth of prefix tokens, returned by the feature-serving server.
    """
    with torch.no_grad():
        base = torch.from_numpy(np.asarray(features["base_image"], dtype=np.float32)).unsqueeze(0).to(device)
        wrist = torch.from_numpy(np.asarray(features["wrist_image"], dtype=np.float32)).unsqueeze(0).to(device)
        lang = torch.from_numpy(np.asarray(features["language"], dtype=np.float32)).unsqueeze(0).to(device)
        lang_mask = torch.from_numpy(np.asarray(features["language_mask"])).unsqueeze(0).to(device)
        logits = probe(base, wrist, lang, lang_mask)
        prob = float(torch.sigmoid(logits).item())
    base_weights = attention_pool_weights(probe.pool_base, torch, base)
    wrist_weights = attention_pool_weights(probe.pool_wrist, torch, wrist)
    return prob, base_weights, wrist_weights


def overlay_attention_heatmap(
    frame: np.ndarray, weights: np.ndarray | None, cv2: Any, alpha: float
) -> np.ndarray:
    """Reshapes the (256,) AttentionPool weights back into their 16x16 spatial
    patch grid, resizes to frame resolution, and alpha-blends a JET colormap
    over the frame (red/yellow = high attention, blue = low). No-op (returns
    the frame unchanged) if there's no prediction yet."""
    if weights is None:
        return frame
    grid = weights.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)
    grid = grid - grid.min()
    peak = grid.max()
    if peak > 0:
        grid = grid / peak
    grid_u8 = (grid * 255).astype(np.uint8)
    size = frame.shape[0]
    resized = cv2.resize(grid_u8, (size, size), interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    heatmap_rgb = heatmap_bgr[:, :, ::-1]  # frames are RGB; applyColorMap outputs BGR
    return cv2.addWeighted(frame, 1.0 - alpha, heatmap_rgb, alpha, 0)


def overlay_probe_prediction(
    frame: np.ndarray, prob: float | None, abstain_threshold: float, cv2: Any
) -> np.ndarray:
    """Burns the probe's current selective prediction into an RGB frame as
    text plus a colored border, so the prediction is visible even after the
    video is downscaled/compressed.

    Follows the same rule calibrate_probe.py's selective risk is computed
    against: the predicted class is fixed (success iff prob > 0.5, i.e.
    whichever of P(success)=prob, P(failure)=1-prob is larger), but the
    border only commits to green (predict success) or red (predict failure)
    when the winning class's probability -- confidence = max(prob, 1-prob)
    -- is at least abstain_threshold. Below that, the border is gray and
    labeled ABSTAIN: the confidence calibration says this call isn't reliable
    enough to color-code as one class or the other.

    Gray with "PROBE: n/a" is reserved for before the first inference call
    of the episode (prob is None), distinct from an in-episode abstention.
    """
    frame = frame.copy()
    if prob is None:
        label, color = "PROBE: n/a", (160, 160, 160)
    else:
        pred = int(prob > 0.5)
        confidence = max(prob, 1.0 - prob)
        if confidence < abstain_threshold:
            label = f"PROBE: ABSTAIN ({prob:.2f})"
            color = (160, 160, 160)
        else:
            label = f"PROBE: {'SUCCESS' if pred == 1 else 'FAILURE'} ({prob:.2f})"
            color = (0, 220, 0) if pred == 1 else (220, 0, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 4)
    cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def overlay_frame(
    frame: np.ndarray,
    weights: np.ndarray | None,
    prob: float | None,
    abstain_threshold: float,
    cv2: Any,
    alpha: float,
) -> np.ndarray:
    frame = overlay_attention_heatmap(frame, weights, cv2, alpha)
    return overlay_probe_prediction(frame, prob, abstain_threshold, cv2)


def episode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["suite"]), str(row["task"]), int(row["episode"])


def read_episode_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_episode_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def upsert_episode_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = episode_key(row)
    for index, existing in enumerate(rows):
        if episode_key(existing) == key:
            rows[index] = row
            return
    rows.append(row)


def as_bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def summarize_group(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    successes = sum(as_bool(row["success"]) for row in rows)
    errors = sum(str(row.get("status", "")) == "error" for row in rows)
    control_frames = [int(row["control_frames"]) for row in rows]
    successful_frames = [
        int(row["control_frames"]) for row in rows if as_bool(row["success"])
    ]
    return {
        "episodes": len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "errors": errors,
        "success_rate": successes / len(rows) if rows else 0.0,
        "mean_control_frames": float(np.mean(control_frames)) if control_frames else 0.0,
        "median_control_frames": float(np.median(control_frames)) if control_frames else 0.0,
        "mean_frames_to_success": float(np.mean(successful_frames)) if successful_frames else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_summaries(output_dir: Path, episode_rows: list[dict[str, Any]]) -> None:
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    by_suite: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in episode_rows:
        by_task[(str(row["suite"]), str(row["task"]))].append(row)
        by_suite[str(row["suite"])].append(row)

    task_rows = [
        {"suite": suite, "task": task, **summarize_group(rows)}
        for (suite, task), rows in sorted(by_task.items())
    ]
    suite_rows = [
        {"suite": suite, **summarize_group(rows)}
        for suite, rows in sorted(by_suite.items())
    ]
    write_csv(output_dir / "task_summary.csv", task_rows)
    write_csv(output_dir / "suite_summary.csv", suite_rows)
    summary = {
        "overall": summarize_group(episode_rows),
        "suites": {row["suite"]: {k: v for k, v in row.items() if k != "suite"} for row in suite_rows},
        "tasks": task_rows,
    }
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "summary.json")


def benchmark_selection(
    occluded_suite: str, scene_variant: str
) -> tuple[Path, str]:
    if scene_variant == "occluded":
        return BENCHMARK_ROOT, occluded_suite
    if scene_variant == "normal":
        return STANDARD_BENCHMARK_ROOT, occluded_suite.removesuffix("_occluded")
    raise ValueError(f"Unknown scene variant: {scene_variant}")


def task_files(suite: str, benchmark_root: Path = BENCHMARK_ROOT) -> list[Path]:
    paths = sorted((benchmark_root / "bddl_files" / suite).glob("*.bddl"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected 10 BDDL files for {suite}, found {len(paths)}")
    return paths


def load_calibration_keys(
    split_json: Path, scene_variant: str, max_episodes: int | None = None
) -> dict[str, dict[str, list[int]]]:
    """Reads split.json's "calibration" episodes and returns, for the given
    scene_variant only, {occluded_suite_name: {task_stem: [episode_index, ...]}}.

    occluded_suite_name is derived from each task's BDDL file location (via
    the same task-stem -> suite lookup collect_features.py uses to backfill
    manifest.csv's suite column) rather than read off the calibration row
    directly -- split.json's per-episode records predate that fix and don't
    carry a suite field, only scene_variant/task/episode/success.

    max_episodes, if given, caps the number of (suite, task, episode) keys
    returned, taken in split.json's calibration-list order (after filtering
    to scene_variant) -- e.g. for a quick smoke test without running the
    full calibration set.
    """
    if not split_json.is_file():
        raise FileNotFoundError(
            f"{split_json} not found -- run train_probe.py first, it writes split.json alongside "
            "the checkpoint so this script knows exactly which episodes are calibration."
        )
    calibration_rows = json.loads(split_json.read_text())["calibration"]

    task_to_suite: dict[str, str] = {}
    for suite in SUITES:
        for bddl_path in task_files(suite, BENCHMARK_ROOT):
            task_to_suite[bddl_path.stem] = suite

    matching_rows = [row for row in calibration_rows if row["scene_variant"] == scene_variant]
    if max_episodes is not None:
        matching_rows = matching_rows[:max_episodes]

    keys: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in matching_rows:
        suite = task_to_suite.get(row["task"])
        if suite is None:
            raise KeyError(f"Task {row['task']!r} (from {split_json}) matches no known BDDL file")
        keys[suite][row["task"]].append(int(row["episode"]))

    return {suite: {task: sorted(episodes) for task, episodes in tasks.items()} for suite, tasks in keys.items()}


def evaluate_episode(
    *,
    env: Any,
    initial_state: Any,
    prompt: str,
    max_steps: int,
    args: argparse.Namespace,
    client: Any,
    agent_video_path: Path,
    wrist_video_path: Path,
    cv2: Any,
    torch: Any,
    probe: Any,
    probe_device: str,
) -> dict[str, Any]:
    started = time.monotonic()
    agent_recorder = VideoRecorder(
        agent_video_path, args.video_resolution, args.video_fps, cv2
    )
    try:
        wrist_recorder = VideoRecorder(
            wrist_video_path, args.video_resolution, args.video_fps, cv2
        )
    except Exception:
        agent_recorder.close()
        raise
    success = False
    control_frames = 0
    sim_frames = 0
    inference_calls = 0
    error = ""
    probe_prob: float | None = None
    base_weights: np.ndarray | None = None
    wrist_weights: np.ndarray | None = None
    try:
        env.reset()
        client.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            observation, _, success, _ = env.step(DUMMY_ACTION)
            sim_frames += 1

        action_plan: collections.deque[np.ndarray] = collections.deque()
        agent_recorder.add(
            overlay_frame(
                rotate_camera_image(observation, "agentview_image"),
                base_weights, probe_prob, args.probe_abstain_threshold, cv2, args.heatmap_alpha,
            )
        )
        wrist_recorder.add(
            overlay_frame(
                rotate_camera_image(observation, "robot0_eye_in_hand_image"),
                wrist_weights, probe_prob, args.probe_abstain_threshold, cv2, args.heatmap_alpha,
            )
        )
        while not success and control_frames < max_steps:
            if not action_plan:
                result = client.infer(
                    make_policy_observation(observation, prompt, args.policy_image_size, cv2)
                )
                action_chunk = np.asarray(result["actions"])
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than "
                        f"--replan-steps={args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])
                inference_calls += 1
                probe_prob, base_weights, wrist_weights = run_probe_with_attention(
                    probe, torch, probe_device, result["prefix_features"]
                )

            action = np.asarray(action_plan.popleft())
            observation, _, success, _ = env.step(action.tolist())
            control_frames += 1
            sim_frames += 1
            agent_recorder.add(
                overlay_frame(
                    rotate_camera_image(observation, "agentview_image"),
                    base_weights, probe_prob, args.probe_abstain_threshold, cv2, args.heatmap_alpha,
                )
            )
            wrist_recorder.add(
                overlay_frame(
                    rotate_camera_image(observation, "robot0_eye_in_hand_image"),
                    wrist_weights, probe_prob, args.probe_abstain_threshold, cv2, args.heatmap_alpha,
                )
            )
    except Exception as exception:  # Preserve partial videos and make long runs resumable.
        error = f"{type(exception).__name__}: {exception}"
        logging.exception("Episode failed")
    finally:
        agent_recorder.close()
        wrist_recorder.close()

    return {
        "status": "error" if error else "completed",
        "success": bool(success and not error),
        "control_frames": control_frames,
        "sim_frames": sim_frames,
        "video_frames": agent_recorder.frames,
        "inference_calls": inference_calls,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "error": error,
    }


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "replan-steps": args.replan_steps,
        "policy-image-size": args.policy_image_size,
        "env-resolution": args.env_resolution,
        "video-resolution": args.video_resolution,
        "video-fps": args.video_fps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.num_steps_wait < 0:
        raise ValueError(f"Invalid non-positive arguments: {', '.join(invalid)}")
    if not (0.5 <= args.probe_abstain_threshold <= 1.0):
        raise ValueError(
            f"--probe-abstain-threshold must be in [0.5, 1.0] (it's a cutoff on "
            f"max(P(success), P(failure)), which is never below 0.5), got "
            f"{args.probe_abstain_threshold}"
        )
    if not (0.0 <= args.heatmap_alpha <= 1.0):
        raise ValueError("--heatmap-alpha must be in [0, 1]")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not BENCHMARK_ROOT.is_dir() or not STANDARD_BENCHMARK_ROOT.is_dir():
        raise FileNotFoundError("Missing submodules; run `git submodule update --init --recursive`")
    if not args.probe_checkpoint.is_file():
        raise FileNotFoundError(
            f"Probe checkpoint not found: {args.probe_checkpoint} (train one with train_probe.py)"
        )

    calibration_keys = load_calibration_keys(args.split_json, args.scene_variant, args.max_episodes)
    calibration_keys = {
        suite: tasks for suite, tasks in calibration_keys.items() if suite in args.suites
    }
    total_calibration_episodes = sum(
        len(episodes) for tasks in calibration_keys.values() for episodes in tasks.values()
    )
    if total_calibration_episodes == 0:
        raise RuntimeError(
            f"No calibration episodes match --scene-variant={args.scene_variant} and "
            f"--suites={args.suites} in {args.split_json}"
        )

    if args.output_dir is None:
        directory = (
            "pi05_libero_occ_probe_calibration_heatmap"
            if args.scene_variant == "occluded"
            else "pi05_libero_matched_normal_probe_calibration_heatmap"
        )
        args.output_dir = REPO_ROOT / "outputs" / directory

    selected_root, _ = benchmark_selection(next(iter(calibration_keys)), args.scene_variant)
    configure_libero(selected_root)
    make_optional_matplotlib_stub()
    try:
        import cv2
        import torch
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as exception:
        print(f"Missing evaluation dependency ({exception}). Run `uv sync` first.", file=sys.stderr)
        return 2

    probe_device = args.probe_device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe = PerceptionSuccessProbe().to(probe_device)
    probe.load_state_dict(torch.load(args.probe_checkpoint, map_location=probe_device))
    probe.eval()
    logging.info("Loaded probe from %s onto %s", args.probe_checkpoint, probe_device)

    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    existing_rows = read_episode_rows(episodes_path)
    recorded_keys = {episode_key(row) for row in existing_rows}
    completed_keys = (
        set()
        if args.overwrite
        else {
            episode_key(row)
            for row in existing_rows
            if str(row.get("status")) == "completed"
            and bool(row.get("wrist_video"))
            and (output_dir / str(row["wrist_video"])).is_file()
        }
    )
    if recorded_keys and args.no_resume and not args.overwrite:
        raise RuntimeError(
            f"{episodes_path} already contains {len(recorded_keys)} episodes; "
            "choose a new --output-dir, pass --overwrite, or omit --no-resume"
        )

    logging.info("Connecting to feature-serving policy server at %s:%d", args.host, args.port)
    client = WebsocketClientPolicy(args.host, args.port)
    run_config = {
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "policy_config": "pi05_libero",
        "probe_checkpoint": str(args.probe_checkpoint),
        "probe_abstain_threshold": args.probe_abstain_threshold,
        "heatmap_alpha": args.heatmap_alpha,
        "split_json": str(args.split_json),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "server_metadata": client.get_server_metadata(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    logging.info(
        "Evaluating %d calibration-split %s-scene episodes with attention heatmap + "
        "confidence-gated probe overlay (abstain below %.2f); %d already recorded",
        total_calibration_episodes,
        args.scene_variant,
        args.probe_abstain_threshold,
        len(completed_keys),
    )
    for occluded_suite in args.suites:
        if occluded_suite not in calibration_keys:
            continue
        benchmark_root, suite = benchmark_selection(occluded_suite, args.scene_variant)
        for task_id, bddl_path in enumerate(task_files(suite, benchmark_root)):
            episode_indices = calibration_keys[occluded_suite].get(bddl_path.stem)
            if not episode_indices:
                continue

            init_path = benchmark_root / "init_files" / suite / f"{bddl_path.stem}.pruned_init"
            initial_states = load_initial_states(init_path, torch)
            if len(initial_states) <= max(episode_indices):
                raise RuntimeError(
                    f"Calibration split references init state {max(episode_indices)} for "
                    f"{bddl_path.stem}, but only {len(initial_states)} exist in {init_path}"
                )

            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=args.env_resolution,
                camera_widths=args.env_resolution,
                camera_names=["agentview", "robot0_eye_in_hand"],
                render_gpu_device_id=args.render_gpu_device_id,
            )
            try:
                env.seed(args.seed)
                prompt = str(env.language_instruction)
                for episode in episode_indices:
                    key = (suite, bddl_path.stem, episode)
                    if key in completed_keys:
                        logging.info("Skipping recorded episode %s/%s/%03d", suite, bddl_path.stem, episode)
                        continue

                    relative_video = (
                        Path("videos")
                        / suite
                        / f"{task_id + 1:02d}_{bddl_path.stem}"
                        / f"episode_{episode:03d}.mp4"
                    )
                    relative_wrist_video = relative_video.with_name(
                        f"episode_{episode:03d}_wrist.mp4"
                    )
                    logging.info("Running %s/%s episode %d", suite, bddl_path.stem, episode)
                    metrics = evaluate_episode(
                        env=env,
                        initial_state=initial_states[episode],
                        prompt=prompt,
                        max_steps=MAX_STEPS[occluded_suite],
                        args=args,
                        client=client,
                        agent_video_path=output_dir / relative_video,
                        wrist_video_path=output_dir / relative_wrist_video,
                        cv2=cv2,
                        torch=torch,
                        probe=probe,
                        probe_device=probe_device,
                    )
                    row = {
                        "suite": suite,
                        "scene_variant": args.scene_variant,
                        "task_id": task_id,
                        "task": bddl_path.stem,
                        "prompt": prompt,
                        "episode": episode,
                        "init_state_index": episode,
                        "seed": args.seed,
                        **metrics,
                        "video": str(relative_video),
                        "wrist_video": str(relative_wrist_video),
                    }
                    upsert_episode_row(existing_rows, row)
                    write_episode_rows(episodes_path, existing_rows)
                    if metrics["status"] == "completed":
                        completed_keys.add(key)
                    write_summaries(output_dir, existing_rows)
                    logging.info(
                        "Episode result: success=%s, control_frames=%d, aggregate_success=%.2f%%",
                        metrics["success"],
                        metrics["control_frames"],
                        100.0 * summarize_group(existing_rows)["success_rate"],
                    )
                    if metrics["status"] == "error" and not args.continue_on_error:
                        raise RuntimeError(
                            "Stopping after an episode error. Fix the cause and rerun the same "
                            "command; completed episodes will be skipped and this episode retried."
                        )
            finally:
                env.close()

    write_summaries(output_dir, existing_rows)
    overall = summarize_group(existing_rows)
    logging.info(
        "Finished: %d/%d successes (%.2f%%); results in %s",
        overall["successes"],
        overall["episodes"],
        100.0 * overall["success_rate"],
        output_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
