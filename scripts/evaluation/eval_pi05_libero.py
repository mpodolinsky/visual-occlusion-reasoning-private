#!/usr/bin/env python3
"""Evaluate an OpenPI policy server on every released LIBERO-Occ task."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="OpenPI policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="OpenPI policy server port.")
    parser.add_argument(
        "--scene-variant",
        choices=("occluded", "normal"),
        default="occluded",
        help="Use LIBERO-Occ scenes or their index-matched standard LIBERO counterparts.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITES,
        default=list(SUITES),
        help="Suites to evaluate (defaults to all four).",
    )
    parser.add_argument(
        "--num-trials-per-task",
        type=int,
        default=50,
        help="Initial states evaluated per task; 50 uses every released state.",
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
        default=128,
        help="Square resolution of saved review videos.",
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
    try:
        env.reset()
        client.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            observation, _, success, _ = env.step(DUMMY_ACTION)
            sim_frames += 1

        action_plan: collections.deque[np.ndarray] = collections.deque()
        agent_recorder.add(rotate_camera_image(observation, "agentview_image"))
        wrist_recorder.add(rotate_camera_image(observation, "robot0_eye_in_hand_image"))
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

            action = np.asarray(action_plan.popleft())
            observation, _, success, _ = env.step(action.tolist())
            control_frames += 1
            sim_frames += 1
            agent_recorder.add(rotate_camera_image(observation, "agentview_image"))
            wrist_recorder.add(rotate_camera_image(observation, "robot0_eye_in_hand_image"))
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
        "num-trials-per-task": args.num_trials_per_task,
        "replan-steps": args.replan_steps,
        "policy-image-size": args.policy_image_size,
        "env-resolution": args.env_resolution,
        "video-resolution": args.video_resolution,
        "video-fps": args.video_fps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid or args.num_steps_wait < 0:
        raise ValueError(f"Invalid non-positive arguments: {', '.join(invalid)}")
    if args.num_trials_per_task > 50:
        raise ValueError("The released LIBERO-Occ tasks contain 50 initial states each")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not BENCHMARK_ROOT.is_dir() or not STANDARD_BENCHMARK_ROOT.is_dir():
        raise FileNotFoundError("Missing submodules; run `git submodule update --init --recursive`")

    if args.output_dir is None:
        directory = (
            "pi05_libero_occ"
            if args.scene_variant == "occluded"
            else "pi05_libero_matched_normal"
        )
        args.output_dir = REPO_ROOT / "outputs" / directory

    selected_root, _ = benchmark_selection(args.suites[0], args.scene_variant)
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

    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    existing_rows = read_episode_rows(episodes_path)
    recorded_keys = {episode_key(row) for row in existing_rows}
    completed_keys = {
        episode_key(row)
        for row in existing_rows
        if str(row.get("status")) == "completed"
        and bool(row.get("wrist_video"))
        and (output_dir / str(row["wrist_video"])).is_file()
    }
    if recorded_keys and args.no_resume:
        raise RuntimeError(
            f"{episodes_path} already contains {len(recorded_keys)} episodes; "
            "choose a new --output-dir or omit --no-resume"
        )

    logging.info("Connecting to OpenPI policy server at %s:%d", args.host, args.port)
    client = WebsocketClientPolicy(args.host, args.port)
    run_config = {
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "policy_config": "pi05_libero",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "server_metadata": client.get_server_metadata(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    total_requested = len(args.suites) * 10 * args.num_trials_per_task
    logging.info(
        "Evaluating %d %s-scene episodes; %d already recorded",
        total_requested,
        args.scene_variant,
        len(completed_keys),
    )
    for occluded_suite in args.suites:
        benchmark_root, suite = benchmark_selection(occluded_suite, args.scene_variant)
        for task_id, bddl_path in enumerate(task_files(suite, benchmark_root)):
            init_path = benchmark_root / "init_files" / suite / f"{bddl_path.stem}.pruned_init"
            initial_states = load_initial_states(init_path, torch)
            if len(initial_states) < args.num_trials_per_task:
                raise RuntimeError(f"Only {len(initial_states)} states found in {init_path}")

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
                for episode in range(args.num_trials_per_task):
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
