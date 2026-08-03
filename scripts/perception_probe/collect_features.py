#!/usr/bin/env python3
"""Roll out pi05_libero on a single task (both scene variants) against the
feature-serving websocket server, caching the raw per-token prefix features
from every inference call for later probe training.

Run with the top-level project venv (has robosuite/mujoco/openpi-client, not
JAX/openpi -- this talks to the model over the websocket, it doesn't load it):
    .venv/bin/python scripts/perception_probe/collect_features.py \
        --suite libero_spatial_occluded --task-id 0 --num-trials 20

Or omit --suite/--task-id to auto-detect the next suite/task under --output-dir
that doesn't yet have --num-trials episodes for both scene variants, and just
keep running the same command to work through whatever's left:
    .venv/bin/python scripts/perception_probe/collect_features.py \
        --output-dir outputs/perception_probe/features_protect --num-trials 25
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
from pathlib import Path
import sys

import numpy as np
import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
import eval_pi05_libero as E  # noqa: E402

MANIFEST_FIELDS = (
    "scene_variant",
    "suite",
    "task",
    "prompt",
    "episode",
    "success",
    "control_frames",
    "inference_calls",
    "npz_path",
)

# LIBERO runs as 20Hz (50ms/step)
# pi0.5 predicts 10 actions per chunk -- the eval script uses 5 and discard the rest, calling the policy again
# Meaning that each call produces 0.25s of actions.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--suite",
        default=None,
        choices=list(E.SUITES),
        help="Occluded-suite name; the matched normal-scene suite is derived automatically. "
        "Omit together with --task-id to auto-detect the next suite/task that doesn't yet "
        "have --num-trials episodes collected for both scene variants under --output-dir.",
    )    # Name of the occluded suite name (e.g. libero_spatial_occluded), the script will roll out the regular and occluded suits
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="0-indexed task within the suite (10 per suite). Omit together with --suite to "
        "auto-detect.",
    )   # 0-indexed task id (10 tasks per suite)
    parser.add_argument("--num-trials", type=int, default=20, help="Initial states per scene variant (<= 50).")     # Number of distinc initial states to sample per task (x2 since we do it for both occluded and non-occluded variants)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)   # Number of no-op steps taken before the policy takes over to let physics settle
    parser.add_argument("--replan-steps", type=int, default=5)      # how many actions from each predicted action chunk get executed before replanning
    parser.add_argument("--policy-image-size", type=int, default=224) # Resolution of images before being sent to the policy
    parser.add_argument("--env-resolution", type=int, default=256)    # Resolution LIBERO renders the frames (before resizing to for the policy)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features"
    )
    parser.add_argument("--wandb-project", default="pi05-perception-probe")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    return parser.parse_args()


def build_task_suite_map() -> dict[str, str]:
    """task stem -> occluded suite name, from the BDDL files each suite actually contains.
    Task stems are unique across all 4 suites and shared between the occluded/normal scene
    variants of the same task, so one pass over the occluded suites' BDDL files covers both."""
    mapping: dict[str, str] = {}
    for suite in E.SUITES:
        for bddl_path in E.task_files(suite, E.BENCHMARK_ROOT):
            mapping[bddl_path.stem] = suite
    return mapping


def find_next_incomplete_task(output_dir: Path, num_trials: int) -> tuple[str, int] | None:
    """Returns the first (suite, task_id) that doesn't yet have num_trials episodes cached
    for BOTH scene variants under output_dir, in the same suite/task order collect_all.sh
    uses -- lets a bare invocation just pick up wherever collection left off, without having
    to track by hand which (suite, task-id) pairs are already done. Returns None once every
    suite/task combination is complete."""
    for suite in E.SUITES:
        for task_id in range(10):
            complete = True
            for scene_variant in ("occluded", "normal"):
                benchmark_root, selected_suite = E.benchmark_selection(suite, scene_variant)
                bddl_path = E.task_files(selected_suite, benchmark_root)[task_id]
                episode_dir = output_dir / scene_variant / f"{task_id + 1:02d}_{bddl_path.stem}"
                existing = len(list(episode_dir.glob("episode_*.npz"))) if episode_dir.is_dir() else 0
                if existing < num_trials:
                    complete = False
                    break
            if not complete:
                return suite, task_id
    return None


def rebuild_manifest(output_dir: Path) -> list[dict]:
    """Scans every cached episode_*.npz under output_dir and rebuilds manifest.csv from
    scratch. Called after each episode so an interrupted run still leaves a valid,
    complete manifest for whatever data actually made it to disk."""
    task_suite_map = build_task_suite_map()
    rows = []
    for npz_path in sorted(output_dir.glob("*/*/episode_*.npz")):
        scene_variant = npz_path.parent.parent.name
        task = npz_path.parent.name.split("_", 1)[1]
        episode = int(npz_path.stem.split("_")[1])
        with np.load(npz_path) as data:
            rows.append(
                {
                    "scene_variant": scene_variant,
                    "suite": task_suite_map.get(task, ""),
                    "task": task,
                    "prompt": "",
                    "episode": episode,
                    "success": bool(data["success"]),
                    "control_frames": "",
                    "inference_calls": int(data["base_image"].shape[0]),
                    "npz_path": str(npz_path.relative_to(output_dir)),
                }
            )
    manifest_path = output_dir / "manifest.csv"
    temp_path = manifest_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(manifest_path)
    return rows


def rollout_episode(
    *, env, initial_state, prompt, max_steps, args, client, cv2
) -> tuple[dict, dict]:
    """Returns (metrics, per_step_features). per_step_features arrays have shape
    (T, ...) stacked across every inference call in the episode."""
    env.reset()
    client.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(args.num_steps_wait):
        observation, _, success, _ = env.step([0.0] * 6 + [-1.0])

    action_plan: collections.deque = collections.deque()
    steps = {"base_image": [], "wrist_image": [], "language": [], "language_mask": []}
    control_frames = 0
    inference_calls = 0
    while not success and control_frames < max_steps:
        if not action_plan:
            result = client.infer(
                E.make_policy_observation(observation, prompt, args.policy_image_size, cv2)
            )
            action_chunk = np.asarray(result["actions"])
            action_plan.extend(action_chunk[: args.replan_steps])
            feats = result["prefix_features"]
            for key in steps:
                steps[key].append(np.asarray(feats[key]))
            inference_calls += 1

        action = np.asarray(action_plan.popleft())
        observation, _, success, _ = env.step(action.tolist())
        control_frames += 1

    per_step = {key: np.stack(values, axis=0) for key, values in steps.items()}
    metrics = {
        "success": bool(success),
        "control_frames": control_frames,
        "inference_calls": inference_calls,
    }
    return metrics, per_step


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if not E.BENCHMARK_ROOT.is_dir() or not E.STANDARD_BENCHMARK_ROOT.is_dir():
        raise FileNotFoundError("Missing submodules; run `git submodule update --init --recursive`")

    if (args.suite is None) != (args.task_id is None):
        raise ValueError("Pass both --suite and --task-id, or neither (to auto-detect).")

    output_dir = args.output_dir.resolve()
    if args.suite is None:
        detected = find_next_incomplete_task(output_dir, args.num_trials)
        if detected is None:
            logging.info(
                "Every suite/task already has >= %d episodes collected per scene variant "
                "under %s -- nothing to do.",
                args.num_trials, output_dir,
            )
            return 0
        args.suite, args.task_id = detected
        logging.info("Auto-detected next incomplete task: --suite %s --task-id %d", args.suite, args.task_id)

    E.configure_libero(E.BENCHMARK_ROOT)  # puts LIBERO_ROOT on sys.path before we import it below
    E.make_optional_matplotlib_stub()
    import cv2
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to feature-serving server: %s", client.get_server_metadata())

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.suite}_task{args.task_id:02d}",
            config=vars(args),
        )
    episode_count, success_count = 0, 0

    try:
        for scene_variant in ("occluded", "normal"):
            benchmark_root, suite = E.benchmark_selection(args.suite, scene_variant)
            E.configure_libero(benchmark_root)
            E.make_optional_matplotlib_stub()
            bddl_path = E.task_files(suite, benchmark_root)[args.task_id]
            init_path = benchmark_root / "init_files" / suite / f"{bddl_path.stem}.pruned_init"
            initial_states = E.load_initial_states(init_path, torch)
            if len(initial_states) < args.num_trials:
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
                max_steps = E.MAX_STEPS[args.suite]
                episode_dir = output_dir / scene_variant / f"{args.task_id + 1:02d}_{bddl_path.stem}"
                episode_dir.mkdir(parents=True, exist_ok=True)

                for episode in range(args.num_trials):
                    npz_path = episode_dir / f"episode_{episode:03d}.npz"
                    if npz_path.is_file():
                        logging.info(
                            "[%s] %s episode %d/%d already collected, skipping",
                            scene_variant,
                            bddl_path.stem,
                            episode + 1,
                            args.num_trials,
                        )
                        continue
                    logging.info("[%s] %s episode %d/%d", scene_variant, bddl_path.stem, episode + 1, args.num_trials)
                    metrics, per_step = rollout_episode(
                        env=env,
                        initial_state=initial_states[episode],
                        prompt=prompt,
                        max_steps=max_steps,
                        args=args,
                        client=client,
                        cv2=cv2,
                    )
                    npz_path = episode_dir / f"episode_{episode:03d}.npz"
                    np.savez_compressed(
                        npz_path,
                        base_image=per_step["base_image"],
                        wrist_image=per_step["wrist_image"],
                        language=per_step["language"],
                        language_mask=per_step["language_mask"],
                        success=metrics["success"],
                    )
                    logging.info(
                        "  success=%s control_frames=%d inference_calls=%d",
                        metrics["success"],
                        metrics["control_frames"],
                        metrics["inference_calls"],
                    )
                    episode_count += 1
                    success_count += int(metrics["success"])
                    if not args.no_wandb:
                        wandb.log(
                            {
                                "suite": args.suite,
                                "task": bddl_path.stem,
                                "scene_variant": scene_variant,
                                "episode": episode,
                                "success": metrics["success"],
                                "control_frames": metrics["control_frames"],
                                "inference_calls": metrics["inference_calls"],
                                "running_success_rate": success_count / episode_count,
                            }
                        )
                    rebuild_manifest(output_dir)
            finally:
                env.close()
    finally:
        if not args.no_wandb:
            wandb.finish()

    rows = rebuild_manifest(output_dir)
    logging.info("Wrote %d episodes to %s", len(rows), output_dir / "manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
