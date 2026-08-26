#!/usr/bin/env python3
"""For a train_probe_time_dependent.py run: roll out --num-episodes random
initial states from each of that run's held-out unseen tasks against the
live policy (the same feature-serving websocket server collect_features.py
talks to), scoring every step with the trained probe checkpoint as it goes.
Saves an agentview review video and a plot of the probe's raw per-step score
and accumulated score over time for each rollout, and flags any episode
where the policy failed the task.

Needs a running feature-serving server (see serve_pi05_with_features.py) --
this script only makes websocket calls, it doesn't load pi0.5 itself:
    submodules/openpi/.venv/bin/python scripts/perception_probe/serve_pi05_with_features.py
    .venv/bin/python scripts/perception_probe/rollout_unseen_with_scores.py \
        outputs/perception_probe/probe_time_dependent_rmean/20260824_181247 --rmean \
        --num-episodes 10

Run with the top-level project venv (has robosuite/mujoco/openpi-client + torch,
not JAX/openpi).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pi05_libero as E  # noqa: E402
from probe_model import PerceptionSuccessProbe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A train_probe_time_dependent.py output run folder.")
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--scene-variant", choices=("occluded", "normal"), default="occluded",
        help="Which scene variant to roll out for each unseen task (each unseen task in split.json "
        "covers both -- this picks one rollout per task, not per (task, variant) pair).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking each task's random initial state(s).")
    parser.add_argument(
        "--num-episodes", type=int, default=5,
        help="Random initial states to roll out per unseen task (drawn without replacement, so must "
        "be <= the number of released initial states per task, typically 50).",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--policy-image-size", type=int, default=224)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--video-resolution", type=int, default=256)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <run_dir>/rollout_videos.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # score_sequence() hyperparameters -- must match how the checkpoint was trained; not persisted
    # on disk from the original run (see eval_run.py's docstring for the same caveat).
    parser.add_argument("--rmean", action="store_true", default=False)
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    return parser.parse_args()


def load_probe(run_dir: Path, checkpoint: str, device: str) -> torch.nn.Module:
    model = PerceptionSuccessProbe().to(device)
    state = torch.load(run_dir / checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def score_step(model: torch.nn.Module, prefix_features: dict, device: str) -> float:
    """Raw sigmoid(logit) for one timestep's prefix features (batch of 1)."""
    base = torch.from_numpy(np.asarray(prefix_features["base_image"])).unsqueeze(0).float().to(device)
    wrist = torch.from_numpy(np.asarray(prefix_features["wrist_image"])).unsqueeze(0).float().to(device)
    lang = torch.from_numpy(np.asarray(prefix_features["language"])).unsqueeze(0).float().to(device)
    lang_mask = torch.from_numpy(np.asarray(prefix_features["language_mask"])).unsqueeze(0).to(device)
    logit = model(base, wrist, lang, lang_mask)
    return torch.sigmoid(logit).item()


def rollout_with_scores(
    *, env, initial_state, prompt, max_steps, args, client, model, cv2,
    agent_video_path: Path, wrist_video_path: Path,
) -> dict:
    agent_recorder = E.VideoRecorder(agent_video_path, args.video_resolution, args.video_fps, cv2)
    try:
        wrist_recorder = E.VideoRecorder(wrist_video_path, args.video_resolution, args.video_fps, cv2)
    except Exception:
        agent_recorder.close()
        raise

    success = False
    control_frames = 0
    inference_calls = 0
    accumulated = 0.0
    trace = []  # list of (control_frame_at_call, raw_score, accumulated_score)
    try:
        env.reset()
        client.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            observation, _, success, _ = env.step(E.DUMMY_ACTION)

        action_plan: collections.deque = collections.deque()
        agent_recorder.add(E.rotate_camera_image(observation, "agentview_image"))
        wrist_recorder.add(E.rotate_camera_image(observation, "robot0_eye_in_hand_image"))
        while not success and control_frames < max_steps:
            if not action_plan:
                result = client.infer(
                    E.make_policy_observation(observation, prompt, args.policy_image_size, cv2)
                )
                action_chunk = np.asarray(result["actions"])
                action_plan.extend(action_chunk[: args.replan_steps])
                inference_calls += 1

                raw_score = score_step(model, result["prefix_features"], args.device)
                accumulated = accumulated + raw_score if args.cumsum or args.rmean else raw_score
                reported = accumulated / inference_calls if args.rmean else accumulated
                trace.append((control_frames, raw_score, reported))

            action = np.asarray(action_plan.popleft())
            observation, _, success, _ = env.step(action.tolist())
            control_frames += 1
            agent_recorder.add(E.rotate_camera_image(observation, "agentview_image"))
            wrist_recorder.add(E.rotate_camera_image(observation, "robot0_eye_in_hand_image"))
    finally:
        agent_recorder.close()
        wrist_recorder.close()

    return {
        "success": bool(success),
        "control_frames": control_frames,
        "inference_calls": inference_calls,
        "trace": trace,
    }


def plot_trace(trace: list[tuple[int, float, float]], success: bool, task: str, scene_variant: str, args, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control_frames = [t[0] for t in trace]
    seconds = [f / 20.0 for f in control_frames]  # LIBERO runs at 20Hz
    raw_scores = [t[1] for t in trace]
    accumulated = [t[2] for t in trace]

    fig, (ax_raw, ax_acc) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_raw.plot(seconds, raw_scores, color="tab:blue", marker=".")
    ax_raw.set_ylabel("raw p(failure)")
    ax_raw.set_ylim(-0.05, 1.05)
    ax_raw.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    acc_label = "running mean" if args.rmean else ("cumsum" if args.cumsum else "raw")
    ax_acc.plot(seconds, accumulated, color="tab:red", marker=".")
    ax_acc.set_ylabel(f"accumulated score ({acc_label})")
    ax_acc.set_xlabel("time (s)")
    if args.rmean:
        ax_acc.set_ylim(-0.05, 1.05)

    outcome = "SUCCESS" if success else "FAILURE"
    fig.suptitle(f"{task} [{scene_variant}] -- {outcome}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _group_mean_curve(group: list[list[tuple[int, float, float]]]) -> tuple[list[float], list[float], list[float]] | None:
    """Averages raw/accumulated scores INDEX-by-INDEX (not by wall-clock time)
    across a group of same-outcome episodes, so a longer episode's tail just
    drops out of the average once shorter episodes in the group have ended,
    rather than the whole curve needing a common length. Index i's x
    position is taken from whichever episode reaches that index first,
    since every trace advances one inference call at a time regardless of
    episode length (same replan_steps throughout)."""
    if not group:
        return None
    max_len = max(len(trace) for trace in group)
    xs, mean_raw, mean_acc = [], [], []
    for i in range(max_len):
        at_i = [trace[i] for trace in group if i < len(trace)]
        xs.append(at_i[0][0] / 20.0)
        mean_raw.append(sum(p[1] for p in at_i) / len(at_i))
        mean_acc.append(sum(p[2] for p in at_i) / len(at_i))
    return xs, mean_raw, mean_acc


def plot_overlay(
    traces: list[tuple[list[tuple[int, float, float]], bool]], args, path: Path, title: str | None = None,
) -> None:
    """All rollouts' score-over-time curves on one pair of axes -- failures in
    red, successes in blue -- so patterns that only show up across many
    episodes (e.g. failures separating from successes earlier/later, or not
    at all) are visible at a glance instead of comparing plots one at a time.
    Each group's index-wise mean is overlaid as a bold line on top of the
    thin individual ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Individual episodes drawn light/muted so they read as background context;
    # the group means use much deeper, fully-opaque colors so they stay
    # legible on top regardless of how many thin lines are underneath.
    individual_success, individual_fail = "cornflowerblue", "lightcoral"
    mean_success, mean_fail = "navy", "darkred"

    fig, (ax_raw, ax_acc) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    success_traces = [trace for trace, success in traces if success]
    fail_traces = [trace for trace, success in traces if not success]
    for trace, success in traces:
        seconds = [t[0] / 20.0 for t in trace]  # LIBERO runs at 20Hz
        raw_scores = [t[1] for t in trace]
        accumulated = [t[2] for t in trace]
        color = individual_success if success else individual_fail
        ax_raw.plot(seconds, raw_scores, color=color, alpha=0.35, linewidth=1)
        ax_acc.plot(seconds, accumulated, color=color, alpha=0.35, linewidth=1)

    for group, color in ((success_traces, mean_success), (fail_traces, mean_fail)):
        mean_curve = _group_mean_curve(group)
        if mean_curve is None:
            continue
        xs, mean_raw, mean_acc = mean_curve
        ax_raw.plot(xs, mean_raw, color=color, alpha=1.0, linewidth=3, zorder=5)
        ax_acc.plot(xs, mean_acc, color=color, alpha=1.0, linewidth=3, zorder=5)

    ax_raw.set_ylabel("raw p(failure)")
    ax_raw.set_ylim(-0.05, 1.05)
    ax_raw.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    acc_label = "running mean" if args.rmean else ("cumsum" if args.cumsum else "raw")
    ax_acc.set_ylabel(f"accumulated score ({acc_label})")
    ax_acc.set_xlabel("time (s)")
    if args.rmean:
        ax_acc.set_ylim(-0.05, 1.05)

    handles = [
        plt.Line2D([0], [0], color=individual_success, linewidth=1, alpha=0.6, label=f"success (n={len(success_traces)})"),
        plt.Line2D([0], [0], color=individual_fail, linewidth=1, alpha=0.6, label=f"failure (n={len(fail_traces)})"),
        plt.Line2D([0], [0], color=mean_success, linewidth=3, label="success mean"),
        plt.Line2D([0], [0], color=mean_fail, linewidth=3, label="failure mean"),
    ]
    ax_raw.legend(handles=handles, loc="upper right")
    fig.suptitle(title or f"All rollouts overlaid (n={len(traces)})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    split = json.loads((args.run_dir / "split.json").read_text())
    suite = split["suite"]
    unseen_tasks = sorted(split["unseen_tasks"])
    logging.info("Suite %s, %d unseen tasks: %s", suite, len(unseen_tasks), unseen_tasks)

    output_dir = args.output_dir or (args.run_dir / "rollout_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not E.BENCHMARK_ROOT.is_dir() or not E.STANDARD_BENCHMARK_ROOT.is_dir():
        raise FileNotFoundError("Missing submodules; run `git submodule update --init --recursive`")

    E.configure_libero(E.BENCHMARK_ROOT)
    E.make_optional_matplotlib_stub()
    import cv2
    import torch as _torch  # noqa: F401 -- load_initial_states expects a torch module argument
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    model = load_probe(args.run_dir, args.checkpoint, args.device)
    client = WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to feature-serving server: %s", client.get_server_metadata())

    benchmark_root, resolved_suite = E.benchmark_selection(suite, args.scene_variant)
    E.configure_libero(benchmark_root)
    E.make_optional_matplotlib_stub()
    bddl_paths = E.task_files(resolved_suite, benchmark_root)
    stem_to_task_id = {p.stem: i for i, p in enumerate(bddl_paths)}
    max_steps = E.MAX_STEPS[suite]

    results = []
    all_traces = []  # (trace, success) for every rollout, across every task -- for the overlay plot
    for task in unseen_tasks:
        if task not in stem_to_task_id:
            logging.warning("Task %s not found in suite %s bddl files -- skipping", task, resolved_suite)
            continue
        task_id = stem_to_task_id[task]
        bddl_path = bddl_paths[task_id]
        init_path = benchmark_root / "init_files" / resolved_suite / f"{bddl_path.stem}.pruned_init"
        initial_states = E.load_initial_states(init_path, _torch)

        if args.num_episodes > len(initial_states):
            raise ValueError(
                f"--num-episodes {args.num_episodes} exceeds the {len(initial_states)} initial states "
                f"released for task {bddl_path.stem}"
            )
        rng = np.random.default_rng(args.seed + task_id)
        episode_indices = rng.choice(len(initial_states), size=args.num_episodes, replace=False)

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
            task_root = output_dir / f"{task_id + 1:02d}_{bddl_path.stem}"
            for episode, episode_idx in enumerate(episode_indices):
                episode_idx = int(episode_idx)
                task_dir = task_root / f"episode_{episode:02d}"
                logging.info(
                    "[%s] %s -- episode %d/%d, initial state %d/%d", args.scene_variant, bddl_path.stem,
                    episode + 1, args.num_episodes, episode_idx, len(initial_states),
                )
                rollout = rollout_with_scores(
                    env=env, initial_state=initial_states[episode_idx], prompt=prompt, max_steps=max_steps,
                    args=args, client=client, model=model, cv2=cv2,
                    agent_video_path=task_dir / "agentview.mp4", wrist_video_path=task_dir / "wrist.mp4",
                )
                plot_trace(
                    rollout["trace"], rollout["success"], bddl_path.stem, args.scene_variant, args,
                    task_dir / "scores.png",
                )
                all_traces.append((rollout["trace"], rollout["success"]))
                (task_dir / "trace.json").write_text(
                    json.dumps(
                        {
                            "task": bddl_path.stem, "scene_variant": args.scene_variant,
                            "init_state_index": episode_idx, "success": rollout["success"],
                            "control_frames": rollout["control_frames"],
                            "inference_calls": rollout["inference_calls"], "trace": rollout["trace"],
                        },
                        indent=2,
                    )
                )
                logging.info(
                    "  success=%s control_frames=%d inference_calls=%d -> %s",
                    rollout["success"], rollout["control_frames"], rollout["inference_calls"], task_dir,
                )
                results.append(
                    {
                        "task": bddl_path.stem, "scene_variant": args.scene_variant,
                        "episode": episode, "init_state_index": episode_idx,
                        "success": rollout["success"], "video": str(task_dir / "agentview.mp4"),
                        "plot": str(task_dir / "scores.png"),
                    }
                )
        finally:
            env.close()

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
    if all_traces:
        plot_overlay(all_traces, args, output_dir / "scores_overlay.png")
        logging.info("Wrote overlay plot (n=%d) to %s", len(all_traces), output_dir / "scores_overlay.png")
    failures = [r for r in results if not r["success"]]
    logging.info(
        "Wrote %d rollouts to %s (%d succeeded, %d failed)",
        len(results), output_dir, len(results) - len(failures), len(failures),
    )
    if failures:
        logging.warning("FAILED episodes (%d/%d):", len(failures), len(results))
        for r in failures:
            logging.warning("  [%s] %s episode %d (init_state %d) -> %s", r["scene_variant"], r["task"], r["episode"], r["init_state_index"], r["video"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
