#!/usr/bin/env python3
"""Collect row-aligned libero_10 rollouts (normal + occluded) against the
feature-serving pi0.5 websocket server.

Each episode is saved as an aligned ``rollout.json`` + ``rollout.npz`` +
``rollout.mp4`` (20 fps, one frame per control step). A global ``manifest.csv``
is rewritten after every episode, and collection auto-resumes: any episode dir
that is already complete is skipped, and with no ``--task-id`` / ``--episode-index``
the script walks to the next incomplete ``(scene_variant, task_id, episode)``.

Run with 12's top-level venv (robosuite/mujoco/openpi-client -- this talks to
the model over the websocket, it does not load JAX):

    MUJOCO_GL=egl uv run python scripts/semantic_failure/collect.py \
        --replan-steps 5 --scene-variant both --num-trials 50

Start the feature server first:

    submodules/openpi/.venv/bin/python scripts/perception_probe/serve_pi05_with_features.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG_NAME,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OUTPUT_DIR,
    OCCLUDED_SUITE,
    PipelineConfig,
    add_replan_argument,
)
from libero_env import load_eval_module, open_task  # noqa: E402
from manifest import SCENE_VARIANTS, episode_dir, episode_is_complete, find_next_incomplete, rebuild_manifest  # noqa: E402
from pi05_client import FeatureClient  # noqa: E402
from recorder import save_episode  # noqa: E402
from rollout_runner import run_episode  # noqa: E402
from serialization import load_rollout  # noqa: E402
from validation import validate_rollout  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scene-variant", choices=("normal", "occluded", "both"), default="both")
    p.add_argument("--task-id", type=int, default=None, help="0-9; omit to sweep all 10.")
    p.add_argument("--episode-index", type=int, default=None, help="Omit to loop range(--num-trials).")
    p.add_argument("--num-trials", type=int, default=50, help="Initial states per task (<= 50).")
    add_replan_argument(p, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--policy-image-size", type=int, default=224)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=None, help="Override suite default (520).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fresh", action="store_true", help="Re-collect episodes even if already complete.")
    p.add_argument("--keep-going", action="store_true", help="Do not stop on a failed alignment check.")
    p.add_argument("--label", action="store_true", help="Run Gemini labeling right after each episode.")
    p.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model for --label.")
    p.add_argument("--no-refine", action="store_true", help="--label: skip Dan's refine pass.")
    p.add_argument("--model-id", default=DEFAULT_CONFIG_NAME)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="pi05-semantic-failure")
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args(argv)


def build_task_stems(variants: tuple[str, ...]) -> dict[tuple[str, int], str]:
    E = load_eval_module()
    stems: dict[tuple[str, int], str] = {}
    for scene_variant in variants:
        benchmark_root, suite = E.benchmark_selection(OCCLUDED_SUITE, scene_variant)
        files = E.task_files(suite, benchmark_root)
        for task_id, bddl in enumerate(files):
            stems[(scene_variant, task_id)] = bddl.stem
    return stems


def resolve_worklist(args: argparse.Namespace, variants: tuple[str, ...], stems) -> list[tuple[str, int, int]]:
    """Explicit selection if given, else the full grid (auto-resume skips
    completed episodes later)."""
    if args.task_id is not None or args.episode_index is not None:
        task_ids = [args.task_id] if args.task_id is not None else list(range(10))
        episodes = [args.episode_index] if args.episode_index is not None else list(range(args.num_trials))
        return [(v, t, e) for t in task_ids for v in variants for e in episodes]
    # Nothing pinned: walk the grid task-major/variant/episode.
    return [
        (v, t, e)
        for t in range(10)
        for v in variants
        for e in range(args.num_trials)
    ]


def maybe_label(directory: Path, args: argparse.Namespace) -> None:
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        logging.warning("--label set but GEMINI_API_KEY missing; skipping labels for %s", directory)
        return
    from dan_label_with_vlm import create_backend
    from episode import build_labeler_meta, label_episode
    from present import format_example
    from serialization import save_labels

    rollout = load_rollout(directory)
    backend = create_backend("gemini", args.model)
    refine = not args.no_refine
    label_episode(rollout, backend, refine=refine)
    save_labels(directory, rollout, build_labeler_meta(backend, refine=refine))
    (directory / "example.md").write_text(format_example(rollout), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    expect_video = not args.no_video
    variants = SCENE_VARIANTS if args.scene_variant == "both" else (args.scene_variant,)

    client = FeatureClient(args.host, args.port)
    logging.info("Feature server: %s", client.metadata)

    stems = build_task_stems(variants)
    worklist = resolve_worklist(args, variants, stems)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    n_done = n_skipped = n_failed_gate = n_errored = 0
    broken: set[tuple[str, int]] = set()
    env = None
    env_key: tuple[str, int] | None = None
    env_ctx: tuple = ()

    def _drop_env() -> None:
        nonlocal env, env_key
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("env.close() failed")
        env = None
        env_key = None

    try:
        for scene_variant, task_id, episode in worklist:
            key = (scene_variant, task_id)
            if key in broken:
                continue
            stem = stems[key]
            directory = episode_dir(output_dir, scene_variant, task_id, stem, episode)
            if not args.fresh and episode_is_complete(directory, expect_video=expect_video):
                n_skipped += 1
                continue

            try:
                if env_key != key:
                    _drop_env()
                    cfg0 = PipelineConfig(
                        occluded_suite=OCCLUDED_SUITE, scene_variant=scene_variant,
                        task_id=task_id, seed=args.seed,
                        env_resolution=args.env_resolution, max_steps=args.max_steps,
                    )
                    env, bddl_path, initial_states, instruction, max_steps = open_task(cfg0)
                    env_key = key
                    env_ctx = (bddl_path, initial_states, instruction, max_steps)
                    logging.info(
                        "opened %s task %d (%s) -- %d init states, max_steps=%d",
                        scene_variant, task_id, bddl_path.stem, len(initial_states), max_steps,
                    )
                bddl_path, initial_states, instruction, max_steps = env_ctx

                if episode >= len(initial_states):
                    logging.warning("episode %d >= %d init states for %s t%d; skipping",
                                    episode, len(initial_states), scene_variant, task_id)
                    continue

                cfg = PipelineConfig(
                    host=args.host, port=args.port,
                    occluded_suite=OCCLUDED_SUITE, scene_variant=scene_variant,
                    task_id=task_id, episode_index=episode, seed=args.seed,
                    num_steps_wait=args.num_steps_wait, replan_steps=int(args.replan_steps),
                    policy_image_size=args.policy_image_size, env_resolution=args.env_resolution,
                    max_steps=args.max_steps, save_video=expect_video,
                    model_id=args.model_id, checkpoint=args.checkpoint,
                )
                logging.info("collect %s t%d ep%d  replan=%d",
                             scene_variant, task_id, episode, cfg.replan_steps)
                rollout = run_episode(
                    env=env, client=client, initial_state=initial_states[episode],
                    instruction=instruction, cfg=cfg, task_file=str(bddl_path),
                    max_steps=max_steps, feature_server_version=str(client.metadata),
                )
                save_episode(directory, rollout, save_video=expect_video)

                report = validate_rollout(load_rollout(directory))
                report.add("Reload test", True)
                if not report.all_passed:
                    n_failed_gate += 1
                    logging.error("ALIGNMENT GATE FAILED for %s\n%s", directory, report.format())
                    if not args.keep_going:
                        logging.error("stopping (pass --keep-going to continue)")
                        return 1
                else:
                    logging.info("aligned OK: %s (success=%s, n_control=%d)",
                                 rollout.rollout_id, rollout.success, rollout.n_control)

                if args.label:
                    try:
                        maybe_label(directory, args)
                    except Exception:
                        logging.exception("labeling failed for %s", directory)

                n_done += 1
                rebuild_manifest(output_dir)
                if run is not None:
                    run.log({
                        "scene_variant": scene_variant, "task_id": task_id, "episode": episode,
                        "success": int(rollout.success), "n_control": rollout.n_control,
                        "n_policy": rollout.n_policy,
                    })
            except KeyboardInterrupt:
                raise
            except Exception:
                n_errored += 1
                logging.exception("EPISODE ERRORED: %s t%d ep%d", scene_variant, task_id, episode)
                if not args.keep_going:
                    return 1
                if env_key != key:
                    # env never came up for this task -- don't retry it every episode
                    broken.add(key)
                    logging.error("marking %s task %d broken; skipping its remaining episodes",
                                  scene_variant, task_id)
                else:
                    _drop_env()  # rebuild a fresh env for the next episode
                rebuild_manifest(output_dir)
    finally:
        _drop_env()
        if run is not None:
            run.finish()

    rows = rebuild_manifest(output_dir)
    remaining = find_next_incomplete(output_dir, stems, args.num_trials, variants=variants, expect_video=expect_video)
    logging.info(
        "collected=%d skipped=%d gate_failed=%d errored=%d  manifest rows=%d  next_incomplete=%s",
        n_done, n_skipped, n_failed_gate, n_errored, len(rows), remaining,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
