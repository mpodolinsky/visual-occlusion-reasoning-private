#!/usr/bin/env python3
"""Run GR00T-N1.7 on libero_10 (normal + occluded) against the GR00T websocket
policy server, saving one aligned episode dir per rollout.

Each episode: ``rollout.json`` + ``rollout.npz`` + ``rollout.mp4`` (agentview,
20 fps) + ``wrist.mp4``. A global ``manifest.csv`` is rewritten after every
episode, and collection auto-resumes: a complete episode dir is skipped and,
with no ``--task-id`` / ``--episode-index``, the script walks to the next
incomplete ``(scene_variant, task_id, episode)``.

Run with 12's top-level venv (this is the websocket *client*; the GR00T model
runs in ``submodules/Isaac-GR00T/.venv`` behind ``scripts/groot/serve.sh``):

    MUJOCO_GL=egl uv run python scripts/groot/collect.py \\
        --replan-steps 8 --scene-variant both --num-trials 25
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from constants import (  # noqa: E402
    DEFAULT_CHECKPOINT_LABEL,
    DEFAULT_OUTPUT_DIR,
    OCCLUDED_SUITE,
    EvalConfig,
    add_replan_argument,
)
from libero_env import load_eval_module, open_task  # noqa: E402
from manifest import (  # noqa: E402
    SCENE_VARIANTS,
    episode_dir,
    episode_is_complete,
    find_next_incomplete,
    rebuild_manifest,
)
from recorder import save_episode  # noqa: E402
from rollout import run_episode  # noqa: E402
from serialization import load_rollout  # noqa: E402
from validation import validate_rollout  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scene-variant", choices=("normal", "occluded", "both"), default="both")
    p.add_argument("--task-id", type=int, default=None, help="0-9; omit to sweep all 10.")
    p.add_argument("--episode-index", type=int, default=None, help="Omit to loop range(--num-trials).")
    p.add_argument("--num-trials", type=int, default=25, help="Initial states per task (<= 50).")
    add_replan_argument(p, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--env-resolution", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=None, help="Override suite default (520).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fresh", action="store_true", help="Re-collect episodes even if already complete.")
    p.add_argument("--keep-going", action="store_true", help="Continue past a per-episode error.")
    p.add_argument("--checkpoint-label", default=DEFAULT_CHECKPOINT_LABEL, help="Recorded in rollout.json.")
    p.add_argument(
        "--with-features",
        action="store_true",
        help="Require + save the layer-16 backbone hidden states (server must run with --with-features).",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="groot-libero-10")
    p.add_argument("--wandb-entity", default=None)
    return p.parse_args(argv)


def build_task_stems(variants: tuple[str, ...]) -> dict[tuple[str, int], str]:
    E = load_eval_module()
    stems: dict[tuple[str, int], str] = {}
    for scene_variant in variants:
        benchmark_root, suite = E.benchmark_selection(OCCLUDED_SUITE, scene_variant)
        for task_id, bddl in enumerate(E.task_files(suite, benchmark_root)):
            stems[(scene_variant, task_id)] = bddl.stem
    return stems


def resolve_worklist(args: argparse.Namespace, variants: tuple[str, ...]) -> list[tuple[str, int, int]]:
    if args.task_id is not None or args.episode_index is not None:
        task_ids = [args.task_id] if args.task_id is not None else list(range(10))
        episodes = [args.episode_index] if args.episode_index is not None else list(range(args.num_trials))
        return [(v, t, e) for t in task_ids for v in variants for e in episodes]
    return [(v, t, e) for t in range(10) for v in variants for e in range(args.num_trials)]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    expect_video = not args.no_video
    variants = SCENE_VARIANTS if args.scene_variant == "both" else (args.scene_variant,)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata() or {}
    server_metadata = str(meta)
    logging.info("GR00T policy server: %s", server_metadata)
    if args.with_features and not meta.get("with_features"):
        logging.error("--with-features requested but the server was not started with --with-features")
        return 2

    stems = build_task_stems(variants)
    worklist = resolve_worklist(args, variants)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    n_done = n_skipped = n_errored = n_gate_failed = 0
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
                    cfg0 = EvalConfig(
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

                cfg = EvalConfig(
                    host=args.host, port=args.port,
                    occluded_suite=OCCLUDED_SUITE, scene_variant=scene_variant,
                    task_id=task_id, episode_index=episode, seed=args.seed,
                    num_steps_wait=args.num_steps_wait, replan_steps=int(args.replan_steps),
                    env_resolution=args.env_resolution, max_steps=args.max_steps,
                    save_video=expect_video, checkpoint_label=args.checkpoint_label,
                )
                logging.info("collect %s t%d ep%d  replan=%d",
                             scene_variant, task_id, episode, cfg.replan_steps)
                rollout = run_episode(
                    env=env, client=client, initial_state=initial_states[episode],
                    instruction=instruction, cfg=cfg, task_file=str(bddl_path),
                    max_steps=max_steps, server_metadata=server_metadata,
                )
                if args.with_features and rollout.policies and rollout.policies[0].features is None:
                    raise RuntimeError(
                        "--with-features set but the server returned no backbone features"
                    )
                save_episode(directory, rollout, save_video=expect_video)

                reloaded = load_rollout(directory)  # reload + alignment gate
                report = validate_rollout(reloaded)
                report.add("Reload test", True)
                if not report.all_passed:
                    n_gate_failed += 1
                    logging.error("ALIGNMENT GATE FAILED for %s\n%s", directory, report.format())
                    if not args.keep_going:
                        logging.error("stopping (pass --keep-going to continue)")
                        return 1
                else:
                    logging.info(
                        "aligned OK: %s (success=%s, n_control=%d, n_policy=%d, features=%s, %.1fs)",
                        rollout.rollout_id, rollout.success, rollout.n_control, rollout.n_policy,
                        reloaded.policies[0].features is not None if reloaded.policies else False,
                        rollout.elapsed_seconds,
                    )

                if run is not None:
                    run.log({
                        "scene_variant": scene_variant, "task_id": task_id, "episode": episode,
                        "success": int(rollout.success), "n_control": rollout.n_control,
                        "n_policy": rollout.n_policy,
                    })

                n_done += 1
                rebuild_manifest(output_dir)
            except KeyboardInterrupt:
                raise
            except Exception:
                n_errored += 1
                logging.exception("EPISODE ERRORED: %s t%d ep%d", scene_variant, task_id, episode)
                if not args.keep_going:
                    return 1
                if env_key != key:
                    broken.add(key)
                    logging.error("marking %s task %d broken; skipping its remaining episodes",
                                  scene_variant, task_id)
                else:
                    _drop_env()
                rebuild_manifest(output_dir)
    finally:
        _drop_env()
        if run is not None:
            run.finish()

    rows = rebuild_manifest(output_dir)
    remaining = find_next_incomplete(output_dir, stems, args.num_trials, variants=variants, expect_video=expect_video)
    logging.info(
        "collected=%d skipped=%d gate_failed=%d errored=%d  manifest rows=%d  next_incomplete=%s",
        n_done, n_skipped, n_gate_failed, n_errored, len(rows), remaining,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
