"""Zero-shot GR00T-N1.7-LIBERO success-rate gauge on LIBERO-X.

Ported from ``16-LIBERO-X-GR00T-ZeroShot/sim/run_eval.py``, but talks to
*this repo's* ``scripts/groot/server/serve_groot_ws.py`` (unchanged, started
via ``scripts/groot/serve.sh``) and reuses its obs/action bridge
(``scripts/groot/groot_obs.py``) instead of duplicating it -- both already
implement exactly the wire protocol and LIBERO_PANDA convention this needs.

Runs in this repo's top-level ``.venv`` (same as ``scripts/evaluation/`` and
the ``scripts/groot/`` collection client). Reports success rate, overall and
per task.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from constants import REPO_ROOT, EvalConfig
from libero_x_env import open_task, reset_from_init
from sample_tasks import sample as sample_tasks

_GROOT_DIR = str(REPO_ROOT / "scripts" / "groot")
if _GROOT_DIR not in sys.path:
    sys.path.insert(0, _GROOT_DIR)
from groot_obs import build_flat_obs, decode_action_step  # noqa: E402


def _load_tasks(cfg: EvalConfig, tasks_file: Path) -> list[str]:
    if tasks_file.is_file():
        return [line.strip() for line in tasks_file.read_text().splitlines() if line.strip()]
    chosen = sample_tasks(cfg)
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_file.write_text("\n".join(chosen) + "\n")
    return chosen


def _save_video(frames: list[np.ndarray], path: Path) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(str(path), [np.asarray(f).astype(np.uint8) for f in frames], fps=20)
    except Exception as exc:  # noqa: BLE001
        print(f"  (video save failed: {exc})")


def run_episode(env, client, state_vec, prompt, cfg: EvalConfig, record: bool):
    client.reset()
    obs = reset_from_init(env, state_vec, cfg)
    frames: list[np.ndarray] = []
    success = False
    steps = 0
    plan: list[np.ndarray] = []

    while not success and steps < cfg.max_steps:
        if not plan:
            result = client.infer(build_flat_obs(obs, prompt))
            if "actions" not in result:
                raise RuntimeError(f"policy response missing 'actions': keys={list(result)}")
            chunk = np.asarray(result["actions"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 7:
                raise RuntimeError(f"bad action chunk shape {chunk.shape}, expected (H, 7)")
            n_take = min(cfg.replan_steps, cfg.max_steps - steps, chunk.shape[0])
            plan = [chunk[j] for j in range(n_take)]

        action = decode_action_step(plan.pop(0))
        obs, _, done, _ = env.step(action)
        success = bool(done)
        steps += 1
        if record and steps % 2 == 0:
            frames.append(obs["agentview_image"][::-1])

    return success, steps, frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--level", default="LEVEL1", choices=("LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="1 task x 1 rollout")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--save-all-videos", action="store_true")
    p.add_argument("--tasks-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--video-tag", default="", help="suffix on video filenames to avoid clobber")
    args = p.parse_args(argv)

    cfg = EvalConfig(
        host=args.host,
        port=args.port,
        level=args.level,
        seed=args.seed,
        n_tasks=args.n_tasks,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
    )
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    tasks_file = Path(args.tasks_file) if args.tasks_file else cfg.outputs_dir / "tasks.txt"
    tasks = _load_tasks(cfg, tasks_file)

    n_rollouts = 1 if args.smoke else cfg.n_rollouts
    if args.smoke:
        tasks = tasks[:1]

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    print(f"connecting to GR00T policy at {cfg.host}:{cfg.port} ...")
    client = WebsocketClientPolicy(cfg.host, cfg.port)
    print(f"connected. server metadata: {client.get_server_metadata()}")

    t0 = time.time()
    per_task: dict[str, dict] = {}
    episodes: list[dict] = []

    for ti, task_file in enumerate(tasks):
        env, bddl_path, states, prompt = open_task(cfg, task_file)
        print(f"\n[{ti + 1}/{len(tasks)}] {bddl_path.stem}\n    prompt: {prompt!r}")
        succ = 0
        try:
            for ep in range(n_rollouts):
                state_vec = states[ep % len(states)]
                record = not args.no_videos
                ok, steps, frames = run_episode(env, client, state_vec, prompt, cfg, record)
                succ += int(ok)
                episodes.append(
                    {"task": task_file, "episode": ep, "success": bool(ok), "steps": steps}
                )
                print(f"    ep {ep:2d}: {'SUCCESS' if ok else 'fail   '}  ({steps} steps)")
                if frames and (args.save_all_videos or not ok):
                    tag = "success" if ok else "fail"
                    vt = f"_{args.video_tag}" if args.video_tag else ""
                    _save_video(
                        frames,
                        cfg.outputs_dir / "videos" / f"{bddl_path.stem}{vt}__ep{ep:02d}_{tag}.mp4",
                    )
        finally:
            env.close()
        per_task[task_file] = {
            "prompt": prompt,
            "successes": succ,
            "trials": n_rollouts,
            "sr": succ / n_rollouts,
        }
        print(f"    -> task SR: {succ}/{n_rollouts} = {succ / n_rollouts:.2%}")

    total_succ = sum(v["successes"] for v in per_task.values())
    total_eps = sum(v["trials"] for v in per_task.values())
    results = {
        "checkpoint": cfg.checkpoint,
        "level": cfg.level,
        "seed": cfg.seed,
        "n_tasks": len(tasks),
        "n_rollouts_per_task": n_rollouts,
        "max_steps": cfg.max_steps,
        "replan_steps": cfg.replan_steps,
        "smoke": args.smoke,
        "tasks": tasks,
        "per_task": per_task,
        "overall_sr": (total_succ / total_eps) if total_eps else 0.0,
        "total_successes": total_succ,
        "total_episodes": total_eps,
        "episodes": episodes,
        "wall_time_s": round(time.time() - t0, 1),
    }
    out = Path(args.out) if args.out else cfg.outputs_dir / (
        "results_smoke.json" if args.smoke else "results.json"
    )
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 64)
    print(f"GR00T-N1.7-LIBERO/libero_10  zero-shot on LIBERO-X {cfg.level}")
    print("=" * 64)
    for tf, v in per_task.items():
        print(f"  {v['successes']:2d}/{v['trials']:<2d}  {v['sr']:6.1%}  {tf}")
    print("-" * 64)
    print(
        f"  OVERALL: {total_succ}/{total_eps} = {results['overall_sr']:.1%}"
        f"   ({results['wall_time_s']}s)"
    )
    print(f"  results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
