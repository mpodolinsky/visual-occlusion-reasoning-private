# Evaluation scripts

These scripts run OpenPI pi0.5 on the released LIBERO-Occ tasks, run the
filename- and initial-state-matched vanilla LIBERO baseline, and compare the
two sets of episode outcomes.

Complete the repository [setup](../../README.md#setup) before using them. Run
all commands below from the repository root.

## Start the policy server

The evaluator is a client for OpenPI's policy server. Start the server in its
own terminal:

```bash
cd submodules/openpi
uv run scripts/serve_policy.py --env LIBERO
```

The `LIBERO` server configuration downloads and caches
`gs://openpi-assets/checkpoints/pi05_libero` when needed. By default the
evaluator connects to `127.0.0.1:8000`; use `--host` and `--port` to connect to
a different server.

## Smoke-test both scene variants

Run one initial state for every task before committing to the full benchmark:

```bash
MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py \
  --scene-variant occluded \
  --num-trials-per-task 1 \
  --output-dir outputs/pi05_libero_occ_smoke

MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py \
  --scene-variant normal \
  --num-trials-per-task 1 \
  --output-dir outputs/pi05_libero_matched_normal_smoke
```

Each smoke test covers all 40 environments. Use a separate output directory,
as above, so smoke-test rows are not mixed into full-run results.

## Run the full matched evaluation

Occluded LIBERO-Occ scenes:

```bash
MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py \
  --scene-variant occluded \
  --num-trials-per-task 50
```

Matched vanilla LIBERO scenes:

```bash
MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py \
  --scene-variant normal \
  --num-trials-per-task 50
```

The default output directories are `outputs/pi05_libero_occ/` and
`outputs/pi05_libero_matched_normal/`. Every released LIBERO-Occ task maps by
filename to a standard LIBERO task, and both runs use the same initial-state
indices. This makes `(canonical suite, task, init_state_index)` the episode
pairing key.

### Resume an interrupted run

The evaluator updates `episodes.csv` after each rollout and resumes by default.
Rerun the same command to skip completed episodes whose agent-view and wrist
videos are still present. `--no-resume` instead refuses to run when completed
episode keys already exist. Use `--continue-on-error` only when one failed
episode should not stop the remaining run.

### Run detached with tmux

The following starts a server and full occluded evaluation in a detached tmux
session. It also preserves a log from each process:

```bash
REPO_DIR="$PWD"; mkdir -p "$REPO_DIR/outputs/pi05_libero_occ"; \
tmux new-session -d -s pi05-libero-occ -n server -c "$REPO_DIR/submodules/openpi" \
  "set -o pipefail; XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py --env LIBERO 2>&1 | tee '$REPO_DIR/outputs/pi05_libero_occ/server.log'" \; \
  new-window -t pi05-libero-occ -n eval -c "$REPO_DIR" \
  "set -o pipefail; until nc -z 127.0.0.1 8000; do sleep 5; done; MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py --scene-variant occluded --num-trials-per-task 50 2>&1 | tee '$REPO_DIR/outputs/pi05_libero_occ/eval.log'"
```

Attach with `tmux attach -t pi05-libero-occ`. To use this pattern for the
normal run, use a different tmux session name, add `--scene-variant normal`,
and update the log directory.

## Compare matched outcomes

After both full runs complete:

```bash
uv run python scripts/evaluation/compare_pi05_libero_runs.py
```

The comparison is strict by default: both CSVs must contain the same complete
set of pairing keys. Pass `--allow-partial` to compare only their intersection.
Custom inputs and destinations can be supplied with `--occluded-csv`,
`--normal-csv`, and `--output-dir`.

The default comparison directory, `outputs/pi05_libero_comparison/`, contains:

- `occlusion_failures.md`: normal successes that failed under occlusion,
  grouped by suite and task with links to both camera views.
- `occlusion_failures.csv`: the same regressions in tabular form.
- `comparison_summary.json`: overall and per-suite outcome counts.

## Evaluation output

Each evaluation directory contains:

- `videos/<suite>/<task>/episode_NNN.mp4`: agent-view rollout.
- `videos/<suite>/<task>/episode_NNN_wrist.mp4`: synchronized wrist-camera
  rollout.
- `episodes.csv`: one durable row per rollout, including success, frame counts,
  inference calls, runtime, and errors.
- `task_summary.csv` and `suite_summary.csv`: grouped success and frame stats.
- `summary.json`: aggregate, suite, and task summaries.
- `run_config.json`: evaluator arguments, server metadata, and checkpoint
  identity.

Review videos are 128x128 by default and use H.264, `yuv420p`, and fast-start
MP4 metadata. The `ffmpeg` executable must be available on `PATH`.
`control_frames` excludes the 10 default stabilization steps; `sim_frames`
includes them.

Run the built-in help for the complete option list:

```bash
uv run python scripts/evaluation/eval_pi05_libero.py --help
uv run python scripts/evaluation/compare_pi05_libero_runs.py --help
```

Useful evaluator controls include `--suites`, `--replan-steps`, image and video
resolutions, rendering device selection, and the output directory. On a
CPU-only machine with OSMesa installed, set `MUJOCO_GL=osmesa` instead of
`MUJOCO_GL=egl`.
