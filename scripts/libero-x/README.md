# LIBERO-X x GR00T-N1.7 -- zero-shot success-rate gauge

Measures how the already-downloaded **`nvidia/GR00T-N1.7-LIBERO`** (`libero_10`)
checkpoint (see [`scripts/groot/`](../groot/README.md)) does on
**[LIBERO-X](https://github.com/meituan/LIBERO-X)** with **no fine-tuning** --
i.e. cross-benchmark zero-shot transfer, the same way
[`scripts/groot/`](../groot/README.md) measures occlusion robustness on
`libero_10`/`libero_10_occluded`.

LIBERO-X (RSS 2026) extends the original LIBERO with 5 progressively harder
difficulty levels (`LEVEL1`-`LEVEL5`), new objects/textures, and new goal
predicates (`ExactIn`, `UprightOn`, `SideOn`). NVIDIA reports ~94-98%
*in-distribution* success on the original `libero_10` suite; LIBERO-X is a
different benchmark, so a much lower number here is the expected result, not
a bug -- see the reference run in `16-LIBERO-X-GR00T-ZeroShot/RESULTS.md`
(0/50 on `LEVEL1`).

## Architecture

```
this repo's top-level .venv        Isaac-GR00T uv venv (submodules/Isaac-GR00T)
  LIBERO-X OffScreenRenderEnv        Gr00tPolicy(libero_10, LIBERO_PANDA)
  scripts/libero-x/run_eval.py  ──ws://127.0.0.1:8000──►  scripts/groot/server/serve_groot_ws.py
  (openpi WebsocketClientPolicy)     (openpi msgpack protocol shim +
                                      Gr00tSimPolicyWrapper)
```

Unlike `16-LIBERO-X-GR00T-ZeroShot`, this folder does **not** duplicate the
GR00T websocket server or the obs/action bridge: `scripts/groot/` already
implements exactly this wire protocol (embodiment `LIBERO_PANDA`, the same
180deg-rotated 256px agentview + wrist images, 8-dim EEF-pose+gripper state,
7-dim delta-EEF action with the gripper normalised/inverted) and is reused
as-is (`scripts/groot/server/serve_groot_ws.py` unchanged;
`scripts/groot/groot_obs.build_flat_obs`/`decode_action_step` imported
directly). Only the LIBERO-X-specific env plumbing is new here, because
LIBERO-X ships its own fork of the `libero` package (new predicates/objects)
that must be imported instead of `submodules/openpi/third_party/libero`'s.

## Run

One-time setup (assumes `scripts/groot/setup.sh` has already been run --
same checkpoint, same server):

```bash
scripts/libero-x/setup.sh      # init submodules/LIBERO-X (~785MB)
```

All-in-one (starts the GR00T server if not already running, smoke test, then
a sample run):

```bash
scripts/libero-x/run_all.sh                       # LEVEL1, 5 tasks x 10 rollouts
LEVEL=LEVEL2 N_TASKS=5 N_ROLLOUTS=10 scripts/libero-x/run_all.sh
```

Manual / piecewise:

```bash
scripts/groot/serve.sh                                        # terminal 1
scripts/libero-x/run_eval.sh --smoke                          # terminal 2: 1 episode
scripts/libero-x/run_eval.sh --level LEVEL1 --n-tasks 5 --n-rollouts 10
```

`STOP_PI05=1 scripts/libero-x/run_all.sh` frees the GPU by stopping the pi0.5
feature server first, same convention as `scripts/groot/run.sh`.

## Files

| Path | Role |
|---|---|
| `constants.py` | `EvalConfig` dataclass -- level/seed/task-sample/rollout knobs, paths into `submodules/LIBERO-X` |
| `libero_x_env.py` | LIBERO-X env construction/reset (`configure_libero_x`, `open_task`, `reset_from_init`) |
| `run_eval.py` | rollout loop + success-rate aggregation -> `outputs/libero-x/results.json` |
| `sample_tasks.py` | seeded uniform-random task sample per level -> `outputs/libero-x/tasks.txt` |
| `setup.sh` | init `submodules/LIBERO-X` |
| `run_eval.sh` / `run_all.sh` | run / full orchestration |

Output layout under `outputs/libero-x/`: `tasks.txt` (sampled task list),
`results.json` / `results_smoke.json`, `videos/*.mp4` (failed episodes by
default, `--save-all-videos` for all), `.libero/config.yaml` (generated,
LIBERO-X-only, distinct from `scripts/evaluation/`'s `.libero/config.yaml`).

Reads (never writes) `submodules/LIBERO-X`, pinned at a fixed commit -- see
`VENDOR.md`.
