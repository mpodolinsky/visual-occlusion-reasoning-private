# `scripts/groot/`

Run **NVIDIA GR00T-N1.7** on the `libero_10` suite (normal + LIBERO-Occ) and
compare occlusion robustness against the pi0.5 runs: rollout success rate,
aligned action/clock arrays, review videos, and -- with `--with-features` --
the frozen VLM backbone hidden states (the pi0.5 SAVE-A analog).

Ported from `16-LIBERO-X-GR00T-ZeroShot/` (which itself mirrors NVIDIA's
`gr00t/eval/sim/LIBERO/libero_env.py`) -- see [`VENDOR.md`](VENDOR.md). The
LIBERO / LIBERO-Occ plumbing is imported **read-only** from
`scripts/evaluation/eval_pi05_libero.py`; nothing outside this folder changes
except `.gitmodules` (adds `submodules/Isaac-GR00T`), `.gitignore` (adds
`checkpoints/`), and the top-level README.

## Architecture

Two processes, two venvs, bridged on `ws://127.0.0.1:8000` -- the same
client/server split the pi0.5 eval uses:

```
repo-12 top-level .venv (py3.11)              submodules/Isaac-GR00T/.venv (py3.10)
  scripts/groot/collect.py                       scripts/groot/server/serve_groot_ws.py
  - libero_10 env via eval_pi05_libero (RO)      - Gr00tPolicy(LIBERO_PANDA,
  - openpi_client.WebsocketClientPolicy  ──ws──►    checkpoints/GR00T-N1.7-LIBERO/libero_10)
  - groot_obs bridge + rollout loop             - openpi msgpack wire shim
  - writes outputs/groot/libero_10/...          - replies {"actions": (16,7) f32}
```

The client side needs **no new deps** (`openpi-client` already vendors the
websocket client + msgpack). All GR00T-heavy deps live in the submodule's own
`.venv`, so `pyproject.toml` / `uv.lock` are untouched.

## Obs / action convention

`groot_obs.build_flat_obs` feeds GR00T the NVIDIA LIBERO convention:
180deg-rotated (`[::-1, ::-1]`) agentview + wrist images at the raw 256px env
resolution (the GR00T processor resizes), an 8-dim `[eef_pos, axis-angle,
gripper_qpos]` state, and the instruction. That 180deg rotation is the same
correction `eval_pi05_libero.rotate_camera_image` applies for pi0.5.
`decode_action_step` maps GR00T's 7-dim delta-EEF output back to a robosuite
`OSC_POSE` action, normalising + binarising + inverting the gripper dim.
(`LIBEROX_NATIVE_CONVENTION=1` switches to the `meituan/LIBERO-X` convention;
off by default -- the `nvidia/GR00T-N1.7-LIBERO` checkpoint uses NVIDIA's.)

Init follows **12's protocol**: `env.set_init_state` + `num_steps_wait=10`
`DUMMY_ACTION` settle steps (matching `eval_pi05_libero`), so the numbers stay
comparable to the pi0.5 results. GR00T returns a 16-step chunk; `--replan-steps`
(default 8) of it are executed open-loop before the next inference.

## One-time setup

Accept the gated licenses first, logged in as your HF user:
<https://huggingface.co/nvidia/GR00T-N1.7-LIBERO> and
<https://huggingface.co/nvidia/Cosmos-Reason2-2B> (the VLM backbone).

```bash
scripts/groot/setup.sh
```

Inits `submodules/Isaac-GR00T` (`n1.7-release`), builds its `.venv`
(`uv sync --python 3.10` -- torch cu128 + flash-attn, large), adds
`websockets` + `msgpack`, and downloads the `libero_10` checkpoint (~7 GB) to
`checkpoints/GR00T-N1.7-LIBERO/`.

## Collect

Terminal 1 -- GR00T policy server (its own venv; model load ~1-3 min):

```bash
scripts/groot/serve.sh
```

Terminal 2 -- collect (12's top-level venv):

```bash
MUJOCO_GL=egl uv run python scripts/groot/collect.py \
  --replan-steps 8 --scene-variant both --num-trials 25
```

- `--replan-steps` is **required** (recorded, never assumed).
- `--scene-variant {normal,occluded,both}` (default `both`).
- `--task-id` (0-9) and `--episode-index` are optional; omit to sweep all 10
  tasks x `--num-trials` init states.
- `--keep-going` continues past a per-episode error (logs it, `n_errored += 1`).
- On CPU-only machines swap `MUJOCO_GL=egl` for `MUJOCO_GL=osmesa`.

`scripts/groot/run.sh` starts the server if `:8000` is free, runs `collect.py`,
and stops only a server it started. `STOP_PI05=1` first pkills the pi0.5 feature
server (one GPU can't hold both models):

```bash
STOP_PI05=1 scripts/groot/run.sh --replan-steps 8 --scene-variant both --num-trials 25
```

## Backbone feature capture (`--with-features`)

Start the server with `GROOT_WITH_FEATURES=1` (or `serve_groot_ws.py --with-features`)
and pass `--with-features` to `collect.py`:

```bash
GROOT_WITH_FEATURES=1 scripts/groot/serve.sh

MUJOCO_GL=egl uv run python scripts/groot/collect.py \
  --replan-steps 8 --scene-variant both --num-trials 25 --with-features
```

`server/feature_policy.py` is a `Gr00tPolicy` **subclass** (the upstream model is
never modified) that re-runs the pipeline and returns the **layer-16 residual
stream of the Cosmos-Reason2-2B (Qwen3-VL) backbone** -- raw, *before* the action
head's `vlln` + VL self-attention. Per inference it is split by token type using
the fixed chat layout (`server/token_layout.py`):

| npz key | shape (per `n_policy`) | dtype | |
|---|---|---|---|
| `base_image` | `(64, 2048)` | f16 | agentview: 256px / patch 16 -> 16x16 -> 2x2 merge -> 8x8 |
| `wrist_image` | `(64, 2048)` | f16 | wrist camera, same |
| `language` | `(200, 2048)` | f16 | instruction tokens (after the last `<|vision_end|>`, before `<|im_end|>`), zero-padded to 200 |
| `language_mask` | `(200,)` | bool | real vs pad |
| `language_len` | `()` per policy | i32 | real instruction token count (constant within an episode) |
| `state_features` | `(1536,)` | f16 | the action head's embedded proprio vector |

`has_features` (scalar bool) and a `has_features` manifest column mark which
episodes carry them. ~0.45 MB/inference (~10-30 MB/episode). `collect.py`
`--with-features` hard-fails if the server was not started with features.

vs pi0.5 SAVE-A: same 2048 hidden dim, but a **mid-stack** layer (16 of 28),
**64** image tokens/camera (2x2 merge) instead of 256, and the language block is
padded to 200 the same way. The unused right-wrist camera has no slot here.

### Alignment gate

After every episode `collect.py` reloads the saved files and runs
`validation.validate_rollout` (ported from `scripts/semantic_failure/`): the
`policy_step` ids are sequential, each inference's control block is contiguous
and starts where the previous ended, `chunk_index` runs 0,1,2..., `sim_step` is
strictly increasing and starts at `num_steps_wait`, `video_frame_id ==
control_step`, features are finite with `base_image != wrist_image`, and
**`executed_action[t] == decode_action_step(predicted_chunk[policy_step[t],
chunk_index[t]])`** (the decode step is applied because GR00T's gripper dim is
transformed before `env.step`). A failure prints the report and stops the run
unless `--keep-going`, which counts it in the final `gate_failed=` line. Each
`--with-features` reply is also checked by `feature_schema.identify_groot_features`.

### Auto-resume + manifest

`manifest.csv` at the output root is rewritten **after every episode** (atomic
`.tmp` -> rename). Re-running `collect.py` **skips any episode dir already
complete** (`rollout.json` + `rollout.npz` + `rollout.mp4` + `wrist.mp4` present
and parseable) and continues at the next incomplete
`(scene_variant, task_id, episode)`. `--fresh` re-collects.

## Compare

```bash
uv run python scripts/groot/compare.py [--run-dir outputs/groot/libero_10]
```

Pairs episodes on `(task, episode)` and writes `comparison/summary.json`
(per-task + overall normal SR, occluded SR, SR drop) and
`comparison/occlusion_failures.md` (normal-success / occluded-fail, with video
links).

## Output layout

```
outputs/groot/libero_10/                       (gitignored via outputs/*)
  manifest.csv
  <scene_variant>/<NN>_<task_stem>/ep<NNN>/
      rollout.json   metadata + per-policy clock summary
      rollout.npz    executed_actions + predicted_action_chunks + control/sim/policy/chunk clocks
                     (+ base_image / wrist_image / language / state_features with --with-features)
      rollout.mp4    agentview, 20 fps, one frame per control step
      wrist.mp4      robot0_eye_in_hand, same timing
  comparison/        (after compare.py)
  logs/              (server log, when started via run.sh)
```

`rollout.json` fields mirror `scripts/semantic_failure/`: identity (`rollout_id`,
`suite`, `scene_variant`, `task_id`, `task_file`, `episode_index`,
`instruction`), run config (`replan_steps`, `max_steps`, `seed`,
`num_steps_wait`, `checkpoint`, `model_id`, `feature_source`, `feature_server`,
`feature_server_version`, `control_hz`), outcome (`success`,
`sim_failure_category`, `failing_predicate`, `failure_detail` -- the
*simulator's* verdict), `elapsed_seconds`, `has_features`, `n_policy`,
`n_control`, and a `policies` list (`policy_step`, executed control-step range,
`predicted_chunk_length`, and per-policy `feature_source` / `feature_module` /
`feature_shapes` / `language_len` when features are captured).

`rollout.npz` carries **every array `scripts/semantic_failure/` writes** --
`executed_actions (n_control, 7)`, `predicted_action_chunks (n_policy, 16, 7)`
(GR00T's real horizon vs pi0.5's 10), `predicted_chunk_len`, the per-control-step
clocks `control_step` / `sim_step` / `policy_step` / `chunk_index` /
`video_frame_id`, and the scalars `success` / `replan_steps` / `n_policy` /
`n_control` / `img_tokens` (64) / `hidden` (2048) / `control_hz`. Plus GR00T
extras `has_features`, and with `--with-features` the backbone hidden states
(table above; `img_tokens` is 64 not pi0.5's 256), `language_len`, and
`state_features` -- all keyed row-for-row to `policy_step`.

`executed_actions[t]` is the **decoded** action (`decode_action_step` applied to
`predicted_action_chunks[policy_step[t], chunk_index[t]]`), so it does not equal
the raw chunk row on the gripper dim.

## Perception-probe sweep

`probe_*.py` here is a **byte-for-byte fork of `scripts/perception_probe/`'s
time-dependent failure probe + 7-phase sweep**, with only the feature source
swapped from pi0.5 to GR00T-N1.7 — so the two VLAs' probeability and
occlusion-failure signal can be compared directly
(`outputs/perception_probe/sweeps/` vs `outputs/groot/probe_sweeps/`).

The probe is SAFE-style: per-timestep attention-pool the frozen backbone tokens
→ one scalar → sigmoid → cumsum over time → "does this rollout fail?". Target is
env `success` (GR00T: 333 succ / 167 fail). No Gemini/VLM labels. 3 of the 10
tasks are held out entirely for a zero-shot **unseen** eval; runs are ranked by
unseen-split **`separation`** (standardized failure-vs-success score gap).

What differs from the pi0.5 sweep (and why it needs no probe-model change):
`PerceptionSuccessProbe` pools over the token axis, so GR00T's **64** image
tokens/camera (vs pi0.5's 256) and fixed **200** language tokens drop straight
in. GR00T taps a **mid-stack layer (16 of 28)**, pi0.5 the final prefix layer.
GR00T's `state_features` is **deliberately unused** — strict `base`/`wrist`/`lang`
parity. Reads `outputs/groot/libero_10/**/rollout.npz` directly (via
`probe_data.read_probe_manifest`), no feature re-collection.

Runs in the **top-level `.venv`** (plain PyTorch — no policy server, no submodule
venv; the `rollout.npz` files are already on disk).

```bash
# smoke train (2 epochs)
.venv/bin/python scripts/groot/probe_train.py --epochs 2 --no-wandb

# see the plan (79 runs, seed 0)
.venv/bin/python scripts/groot/probe_sweep.py --phase 1 2 3 4 5 6 7 --seeds 0 --dry-run

# run it (resumable; skips cells with latest/test_metrics.json)
bash scripts/groot/probe_sweep_parallel.sh      # phases 1-3 concurrent
bash scripts/groot/probe_sweep_phase4.sh
bash scripts/groot/probe_sweep_phase5.sh
bash scripts/groot/probe_sweep_phase67.sh
```

Output: `outputs/groot/probe_time_dependent/<ts>/` per run (`probe_best.pt`,
`test_metrics.json`, `unseen_metrics.json`, ROC + score-overlay PNGs, `probe.onnx`,
`split.json`); `outputs/groot/probe_sweeps/phaseN/summary.csv` ranked by unseen
`separation`, plus `results.csv` and `plots/`.

## Files

| file | role |
|---|---|
| `collect.py` | collector CLI: worklist, per-`(variant,task)` env reuse, manifest, auto-resume |
| `compare.py` | normal vs occluded success-rate summary from `manifest.csv` |
| `run.sh` | server-launch + collect wrapper (`STOP_PI05=1` frees the GPU) |
| `serve.sh` / `setup.sh` | launch the GR00T server / one-time submodule+venv+checkpoint setup |
| `server/serve_groot_ws.py` | GR00T policy behind the openpi msgpack websocket (runs in the submodule venv); `--with-features` |
| `server/feature_policy.py` | `Gr00tFeaturePolicy(Gr00tPolicy)` -- subclass that also returns layer-16 backbone hidden states |
| `server/token_layout.py` | pure chat-token split helpers (`image_runs`, `instruction_span`) |
| `server/msgpack_numpy.py` / `server/smoke_server.py` | wire codec / one-shot server sanity check |
| `groot_obs.py` | `build_flat_obs` / `decode_action_step` -- the NVIDIA LIBERO obs/action bridge |
| `libero_env.py` | `open_task` / `goal_predicate_strings` via 12's `eval_pi05_libero` (read-only) |
| `rollout.py` | control loop + the four clocks + feature identity check |
| `recorder.py` / `serialization.py` / `video.py` | write `rollout.{mp4,json,npz}` + `wrist.mp4` |
| `validation.py` | `validate_rollout` -- the alignment gate (ported from `semantic_failure`) |
| `feature_schema.py` | `identify_groot_features` -- backbone-feature identity check |
| `present.py` | `example.md` / `EXAMPLES.md` cards |
| `records.py` | dataclasses (`RolloutRecord`, `PolicyRecord`, `ControlRecord`, `GrootFeatures`) |
| `manifest.py` | `rebuild_manifest`, `episode_is_complete`, `find_next_incomplete` |
| `constants.py` | shared constants + `EvalConfig` |
| `probe_model.py` | verbatim copy of `scripts/perception_probe/probe_model.py` (`PerceptionSuccessProbe`) |
| `probe_utils.py` | vendored metric / ROC / split / overlay helpers (verbatim from `scripts/perception_probe/`) |
| `probe_data.py` | `read_probe_manifest` (groot manifest → trainer row shape) + `EpisodeSequenceDataset` / `collate_pad_episodes` |
| `probe_train.py` | fork of `train_probe_time_dependent.py` — SAFE-style probe trainer on GR00T backbone features |
| `probe_sweep.py` + `probe_sweep_*.sh` | fork of `sweep.py` — 7-phase sweep (grids identical to the pi0.5 sweep) |
| `tests/` | `test_groot_obs` / `test_manifest` / `test_token_layout` / `test_validation` / `test_feature_schema` / `test_probe_data` (pure; no sim / network / GR00T venv) |

```bash
uv run python -m unittest discover -s scripts/groot/tests
```
