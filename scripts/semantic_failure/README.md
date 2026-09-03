# `scripts/semantic_failure/`

Collect **row-aligned** `libero_10` rollouts (normal + LIBERO-Occ) and label
them with Gemini. Ported from `17-LIBERO-10-Semantic-Failure-Pipeline`; the
labeling half (`dan_label_with_vlm.py`) is vendored from
[dtl184/liberox-evals](https://github.com/dtl184/liberox-evals) -- see
[`VENDOR.md`](VENDOR.md).

Scope: **`libero_10` only** (both scene variants). Nothing else in 12 is
modified -- this folder imports `scripts/evaluation/eval_pi05_libero.py`
read-only (the same `sys.path` trick `collect_features.py` uses) and reuses
`scripts/perception_probe/serve_pi05_with_features.py` as the policy server.

## What each episode records

For one rollout, joinable by array row:

- **SAVE-A prefix features** from every pi0.5 inference (`base_image` /
  `wrist_image` 256x2048, `language`, `language_mask`) -- validated
  (`identify_save_a`: shapes, finite, base != wrist; a stock server is rejected).
- **Executed action** for every control step, and the **full predicted action
  chunk** for every inference.
- **Four clocks** per control step: `control_step`, `sim_step` (wait steps
  counted), `policy_step`, `chunk_index`.
- **Two 20 fps videos** -- `rollout.mp4` (agentview) and `wrist.mp4`
  (`robot0_eye_in_hand`), both 180deg-corrected, exactly one frame per control
  step (`video_frame_id == control_step`). Wait steps are `sim_step`-only and
  are not in either video. Gemini labeling watches `rollout.mp4` only.
- On failure: the unsatisfied simulator goal predicate strings.
- After labeling: a 3-second keyword-phrase `semantic_timeline` (every episode)
  and Dan's `vlm_failure` + `failure_annotation` (failed episodes; onset frame
  mapped onto `control` / `policy` / `chunk`). These -- with the Gemini model and
  prompt templates used -- go into a **separate `labels.json` + `labels.npz`**;
  `rollout.{json,npz,mp4}` are never rewritten.

`validate_rollout` asserts the whole alignment graph after each save and prints
a PASS/FAIL report.

## Install

The extra deps (`google-genai`, `imageio`, `imageio-ffmpeg`) are in the repo's
`pyproject.toml`:

```bash
uv sync
```

## Collect

Terminal 1 -- feature server (openpi's JAX venv):

```bash
submodules/openpi/.venv/bin/python scripts/perception_probe/serve_pi05_with_features.py
```

Terminal 2 -- collect (12's top-level venv):

```bash
MUJOCO_GL=egl uv run python scripts/semantic_failure/collect.py \
  --replan-steps 5 --scene-variant both --num-trials 50
```

- `--replan-steps` is **required** (recorded, never assumed).
- `--scene-variant {normal,occluded,both}` (default `both`).
- `--task-id` (0-9) and `--episode-index` are optional; omit to sweep all 10
  tasks x `--num-trials` init states.
- `--label` also runs Gemini right after each episode (needs `GEMINI_API_KEY`);
  default is collect-now / label-later.
- On CPU-only machines swap `MUJOCO_GL=egl` for `MUJOCO_GL=osmesa`.

`scripts/semantic_failure/run.sh` starts the feature server if `:8000` is free,
runs `collect.py "$@"`, and stops only a server it started.

### Auto-resume + manifest

`manifest.csv` at the output root is rewritten **after every episode** (atomic
`.tmp` -> rename), so an interrupted run still leaves a complete manifest.
Columns include `success`, `n_control`, `sim_failure_category`,
`failing_predicate`, `labeled_captions`, `labeled_failure`,
`vlm_failure_onset_frame`, `dir`.

Re-running `collect.py` **skips any episode dir that is already complete**
(`rollout.json` + `rollout.npz` + `rollout.mp4` + `wrist.mp4` present and
parseable) and continues with the next incomplete
`(scene_variant, task_id, episode)`. Pass
`--fresh` to re-collect, `--keep-going` to continue past an alignment-gate
failure.

## Label

```bash
GEMINI_API_KEY=... uv run python scripts/semantic_failure/label_run.py \
  [outputs/semantic_failure/libero_10]
```

Labels every episode under the run dir, **skipping already-labeled ones**
(`--force` to redo), refreshes `manifest.csv`, and writes `EXAMPLES.md`. Single
episode: `label.py <episode_dir>` (`--no-refine` to skip Dan's refine pass).

Labeling needs **no GPU / simulator / feature server** -- only each episode's
`rollout.json` + `rollout.mp4` and `GEMINI_API_KEY`. To label a dataset pulled
from HF rather than a local collect run, a minimal env is enough (no `uv sync`):

```bash
pip install numpy google-genai imageio imageio-ffmpeg huggingface_hub
hf download podolinsky/pi0.5-libero-10-features --repo-type dataset \
  --local-dir data/pi05-libero-10 \
  --include "*rollout.json" "*rollout.mp4" "manifest.csv"
GEMINI_API_KEY=... python scripts/semantic_failure/label_run.py data/pi05-libero-10
hf upload podolinsky/pi0.5-libero-10-features data/pi05-libero-10 . \
  --repo-type dataset \
  --include "*labels.json" "*labels.npz" "*example.md" "manifest.csv" "EXAMPLES.md"
```

The top-level [`README.md`](../../README.md) has the same steps in more detail.

Per episode (`episode.label_episode`): the episode mp4 is uploaded to Gemini
**once** (with an explicit cache when the account allows it); Dan's failure pass
runs first (coarse on the full video, then a ±3-second 1-FPS refine clip as a
separate one-shot), then the 3-second keyword phrases run as a **second turn on
the same session**, handed a compact failure summary as context but told not to
copy it into the phrases. Collection never calls Gemini unless `--label` is
passed.

## Output layout

```
outputs/semantic_failure/libero_10/            (gitignored via outputs/*)
  manifest.csv
  <scene_variant>/<NN>_<task_stem>/ep<NNN>/
      rollout.json   collection metadata + per-policy/per-control clock records
      rollout.npz    float16 feature tensors + action chunks + executed + clock arrays
      rollout.mp4    agentview, 20 fps, one frame per control step
      wrist.mp4      robot0_eye_in_hand, same timing
      labels.json    Gemini model + prompts + phrases + failure  (only after labeling)
      labels.npz     sem_* phrases + fail_* fields as arrays     (only after labeling)
      example.md     (after labeling)
  EXAMPLES.md        (after label_run.py)
```

**`rollout.{json,npz,mp4}` are written once by collection and never touched
again.** Labeling adds only `labels.json` + `labels.npz` + `example.md` (a few KB
per episode), so a labeled dataset is a cheap diff on top of the raw one.

### `rollout.json` (collection; frozen)

Flat metadata: `rollout_id`, `suite`, `scene_variant`, `task_id`, `task_file`,
`episode_index`, `instruction`, `replan_steps`, `success`, `max_steps`, `seed`,
`model_id`, `checkpoint`, `feature_source`, `feature_server`,
`feature_server_version`, `control_hz`, `num_steps_wait`,
`sim_failure_category`, `failing_predicate`, `failure_detail` (these three are
the *simulator's* verdict, not Gemini's), `video_path`, `wrist_video_path`,
`n_policy`, `n_control`.
Then `policies` -- a per-inference summary: `{policy_step,
executed_control_step_start, executed_control_step_end, predicted_chunk_length,
feature_source, feature_module, feature_shapes}`.

Per-control-step arrays and the feature tensors are **not** here -- they are in
`rollout.npz`.

### `labels.json` (labeling; additive)

- `rollout_id` / `scene_variant` / `task_id` / `task_file` / `instruction` /
  `success` -- enough to identify the episode standalone.
- `labeler` -- `{backend, model, refine, labeled_at, pipeline, prompts}`, where
  `prompts` holds the verbatim templates used: `keyword_phrases`,
  `failure_coarse`, `failure_refine`, `failure_taxonomy`.
- `semantic_timeline` -- list of `{segment_index, t_start_sec, t_end_sec,
  control_step_start, control_step_end, policy_step_start, policy_step_end,
  phrase}` (`description` also written as an alias).
- `vlm_failure` -- the full dict from Dan's localizer (mode, onset
  type/seconds/timestamp/frame, coarse vs refined, reason, recovery,
  justification, token usage, raw responses); `{}` on success.
- `failure_annotation` -- `{failure_control_step, failure_sim_step,
  failure_policy_step, failure_chunk_index, first_post_failure_policy_step,
  failure_type, correction_action}` (the VLM onset mapped onto the clocks).

`labels.npz` is the array-aligned copy of the phrases + failure fields for
downstream loading. `serialization.load_rollout(dir)` reads `labels.json` when
present and repopulates `rollout.semantic_timeline` / `.vlm_failure` / `.failure`;
`load_label_document(dir)` / `load_labels(dir)` return the raw json / npz.

## Files

| file | role |
|---|---|
| `collect.py` | collector CLI: worklist, per-`(variant,task)` env reuse, gate, manifest, optional label |
| `label.py` / `label_run.py` | label one episode / a whole run dir (resume-aware) |
| `run.sh` | server-launch + collect wrapper |
| `libero_env.py` | `open_task` / `goal_predicate_strings` via 12's `eval_pi05_libero` (read-only) |
| `rollout_runner.py` | control loop + feature infer + the four clocks |
| `recorder.py` / `serialization.py` / `video.py` | write `rollout.{mp4,json,npz}` (collect) and `labels.{json,npz}` (label) |
| `records.py` | dataclasses (`RolloutRecord`, `PolicyRecord`, `ControlRecord`, ...) |
| `feature_schema.py` / `pi05_client.py` | SAVE-A validation + the websocket client |
| `validation.py` | `validate_rollout` alignment assertions |
| `manifest.py` | `rebuild_manifest`, `episode_is_complete`, `find_next_incomplete` |
| `episode.py` | `label_episode`: Dan failure then 3-second phrases on one shared video session |
| `gemini_session.py` | `GeminiVideoSession`: upload the mp4 once, reuse it (+ cache) across turns |
| `semantic.py` | 3-second keyword-phrase prompt + `caption_timeline_on_session` |
| `failure.py` / `dan_label_with_vlm.py` | Dan's coarse+refine failure localizer + clock mapping |
| `present.py` | `example.md` / `EXAMPLES.md` |
| `constants.py` | shared constants + `PipelineConfig` |
| `tests/` | `test_records.py`, `test_timeline.py` (pure; no sim/network) |

```bash
uv run python -m unittest discover -s scripts/semantic_failure/tests
```
