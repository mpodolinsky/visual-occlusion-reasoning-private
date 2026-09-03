# Steering Vision Language Action Models Using Latent Spaces

Utilities for evaluating an OpenPI VLA on matched LIBERO episodes with and
without the scene occlusions from LIBERO-Occ.

The core workflow evaluates the same 40 tasks and 50 released initial states
in two scene variants:

- `occluded`: the four LIBERO-Occ suites.
- `normal`: the filename- and initial-state-matched standard LIBERO suites.

## Setup

The project targets Python 3.11 and uses `uv` for its environments. Install
Git, [`uv`](https://docs.astral.sh/uv/), and `ffmpeg`; the evaluator uses
`ffmpeg` to write reviewable MP4 rollouts.

From the repository root, fetch the benchmark and OpenPI submodules and create
both environments:

```bash
git submodule update --init --recursive
uv sync
cd submodules/openpi && uv sync
```

The root environment contains the evaluation and analysis tools. OpenPI has a
separate environment for its policy server. The first server launch downloads
and caches the default `pi05_libero` checkpoint.

The commands below use EGL for headless MuJoCo rendering. On a CPU-only machine
with OSMesa installed, replace `MUJOCO_GL=egl` with `MUJOCO_GL=osmesa`.

## Run the core evaluation

Start the stock OpenPI pi0.5 LIBERO policy server in one terminal:

```bash
cd submodules/openpi
uv run scripts/serve_policy.py --env LIBERO
```

In a second terminal, return to this repository's root and evaluate the
occluded episodes:

```bash
MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py \
  --scene-variant occluded \
  --num-trials-per-task 50
```

Then evaluate the matched vanilla LIBERO episodes:

```bash
MUJOCO_GL=egl uv run python scripts/evaluation/eval_pi05_libero.py \
  --scene-variant normal \
  --num-trials-per-task 50
```

Each command runs 2,000 episodes: 4 suites x 10 tasks x 50 initial states.
For a quick end-to-end check, use `--num-trials-per-task 1` and a separate
`--output-dir` before starting the full runs.

After both full runs finish, compare matched outcomes:

```bash
uv run python scripts/evaluation/compare_pi05_libero_runs.py
```

By default, evaluation results are written to
`outputs/pi05_libero_occ/` and `outputs/pi05_libero_matched_normal/`; comparison
artifacts are written to `outputs/pi05_libero_comparison/`.

See [`scripts/evaluation/README.md`](scripts/evaluation/README.md) for smoke
tests, resumable and detached runs, output files, episode-pairing rules, and
evaluator options.

# Collect features and train the perception probe

Start the OpenPI pi0.5 LIBERO policy server (modified to expose internal embeddings) in one terminal:

```bash
submodules/openpi/.venv/bin/python scripts/perception_probe/serve_pi05_with_features.py
```

Then run the policy and save the embeddings:

```bash
NUM_TRIALS=5 scripts/perception_probe/collect_all.sh
```

By default, embedding-success pairs will be saved as [`.npz`] in [`outputs/perception_probe/features/`](outputs/perception_probe/features/). Train the perception probe by running: 

```bash
.venv/bin/python scripts/perception_probe/train_probe.py
```

Depending on your available RAM, you may need to reduce [`--num-workers`] (4 by default). During training, we ramdomly select [`--chunk-size`] episodes and decompress them together. We then randomly sample steps from these episodes to compute the gradient update.

The training loop outputs to [`outputs/perception_probe/probe`](outputs/perception_probe/probe/).

### Time-dependent training (SAFE-style)

[`scripts/perception_probe/train_probe_time_dependent.py`](scripts/perception_probe/train_probe_time_dependent.py)
is a second training loop, ported from
[SAFE (NeurIPS 2025)](https://github.com/tum-vision/safe)'s `IndepModel`: each
episode's per-timestep score is accumulated over time (`cumsum`, optionally a
running mean via `--rmean`) and trained with either SAFE's own hinge loss or a
per-timestep BCE loss (`--raw-target-loss`) that pushes the raw sigmoid score
toward 1 at every timestep of a failure episode and 0 at every timestep of a
success one. It only uses `libero_10_occluded` (both scene variants) and
holds a random subset of tasks out entirely as a zero-shot `unseen` split
(`--num-unseen-tasks`, seeded by `--unseen-task-seed`). Eval-time AUROC is
computed with each episode truncated to its task's shortest observed rollout
length (`--no-task-min-step-eval` to disable), which keeps LIBERO's episode
length -- tied directly to success/failure, since a rollout stops the instant
it succeeds but always runs to the suite's step cap on failure -- from leaking
into the metric:

```bash
.venv/bin/python scripts/perception_probe/train_probe_time_dependent.py \
  --raw-target-loss --epochs 15
```

Each run directory (`outputs/perception_probe/probe_time_dependent[_rawtarget]/<timestamp>/`)
gets, automatically: `probe_best.pt`/`probe_last.pt`, `split.json` (the
train/val/test/calibration/unseen split and per-task `task_min_step`
values), `test_metrics.json`/`unseen_metrics.json` + ROC curves, an ONNX
export of the checkpoint (`probe.onnx`, for inspection e.g. in
[Netron](https://netron.app)), and `test_scores_overlay.png`/
`unseen_scores_overlay.png` (every episode's raw + accumulated score over
time, failures in red / successes in blue, with a bold mean curve per
outcome -- see `plot_overlay()` in `rollout_unseen_with_scores.py`).

Two related scripts aren't run automatically:

- [`eval_run.py`](scripts/perception_probe/eval_run.py): reruns the unseen
  eval + ONNX export standalone, for a run whose training crashed before
  reaching its own final eval pass.
- [`rollout_unseen_with_scores.py`](scripts/perception_probe/rollout_unseen_with_scores.py):
  rolls out `--num-episodes` real episodes per unseen task against a live
  feature-serving policy server (see below), scoring each step with the
  trained probe and saving a review video + score plot per episode, plus an
  overlay across all of them.

### t-SNE of raw features

[`scripts/perception_probe/plot_tsne_features.py`](scripts/perception_probe/plot_tsne_features.py)
projects every cached per-timestep feature (mean-pooled per modality) for a
suite down to 2D with t-SNE (perplexity 30, matching sklearn's -- and SAFE's
own `visualize_features.py`, which never overrides it -- default) and, from
that single embedding (t-SNE is stochastic, so it's fit once and reused),
saves three colored views -- success/failure, task id, and scene variant
(occluded vs. normal) -- each in both an annotated and a bare (`-clean`, no
title/legend) version. Pass `--cache` to skip re-fitting on repeat runs.

# Collect and label semantic + failure rollouts

[`scripts/semantic_failure/`](scripts/semantic_failure/README.md) collects
`libero_10` rollouts (normal + occluded) in which every pi0.5 inference, SAVE-A
prefix tensor, executed action, predicted action chunk, and video frame is
joinable by array row, then labels each episode with Gemini.

Start the feature server (same one the perception probe uses), then collect --
`--replan-steps` is mandatory, both scene variants sweep by default, and a
`manifest.csv` is rewritten after every episode:

```bash
submodules/openpi/.venv/bin/python scripts/perception_probe/serve_pi05_with_features.py

MUJOCO_GL=egl uv run python scripts/semantic_failure/collect.py \
  --replan-steps 5 --scene-variant both --num-trials 50
```

Re-running the command skips episodes already on disk and picks up the next
incomplete `(scene_variant, task_id, episode)`. Then label (needs
`GEMINI_API_KEY`):

```bash
GEMINI_API_KEY=... uv run python scripts/semantic_failure/label_run.py
```

Each episode's mp4 is uploaded to Gemini once and reused across turns: Dan's
coarse+refine failure localizer runs first on the failed episodes, then
3-second keyword phrases (e.g. "reaching for mug") are captioned for every
episode. Labels -- plus the Gemini model and prompt templates used -- are
written to a separate `labels.json` + `labels.npz`; the collection artifacts
(`rollout.json` / `rollout.npz` / `rollout.mp4`) are never rewritten, so a
labeled dataset is a cheap diff on top of the raw one. `label_run.py` skips
already-labeled episodes and writes `EXAMPLES.md`.

`scripts/semantic_failure/run.sh` wraps the server launch + collect into one
command. See that folder's README for the output layout, the alignment
contract, and per-episode labeling.

## Label an already-collected dataset (no GPU / sim / policy server)

A collected run of 500 episodes (10 tasks x 25 normal + 25 occluded) lives on
the Hub at
[`podolinsky/pi0.5-libero-10-features`](https://huggingface.co/datasets/podolinsky/pi0.5-libero-10-features)
(private). Labeling only reads each episode's `rollout.json` + `rollout.mp4` and
calls Gemini -- it needs neither a GPU, the simulator, nor the feature server,
so **do not run `uv sync` for this**; a minimal environment is enough:

```bash
python -m venv .venv-label && . .venv-label/bin/activate
pip install numpy google-genai imageio imageio-ffmpeg huggingface_hub
```

Download the episodes -- either the full ~53 GB, or just the videos + metadata
the labeler reads (the ~50 GB of feature `.npz` is not needed to label; labels
re-attach to the full set later):

```bash
hf download podolinsky/pi0.5-libero-10-features --repo-type dataset \
  --local-dir data/pi05-libero-10 \
  --include "*rollout.json" "*rollout.mp4" "manifest.csv"
```

Label every episode under that directory, then push the labels back:

```bash
export GEMINI_API_KEY=...
python scripts/semantic_failure/label_run.py data/pi05-libero-10

hf upload podolinsky/pi0.5-libero-10-features data/pi05-libero-10 . \
  --repo-type dataset \
  --include "*labels.json" "*labels.npz" "*example.md" "manifest.csv" "EXAMPLES.md"
```

`label_run.py` walks `*/*/ep*/` under the path, writes `labels.json` +
`labels.npz` + `example.md` into each episode dir, refreshes `manifest.csv`, and
writes `EXAMPLES.md`. It **skips episodes that already carry labels**, so it is
safe to interrupt and resume. Flags: `--force` re-labels, `--no-refine` skips
the +-3 s failure-onset refine pass, `--model <id>` picks a different Gemini
model (default `gemini-3.1-pro-preview`). Single episode:
`python scripts/semantic_failure/label.py data/pi05-libero-10/<variant>/<NN>_<task>/ep<NNN>`.

## Script layout

- [`scripts/evaluation/`](scripts/evaluation/README.md): run the policy on
  matched normal/occluded episodes and compare their outcomes.
- [`scripts/capture/`](scripts/capture/README.md): capture initialized
  LIBERO-Occ observations for inspection.
- [`scripts/figures/`](scripts/figures/README.md): assemble suite composites
  and render hand-scripted wrist-camera illustrations.
- [`scripts/sharing/`](scripts/sharing/): turn comparison results into
  collaborator-facing Markdown/PDF reports.
- [`scripts/perception_probe/`](scripts/perception_probe/): collect features and train perception probe.
- [`scripts/semantic_failure/`](scripts/semantic_failure/README.md): collect
  row-aligned `libero_10` rollouts (features + executed actions + predicted
  chunks + `control`/`sim`/`policy`/`chunk` clocks + 20 Hz agentview & wrist
  video) and label them with Gemini 3-second keyword phrases and Dan's two-pass
  failure localizer. Global manifest + auto-resume.

Generated artifacts live below `outputs/`. Benchmark assets and the OpenPI
implementation are pinned in `submodules/Libero-Occ/` and
`submodules/openpi/`, respectively.
