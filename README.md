# visual-occlusion-reasoning

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

Generated artifacts live below `outputs/`. Benchmark assets and the OpenPI
implementation are pinned in `submodules/Libero-Occ/` and
`submodules/openpi/`, respectively.
