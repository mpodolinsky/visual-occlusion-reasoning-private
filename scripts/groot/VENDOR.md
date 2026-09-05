# Vendored / ported code

| file(s) | origin | notes |
|---|---|---|
| `server/serve_groot_ws.py` | `16-LIBERO-X-GR00T-ZeroShot/server/serve_groot_ws.py` | verbatim; default `--model-path` repointed at this repo's `checkpoints/` |
| `server/msgpack_numpy.py` | openpi-client `openpi_client/msgpack_numpy.py` (via repo 16) | verbatim wire codec |
| `server/smoke_server.py` | `16-LIBERO-X-GR00T-ZeroShot/server/smoke_server.py` | verbatim |
| `server/feature_policy.py` | new (not from repo 16) | `Gr00tFeaturePolicy` **subclasses** `gr00t.policy.gr00t_policy.Gr00tPolicy` and re-runs its `_get_action` pipeline to also return the layer-16 backbone hidden states. The GR00T package is imported, never edited. |
| `server/token_layout.py` | new | pure token-id split helpers |
| `groot_obs.py` | `16-LIBERO-X-GR00T-ZeroShot/sim/groot_obs.py` | ported; `quat2axisangle` inlined instead of imported |
| `setup.sh` / `serve.sh` / `run.sh` | `16-LIBERO-X-GR00T-ZeroShot/scripts/{setup_groot,serve_groot,run_all}.sh` | adapted to 12's submodule layout |

Repo 16's obs/action bridge itself mirrors **NVIDIA/Isaac-GR00T @ `n1.7-release`**
`gr00t/eval/sim/LIBERO/libero_env.py` (180deg-rotated 256px agentview + wrist,
8-dim EEF-pose+gripper state, 7-dim delta-EEF action with the gripper
normalised/inverted). The GR00T model code (`gr00t.policy.gr00t_policy`) is the
unmodified upstream package, installed into `submodules/Isaac-GR00T/.venv`.

`constants.py`, `libero_env.py`, `manifest.py`, `recorder.py`, `serialization.py`,
`records.py`, `rollout.py`, `video.py`, `collect.py`, `present.py` follow the
structure of `scripts/semantic_failure/` in this repo, with the Gemini-labeling
machinery removed. `validation.py` (`validate_rollout` alignment gate) and
`feature_schema.py` (`identify_groot_features`) are ported from the matching
`semantic_failure` files, adapted to `GrootFeatures` and the decode-aware
executed<->predicted check. `rollout.json` / `rollout.npz` carry every field the
pi0.5 pipeline writes (see that folder's README), plus GR00T extras
(`state_features`, `language_len`, `has_features`).

`libero_env.py` imports `scripts/evaluation/eval_pi05_libero.py` read-only for
the LIBERO / LIBERO-Occ benchmark plumbing; nothing under `scripts/evaluation/`
is modified.

## Perception-probe sweep fork

| file(s) | origin | notes |
|---|---|---|
| `probe_model.py` | `scripts/perception_probe/probe_model.py` | verbatim (`TokenPool`, `PerceptionSuccessProbe`); already generic over token count |
| `probe_utils.py` | `scripts/perception_probe/train_probe.py` + `rollout_unseen_with_scores.py` | verbatim helpers: `auroc`, `confusion_matrix`, `roc_curve_points`, `plot_roc_curve`, `available_memory_gb`, `split_episodes`, `plot_overlay` |
| `probe_data.py` | new + `scripts/perception_probe/train_probe_time_dependent.py` | `read_probe_manifest` maps the `scripts/groot/manifest.py` CSV (`dir` / `n_policy` / `n_control`) into the row shape the trainer wants (`npz_path` / `inference_calls` / `control_frames`); `EpisodeSequenceDataset` / `_edge_pad` / `collate_pad_episodes` copied verbatim |
| `probe_train.py` | `scripts/perception_probe/train_probe_time_dependent.py` | structural fork. All SAFE-ported losses / `score_sequence` / `run_eval_pass` / seen-unseen split / `separation` metric verbatim. Changes: local vendored imports; `read_manifest` -> `read_probe_manifest`; suite filter via the manifest `suite` column instead of `collect_features.build_task_suite_map`; `NUM_IMAGE_TOKENS=64`, `REPLAN_STEPS=8`; GR00T output/feature default paths. Gemini/label machinery: absent (same as the rest of `scripts/groot/`). |
| `probe_sweep.py` + `probe_sweep_*.sh` | `scripts/perception_probe/sweep.py` + `run_*.sh` | the 7 phase grids (`PHASE1`..`PHASE7`) are byte-for-byte identical; only `TRAIN`, `SWEEPS_ROOT`, and the `--features-dir` default point at GR00T |

`state_features` (GR00T's proprioceptive vector, absent in pi0.5) is deliberately
not wired into the probe -- strict 3-modality (`base`/`wrist`/`lang`) parity with
the pi0.5 sweep so the two VLAs' results compare directly. `scripts/perception_probe/`
itself is not modified.
