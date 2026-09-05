# Vendored / ported code

| file(s) | origin | notes |
|---|---|---|
| `constants.py` | `16-LIBERO-X-GR00T-ZeroShot/sim/config.py` | ported; `EvalConfig` repointed at `submodules/LIBERO-X` and `outputs/libero-x/` |
| `libero_x_env.py` | `16-LIBERO-X-GR00T-ZeroShot/sim/liberox_env.py` | ported near-verbatim; `list_bddl_files` uses plain lexicographic `sorted()` instead of `natsort` (`natsort` is not a top-level dependency of this repo). This is still fully deterministic/reproducible, but **not** the same ordering `natsort` would give for filenames with multi-digit scene numbers (e.g. `SCENE8` sorts after `SCENE34` lexicographically), so a given `--seed` picks a *different* (equally valid) task sample here than in repo 16 -- do not expect identical `tasks.txt` contents across the two. |
| `run_eval.py` | `16-LIBERO-X-GR00T-ZeroShot/sim/run_eval.py` | ported; imports `build_flat_obs`/`decode_action_step` from `scripts/groot/groot_obs.py` (via a process-scoped `sys.path` insert, same pattern `scripts/groot/libero_env.py` uses to reach `scripts/evaluation/`) instead of duplicating them |
| `sample_tasks.py` | `16-LIBERO-X-GR00T-ZeroShot/sim/sample_tasks.py` | verbatim (sort switched to plain `sorted()`, see above) |
| `setup.sh` / `run_eval.sh` / `run_all.sh` | `16-LIBERO-X-GR00T-ZeroShot/scripts/{setup_groot,run_eval,run_all}.sh` | adapted: no separate `Isaac-GR00T` clone/venv or `liberox` conda env (this repo already has both, via `scripts/groot/`); `run_all.sh` follows `scripts/groot/run.sh`'s port-check/cleanup pattern instead of `screen` sessions |

**Not** ported / duplicated here (reused from `scripts/groot/` unchanged instead):
`server/serve_groot_ws.py`, `server/msgpack_numpy.py`, `groot_obs.py` -- these
already implement the exact wire protocol and LIBERO_PANDA obs/action
convention this pipeline needs, including the `LIBEROX_NATIVE_CONVENTION`
env-var toggle placed there for a checkpoint fine-tuned natively on
`meituan/LIBERO-X` (off by default; the `nvidia/GR00T-N1.7-LIBERO` checkpoint
used here expects the NVIDIA convention).

`submodules/LIBERO-X` is a new git submodule -> `https://github.com/meituan/LIBERO-X.git`,
pinned at `f528726421c7211d8eb05fe48e9e5e2535ccc813`. Read-only: BDDL files,
`.init` states, and LIBERO-X's own `libero` package (which forks
`envs/bddl_base_domain.py`, adds `envs/objects/{extension_objects,libero_x_objects}.py`
and `utils/parse_bddl.py` relative to `submodules/openpi/third_party/libero` --
confirmed by diff, hence the need for its own submodule rather than reusing
openpi's `libero` checkout). `libero_x_env.configure_libero_x` inserts this
submodule's root at the front of `sys.path` for the current process only, so
it never conflicts with another script importing a different `libero` fork
in its own process.

`scripts/groot/`, `scripts/evaluation/`, `scripts/perception_probe/`, and
`scripts/semantic_failure/` are not modified.
