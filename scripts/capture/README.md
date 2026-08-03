# Capture scripts

`save_libero_occ_samples.py` renders initialized observations from the four
released LIBERO-Occ suites. Complete the repository [setup](../../README.md#setup)
and run commands from the repository root.

Capture the first task from each suite:

```bash
MUJOCO_GL=egl uv run python scripts/capture/save_libero_occ_samples.py
```

This writes four PNGs below `outputs/libero_occ_samples/`. Capture all 10 tasks
in every suite with:

```bash
MUJOCO_GL=egl uv run python scripts/capture/save_libero_occ_samples.py \
  --all-tasks \
  --output-dir outputs/libero_occ_tasks
```

The 40 output filenames are numbered `1-` through `10-` within each suite so
they can be passed directly to the [figure scripts](../figures/README.md).

Use `MUJOCO_GL=osmesa` instead of `egl` on a CPU-only machine with OSMesa
installed. Run the built-in help for camera, resolution, seed, settling-step,
render-device, and output options:

```bash
uv run python scripts/capture/save_libero_occ_samples.py --help
```
