# Figure scripts

These utilities create suite overview grids and hand-scripted wrist-camera
illustrations. Complete the repository [setup](../../README.md#setup) and run
commands from the repository root.

## Suite composites

First [capture all 40 tasks](../capture/README.md), then build one 2-column by
5-row image grid per suite:

```bash
uv run python scripts/figures/make_libero_occ_composites.py
```

The script reads `outputs/libero_occ_tasks/` and writes suite PNGs to
`outputs/libero_occ_composites/` by default. Use `--input-dir` and
`--output-dir` to override those locations.

## Wrist-camera illustrations

The three wrist-camera scripts use fixed tasks and hand-scripted motions to
render illustrative stills. They do not run the VLA or implement an
active-perception policy.

```bash
MUJOCO_GL=egl uv run python scripts/figures/render_wrist_bowl_demo.py
MUJOCO_GL=egl uv run python scripts/figures/render_wrist_mug_demo.py
MUJOCO_GL=egl uv run python scripts/figures/render_wrist_occlusion_demo.py
```

They write to `outputs/wrist_camera_bowl_demo/`,
`outputs/wrist_camera_mug_demo/`, and
`outputs/wrist_camera_occlusion_demo/`, respectively. On a CPU-only machine
with OSMesa installed, set `MUJOCO_GL=osmesa` instead of `MUJOCO_GL=egl`.
