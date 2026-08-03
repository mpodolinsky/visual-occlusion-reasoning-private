#!/usr/bin/env python3
"""Render a wrist-camera occlusion demo for the black bowl / cookie box task.

Runs the LIBERO-Occ "pick up the black bowl next to the cookie box and place
it on the plate" task (libero_spatial_occluded suite) from the wrist camera
(robot0_eye_in_hand). The scripted trajectory:

1. Right after reset, a short sideways move crops more than half of the
   black bowl out of the wrist camera's view -- the "partially occluded"
   frame.
2. Reversing that move (down/left, then back) brings the bowl fully and
   clearly into frame -- the "completely visible" frame.

The active-perception policy is not implemented yet; this script hand-scripts
a plausible camera motion purely to produce illustrative before/after stills.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "submodules" / "Libero-Occ" / "benchmark_assets"
LIBERO_ROOT = REPO_ROOT / "submodules" / "openpi" / "third_party" / "libero"
OUT_DIR = REPO_ROOT / "outputs" / "wrist_camera_bowl_demo"

SUITE = "libero_spatial_occluded"
TASK = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
RESOLUTION = 512
CAMERA = "robot0_eye_in_hand"


def configure_libero() -> None:
    sys.path.insert(0, str(LIBERO_ROOT))
    config_dir = REPO_ROOT / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config = {
        "benchmark_root": str(BENCHMARK_ROOT),
        "bddl_files": str(BENCHMARK_ROOT / "bddl_files"),
        "init_states": str(BENCHMARK_ROOT / "init_files"),
        "datasets": str(REPO_ROOT / "outputs" / "datasets"),
        "assets": str(BENCHMARK_ROOT),
    }
    (config_dir / "config.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in config.items())
    )


def make_optional_matplotlib_stub() -> None:
    try:
        import matplotlib.cm  # noqa: F401
    except ModuleNotFoundError:
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []  # type: ignore[attr-defined]
        matplotlib.cm = ModuleType("matplotlib.cm")  # type: ignore[attr-defined]
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.cm"] = matplotlib.cm


def load_initial_state(path: Path, torch):
    return torch.load(path, map_location="cpu", weights_only=False)[0]


def save(img_arr, path: Path, image_class) -> None:
    image = img_arr[::-1, ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    image_class.fromarray(image).save(path)


def make_env(offscreen_render_env, torch):
    bddl_path = BENCHMARK_ROOT / "bddl_files" / SUITE / f"{TASK}.bddl"
    init_path = BENCHMARK_ROOT / "init_files" / SUITE / f"{TASK}.pruned_init"
    env = offscreen_render_env(
        bddl_file_name=str(bddl_path),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
        camera_names=[CAMERA],
        render_gpu_device_id=-1,
    )
    env.seed(0)
    env.reset()
    obs = env.set_init_state(load_initial_state(init_path, torch))
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
    return env, obs


def main() -> int:
    if not BENCHMARK_ROOT.is_dir():
        raise FileNotFoundError(
            "LIBERO-Occ assets are missing. Run `git submodule update --init --recursive`."
        )

    configure_libero()
    make_optional_matplotlib_stub()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    image_key = f"{CAMERA}_image"

    env, obs = make_env(OffScreenRenderEnv, torch)
    try:
        # Nudge sideways so more than half of the bowl is cropped out of frame.
        for _ in range(8):
            obs, _, _, _ = env.step([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0])
        save(obs[image_key], OUT_DIR / "01_partially_occluded.png", Image)

        # Reverse, then continue down/left and back, to bring the bowl fully into view.
        for _ in range(30):
            obs, _, _, _ = env.step([0.0, -0.6, -0.2, 0.0, 0.0, 0.0, -1.0])
        for _ in range(20):
            obs, _, _, _ = env.step([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        save(obs[image_key], OUT_DIR / "02_fully_visible.png", Image)
    finally:
        env.close()

    print(f"Saved images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
