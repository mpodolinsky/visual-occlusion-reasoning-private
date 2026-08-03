#!/usr/bin/env python3
"""Render a wrist-camera occlusion demo for the presentation.

Runs the LIBERO-Occ "pick up the alphabet soup" task (libero_object_occluded
suite) from the wrist camera (robot0_eye_in_hand). The scripted trajectory:

1. Descends and sweeps the gripper until a distractor box blocks the wrist
   camera's view of the target object -- this is the "occluded" frame.
2. Branch A ("our method" stand-in): pitches the wrist camera in place to
   look around the distractor, revealing the target can and the basket.
3. Branch B (baseline / no active-perception): makes a small ineffective
   forward creep instead, so the distractor keeps blocking the view.

The active-perception policy itself is not implemented yet; this script
scripts a plausible camera motion by hand purely to produce illustrative
before/after stills for the presentation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "submodules" / "Libero-Occ" / "benchmark_assets"
LIBERO_ROOT = REPO_ROOT / "submodules" / "openpi" / "third_party" / "libero"
OUT_DIR = REPO_ROOT / "outputs" / "wrist_camera_occlusion_demo"

SUITE = "libero_object_occluded"
TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
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


def reach_occlusion(env, obs):
    """Descend to table height, then sweep sideways until a distractor box
    fills the wrist camera's view."""
    for _ in range(25):
        obs, _, _, _ = env.step([0.0, 0.0, -0.8, 0.0, 0.0, 0.0, -1.0])
    for _ in range(20):
        obs, _, _, _ = env.step([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, -1.0])
    return obs


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

    # Shared occluded starting frame, used as frame 1 of both sequences.
    env, obs = make_env(OffScreenRenderEnv, torch)
    try:
        obs = reach_occlusion(env, obs)
        save(obs[image_key], OUT_DIR / "01_occluded.png", Image)
    finally:
        env.close()

    # Branch A: reposition the wrist camera (pitch in place) to look around
    # the distractor and reveal the target can + basket.
    env, obs = make_env(OffScreenRenderEnv, torch)
    try:
        obs = reach_occlusion(env, obs)
        for _ in range(11):
            obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, -0.6, 0.0, -1.0])
        save(obs[image_key], OUT_DIR / "02_ours_resolved.png", Image)
    finally:
        env.close()

    # Branch B: baseline creeps forward without addressing the occlusion.
    env, obs = make_env(OffScreenRenderEnv, torch)
    try:
        obs = reach_occlusion(env, obs)
        for _ in range(11):
            obs, _, _, _ = env.step([0.15, 0.05, 0.0, 0.0, 0.0, 0.0, -1.0])
        save(obs[image_key], OUT_DIR / "03_baseline_still_occluded.png", Image)
    finally:
        env.close()

    print(f"Saved images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
