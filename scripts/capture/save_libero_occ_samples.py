#!/usr/bin/env python3
"""Render sample observations from the released LIBERO-Occ task suites."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "submodules" / "Libero-Occ" / "benchmark_assets"
LIBERO_ROOT = REPO_ROOT / "submodules" / "openpi" / "third_party" / "libero"
SUITES = (
    "libero_spatial_occluded",
    "libero_goal_occluded",
    "libero_object_occluded",
    "libero_10_occluded",
)
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_occ_samples",
        help="Directory in which PNG files are written.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Capture all 10 tasks per suite instead of one task per suite.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="No-op simulation steps before capturing the observation.",
    )
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="GPU passed to robosuite; -1 uses its default selection.",
    )
    return parser.parse_args()


def configure_libero() -> None:
    """Create a repo-local LIBERO config before importing the package."""
    # Upstream LIBERO's package metadata does not install its namespace package.
    # OpenPI therefore uses this source checkout on PYTHONPATH as well.
    sys.path.insert(0, str(LIBERO_ROOT))
    config_dir = REPO_ROOT / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    # LIBERO reads this file as an import side effect. Pointing directly at the
    # released assets avoids copying them into site-packages or another checkout.
    config = {
        "benchmark_root": str(BENCHMARK_ROOT),
        "bddl_files": str(BENCHMARK_ROOT / "bddl_files"),
        "init_states": str(BENCHMARK_ROOT / "init_files"),
        "datasets": str(REPO_ROOT / "outputs" / "datasets"),
        "assets": str(BENCHMARK_ROOT),
    }
    lines = [f"{key}: {value}\n" for key, value in config.items()]
    (config_dir / "config.yaml").write_text("".join(lines), encoding="utf-8")


def make_optional_matplotlib_stub() -> None:
    """Satisfy LIBERO's unused segmentation-only matplotlib import."""
    try:
        import matplotlib.cm  # noqa: F401
    except ModuleNotFoundError:
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []  # type: ignore[attr-defined]
        matplotlib.cm = ModuleType("matplotlib.cm")  # type: ignore[attr-defined]
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.cm"] = matplotlib.cm


def load_initial_state(path: Path, torch: Any) -> Any:
    # torch 2.6 changed weights_only's default; these trusted benchmark files
    # contain ordinary serialized tensors rather than model weights.
    try:
        states = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older torch releases.
        states = torch.load(path, map_location="cpu")
    if len(states) == 0:
        raise ValueError(f"No initial states found in {path}")
    return states[0]


def tasks_for_suite(suite: str, all_tasks: bool) -> list[Path]:
    tasks = sorted((BENCHMARK_ROOT / "bddl_files" / suite).glob("*.bddl"))
    if not tasks:
        raise FileNotFoundError(f"No BDDL tasks found for {suite}")
    return tasks if all_tasks else tasks[:1]


def capture_task(
    bddl_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    offscreen_render_env: Any,
    image_class: Any,
    torch: Any,
) -> None:
    init_path = (
        BENCHMARK_ROOT
        / "init_files"
        / bddl_path.parent.name
        / f"{bddl_path.stem}.pruned_init"
    )
    if not init_path.is_file():
        raise FileNotFoundError(f"Missing initial-state file: {init_path}")

    env = offscreen_render_env(
        bddl_file_name=str(bddl_path),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        camera_names=[args.camera],
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(load_initial_state(init_path, torch))
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)

        observation_key = f"{args.camera}_image"
        if observation_key not in observation:
            available = sorted(key for key in observation if key.endswith("_image"))
            raise KeyError(
                f"Camera observation {observation_key!r} not found; available: {available}"
            )

        # LIBERO camera observations require a 180-degree rotation to match the
        # orientation used by its training and evaluation pipelines.
        image = observation[observation_key][::-1, ::-1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_class.fromarray(image).save(output_path)
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    if args.resolution <= 0 or args.settle_steps < 0:
        raise ValueError("resolution must be positive and settle-steps non-negative")
    if not BENCHMARK_ROOT.is_dir():
        raise FileNotFoundError(
            "LIBERO-Occ assets are missing. Run `git submodule update --init --recursive`."
        )
    if not (LIBERO_ROOT / "libero" / "libero" / "__init__.py").is_file():
        raise FileNotFoundError(
            "OpenPI's LIBERO submodule is missing. Run "
            "`git submodule update --init --recursive`."
        )

    configure_libero()
    make_optional_matplotlib_stub()
    try:
        import torch
        from libero.libero.envs import OffScreenRenderEnv
        from PIL import Image
    except ImportError as error:
        print(
            f"Missing simulator dependency ({error}). Run `uv sync` first.",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir.resolve()
    captured = 0
    for suite in SUITES:
        tasks = tasks_for_suite(suite, args.all_tasks)
        for task_number, bddl_path in enumerate(tasks, start=1):
            task_slug = bddl_path.stem.replace("_", "-")
            suffix = (
                f"{task_number}-{task_slug}"
                if args.all_tasks
                else "sample"
            )
            output_path = output_dir / suite / f"{suffix}.png"
            print(f"Rendering {suite}/{bddl_path.stem} -> {output_path}")
            capture_task(
                bddl_path,
                output_path,
                args,
                OffScreenRenderEnv,
                Image,
                torch,
            )
            captured += 1

    print(f"Saved {captured} image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
