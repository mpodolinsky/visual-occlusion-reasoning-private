"""LIBERO-X environment construction + reset.

Ported (near-verbatim) from
``16-LIBERO-X-GR00T-ZeroShot/sim/liberox_env.py``, which itself follows Dan's
``liberox-evals/src/eval_subgoals.py`` protocol. ``configure_libero_x`` is
repointed at this repo's ``submodules/LIBERO-X`` and ``outputs/libero-x/``.
Process-scoped only: ``sys.path`` is mutated for the current interpreter, same
as ``scripts/evaluation/eval_pi05_libero.configure_libero`` does for the
plain-LIBERO / LIBERO-Occ eval -- since each eval run is its own process,
this never collides with another script importing a different ``libero``
package fork in its own process.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from constants import EvalConfig


def quat2axisangle(quat) -> np.ndarray:
    """robosuite quaternion (x, y, z, w) -> axis-angle exponential coords."""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(float(q[3])) / den


def list_bddl_files(cfg: EvalConfig) -> list[str]:
    files = [f.name for f in cfg.bddl_dir.glob("*.bddl")]
    if not files:
        raise FileNotFoundError(f"No BDDL files in {cfg.bddl_dir}")
    return sorted(files)


def configure_libero_x(cfg: EvalConfig) -> None:
    """Put LIBERO-X's own ``libero`` package on ``sys.path`` (index 0, ahead
    of any other ``libero`` fork already on the path) and write a config file
    only under this repo's ``outputs/``.
    """
    root = cfg.libero_x_root
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    config_dir = cfg.libero_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    (config_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {root / 'libero' / 'libero_x'}",
                f"bddl_files: {cfg.bddl_dir.parent}",
                f"init_states: {cfg.init_dir.parent}",
                f"datasets: {cfg.outputs_dir / 'datasets'}",
                f"assets: {root / 'libero' / 'libero_x'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def open_task(cfg: EvalConfig, bddl_name: str):
    """Construct an ``OffScreenRenderEnv`` for one LIBERO-X task and load its
    init states. Returns ``(env, bddl_path, init_states, task_description)``.
    """
    configure_libero_x(cfg)
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.parse_bddl import parse_bddl_file

    bddl_path = cfg.bddl_dir / bddl_name
    init_path = cfg.init_dir / f"{bddl_path.stem}.init"
    if not init_path.is_file():
        raise FileNotFoundError(f"Init file not found: {init_path}")
    try:
        states = torch.load(init_path, map_location="cpu", weights_only=False)
    except TypeError:
        states = torch.load(init_path, map_location="cpu")

    task_description = parse_bddl_file(bddl_path)["language"]
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=cfg.env_resolution,
        camera_widths=cfg.env_resolution,
        horizon=cfg.max_steps + 1,
    )
    return env, bddl_path, states, str(task_description)


def reset_from_init(env, state_vec, cfg: EvalConfig):
    """Dan's init path: ``regenerate_obs_from_state``, no dummy settle steps."""
    env.reset()
    vec = state_vec.numpy() if hasattr(state_vec, "numpy") else state_vec
    return env.regenerate_obs_from_state(vec)


def expected_obs_keys() -> tuple[str, ...]:
    return (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
