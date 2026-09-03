"""LIBERO / LIBERO-Occ access for libero_10, via 12's own eval helpers.

Unlike 17 (which vendored a trimmed copy), this pipeline lives inside 12, so it
imports ``scripts/evaluation/eval_pi05_libero`` read-only -- the same
``sys.path`` trick ``scripts/perception_probe/collect_features.py`` uses. Nothing
in ``scripts/evaluation/`` is modified.
"""

from __future__ import annotations

import sys
from typing import Any

from constants import REPO_ROOT, PipelineConfig

_EVAL_DIR = str(REPO_ROOT / "scripts" / "evaluation")


def load_eval_module() -> Any:
    if _EVAL_DIR not in sys.path:
        sys.path.insert(0, _EVAL_DIR)
    import eval_pi05_libero as E  # noqa: WPS433

    return E


def open_task(cfg: PipelineConfig) -> tuple[Any, Any, Any, str, int]:
    """Returns (env, bddl_path, initial_states, instruction, max_steps)."""
    E = load_eval_module()
    E.make_optional_matplotlib_stub()
    benchmark_root, suite = E.benchmark_selection(cfg.occluded_suite, cfg.scene_variant)
    E.configure_libero(benchmark_root)
    E.make_optional_matplotlib_stub()

    import torch
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = E.task_files(suite, benchmark_root)[cfg.task_id]
    init_path = benchmark_root / "init_files" / suite / f"{bddl_path.stem}.pruned_init"
    initial_states = E.load_initial_states(init_path, torch)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=cfg.env_resolution,
        camera_widths=cfg.env_resolution,
        camera_names=["agentview", "robot0_eye_in_hand"],
        render_gpu_device_id=-1,
    )
    env.seed(cfg.seed)
    instruction = str(env.language_instruction)
    max_steps = cfg.max_steps if cfg.max_steps is not None else int(E.MAX_STEPS[cfg.occluded_suite])
    env._pipeline_eval = E  # type: ignore[attr-defined]
    return env, bddl_path, initial_states, instruction, max_steps


def goal_predicate_strings(env: Any) -> list[str]:
    inner = env.env if hasattr(env, "env") else env
    parsed = getattr(inner, "parsed_problem", None) or {}
    goals = parsed.get("goal_state") or []
    return [str(g) for g in goals]
