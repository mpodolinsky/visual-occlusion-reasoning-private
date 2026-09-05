"""Shared constants + per-run config carrier for the GR00T-on-LIBERO-X eval.

LIBERO-X (``meituan/LIBERO-X``, pinned in ``submodules/LIBERO-X``) is a
different benchmark from ``libero_10``/``libero_10_occluded``: it adds new
objects, textures, and goal predicates (``ExactIn``, ``UprightOn``,
``SideOn``) across 5 difficulty levels. This pipeline measures cross-benchmark
zero-shot transfer of the same ``nvidia/GR00T-N1.7-LIBERO`` checkpoint already
downloaded for ``scripts/groot/`` -- it reuses that folder's GR00T websocket
server and obs/action bridge unchanged (see ``VENDOR.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# scripts/libero-x/ -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

LIBERO_X_ROOT = REPO_ROOT / "submodules" / "LIBERO-X"

LEVELS = ("LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5")

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "libero-x"

# Same checkpoint scripts/groot/ already downloads; GR00T-N1.7 was fine-tuned
# on the original libero_10 suite, not LIBERO-X, so this is a genuine
# zero-shot / out-of-distribution eval.
CHECKPOINT_LABEL = "nvidia/GR00T-N1.7-LIBERO/libero_10"


@dataclass
class EvalConfig:
    libero_x_root: Path = LIBERO_X_ROOT
    repo_root: Path = REPO_ROOT

    host: str = "127.0.0.1"
    port: int = 8000

    level: str = "LEVEL1"
    seed: int = 7
    n_tasks: int = 5
    n_rollouts: int = 10

    env_resolution: int = 256
    max_steps: int = 500
    # Actions executed per policy call before replanning. GR00T's LIBERO
    # README uses 8; scripts/groot/collect.py requires this explicitly too.
    replan_steps: int = 8

    checkpoint: str = CHECKPOINT_LABEL

    @property
    def base_level(self) -> str:
        # LIBERO-X ships only 4 BDDL/init level dirs; LEVEL5 reuses LEVEL4's.
        return "LEVEL4" if self.level.upper() == "LEVEL5" else self.level.upper()

    @property
    def bddl_dir(self) -> Path:
        return self.libero_x_root / "libero" / "libero_x" / "bddl" / self.base_level

    @property
    def init_dir(self) -> Path:
        return self.libero_x_root / "libero" / "libero_x" / "init" / self.base_level

    @property
    def libero_config_dir(self) -> Path:
        return self.outputs_dir / ".libero"

    @property
    def outputs_dir(self) -> Path:
        return DEFAULT_OUTPUT_DIR
