"""Pick N BDDL files from a LIBERO-X level uniformly at random, fixed seed."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from constants import EvalConfig
from libero_x_env import list_bddl_files


def sample(cfg: EvalConfig) -> list[str]:
    names = list_bddl_files(cfg)  # sorted, deterministic
    if len(names) < cfg.n_tasks:
        raise ValueError(f"Only {len(names)} tasks in {cfg.level}, need {cfg.n_tasks}")
    chosen = random.Random(cfg.seed).sample(names, cfg.n_tasks)
    return sorted(chosen)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="LEVEL1")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-tasks", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = EvalConfig(level=args.level, seed=args.seed, n_tasks=args.n_tasks)
    chosen = sample(cfg)
    out = args.out or str(cfg.outputs_dir / "tasks.txt")
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(chosen) + "\n")
    print(f"Sampled {len(chosen)} {args.level} tasks (seed {args.seed}) -> {out}")
    for c in chosen:
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
