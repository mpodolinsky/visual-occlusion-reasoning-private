#!/usr/bin/env python3
"""t-SNE plot of every cached per-timestep pi0.5 feature across --suite
(default libero_10_occluded, both scene variants).

Produces TWO plots from the SAME t-SNE embedding (computed exactly once --
t-SNE is stochastic, so a second fit_transform() call would give a
different, non-comparable layout; 13-SAFE/scripts/visualize_features.py
handles this the same way: one projector.fit_transform() call, its output
reused for both of its own two color-coded plots, see lines ~85-160 there):

  <output>-succ.png    -- colored by success/failure (this script's default
                           first request)
  <output>-taskid.png  -- colored by task id (tab10, 10 discrete colors --
                           SAFE's own default when --custom-cmap isn't set)

Coloring for -succ.png matches SAFE exactly: every step of a SUCCESS
episode gets color value 0 (blue, under the 'coolwarm' colormap); every
step of a FAILURE episode gets a value linearly increasing from 0 to 1 over
the episode's length (np.linspace(0, 1, T)) -- so a failure's early steps
start blue like a success and gradually shift to red toward the end.
Perplexity=30 to match sklearn's (and SAFE's own, since their
TSNE(n_components=2) call never overrides it) default.

Each timestep's raw per-token features (256 base-image tokens + 256
wrist-image tokens + up to 200 language tokens, each 2048-dim) are
unweighted-mean-pooled per modality and concatenated into one 6144-dim
vector -- SAFE's own features are already a single pooled vector per step
from their own encoder, so this is the closest unsupervised (no trained
weights involved) equivalent for our per-token representation.

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/plot_tsne_features.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_features import build_task_suite_map  # noqa: E402
from train_probe import read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--suite", default="libero_10_occluded")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs" / "perception_probe" / "tsne_features",
        help="Base path (no extension) -- writes <output>-succ.png and <output>-taskid.png.",
    )
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="Optional .npz path to save/reuse the fitted 2D embedding (skips re-running t-SNE if it "
        "already exists there) -- matches SAFE's own pickle-caching of feats_projected.",
    )
    return parser.parse_args()


def pooled_episode_features(npz_path: Path) -> np.ndarray:
    """(T, 6144): per-timestep mean-pooled [base_image, wrist_image, language]."""
    with np.load(npz_path) as data:
        base = data["base_image"].astype(np.float32).mean(axis=1)  # (T, 2048)
        wrist = data["wrist_image"].astype(np.float32).mean(axis=1)  # (T, 2048)
        lang = data["language"].astype(np.float32)  # (T, 200, 2048)
        mask = data["language_mask"].astype(np.float32)[..., None]  # (T, 200, 1)
        lang_pooled = (lang * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1.0)  # (T, 2048)
    return np.concatenate([base, wrist, lang_pooled], axis=1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    all_rows = read_manifest(args.features_dir)
    task_suite_map = build_task_suite_map()
    rows = [r for r in all_rows if task_suite_map.get(r["task"]) == args.suite]
    if not rows:
        raise ValueError(f"No episodes found for --suite {args.suite!r} in {args.features_dir}/manifest.csv")
    n_success = sum(1 for r in rows if r["success"] == "True")
    logging.info(
        "Suite %s: %d episodes (%d success, %d failure)", args.suite, len(rows), n_success, len(rows) - n_success
    )

    task_names = sorted({r["task"] for r in rows})
    task_id_of = {task: i for i, task in enumerate(task_names)}
    logging.info("%d distinct tasks: %s", len(task_names), task_names)

    # scene_variant is per-episode metadata already in manifest.csv (via "inference_calls" for each
    # episode's length T) -- cheap to (re)derive from `rows` alone, no need to touch the .npz files
    # or the (possibly cached) pooled feature matrix for this.
    scene_variant_list = []
    for row in rows:
        T = int(row["inference_calls"])
        scene_variant_list.append(np.zeros(T) if row["scene_variant"] == "occluded" else np.ones(T))
    scene_variant_ids = np.concatenate(scene_variant_list, axis=0)
    n_occluded = int((scene_variant_ids == 0).sum())
    logging.info(
        "Scene variant: %d occluded steps, %d normal steps", n_occluded, len(scene_variant_ids) - n_occluded
    )

    if args.cache is not None and args.cache.is_file():
        logging.info("Loading cached t-SNE embedding from %s (skipping re-fit)", args.cache)
        cached = np.load(args.cache)
        feats_2d, success_colors, task_ids = cached["feats_2d"], cached["success_colors"], cached["task_ids"]
    else:
        feats_list, success_colors_list, task_ids_list = [], [], []
        for row in rows:
            feats = pooled_episode_features(args.features_dir / row["npz_path"])
            T = feats.shape[0]
            success_colors = np.zeros(T) if row["success"] == "True" else np.linspace(0.0, 1.0, T)
            feats_list.append(feats)
            success_colors_list.append(success_colors)
            task_ids_list.append(np.full(T, task_id_of[row["task"]]))

        feats = np.concatenate(feats_list, axis=0)
        success_colors = np.concatenate(success_colors_list, axis=0)
        task_ids = np.concatenate(task_ids_list, axis=0)
        logging.info("Pooled feature matrix: %s", feats.shape)

        from sklearn.manifold import TSNE

        logging.info(
            "Running t-SNE (perplexity=%.1f) on %d points -- this may take a while...", args.perplexity, len(feats)
        )
        projector = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, verbose=1)
        feats_2d = projector.fit_transform(feats)
        logging.info("Done: %s", feats_2d.shape)

        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.cache, feats_2d=feats_2d, success_colors=success_colors, task_ids=task_ids)
            logging.info("Cached embedding to %s", args.cache)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_scatter(
        suffix: str, colors: np.ndarray, cmap, vmin: float, vmax: float, alpha: float,
        title: str, handles: list | None, annotated: bool,
    ) -> None:
        """Same embedding (feats_2d) reused for every plot -- only the color
        channel (and whether title/legend are drawn) changes. `annotated`
        selects between the titled+legended version (suffix as given) and a
        bare version for figures/slides (suffix + "-clean", no title, no
        legend, tighter crop since there's no legend eating margin)."""
        path = args.output.with_name(args.output.name + suffix + ("" if annotated else "-clean") + ".png")
        plt.figure(figsize=(8, 8), dpi=200)
        plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=colors, cmap=cmap, vmin=vmin, vmax=vmax, s=0.5, alpha=alpha)
        plt.axis("off")
        plt.gca().set_aspect("equal", adjustable="box")
        if annotated:
            plt.title(title, fontsize=9)
            if handles is not None:
                plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logging.info("Saved %s", path)

    succ_title = (
        f"{args.suite}: t-SNE of pooled features (perplexity={args.perplexity:.0f})\n"
        f"blue=success, blue→red over episode progress=failure (n={len(rows)} episodes, {len(feats_2d)} steps)"
    )
    taskid_title = (
        f"{args.suite}: t-SNE of pooled features (perplexity={args.perplexity:.0f})\n"
        f"colored by task id (n={len(rows)} episodes, {len(feats_2d)} steps)"
    )
    taskid_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=plt.get_cmap("tab10")(i / 9), label=f"{i}: {name[:30]}")
        for i, name in enumerate(task_names)
    ]
    scene_variant_title = (
        f"{args.suite}: t-SNE of pooled features (perplexity={args.perplexity:.0f})\n"
        f"colored by scene variant (n={len(rows)} episodes, {len(feats_2d)} steps)"
    )
    scene_variant_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:orange", label=f"occluded (n={n_occluded})"),
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color="tab:green",
            label=f"normal (n={len(scene_variant_ids) - n_occluded})",
        ),
    ]
    scene_variant_cmap = matplotlib.colors.ListedColormap(["tab:orange", "tab:green"])

    for annotated in (True, False):
        save_scatter("-succ", success_colors, "coolwarm", 0.0, 1.0, 0.5, succ_title, None, annotated)
        save_scatter("-taskid", task_ids, "tab10", 0, 9, 0.7, taskid_title, taskid_handles, annotated)
        save_scatter(
            "-scenevariant", scene_variant_ids, scene_variant_cmap, 0.0, 1.0, 0.5,
            scene_variant_title, scene_variant_handles, annotated,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
