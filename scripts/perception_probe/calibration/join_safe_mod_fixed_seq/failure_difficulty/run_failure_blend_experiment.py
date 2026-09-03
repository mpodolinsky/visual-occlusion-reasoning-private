#!/usr/bin/env python3
"""Demonstrates the practical consequence of SAFE's h being a function of successful
episodes ONLY (see construct_cp_band.py / construct_joint_threshold_fixed_seq.py's own
module docstrings): it cannot adapt to how hard failures are to detect, because it never
looks at a single failure trajectory when choosing h. Our own joint procedure looks at
both classes and either adapts (a smaller h*, trading against FPR) or correctly REFUSES
(fixed sequence testing finds no valid h) once the model can no longer support the
guarantee -- SAFE just keeps outputting the same number, silently wrong.

Method: take ONE model's real seen calibration+test episodes. Freeze mu_t/s(t) (fit from
D_fit successes only, exactly as construct_joint_threshold_fixed_seq.py does) and SAFE's
own h (also computed once, from D_test successes, at gamma=0) -- neither ever changes
across this experiment, since neither is refit against failures. Then, for a sweep of
gamma in [0, 1), blend every REAL failure trajectory toward mu_t (the frozen "normal"
reference line):

    score'_i(t) = (1 - gamma) * score_i(t) + gamma * mu_t(t)

gamma=0 is the real, unmodified failures (today's baseline); gamma->1 makes failures
increasingly subtle/late-onset, until they're indistinguishable from average success
behavior. Success trajectories are never touched.

At each gamma:
  - SAFE's h stays exactly what it was at gamma=0 (by construction -- it's a fact about
    mu_t/s(t)/D_test successes, none of which change). Its TRUE FPR/FNR/detection-time
    against the blended failures is recomputed and reported -- this is what degrades.
  - Our own h* is recalibrated from scratch each gamma (the SAME h-sweep + Learn-then-Test
    p-values + multi-start fixed sequence testing as construct_joint_threshold_fixed_seq.py,
    but reusing the frozen mu_t/s(t) rather than refitting them -- only the FPR/FNR/p-values
    depend on gamma here) -- so it can shrink (tighten the band to keep catching harder
    failures) or, once no h in the grid controls both risks, correctly report that no
    certified threshold exists at this difficulty level.

Run with the top-level project venv:
    .venv/bin/python scripts/perception_probe/calibration/join_safe_mod_fixed_seq/failure_difficulty/run_failure_blend_experiment.py \\
        --models-json scripts/perception_probe/calibration/libero-10-occ-models/models.json \\
        --model-name best-sweep-3
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
CALIBRATION_DIR = Path(__file__).resolve().parents[2]
FIXED_SEQ_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIXED_SEQ_DIR))
sys.path.insert(0, str(CALIBRATION_DIR))
sys.path.insert(0, str(CALIBRATION_DIR.parent))
from construct_joint_threshold_fixed_seq import edge_pad, fixed_sequence_test, tfunc_modulation  # noqa: E402
from run_joint_threshold_fixed_seq_sweep import resolve_run_dir, score_dataset  # noqa: E402
import compute_calibration_scores as ccs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--model-name", required=True, help="Must match a \"name\" entry in --models-json.")
    parser.add_argument(
        "--gammas", type=float, nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        help="Blend fractions -- 0 = real failures, ->1 = indistinguishable from mu_t.",
    )
    parser.add_argument("--checkpoint", default="probe_best.pt")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "outputs" / "perception_probe" / "features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rmean", action="store_true", default=False)
    parser.add_argument("--cumsum", dest="cumsum", action="store_true", default=True)
    parser.add_argument("--no-cumsum", dest="cumsum", action="store_false")
    parser.add_argument("--scores-cache-dir", type=Path, default=CALIBRATION_DIR)
    parser.add_argument("--force-rescore", action="store_true")

    parser.add_argument("--alpha-fpr", type=float, default=0.25)
    parser.add_argument("--alpha-fnr", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--h-min", type=float, default=0.0)
    parser.add_argument("--h-max", type=float, default=4.0)
    parser.add_argument("--num-h", type=int, default=40)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--modulation-alpha", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--ab-split-fraction", type=float, default=0.3)
    parser.add_argument(
        "--traj-plot-gammas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
        help="Which gamma snapshots to draw failure trajectories at in the visualization plot "
        "(a subset of --gammas is typical, to keep the plot legible).",
    )
    parser.add_argument(
        "--overlay-gammas", type=float, nargs="+", default=[0.5, 0.6, 0.8],
        help="Gamma snapshots for the single-panel IEEE-ready overlay figure.",
    )

    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parent / "runs",
        help="Each run gets its own output_root/<model-name>_<timestamp>/ folder.",
    )
    return parser.parse_args()


def evaluate_at_gamma(
    mu_t: np.ndarray, step_size: np.ndarray, succ_scores: np.ndarray,
    fail_scores_orig: np.ndarray, fail_lengths: np.ndarray, gamma: float,
    *, alpha_fpr: float, alpha_fnr: float, delta: float,
    h_min: float, h_max: float, num_h: int, n_starts: int,
    safe_h: float,
) -> dict:
    """One gamma's worth of: blend failures, run our full h-sweep + LTT + fixed-sequence
    selection (reusing the FROZEN mu_t/step_size, not refitting them), and evaluate SAFE's
    frozen h against the same blended failures. Returns a flat dict of results."""
    fail_scores = (1 - gamma) * fail_scores_orig + gamma * mu_t[None, :]
    n_succ, n_fail = len(succ_scores), len(fail_scores)

    h_sweep = np.linspace(h_min, h_max, num_h)
    fpr_sweep = np.empty_like(h_sweep)
    fnr_sweep = np.empty_like(h_sweep)
    det_time_sweep = np.empty_like(h_sweep)
    for i, h in enumerate(h_sweep):
        eta_h = mu_t + h * step_size
        fpr_sweep[i] = np.mean(np.any(succ_scores > eta_h, axis=1))
        fnr_sweep[i] = np.mean(np.all(fail_scores <= eta_h, axis=1))
        detection_mask = fail_scores > eta_h
        has_detection = detection_mask.any(axis=1)
        first_detection = detection_mask.argmax(axis=1)
        detection_times = np.where(has_detection, first_detection, fail_lengths)
        det_time_sweep[i] = np.mean(detection_times / fail_lengths)

    violations_fpr = np.round(fpr_sweep * n_succ).astype(int)
    violations_fnr = np.round(fnr_sweep * n_fail).astype(int)
    p_fpr_sweep = binom.cdf(violations_fpr, n_succ, alpha_fpr)
    p_fnr_sweep = binom.cdf(violations_fnr, n_fail, alpha_fnr)
    p_joint_sweep = np.maximum(p_fpr_sweep, p_fnr_sweep)

    accepted, threshold, start_indices = fixed_sequence_test(p_joint_sweep, delta, n_starts)

    result = {
        "gamma": gamma, "n_succ": n_succ, "n_fail": n_fail,
        "fixed_seq_threshold": threshold, "n_accepted": int(accepted.sum()),
    }
    if accepted.any():
        valid_indices = np.flatnonzero(accepted)
        selected_idx = int(valid_indices[np.argmin(det_time_sweep[accepted])])
        result.update({
            "our_h": float(h_sweep[selected_idx]),
            "our_fpr": float(fpr_sweep[selected_idx]),
            "our_fnr": float(fnr_sweep[selected_idx]),
            "our_detection_time": float(det_time_sweep[selected_idx]),
        })
    else:
        result.update({"our_h": None, "our_fpr": None, "our_fnr": None, "our_detection_time": None})

    # SAFE's frozen h, evaluated against these SAME blended failures.
    eta_safe = mu_t + safe_h * step_size
    safe_fpr = float(np.mean(np.any(succ_scores > eta_safe, axis=1)))
    safe_fnr = float(np.mean(np.all(fail_scores <= eta_safe, axis=1)))
    safe_detection_mask = fail_scores > eta_safe
    safe_has_detection = safe_detection_mask.any(axis=1)
    safe_first_detection = safe_detection_mask.argmax(axis=1)
    safe_detection_times = np.where(safe_has_detection, safe_first_detection, fail_lengths)
    safe_detection_time = float(np.mean(safe_detection_times / fail_lengths))
    result.update({
        "safe_h": safe_h, "safe_fpr": safe_fpr, "safe_fnr": safe_fnr,
        "safe_detection_time": safe_detection_time,
    })
    return result


def plot_blended_trajectories(
    mu_t: np.ndarray, succ_scores: np.ndarray, fail_scores_orig: np.ndarray,
    gammas: list[float], output_path: Path, model_name: str,
) -> None:
    """Visualizes what the blend score' = (1-gamma)*score + gamma*mu_t actually does to
    real failure trajectories -- one panel per gamma snapshot, all sharing the same
    (unchanged) D_test successes and mu_t in the background for reference, so it's visually
    obvious how failures collapse toward "normal" as gamma increases."""
    n_panels = len(gammas)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    t_axis = np.arange(len(mu_t))

    for idx, gamma in enumerate(gammas):
        ax = axes[idx // ncols][idx % ncols]
        blended_fail = (1 - gamma) * fail_scores_orig + gamma * mu_t[None, :]

        succ_labeled = False
        for traj in succ_scores:
            ax.plot(t_axis, traj, color="cornflowerblue", alpha=0.25, linewidth=1,
                    label="success (D_test)" if not succ_labeled else None)
            succ_labeled = True
        fail_labeled = False
        for traj in blended_fail:
            ax.plot(t_axis, traj, color="lightcoral", alpha=0.4, linewidth=1,
                    label="failure (blended)" if not fail_labeled else None)
            fail_labeled = True
        ax.plot(t_axis, mu_t, color="black", linewidth=2.5, label="mu_t")

        ax.set_title(f"gamma = {gamma:.2f}" + ("  (real, unmodified)" if gamma == 0 else ""))
        ax.set_xlabel("inference call index t")
        ax.set_ylabel("accumulated score")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"{model_name}: failure trajectories blended toward mu_t -- "
        f"score'(t) = (1-gamma)*score(t) + gamma*mu_t(t)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_blend_overlay_ieee(
    mu_t: np.ndarray, succ_scores: np.ndarray, fail_scores_orig: np.ndarray,
    gammas: list[float], output_path: Path,
) -> None:
    """Single-panel, publication-ready overlay of the blended failure trajectories at a few
    gamma snapshots (default 0.5/0.6/0.8) plus mu_t and the real (unmodified) D_test
    successes for reference -- no title, no per-episode legend clutter (one legend entry
    per gamma + successes + mu_t only), legend placed BELOW the axes so the plot area
    itself stays clean for direct inclusion in an IEEE two-column figure. Color runs
    yellow (easiest / lowest gamma, closest to the real failures) -> red (hardest /
    highest gamma, closest to indistinguishable from success), matching the intuitive
    caution->danger reading of that gradient; the lowest and highest gamma given are
    annotated "(easier)"/"(harder)" in the legend."""
    t_axis = np.arange(len(mu_t))
    colors = ["#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026"]  # sequential yellow->red
    gamma_min, gamma_max = min(gammas), max(gammas)

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for i, traj in enumerate(succ_scores):
        ax.plot(t_axis, traj, color="#4292c6", alpha=0.35, linewidth=0.6,
                label="success" if i == 0 else None, zorder=1)
    for gamma, color in zip(gammas, colors):
        blended = (1 - gamma) * fail_scores_orig + gamma * mu_t[None, :]
        suffix = " (easier)" if gamma == gamma_min else " (harder)" if gamma == gamma_max else ""
        for i, traj in enumerate(blended):
            ax.plot(t_axis, traj, color=color, alpha=0.55, linewidth=0.8,
                    label=rf"$\gamma={gamma:.1f}${suffix}" if i == 0 else None, zorder=2)
    ax.plot(t_axis, mu_t, color="black", linewidth=1.6, label=r"$\mu_t$", zorder=3)

    ax.set_xlabel("inference call index $t$", fontsize=9)
    ax.set_ylabel("accumulated score", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(gammas) + 2, fontsize=7.5,
        frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.4,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fnr_ieee(
    gammas: np.ndarray, our_fnr: np.ndarray, safe_fnr: np.ndarray, alpha_fnr: float,
    refused: np.ndarray, output_path: Path,
) -> None:
    """Single-panel, publication-ready summary: true FNR vs. failure difficulty gamma, for
    our adaptive/refusing procedure vs. SAFE's frozen h. No title; legend below the axes;
    sized to match plot_blend_overlay_ieee(). Neutral color coding (steel blue = ours, gray
    = SAFE, dashed gray = beta -- renamed from alpha_fnr to match the paper's alpha/beta
    notation). Every CERTIFIED point (both methods) gets a dark green outline if its true
    FNR <= beta (risk actually controlled there) or a dark red outline if not (risk
    violated). Refusing to certify is ALSO a risk-controlled outcome -- Learn-then-Test
    declining to name an h never asserts anything false, so it can never violate beta --
    so refused points get the SAME green highlight ring, drawn around a blue hollow circle
    (blue = still "ours", hollow = no specific h was certified). The legend's method
    entries (Ours: certified/refused, Single-Sided Conformal) are plain proxies with no
    green/red coloring of their own -- risk-controlled/violated status is communicated
    ONLY by the two dedicated color-coded legend entries, not baked into every swatch."""
    controlled_color, violated_color = "#1a6b1a", "#8b1a1a"
    ours_color, safe_color = "#3b6faa", "#7f7f7f"

    def edge_colors(values: np.ndarray) -> list[str]:
        return [controlled_color if v <= alpha_fnr else violated_color for v in values]

    our_line_y = np.where(refused, 0.0, our_fnr)

    # Small constant horizontal offset (in gamma units) so the two series' markers don't
    # sit exactly on top of each other where they coincide (e.g. both at FNR=0 for small
    # gamma) -- shifting x rather than y so the plotted FNR values stay visually exact.
    offset = 0.012
    gammas_safe, gammas_ours = gammas - offset, gammas + offset

    fig, ax = plt.subplots(figsize=(4.8, 2.6))

    ax.plot(gammas_ours, our_line_y, color=ours_color, linewidth=1.5, zorder=1)
    ax.axhline(alpha_fnr, color="#4d4d4d", linestyle="--", linewidth=1)
    ax.plot(gammas_safe, safe_fnr, color=safe_color, linewidth=1.5, linestyle=":", zorder=1)
    ax.scatter(gammas_safe, safe_fnr, marker="s", s=26, facecolors=safe_color,
               edgecolors=edge_colors(safe_fnr), linewidths=1.3, zorder=2)
    if (~refused).any():
        ax.scatter(gammas_ours[~refused], our_fnr[~refused], marker="o", s=30, facecolors=ours_color,
                   edgecolors=edge_colors(our_fnr[~refused]), linewidths=1.3, zorder=3)
    if refused.any():
        # Green highlight ring first (refusing is itself risk-controlled), then the smaller
        # blue hollow "ours" marker on top.
        ax.scatter(gammas_ours[refused], np.zeros(refused.sum()), marker="o", s=55, facecolors="none",
                   edgecolors=controlled_color, linewidths=1.3, zorder=4)
        ax.scatter(gammas_ours[refused], np.zeros(refused.sum()), marker="o", s=22, facecolors="white",
                   edgecolors=ours_color, linewidths=1.2, zorder=5)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Difficulty Factor [$\gamma$]", fontsize=9)
    ax.set_ylabel("False-Negative Rate", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1),
        Line2D([0], [0], marker="s", linestyle=":", color=safe_color, markerfacecolor=safe_color,
               markeredgecolor=safe_color, markersize=5),
        Line2D([0], [0], marker="o", linestyle="-", color=ours_color, markerfacecolor=ours_color,
               markeredgecolor=ours_color, markersize=5),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=ours_color, markeredgewidth=1.2, markersize=5),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=controlled_color, markeredgewidth=1.3, markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=violated_color, markeredgewidth=1.3, markersize=6),
    ]
    labels = [
        r"$\beta$", "Single-Sided Conformal", "Ours: certified", "Ours: refused to certify",
        "risk controlled", "risk violated",
    ]
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.5,
        frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.4,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    models = json.loads(args.models_json.read_text())["models"]
    model_cfg = next((m for m in models if m["name"] == args.model_name), None)
    if model_cfg is None:
        raise ValueError(f"'{args.model_name}' not found in {args.models_json} (have: {[m['name'] for m in models]})")
    run_dir = resolve_run_dir(model_cfg["run_dir"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{args.model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_npz = score_dataset(model_cfg, args, run_dir, ["calibration", "test"])
    cache = np.load(seen_npz, allow_pickle=True)
    scores, lengths, success = cache["scores"], cache["lengths"], cache["success"]
    n_episodes = len(scores)
    logging.info("Loaded %d episodes (%d success, %d failure) from %s", n_episodes, int(success.sum()), int((~success).sum()), seen_npz)

    max_len = int(lengths.max())
    D_all = np.stack([edge_pad(s, max_len) for s in scores])
    succ_idx_all = np.flatnonzero(success)
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(len(succ_idx_all))
    n_fit = max(1, int(len(succ_idx_all) * args.ab_split_fraction))
    fit_idx, test_idx = succ_idx_all[perm[:n_fit]], succ_idx_all[perm[n_fit:]]
    D_fit_succ, D_test_succ = D_all[fit_idx], D_all[test_idx]
    logging.info("Success split: %d D_fit (mu_t/s(t), FROZEN for the whole gamma sweep) / %d D_test", len(fit_idx), len(test_idx))

    mu_t = D_fit_succ.mean(axis=0)
    step_size, H_mask = tfunc_modulation(D_fit_succ, mu_t, args.modulation_alpha)
    logging.info("Tfunc modulation: |H|=%d/%d D_fit kept, s(t) range=[%.4f, %.4f]", int(H_mask.sum()), len(H_mask), step_size.min(), step_size.max())

    succ_scores = D_test_succ
    fail_idx_all = np.flatnonzero(~success)
    fail_scores_orig = D_all[~success]
    fail_lengths = lengths[~success]

    # SAFE's own h -- computed ONCE, from D_test successes only, at gamma=0. Frozen for the
    # whole sweep (it's provably invariant to gamma since it never touches failure data).
    D_j_safe = np.max((succ_scores - mu_t) / step_size, axis=1)
    safe_h = float(np.quantile(D_j_safe, 1 - args.modulation_alpha))
    logging.info("SAFE's own h = %.4f (frozen -- provably invariant to failure blending)", safe_h)

    traj_plot_path = output_dir / f"{args.model_name}_blended_trajectories.png"
    plot_blended_trajectories(mu_t, succ_scores, fail_scores_orig, args.traj_plot_gammas, traj_plot_path, args.model_name)
    logging.info("Wrote blended-trajectory visualization to %s", traj_plot_path)

    overlay_plot_path = output_dir / f"{args.model_name}_blend_overlay_ieee.png"
    plot_blend_overlay_ieee(mu_t, succ_scores, fail_scores_orig, args.overlay_gammas, overlay_plot_path)
    logging.info("Wrote IEEE-ready overlay figure to %s", overlay_plot_path)

    results = []
    for gamma in args.gammas:
        r = evaluate_at_gamma(
            mu_t, step_size, succ_scores, fail_scores_orig, fail_lengths, gamma,
            alpha_fpr=args.alpha_fpr, alpha_fnr=args.alpha_fnr, delta=args.delta,
            h_min=args.h_min, h_max=args.h_max, num_h=args.num_h, n_starts=args.n_starts,
            safe_h=safe_h,
        )
        results.append(r)
        if r["our_h"] is not None:
            logging.info(
                "[gamma=%.2f] ours: h*=%.2f FPR=%.3f FNR=%.3f det=%.3f | SAFE: FPR=%.3f FNR=%.3f det=%.3f",
                gamma, r["our_h"], r["our_fpr"], r["our_fnr"], r["our_detection_time"],
                r["safe_fpr"], r["safe_fnr"], r["safe_detection_time"],
            )
        else:
            logging.info(
                "[gamma=%.2f] ours: NO VALID h (correctly refuses) | SAFE: FPR=%.3f FNR=%.3f det=%.3f",
                gamma, r["safe_fpr"], r["safe_fnr"], r["safe_detection_time"],
            )

    summary = {
        "model_name": args.model_name, "run_dir": str(run_dir), "safe_h": safe_h,
        "n_fit": len(fit_idx), "n_test": len(test_idx), "n_fail": len(fail_idx_all),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Wrote summary to %s", output_dir / "summary.json")

    # --- Plot: h(gamma) and FNR(gamma), ours vs. SAFE ---
    gammas = np.array([r["gamma"] for r in results])
    our_h = np.array([r["our_h"] if r["our_h"] is not None else np.nan for r in results])
    our_fnr = np.array([r["our_fnr"] if r["our_fnr"] is not None else np.nan for r in results])
    our_fpr = np.array([r["our_fpr"] if r["our_fpr"] is not None else np.nan for r in results])
    safe_fnr = np.array([r["safe_fnr"] for r in results])
    safe_fpr = np.array([r["safe_fpr"] for r in results])
    refused = np.array([r["our_h"] is None for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(gammas, our_h, marker="o", color="darkviolet", linewidth=2, label="our h* (adapts / shrinks)")
    ax.axhline(safe_h, color="darkorange", linestyle=":", linewidth=2.5, label=f"SAFE's h (frozen, blind to failures)")
    if refused.any():
        ax.scatter(gammas[refused], np.full(refused.sum(), ax.get_ylim()[0]), marker="x", s=80, color="red",
                   zorder=5, label="ours: no valid h (correctly refuses)")
    ax.set_xlabel("gamma (failure blend toward mu_t -- 0=real, ->1=indistinguishable from success)")
    ax.set_ylabel("selected h")
    ax.set_title(f"{args.model_name}: selected h vs. failure difficulty")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(gammas, our_fnr, marker="o", color="darkviolet", linewidth=2, label="our FNR (at our h*)")
    ax.plot(gammas, safe_fnr, marker="s", color="darkorange", linewidth=2, linestyle=":", label="SAFE FNR (at SAFE's frozen h)")
    ax.axhline(args.alpha_fnr, color="gray", linestyle="--", linewidth=1, label=f"alpha_fnr={args.alpha_fnr}")
    if refused.any():
        ax.scatter(gammas[refused], np.zeros(refused.sum()), marker="x", s=80, color="red", zorder=5,
                   label="ours: no valid h (correctly refuses)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("gamma (failure blend toward mu_t)")
    ax.set_ylabel("FNR (fraction of failures MISSED)")
    ax.set_title(f"{args.model_name}: true FNR vs. failure difficulty")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / f"{args.model_name}_failure_blend.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info("Wrote plot to %s", plot_path)

    ieee_fnr_path = output_dir / f"{args.model_name}_fnr_ieee.png"
    plot_fnr_ieee(gammas, our_fnr, safe_fnr, args.alpha_fnr, refused, ieee_fnr_path)
    logging.info("Wrote IEEE-ready FNR summary figure to %s", ieee_fnr_path)

    print(f"\n{'gamma':>6s} {'our_h':>8s} {'our_FPR':>8s} {'our_FNR':>8s} {'our_det':>8s} | "
          f"{'safe_h':>7s} {'safe_FPR':>9s} {'safe_FNR':>9s} {'safe_det':>9s}")
    for r in results:
        our_h_s = f"{r['our_h']:.2f}" if r["our_h"] is not None else "REFUSED"
        our_fpr_s = f"{r['our_fpr']:.3f}" if r["our_fpr"] is not None else "--"
        our_fnr_s = f"{r['our_fnr']:.3f}" if r["our_fnr"] is not None else "--"
        our_det_s = f"{r['our_detection_time']:.3f}" if r["our_detection_time"] is not None else "--"
        print(f"{r['gamma']:>6.2f} {our_h_s:>8s} {our_fpr_s:>8s} {our_fnr_s:>8s} {our_det_s:>8s} | "
              f"{r['safe_h']:>7.3f} {r['safe_fpr']:>9.3f} {r['safe_fnr']:>9.3f} {r['safe_detection_time']:>9.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
