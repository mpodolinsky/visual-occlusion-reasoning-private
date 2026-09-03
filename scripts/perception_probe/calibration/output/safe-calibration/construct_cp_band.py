#!/usr/bin/env python3
"""Plain-function port of construct_cp_band.ipynb (FAIL-Detect Appendix B / SAFE's
FunctionalPredictor with ModulationType.Tfunc, one-sided upper band): builds a
time-varying conformal-prediction band eta_t = mu_t + h*s_A(t) from a run's SEEN
(successful) calibration-split score trajectories, giving a whole-trajectory (not just
pointwise) false-alarm guarantee P(score_t <= eta_t for all t) >= 1 - alpha for a fresh
successful rollout drawn exchangeably with the calibration set.

Two deliberate deviations from the notebook, matching the conventions established for
construct_joint_threshold.py:
  - alpha/seed/output paths are function parameters, not hardcoded constants.
  - Everything the notebook printed is also returned in the result dict / written to a
    _result.json, so this is safe to call unattended from a sweep script.

Two public entry points:
  run_cp_band(...)              -- fits mu_t/s_A/h/eta_t on ONE dataset's successful
                                    episodes (the "seen" pool: calibration+test), saves
                                    the band + a diagnostic plot + a _result.json, and
                                    persists mu_t/s_A/upper_t/h/max_len to a .npz so the
                                    band can be re-applied later without refitting.
  evaluate_band_on_dataset(...) -- applies an ALREADY-FITTED band as-is to a different
                                    dataset's episodes (e.g. unseen tasks) -- edge-pads/
                                    crops trajectories to the band's max_len, then reports
                                    empirical coverage on successes and flag rate/first-
                                    crossing-time on failures. No refitting: this is a
                                    transfer check, mirroring
                                    construct_joint_threshold.py's evaluate_threshold_on_dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-8


def edge_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (repeat last value) or crop arr to target_len -- matches SAFE's own
    align_method="extend" and how train_probe_time_dependent.py batches variable-length
    episodes (edge-padding, not zero-padding)."""
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


def tfunc_modulation(cal_A: np.ndarray, mu_t: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Port of SAFE's FunctionalPredictor._get_modulation_trajectory(ModulationType.Tfunc).
    Returns (modulation_trajectory, H_mask) where H_mask marks which D_cal_A episodes
    survived trimming (episodes whose own worst-timestep deviation from mu_t is extreme
    are dropped so a single outlier can't blow up the band width at every timestep)."""
    n1 = cal_A.shape[0]
    per_episode_max_dev = np.max(np.abs(cal_A - mu_t), axis=1)  # (N1,)

    if int(np.ceil((n1 + 1) * (1 - alpha))) > n1:
        H_mask = np.ones(n1, dtype=bool)  # H = [N1], no trimming
    else:
        gamma = np.sort(per_episode_max_dev)[int(np.ceil((n1 + 1) * (1 - alpha))) - 1]
        H_mask = per_episode_max_dev <= gamma

    modulation = np.max(np.abs(cal_A[H_mask] - mu_t), axis=0) + EPS
    return modulation, H_mask


def stays_under_band(traj: np.ndarray, band: np.ndarray) -> bool:
    return bool(np.all(traj <= band))


def compute_detection_times(fail_idx, fail_trajectories, upper_t, lengths, tasks) -> tuple[list[dict], float | None]:
    """Per failure episode: first timestep its trajectory crosses above the band, as a
    FRACTION of that episode's own (unpadded) length -- 0 = flagged immediately, 1 = never
    flagged before the episode ended. Undetected episodes are scored at fraction 1.0 (i.e.
    as if "detected" only at the very last step) -- same convention as
    construct_joint_threshold.py's det_time_sweep, so the two pipelines' detection-time
    numbers stay directly comparable. Returns (per-episode records, mean fraction)."""
    records = []
    fractions = []
    for i, traj in zip(fail_idx, fail_trajectories):
        length = int(lengths[i])
        above = np.flatnonzero(traj > upper_t)
        first_crossing = int(above[0]) if len(above) else None
        detection_step = first_crossing if first_crossing is not None else length
        fraction = min(detection_step, length) / length if length > 0 else None
        records.append({
            "task": str(tasks[i]), "length": length,
            "first_crossing": first_crossing,
            "detection_time_fraction": fraction,
        })
        if fraction is not None:
            fractions.append(fraction)
    mean_fraction = float(np.mean(fractions)) if fractions else None
    return records, mean_fraction


def run_cp_band(
    scores_npz_path: Path,
    output_dir: Path,
    *,
    alpha: float = 0.15,
    seed: int = 0,
    ab_split_fraction: float = 0.3,
    dataset_label: str = "seen",
) -> dict:
    """Fits mu_t/s_A/h/eta_t from scores_npz_path's SUCCESSFUL episodes (D_cal, split
    30/70 into D_cal_A/D_cal_B by seed), following Appendix B step-for-step. Saves a
    diagnostic plot (mu_t, eta_t, D_cal_A/D_cal_B/failure trajectories overlaid),
    persists the fitted band to a .npz, and writes a _result.json. Returns the result dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(scores_npz_path, allow_pickle=True)
    scores = cache["scores"]
    lengths = cache["lengths"]
    tasks = cache["tasks"]
    success = cache["success"]

    success_idx = np.flatnonzero(success)
    succ_trajectories = [scores[i] for i in success_idx]
    if not succ_trajectories:
        raise ValueError(f"{scores_npz_path} has no successful episodes -- cannot fit a band.")
    max_len = max(len(s) for s in succ_trajectories)
    D_cal = np.stack([edge_pad(s, max_len) for s in succ_trajectories])  # (N_success, max_len)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(D_cal))
    n1 = int(len(D_cal) * ab_split_fraction)
    if n1 < 1:
        raise ValueError(
            f"Only {len(D_cal)} successful episodes in {scores_npz_path} -- not enough for a "
            f"D_cal_A/D_cal_B split at ab_split_fraction={ab_split_fraction}."
        )
    idx_A, idx_B = perm[:n1], perm[n1:]
    D_cal_A, D_cal_B = D_cal[idx_A], D_cal[idx_B]
    N1, N2 = D_cal_A.shape[0], D_cal_B.shape[0]
    n2_warning = None
    if N2 < 20:
        n2_warning = f"N2={N2} is small -- the (1-alpha)-quantile at alpha={alpha} is coarse (near-max of D_cal_B)."

    # Step 1: mean successful trajectory.
    mu_t = D_cal_A.mean(axis=0)

    # Step 2: modulation function s_A(t).
    s_A, H_mask = tfunc_modulation(D_cal_A, mu_t, alpha)

    # Step 3: per-episode max normalized deviation on D_cal_B, band width h, upper band.
    D_j = np.max((D_cal_B - mu_t) / s_A, axis=1)  # (N2,)
    h = float(np.quantile(D_j, 1 - alpha))
    upper_t = mu_t + h * s_A  # eta_t

    # Sanity checks (mirrors the notebook's own coverage self-check).
    cal_B_covered = sum(stays_under_band(t, upper_t) for t in D_cal_B)
    cal_B_coverage = cal_B_covered / N2

    fail_idx = np.flatnonzero(~success)
    fail_trajectories = [edge_pad(scores[i], max_len) for i in fail_idx]
    fail_flagged = sum(not stays_under_band(t, upper_t) for t in fail_trajectories)
    first_crossings, avg_detection_time = compute_detection_times(fail_idx, fail_trajectories, upper_t, lengths, tasks)

    # Plot.
    t_axis = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, traj in enumerate(D_cal_A):
        ax.plot(t_axis, traj, color="cornflowerblue", alpha=0.5, linewidth=1,
                label="D_cal_A (success, N1)" if i == 0 else None)
    for i, traj in enumerate(D_cal_B):
        ax.plot(t_axis, traj, color="mediumseagreen", alpha=0.5, linewidth=1,
                label="D_cal_B (success, N2)" if i == 0 else None)
    for i, traj in enumerate(fail_trajectories):
        ax.plot(t_axis, traj, color="lightcoral", alpha=0.5, linewidth=1,
                label="failure (not used to fit the band)" if i == 0 else None)
    ax.plot(t_axis, mu_t, color="black", linewidth=2, label="mu_t (mean of D_cal_A)")
    ax.plot(t_axis, upper_t, color="darkred", linewidth=2, linestyle="--",
            label=f"eta_t = mu_t + h*s_A(t)  (alpha={alpha})")
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(f"Functional CP band ({dataset_label}) -- N1={N1}, N2={N2}, h={h:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / f"cp_band_{dataset_label}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    band_path = output_dir / f"cp_band_{dataset_label}_band.npz"
    np.savez(
        band_path, mu_t=mu_t, s_A=s_A, upper_t=upper_t, h=h, alpha=alpha, max_len=max_len,
        N1=N1, N2=N2, ab_split_fraction=ab_split_fraction, split_seed=seed,
        source_cache=str(scores_npz_path),
    )

    result = {
        "dataset_label": dataset_label,
        "scores_npz_path": str(scores_npz_path),
        "alpha": alpha,
        "seed": seed,
        "ab_split_fraction": ab_split_fraction,
        "n_success_total": int(len(D_cal)),
        "n_failure_total": int(len(fail_idx)),
        "N1": N1,
        "N2": N2,
        "n2_warning": n2_warning,
        "max_len": int(max_len),
        "h": h,
        "cal_B_coverage": cal_B_coverage,
        "cal_B_coverage_target": 1 - alpha,
        "n_failures_flagged": int(fail_flagged),
        "n_failures_total": len(fail_trajectories),
        "failure_flag_rate": (fail_flagged / len(fail_trajectories)) if fail_trajectories else None,
        "avg_detection_time": avg_detection_time,
        "failure_first_crossings": first_crossings,
        "plot_path": str(plot_path),
        "band_path": str(band_path),
    }
    (output_dir / f"cp_band_{dataset_label}_result.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate_band_on_dataset(
    scores_npz_path: Path,
    output_dir: Path,
    *,
    band_path: Path,
    dataset_label: str = "unseen",
) -> dict:
    """Applies an ALREADY-FITTED band (from run_cp_band, e.g. fit on seen data) AS-IS to
    a different dataset's episodes -- no refitting. Trajectories are edge-padded/cropped
    to the band's own max_len (its eta_t is only defined out to there). Reports empirical
    coverage on successes (target: >= 1 - alpha) and the flag rate / first-crossing time
    on failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    band = np.load(band_path, allow_pickle=True)
    upper_t = band["upper_t"]
    max_len = int(band["max_len"])
    alpha = float(band["alpha"])
    h = float(band["h"])

    cache = np.load(scores_npz_path, allow_pickle=True)
    scores = cache["scores"]
    lengths = cache["lengths"]
    tasks = cache["tasks"]
    success = cache["success"]

    success_idx = np.flatnonzero(success)
    fail_idx = np.flatnonzero(~success)
    succ_trajectories = [edge_pad(scores[i], max_len) for i in success_idx]
    fail_trajectories = [edge_pad(scores[i], max_len) for i in fail_idx]

    succ_covered = sum(stays_under_band(t, upper_t) for t in succ_trajectories)
    coverage = (succ_covered / len(succ_trajectories)) if succ_trajectories else None
    guarantee_holds = (coverage is not None) and (coverage >= 1 - alpha)

    fail_flagged = sum(not stays_under_band(t, upper_t) for t in fail_trajectories)
    flag_rate = (fail_flagged / len(fail_trajectories)) if fail_trajectories else None
    first_crossings, avg_detection_time = compute_detection_times(fail_idx, fail_trajectories, upper_t, lengths, tasks)

    # Plot: overlay the new dataset's trajectories against the fixed (already-fit) band.
    t_axis = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, traj in enumerate(succ_trajectories):
        ax.plot(t_axis, traj, color="cornflowerblue", alpha=0.5, linewidth=1,
                label=f"success ({dataset_label})" if i == 0 else None)
    for i, traj in enumerate(fail_trajectories):
        ax.plot(t_axis, traj, color="lightcoral", alpha=0.5, linewidth=1,
                label=f"failure ({dataset_label})" if i == 0 else None)
    ax.plot(t_axis, band["mu_t"], color="black", linewidth=2, label="mu_t (fixed, from seen fit)")
    ax.plot(t_axis, upper_t, color="darkred", linewidth=2, linestyle="--",
            label=f"eta_t (fixed, h={h:.3f}, alpha={alpha})")
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(f"CP band transfer check on {dataset_label} -- coverage={coverage:.3f}" if coverage is not None
                 else f"CP band transfer check on {dataset_label}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / f"cp_band_{dataset_label}_transfer_check.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    result = {
        "dataset_label": dataset_label,
        "scores_npz_path": str(scores_npz_path),
        "band_path": str(band_path),
        "alpha": alpha,
        "h": h,
        "max_len": max_len,
        "n_success": len(succ_trajectories),
        "n_failure": len(fail_trajectories),
        "coverage": coverage,
        "coverage_target": 1 - alpha,
        "guarantee_holds": guarantee_holds,
        "n_failures_flagged": int(fail_flagged),
        "failure_flag_rate": flag_rate,
        "avg_detection_time": avg_detection_time,
        "failure_first_crossings": first_crossings,
        "plot_path": str(plot_path),
    }
    (output_dir / f"cp_band_{dataset_label}_transfer_check_result.json").write_text(json.dumps(result, indent=2))
    return result
