#!/usr/bin/env python3
"""The construct_joint_threshold.ipynb algorithm as a plain, callable function -- a single
threshold h that simultaneously bounds FPR and FNR (Learn-then-Test p-values + Bonferroni
correction over an h-sweep, then the fastest-detecting Bonferroni-valid h), plus a Bayes'-
rule guarantee and its empirical counterpart for P(failure | flagged) at that h.

Ported cell-for-cell from the notebook (this session read every cell verbatim before
converting), with two changes made for unattended/repeated sweep use:
  - the "3 example heights" illustrative plot is dropped (notebook-only exploration, adds
    nothing to a scripted run);
  - the h-sweep point count is a parameter (`num_h`, default 20) instead of the notebook's
    inconsistent hardcoded value.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402


def edge_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


def run_joint_threshold(
    scores_npz_path: Path,
    output_dir: Path,
    *,
    alpha_fpr: float = 0.25,
    alpha_fnr: float = 0.25,
    delta: float = 0.1,
    h_min: float = 0.0,
    h_max: float = 2000.0,
    num_h: int = 20,
    t_max_zoom: int = 60,
    dataset_label: str = "seen",
) -> dict:
    """Runs the full pipeline against one cached score-trajectory .npz (as produced by
    compute_calibration_scores.py) and writes its plots into output_dir. Returns a dict with
    everything a sweep summary needs: selected_h, detection_time, the Bayes'-rule guarantee,
    and its empirical counterpart, plus the supporting numbers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(scores_npz_path, allow_pickle=True)
    scores = cache["scores"]
    lengths = cache["lengths"]
    success = cache["success"]
    n_episodes = len(scores)
    logging.info(
        "[%s] Loaded %d episodes (%d success, %d failure) from %s",
        dataset_label, n_episodes, int(success.sum()), int((~success).sum()), scores_npz_path,
    )
    if n_episodes == 0 or (~success).sum() == 0 or success.sum() == 0:
        raise ValueError(
            f"[{dataset_label}] Need at least one success AND one failure episode in "
            f"{scores_npz_path} (got {n_episodes} total, {int(success.sum())} success)."
        )

    # --- mu_t: mean trajectory over ALL episodes (both classes pooled) ---
    max_len = int(lengths.max())
    D_all = np.stack([edge_pad(s, max_len) for s in scores])
    mu_t = D_all.mean(axis=0)

    # --- constant modulation, Eq. (1): s(t) = 1/T ---
    T = max_len
    step_size = np.full(T, 1.0 / T)

    succ_scores = D_all[success]
    fail_scores = D_all[~success]
    fail_lengths = lengths[~success]
    n_succ, n_fail = len(succ_scores), len(fail_scores)

    # --- FPR/FNR/detection-time sweep over h ---
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

    eer_idx = int(np.argmin(np.abs(fpr_sweep - fnr_sweep)))
    logging.info(
        "[%s] Approx. equal-error-rate point: h=%.1f, FPR=%.3f, FNR=%.3f, avg detection time=%.3f",
        dataset_label, h_sweep[eer_idx], fpr_sweep[eer_idx], fnr_sweep[eer_idx], det_time_sweep[eer_idx],
    )

    t_axis = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=2, label="FPR(h) -- successes flagged")
    ax.plot(h_sweep, fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=2, label="FNR(h) -- failures missed")
    ax.plot(h_sweep, det_time_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="forestgreen", linewidth=2, linestyle="-.",
            label="avg detection time(h) -- failures, relative to episode length")
    ax.axvline(h_sweep[eer_idx], color="gray", linestyle=":", linewidth=1,
               label=f"equal-error h={h_sweep[eer_idx]:.0f}")
    ax.set_xlabel("h")
    ax.set_ylabel("rate / relative detection time")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"FPR / FNR / avg detection time vs. h ({dataset_label}, constant modulation)")
    ax.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_fpr_fnr_sweep.png", dpi=150)
    plt.close(fig)

    t_max_zoom_eff = min(t_max_zoom, max_len)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=2, label="FPR(h)")
    ax.plot(h_sweep, fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=2, label="FNR(h)")
    ax.plot(h_sweep, det_time_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="forestgreen", linewidth=2, linestyle="-.", label="avg detection time(h)")
    ax.axvline(h_sweep[eer_idx], color="gray", linestyle=":", linewidth=1, label=f"equal-error h={h_sweep[eer_idx]:.0f}")
    ax.set_xlabel("h")
    ax.set_ylabel("rate / relative detection time")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"FPR / FNR / avg detection time vs. h ({dataset_label}, zoomed t=0..{t_max_zoom_eff})")
    ax.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_fpr_fnr_sweep_zoom.png", dpi=150)
    plt.close(fig)

    # --- Learn-then-Test p-values per risk ---
    violations_fpr = np.round(fpr_sweep * n_succ).astype(int)
    violations_fnr = np.round(fnr_sweep * n_fail).astype(int)
    p_fpr_sweep = binom.cdf(violations_fpr, n_succ, alpha_fpr)
    p_fnr_sweep = binom.cdf(violations_fnr, n_fail, alpha_fnr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, p_fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=2, label=f"p_FPR(h)  (alpha_FPR={alpha_fpr})")
    ax.plot(h_sweep, p_fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=2, label=f"p_FNR(h)  (alpha_FNR={alpha_fnr})")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1, label="uncorrected significance = 0.05")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title(f"Learn-then-Test p-values vs. h ({dataset_label})")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_pvalues.png", dpi=150)
    plt.close(fig)

    # --- joint p-value: max(p_fpr, p_fnr), a valid p-value for the union null ---
    p_joint_sweep = np.maximum(p_fpr_sweep, p_fnr_sweep)
    best_idx = int(np.argmin(p_joint_sweep))
    logging.info(
        "[%s] Best (lowest p_joint) h in the sweep: h=%.1f, p_joint=%.4f (FPR=%.3f, FNR=%.3f)",
        dataset_label, h_sweep[best_idx], p_joint_sweep[best_idx], fpr_sweep[best_idx], fnr_sweep[best_idx],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, p_fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=1.5, alpha=0.5, linestyle="--", label="p_FPR(h)")
    ax.plot(h_sweep, p_fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=1.5, alpha=0.5, linestyle="--", label="p_FNR(h)")
    ax.plot(h_sweep, p_joint_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="black", linewidth=2.5, label="p_joint(h) = max(p_FPR, p_FNR)")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1, label="significance = 0.05")
    ax.axvline(h_sweep[best_idx], color="forestgreen", linestyle=":", linewidth=1,
               label=f"best h={h_sweep[best_idx]:.0f} (min p_joint)")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title(f"Joint p-value vs. h ({dataset_label})")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_pvalues_joint.png", dpi=150)
    plt.close(fig)

    # --- Bonferroni correction over the h grid: p_joint(h) <= delta / |H| ---
    bonferroni_threshold = delta / len(h_sweep)
    valid_h_mask = p_joint_sweep <= bonferroni_threshold
    valid_hs = h_sweep[valid_h_mask]
    logging.info(
        "[%s] Bonferroni threshold = %.5f; %d/%d h values control both risks at FWER<=delta",
        dataset_label, bonferroni_threshold, int(valid_h_mask.sum()), len(h_sweep),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, p_fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=1.5, alpha=0.5, linestyle="--", label="p_FPR(h)")
    ax.plot(h_sweep, p_fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=1.5, alpha=0.5, linestyle="--", label="p_FNR(h)")
    ax.plot(h_sweep, p_joint_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="black", linewidth=2.5, label="p_joint(h) = max(p_FPR, p_FNR)")
    ax.plot(h_sweep, det_time_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="forestgreen", linewidth=2, linestyle="-.",
            label="avg detection time(h) -- failures, relative to episode length")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1, label="uncorrected significance = 0.05")
    ax.axhline(bonferroni_threshold, color="red", linestyle="--", linewidth=2,
               label=f"Bonferroni threshold = delta/|H| = {bonferroni_threshold:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("p-value / relative detection time (log scale)")
    ax.set_title(f"Joint p-value + avg detection time vs. h, Bonferroni-corrected ({dataset_label})")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_pvalues_bonferroni.png", dpi=150)
    plt.close(fig)

    if not valid_h_mask.any():
        raise ValueError(
            f"[{dataset_label}] No h in the sweep satisfies the Bonferroni correction "
            f"(min p_joint={p_joint_sweep.min():.4f} at h={h_sweep[np.argmin(p_joint_sweep)]:.1f}, "
            f"vs. threshold {bonferroni_threshold:.5f}) -- nothing to select from."
        )

    # --- select h*: minimum detection time among the Bonferroni-valid set ---
    valid_indices = np.flatnonzero(valid_h_mask)
    selected_idx = int(valid_indices[np.argmin(det_time_sweep[valid_h_mask])])
    selected_h = float(h_sweep[selected_idx])
    logging.info(
        "[%s] Selected h=%.2f (min avg detection time=%.3f, FPR=%.3f, FNR=%.3f, p_joint=%.4f)",
        dataset_label, selected_h, det_time_sweep[selected_idx], fpr_sweep[selected_idx],
        fnr_sweep[selected_idx], p_joint_sweep[selected_idx],
    )

    t_axis_zoom = t_axis[:t_max_zoom_eff]
    eta_selected = mu_t + selected_h * step_size
    fig, ax = plt.subplots(figsize=(11, 6))
    success_labeled, failure_labeled = False, False
    for i in range(len(D_all)):
        if success[i]:
            color, label = "cornflowerblue", (None if success_labeled else "success")
            success_labeled = True
        else:
            color, label = "lightcoral", (None if failure_labeled else "failure")
            failure_labeled = True
        ax.plot(t_axis_zoom, D_all[i, :t_max_zoom_eff], color=color, alpha=0.35, linewidth=1, label=label)
    ax.plot(t_axis_zoom, mu_t[:t_max_zoom_eff], color="black", linewidth=2.5, label="mu_t (all episodes combined)")
    ax.plot(t_axis_zoom, eta_selected[:t_max_zoom_eff], color="darkviolet", linewidth=2.5, linestyle="--",
            label=f"eta_h(t), selected h={selected_h:.1f}")
    ax.set_xlim(0, t_max_zoom_eff)
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(
        f"Bonferroni-selected threshold ({dataset_label}) -- h={selected_h:.1f}, zoomed to t=0..{t_max_zoom_eff}\n"
        f"FPR={fpr_sweep[selected_idx]:.3f}, FNR={fnr_sweep[selected_idx]:.3f}, "
        f"avg detection time={det_time_sweep[selected_idx]:.3f}"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_selected_h_zoom.png", dpi=150)
    plt.close(fig)

    # --- Bayes'-rule guarantee: P(F|E) >= (1-alpha_fnr)*P(F) / ((1-alpha_fnr)*P(F) + alpha_fpr*P(S)) ---
    P_F = n_fail / (n_fail + n_succ)
    P_S = 1 - P_F
    bayes_bound = ((1 - alpha_fnr) * P_F) / ((1 - alpha_fnr) * P_F + alpha_fpr * P_S)

    # --- empirical P(F|E(h*)): among flagged episodes, the actual fraction that are failures ---
    flagged_succ = np.any(succ_scores > eta_selected, axis=1)
    flagged_fail = np.any(fail_scores > eta_selected, axis=1)
    n_flagged = int(flagged_succ.sum() + flagged_fail.sum())
    empirical_p_f_given_e = float(flagged_fail.sum() / n_flagged) if n_flagged > 0 else float("nan")
    logging.info(
        "[%s] P(F|E) >= %.4f (Bayes bound); empirical P(F|E(h*)) = %.4f (%d/%d flagged episodes)",
        dataset_label, bayes_bound, empirical_p_f_given_e, int(flagged_fail.sum()), n_flagged,
    )

    result = {
        "dataset_label": dataset_label,
        "n_episodes": n_episodes,
        "n_succ": n_succ,
        "n_fail": n_fail,
        "alpha_fpr": alpha_fpr,
        "alpha_fnr": alpha_fnr,
        "delta": delta,
        "h_sweep": {"min": h_min, "max": h_max, "num": num_h},
        "bonferroni_threshold": bonferroni_threshold,
        "bonferroni_valid_range": [float(valid_hs.min()), float(valid_hs.max())] if valid_hs.size else None,
        "bonferroni_valid_count": int(valid_h_mask.sum()),
        "selected_h": selected_h,
        "detection_time": float(det_time_sweep[selected_idx]),
        "fpr": float(fpr_sweep[selected_idx]),
        "fnr": float(fnr_sweep[selected_idx]),
        "p_joint": float(p_joint_sweep[selected_idx]),
        "p_f": P_F,
        "p_s": P_S,
        "bayes_bound_p_f_given_e": float(bayes_bound),
        "empirical_p_f_given_e": empirical_p_f_given_e,
        "n_flagged": n_flagged,
        "n_flagged_fail": int(flagged_fail.sum()),
    }
    # Persist the fixed decision rule itself (mu_t, step_size, h*) -- not just the scalar
    # summary -- so evaluate_threshold_on_dataset() can apply this *exact* eta(t) to a
    # different dataset (e.g. unseen tasks) as a transfer check, without recalibrating.
    threshold_curve_path = output_dir / f"joint_threshold_{dataset_label}_threshold_curve.npz"
    np.savez(threshold_curve_path, mu_t=mu_t, step_size=step_size, selected_h=selected_h)
    result["threshold_curve_path"] = str(threshold_curve_path)

    (output_dir / f"joint_threshold_{dataset_label}_result.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate_threshold_on_dataset(
    scores_npz_path: Path,
    output_dir: Path,
    *,
    threshold_curve_path: Path,
    alpha_fpr: float,
    alpha_fnr: float,
    t_max_zoom: int = 60,
    dataset_label: str = "unseen",
) -> dict:
    """Applies a FIXED decision rule eta(t) = mu_t + h*step_size -- calibrated elsewhere via
    run_joint_threshold() on a different (e.g. seen) dataset, saved as threshold_curve_path
    -- to a new dataset, and reports the empirical FPR/FNR/detection-time/P(F|E) it actually
    achieves there. This is a transfer check, NOT a recalibration: h is not re-selected here,
    so there's no h-sweep, no Learn-then-Test p-values, and no Bonferroni step -- the point is
    to see how far the guarantee established on the calibration dataset holds up (or doesn't)
    on genuinely new data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = np.load(threshold_curve_path)
    mu_t_cal, step_size_cal, selected_h = curve["mu_t"], curve["step_size"], float(curve["selected_h"])

    cache = np.load(scores_npz_path, allow_pickle=True)
    scores = cache["scores"]
    lengths = cache["lengths"]
    success = cache["success"]
    n_episodes = len(scores)
    logging.info(
        "[%s] Loaded %d episodes (%d success, %d failure) from %s -- evaluating fixed h*=%.2f",
        dataset_label, n_episodes, int(success.sum()), int((~success).sum()), scores_npz_path, selected_h,
    )
    if n_episodes == 0 or (~success).sum() == 0 or success.sum() == 0:
        raise ValueError(
            f"[{dataset_label}] Need at least one success AND one failure episode in "
            f"{scores_npz_path} (got {n_episodes} total, {int(success.sum())} success)."
        )

    # eta(t) was only ever defined out to len(mu_t_cal) timesteps -- edge-pad (or crop) both
    # the calibrated mu_t/step_size AND this dataset's own trajectories out to a shared
    # length, exactly like run_joint_threshold's own D_all pooling. step_size is constant, so
    # edge-padding it just repeats the same value -- no distortion; mu_t's edge-pad matches
    # the same "hold the last value" convention used throughout this project (SAFE's own).
    max_len = max(len(mu_t_cal), int(lengths.max()))
    mu_t = edge_pad(mu_t_cal, max_len)
    step_size = edge_pad(step_size_cal, max_len)
    D_all = np.stack([edge_pad(s, max_len) for s in scores])
    eta = mu_t + selected_h * step_size

    succ_scores = D_all[success]
    fail_scores = D_all[~success]
    fail_lengths = lengths[~success]
    n_succ, n_fail = len(succ_scores), len(fail_scores)

    fpr = float(np.mean(np.any(succ_scores > eta, axis=1)))
    fnr = float(np.mean(np.all(fail_scores <= eta, axis=1)))
    detection_mask = fail_scores > eta
    has_detection = detection_mask.any(axis=1)
    first_detection = detection_mask.argmax(axis=1)
    detection_times = np.where(has_detection, first_detection, fail_lengths)
    detection_time = float(np.mean(detection_times / fail_lengths))

    flagged_succ = np.any(succ_scores > eta, axis=1)
    flagged_fail = np.any(fail_scores > eta, axis=1)
    n_flagged = int(flagged_succ.sum() + flagged_fail.sum())
    empirical_p_f_given_e = float(flagged_fail.sum() / n_flagged) if n_flagged > 0 else float("nan")

    guarantee_holds_fpr = fpr <= alpha_fpr
    guarantee_holds_fnr = fnr <= alpha_fnr
    logging.info(
        "[%s] At fixed h*=%.2f: FPR=%.3f (target<=%.2f, %s), FNR=%.3f (target<=%.2f, %s), "
        "avg detection time=%.3f, empirical P(F|E)=%.4f (%d/%d flagged)",
        dataset_label, selected_h, fpr, alpha_fpr, "holds" if guarantee_holds_fpr else "VIOLATED",
        fnr, alpha_fnr, "holds" if guarantee_holds_fnr else "VIOLATED",
        detection_time, empirical_p_f_given_e, int(flagged_fail.sum()), n_flagged,
    )

    t_max_zoom_eff = min(t_max_zoom, max_len)
    t_axis_zoom = np.arange(max_len)[:t_max_zoom_eff]
    fig, ax = plt.subplots(figsize=(11, 6))
    success_labeled, failure_labeled = False, False
    for i in range(len(D_all)):
        if success[i]:
            color, label = "cornflowerblue", (None if success_labeled else "success")
            success_labeled = True
        else:
            color, label = "lightcoral", (None if failure_labeled else "failure")
            failure_labeled = True
        ax.plot(t_axis_zoom, D_all[i, :t_max_zoom_eff], color=color, alpha=0.35, linewidth=1, label=label)
    ax.plot(t_axis_zoom, mu_t[:t_max_zoom_eff], color="black", linewidth=2.5, label="mu_t (calibrated elsewhere)")
    ax.plot(t_axis_zoom, eta[:t_max_zoom_eff], color="darkviolet", linewidth=2.5, linestyle="--",
            label=f"eta_h(t), fixed h*={selected_h:.1f} (from seen calibration)")
    ax.set_xlim(0, t_max_zoom_eff)
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(
        f"Transfer check on {dataset_label} -- fixed h*={selected_h:.1f}, zoomed to t=0..{t_max_zoom_eff}\n"
        f"FPR={fpr:.3f} ({'holds' if guarantee_holds_fpr else 'VIOLATED'} vs alpha_fpr={alpha_fpr}), "
        f"FNR={fnr:.3f} ({'holds' if guarantee_holds_fnr else 'VIOLATED'} vs alpha_fnr={alpha_fnr})"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_transfer_check_zoom.png", dpi=150)
    plt.close(fig)

    result = {
        "dataset_label": dataset_label,
        "n_episodes": n_episodes,
        "n_succ": n_succ,
        "n_fail": n_fail,
        "selected_h": selected_h,
        "alpha_fpr": alpha_fpr,
        "alpha_fnr": alpha_fnr,
        "fpr": fpr,
        "fnr": fnr,
        "guarantee_holds_fpr": guarantee_holds_fpr,
        "guarantee_holds_fnr": guarantee_holds_fnr,
        "detection_time": detection_time,
        "empirical_p_f_given_e": empirical_p_f_given_e,
        "n_flagged": n_flagged,
        "n_flagged_fail": int(flagged_fail.sum()),
    }
    (output_dir / f"joint_threshold_{dataset_label}_transfer_check_result.json").write_text(json.dumps(result, indent=2))
    return result
