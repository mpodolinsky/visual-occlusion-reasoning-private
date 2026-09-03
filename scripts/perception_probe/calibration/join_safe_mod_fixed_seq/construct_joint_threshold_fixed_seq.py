#!/usr/bin/env python3
"""A variant of joint_safe_modulation/construct_joint_threshold_safe_modulation.py with
exactly ONE further change: the multiple-testing correction over the h-grid is
MULTI-START FIXED SEQUENCE TESTING (Angelopoulos et al., "Learn then Test", arXiv:
2110.01052, Section 2.3.1, Algorithm 1) instead of Bonferroni.

Why: Bonferroni spends its error budget delta/|H| on EVERY candidate h in the grid,
regardless of whether that h was ever a real contender -- with num_h=40 candidates, that's
delta/40, which is what was making Bonferroni fail on most of our models (best-sweep-2/3/4
and even base, once the D_fit/D_test sample split was applied -- see the split-seed
docstring in the copied run_joint_threshold below). Fixed sequence testing instead commits
to a FIXED (data-independent) walk order over the grid and tests each candidate at the raw
level delta/|J| (J = the set of starting points, not the full grid) -- it only ever risks
wrongly accepting the FIRST candidate a given walk fails on, since it stops there for
good, so it doesn't need monotonicity in the risk to be valid (Proposition 3/4 of the
paper), just a walk order fixed in advance.

Our p_joint(h) is roughly U-shaped, not monotone (bad at h_min from FPR, bad at h_max from
FNR, good somewhere in the middle) -- so a SINGLE walk from one end would fail its very
first test and return nothing. Multi-start fixes this: initialize the walk from n_starts
equally-spaced points across the grid (chosen a priori from domain knowledge -- see
h_min/h_max below -- not from peeking at the calibration data itself), each walking FORWARD
(ascending h) at level delta/n_starts, stopping for good the moment it fails; a start that
lands inside the good middle region gets to walk outward from there "for free."

h_min/h_max default to [0, 4] here (not [0, 2] like joint_safe_modulation) -- this range
comes from OUR OWN domain knowledge of SAFE's Tfunc-modulation band scale, established
empirically from D_fit ("cal set A"): SAFE's own h under this modulation (also computed
from D_fit/D_test) has consistently landed in the 0.6-1.9 range across every model we've
run so far, so [0, 4] is a generous, pre-registered envelope around that -- chosen from
prior runs' behavior, not from this specific run's own p-values, so it doesn't compromise
the fixed-sequence guarantee's validity.

Everything else (successes-only mu_t/s(t) fit from D_fit, SAFE's Tfunc modulation, the
D_fit/D_test sample split, SAFE's own h-selection rule computed for comparison, the h-sweep
itself, Learn-then-Test p-values) is UNCHANGED from joint_safe_modulation -- only the
correction step (Bonferroni -> multi-start fixed sequence testing) differs.

Rationale: constant modulation gives every timestep the same band width, so h alone
controls the whole trajectory uniformly. SAFE's Tfunc modulation instead measures, per
timestep, how much successful episodes actually vary around mu_t there (with an
alpha-quantile trim so one outlier episode can't blow up the modulation, hence the whole
band, at every timestep) -- so the resulting band is naturally wider where normal
rollouts are naturally noisier, and narrower where they're consistent. h then scales
that whole shape uniformly, same as before -- only what h scales has changed.

Concretely: s(t) = max_{k in H} |D_succ^k(t) - mu_t(t)| + EPS, computed from a `D_fit`
subset of successful episodes (mu_t is the mean of that same D_fit subset) -- NOT the
both-classes-pooled mean joint_threshold.py itself uses (that's a deliberate, different
choice for THAT pipeline). The REMAINING successes (`D_test`, held out from fitting) are
what the FPR test and SAFE's own h-quantile are actually evaluated against -- this sample
split (successes only; see run_joint_threshold's docstring for why failures don't need
one) matches construct_cp_band.py's own D_cal_A(fit)/D_cal_B(quantile) split and is
required for the Learn-then-Test binomial p-value to be valid: it's only a valid p-value
if each tested success's "did it cross eta_h?" indicator is independent of eta_h itself,
which fails if eta_h's own mu_t/s(t) were fit from that same success.

Ported cell-for-cell from the notebook (this session read every cell verbatim before
converting), with three changes made for unattended/repeated sweep use:
  - SAFE's adaptive Tfunc modulation replaces the constant s(t)=1/T (this file's own
    modification, see above), with its own alpha (`modulation_alpha`, default 0.15,
    matching construct_cp_band.py's default) governing the outlier-trimming quantile --
    kept separate from alpha_fpr/alpha_fnr, which still govern the Learn-then-Test
    p-values as before;
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

EPS = 1e-8


def edge_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


def tfunc_modulation(cal_succ: np.ndarray, mu_t: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Port of SAFE's FunctionalPredictor._get_modulation_trajectory(ModulationType.Tfunc)
    (same as construct_cp_band.py's copy). Returns (modulation_trajectory, H_mask) where
    H_mask marks which successful episodes survived trimming -- episodes whose own
    worst-timestep deviation from mu_t is extreme (above the (1-alpha)-quantile of all
    episodes' worst deviations) are dropped so a single outlier can't blow up the
    modulation, and therefore the band width, at every timestep."""
    n = cal_succ.shape[0]
    per_episode_max_dev = np.max(np.abs(cal_succ - mu_t), axis=1)  # (n,)

    if int(np.ceil((n + 1) * (1 - alpha))) > n:
        H_mask = np.ones(n, dtype=bool)  # H = [n], no trimming
    else:
        gamma = np.sort(per_episode_max_dev)[int(np.ceil((n + 1) * (1 - alpha))) - 1]
        H_mask = per_episode_max_dev <= gamma

    modulation = np.max(np.abs(cal_succ[H_mask] - mu_t), axis=0) + EPS
    return modulation, H_mask


def fixed_sequence_test(p_joint_sweep: np.ndarray, delta: float, n_starts: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Multi-start fixed sequence testing (Angelopoulos et al. 2022, Algorithm 1): walks
    FORWARD (ascending index) from each of n_starts equally-spaced starting indices into
    p_joint_sweep, accepting (marking True) each index while p_joint_sweep[j] <= delta/|J|,
    and permanently stopping that walk the instant it hits an index that fails -- this is
    what makes it FWER-valid regardless of whether p_joint_sweep is monotone (see module
    docstring). Skips re-walking any index already accepted by an earlier start (mirrors
    the paper's "if λ_j not in Λ̂" check). Starting indices are a structural, equally-spaced
    choice over the grid -- NOT derived from p_joint_sweep itself.

    Returns (accepted_mask, threshold, start_indices)."""
    n = len(p_joint_sweep)
    start_indices = np.unique(np.round(np.linspace(0, n - 1, n_starts)).astype(int))
    threshold = delta / len(start_indices)
    accepted = np.zeros(n, dtype=bool)
    for j0 in start_indices:
        j = int(j0)
        if accepted[j]:
            continue  # already covered by an earlier start's walk
        while j < n and p_joint_sweep[j] <= threshold:
            accepted[j] = True
            j += 1
    return accepted, threshold, start_indices


def run_joint_threshold(
    scores_npz_path: Path,
    output_dir: Path,
    *,
    alpha_fpr: float = 0.25,
    alpha_fnr: float = 0.25,
    delta: float = 0.1,
    h_min: float = 0.0,
    h_max: float = 4.0,
    num_h: int = 40,
    n_starts: int = 5,
    t_max_zoom: int = 60,
    modulation_alpha: float = 0.15,
    safe_reference_h: float | None = None,
    split_seed: int = 0,
    ab_split_fraction: float = 0.3,
    dataset_label: str = "seen",
) -> dict:
    """Runs the full pipeline against one cached score-trajectory .npz (as produced by
    compute_calibration_scores.py) and writes its plots into output_dir. Returns a dict with
    everything a sweep summary needs: selected_h, detection_time, the Bayes'-rule guarantee,
    and its empirical counterpart, plus the supporting numbers.

    Sample splitting (successes only): mu_t/s(t) are fit from a `D_fit` subset of successful
    episodes (fraction `ab_split_fraction`, matching construct_cp_band.py's own convention);
    the REMAINING successes (`D_test`, held out from fitting) are what the FPR
    test/Learn-then-Test p-value/SAFE's own h-quantile all actually get evaluated against.
    This is required for validity: the binomial p-value p_FPR(h) is only a valid p-value if
    each tested success's "did it cross eta_h?" indicator is independent of eta_h itself --
    which fails if eta_h's own mu_t/s(t) were fit from that same success (an unusually high
    trajectory pulls mu_t up, biasing every other success's indicator too). Failures are
    NOT split: mu_t/s(t) never touch failure data at all (they're fit from successes only),
    so every failure is already "fresh" relative to the fixed band regardless of splitting --
    using ALL failures for the FNR test is valid and simply gives it more power (larger
    n_fail in the binomial trial), with no matching-ratio requirement against the success
    split.

    n_starts: number of equally-spaced starting points for multi-start fixed sequence
    testing (see module docstring) -- NOT the number of h values ultimately accepted, and
    not the same as num_h (the grid resolution); each start is its own forward-only walk,
    tested at level delta/n_starts.

    safe_reference_h: optional override for the h SAFE's own selection rule would have
    picked -- by default (None) this is computed INTERNALLY, from this same call's own
    mu_t/step_size and the SAME D_test split as our own FPR test (so the only thing that
    differs from our own fixed-sequence-selected h* is the selection rule itself, not the
    data or the band shape). Every h-axis plot below draws it as a vertical reference line,
    so the two selection procedures' choices of h are visually comparable on the same axes.
    Pass an explicit value only to compare against a DIFFERENT dataset's SAFE h (e.g. for a
    transfer-style check) instead of this call's own.
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

    # --- Sample split (successes only -- see run_joint_threshold's docstring for why):
    # D_fit fits mu_t/s(t); D_test is held out from fitting and is what everything
    # downstream (our FPR test, SAFE's own h-quantile) is actually evaluated against. ---
    max_len = int(lengths.max())
    D_all = np.stack([edge_pad(s, max_len) for s in scores])
    succ_idx_all = np.flatnonzero(success)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(succ_idx_all))
    n_fit = max(1, int(len(succ_idx_all) * ab_split_fraction))
    if n_fit >= len(succ_idx_all):
        raise ValueError(
            f"[{dataset_label}] Only {len(succ_idx_all)} successful episodes -- not enough "
            f"for a D_fit/D_test split at ab_split_fraction={ab_split_fraction}."
        )
    fit_idx, test_idx = succ_idx_all[perm[:n_fit]], succ_idx_all[perm[n_fit:]]
    D_fit_succ, D_test_succ = D_all[fit_idx], D_all[test_idx]
    logging.info(
        "[%s] Success split: %d D_fit (mu_t/s(t)) / %d D_test (FPR test + SAFE h), seed=%d",
        dataset_label, len(fit_idx), len(test_idx), split_seed,
    )

    # --- mu_t: mean trajectory over D_fit successes ONLY -- matching SAFE's own
    # construct_cp_band.py convention (mu_t is fit from D_cal_A, itself a successes-only
    # subset), not the original notebook's both-classes-pooled mean. This file's whole
    # point is to compare our h-selection rule against SAFE's using the SAME band shape
    # (mean + modulation) SAFE would actually use -- so both need to be successes-only,
    # not just the modulation. (Contrast with joint_threshold.py's own pooled mu_t, which
    # is a deliberate, different choice for THAT pipeline -- see joint_succ_mean/ for the
    # standalone successes-only-mean-but-constant-modulation variant.) ---
    mu_t = D_fit_succ.mean(axis=0)

    # --- SAFE's adaptive Tfunc modulation (this file's modification -- see module
    # docstring), in place of the notebook's constant s(t) = 1/T. Fit from D_fit successes
    # only, against the D_fit-only mu_t above -- same split, not the full successful pool. ---
    step_size, modulation_H_mask = tfunc_modulation(D_fit_succ, mu_t, modulation_alpha)
    logging.info(
        "[%s] Tfunc modulation: |H| = %d / %d D_fit successful episodes kept (alpha=%.2f), "
        "s(t) range=[%.4f, %.4f]",
        dataset_label, int(modulation_H_mask.sum()), len(modulation_H_mask), modulation_alpha,
        step_size.min(), step_size.max(),
    )

    # succ_scores is D_test ONLY (held out from fitting mu_t/s(t)) -- this is what our FPR
    # test and SAFE's own h-quantile are evaluated against, below. fail_scores is ALL
    # failures, unsplit -- see the split-seed docstring above for why that's valid.
    succ_scores = D_test_succ
    fail_scores = D_all[~success]
    fail_lengths = lengths[~success]
    n_succ, n_fail = len(succ_scores), len(fail_scores)
    if n_succ < 20:
        logging.warning(
            "[%s] D_test has only %d successful episodes (after holding out %d for D_fit) -- "
            "the FPR test / SAFE h-quantile are correspondingly coarse.",
            dataset_label, n_succ, len(fit_idx),
        )

    # --- SAFE's OWN h-selection rule (the (1-alpha)-quantile of the per-episode max
    # normalized deviation D_j = max_t (score_t - mu_t(t)) / s(t)), applied to the EXACT
    # SAME mu_t/s(t) (fit from D_fit) and the SAME held-out D_test successes our own FPR
    # test uses below -- this now matches construct_cp_band.py's real D_cal_A(fit)/
    # D_cal_B(quantile) split exactly, just reusing our own already-split D_fit/D_test
    # instead of an independently-drawn split, so the comparison isolates the ONE thing
    # that differs from our own fixed-sequence-selected h*: the selection rule itself, not the
    # band shape or which data it's calibrated/tested on.
    D_j_safe = np.max((succ_scores - mu_t) / step_size, axis=1)  # (n_succ,)
    safe_conformal_h = float(np.quantile(D_j_safe, 1 - modulation_alpha))
    logging.info(
        "[%s] SAFE's own h-selection rule, applied to OUR mu_t/s(t)/split: "
        "h=%.4f ((1-alpha)-quantile of D_j, alpha=%.2f, n_succ=%d)",
        dataset_label, safe_conformal_h, modulation_alpha, n_succ,
    )
    if safe_reference_h is None:
        safe_reference_h = safe_conformal_h

    # SAFE's own FPR/FNR/detection-time/empirical P(F|E) at ITS h -- same metrics as our own
    # h* below, evaluated at safe_conformal_h instead, for a like-for-like comparison table.
    eta_safe_conformal = mu_t + safe_conformal_h * step_size
    safe_fpr = float(np.mean(np.any(succ_scores > eta_safe_conformal, axis=1)))
    safe_fnr = float(np.mean(np.all(fail_scores <= eta_safe_conformal, axis=1)))
    safe_detection_mask = fail_scores > eta_safe_conformal
    safe_has_detection = safe_detection_mask.any(axis=1)
    safe_first_detection = safe_detection_mask.argmax(axis=1)
    safe_detection_times = np.where(safe_has_detection, safe_first_detection, fail_lengths)
    safe_detection_time = float(np.mean(safe_detection_times / fail_lengths))
    safe_flagged_succ = np.any(succ_scores > eta_safe_conformal, axis=1)
    safe_flagged_fail = np.any(fail_scores > eta_safe_conformal, axis=1)
    safe_n_flagged = int(safe_flagged_succ.sum() + safe_flagged_fail.sum())
    safe_empirical_p_f_given_e = float(safe_flagged_fail.sum() / safe_n_flagged) if safe_n_flagged > 0 else float("nan")
    logging.info(
        "[%s] SAFE's own h=%.4f: FPR=%.3f, FNR=%.3f, avg detection time=%.3f, empirical P(F|E)=%.4f",
        dataset_label, safe_conformal_h, safe_fpr, safe_fnr, safe_detection_time, safe_empirical_p_f_given_e,
    )

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

    def _draw_safe_reference(ax) -> None:
        """Overlays SAFE's own fully-conformal h (see run_joint_threshold's docstring) as a
        vertical reference line on an h-axis plot, if one was passed in."""
        if safe_reference_h is not None:
            ax.axvline(safe_reference_h, color="darkorange", linestyle="-", linewidth=2.5,
                       label=f"SAFE conformal h*={safe_reference_h:.2f}")

    t_axis = np.arange(max_len)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=2, label="FPR(h) -- successes flagged")
    ax.plot(h_sweep, fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=2, label="FNR(h) -- failures missed")
    ax.plot(h_sweep, det_time_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="forestgreen", linewidth=2, linestyle="-.",
            label="avg detection time(h) -- failures, relative to episode length")
    ax.axvline(h_sweep[eer_idx], color="gray", linestyle=":", linewidth=1,
               label=f"equal-error h={h_sweep[eer_idx]:.0f}")
    _draw_safe_reference(ax)
    ax.set_xlabel("h")
    ax.set_ylabel("rate / relative detection time")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"FPR / FNR / avg detection time vs. h ({dataset_label}, SAFE Tfunc modulation)")
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
    _draw_safe_reference(ax)
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
    _draw_safe_reference(ax)
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
    _draw_safe_reference(ax)
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title(f"Joint p-value vs. h ({dataset_label})")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_pvalues_joint.png", dpi=150)
    plt.close(fig)

    # --- Multi-start fixed sequence testing over the h grid (see module docstring for why
    # this replaces Bonferroni here) ---
    valid_h_mask, fixed_seq_threshold, start_indices = fixed_sequence_test(p_joint_sweep, delta, n_starts)
    valid_hs = h_sweep[valid_h_mask]
    logging.info(
        "[%s] Fixed-seq threshold = delta/|J| = %.5f (J=%d starts at h=%s); "
        "%d/%d h values accepted",
        dataset_label, fixed_seq_threshold, len(start_indices),
        np.round(h_sweep[start_indices], 2).tolist(), int(valid_h_mask.sum()), len(h_sweep),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_sweep, p_fpr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="steelblue", linewidth=1.5, alpha=0.5, linestyle="--", label="p_FPR(h)")
    ax.plot(h_sweep, p_fnr_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="crimson", linewidth=1.5, alpha=0.5, linestyle="--", label="p_FNR(h)")
    ax.plot(h_sweep, p_joint_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="black", linewidth=2.5, label="p_joint(h) = max(p_FPR, p_FNR)")
    ax.plot(h_sweep, det_time_sweep, marker="o", markersize=5, markerfacecolor="yellow", markeredgecolor="black", markeredgewidth=0.5, color="forestgreen", linewidth=2, linestyle="-.",
            label="avg detection time(h) -- failures, relative to episode length")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1, label="uncorrected significance = 0.05")
    ax.axhline(fixed_seq_threshold, color="red", linestyle="--", linewidth=2,
               label=f"Fixed-seq threshold = delta/|J| = {fixed_seq_threshold:.4f}")
    for si in start_indices:
        ax.axvline(h_sweep[si], color="purple", linestyle=":", linewidth=1, alpha=0.6,
                   label="starting point" if si == start_indices[0] else None)
    if valid_h_mask.any():
        ax.scatter(h_sweep[valid_h_mask], p_joint_sweep[valid_h_mask], marker="s", s=45,
                   facecolors="none", edgecolors="limegreen", linewidths=2, zorder=5, label="accepted h")
    _draw_safe_reference(ax)
    ax.set_yscale("log")
    ax.set_xlabel("h")
    ax.set_ylabel("p-value / relative detection time (log scale)")
    ax.set_title(f"Joint p-value + avg detection time vs. h, fixed-seq-tested ({dataset_label})")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_pvalues_fixed_seq.png", dpi=150)
    plt.close(fig)

    if not valid_h_mask.any():
        raise ValueError(
            f"[{dataset_label}] No h in the sweep was accepted by fixed sequence testing "
            f"(threshold delta/|J|={fixed_seq_threshold:.5f}, starts at h="
            f"{np.round(h_sweep[start_indices], 2).tolist()}) -- nothing to select from."
        )

    # --- select h*: minimum detection time among the fixed-sequence-accepted set ---
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
    ax.plot(t_axis_zoom, mu_t[:t_max_zoom_eff], color="black", linewidth=2.5, label="mu_t (successful episodes only)")
    ax.plot(t_axis_zoom, eta_selected[:t_max_zoom_eff], color="darkviolet", linewidth=2.5, linestyle="--",
            label=f"eta_h(t), selected h={selected_h:.1f}")
    if safe_reference_h is not None:
        eta_safe = mu_t + safe_reference_h * step_size
        ax.plot(t_axis_zoom, eta_safe[:t_max_zoom_eff], color="darkorange", linewidth=2.5, linestyle=":",
                label=f"eta_h(t), SAFE conformal h*={safe_reference_h:.2f}")
    ax.set_xlim(0, t_max_zoom_eff)
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(
        f"Fixed-seq-selected threshold ({dataset_label}) -- h={selected_h:.1f}, zoomed to t=0..{t_max_zoom_eff}\n"
        f"FPR={fpr_sweep[selected_idx]:.3f}, FNR={fnr_sweep[selected_idx]:.3f}, "
        f"avg detection time={det_time_sweep[selected_idx]:.3f}"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_selected_h_zoom.png", dpi=150)
    plt.close(fig)

    # Same plot, full unzoomed trajectories (t=0..max_len) -- the zoomed version above crops
    # to t_max_zoom for readability early on, but this shows the whole episode length,
    # including where failure trajectories that outlast the longest success (LIBERO failures
    # run to the suite's step cap) extend past where the band itself was ever fit.
    fig, ax = plt.subplots(figsize=(11, 6))
    success_labeled, failure_labeled = False, False
    for i in range(len(D_all)):
        if success[i]:
            color, label = "cornflowerblue", (None if success_labeled else "success")
            success_labeled = True
        else:
            color, label = "lightcoral", (None if failure_labeled else "failure")
            failure_labeled = True
        ax.plot(t_axis, D_all[i], color=color, alpha=0.35, linewidth=1, label=label)
    ax.plot(t_axis, mu_t, color="black", linewidth=2.5, label="mu_t (successful episodes only)")
    ax.plot(t_axis, eta_selected, color="darkviolet", linewidth=2.5, linestyle="--",
            label=f"eta_h(t), selected h={selected_h:.1f}")
    if safe_reference_h is not None:
        ax.plot(t_axis, eta_safe, color="darkorange", linewidth=2.5, linestyle=":",
                label=f"eta_h(t), SAFE conformal h*={safe_reference_h:.2f}")
    ax.set_xlim(0, max_len)
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(
        f"Fixed-seq-selected threshold ({dataset_label}) -- h={selected_h:.1f}, full t=0..{max_len}\n"
        f"FPR={fpr_sweep[selected_idx]:.3f}, FNR={fnr_sweep[selected_idx]:.3f}, "
        f"avg detection time={det_time_sweep[selected_idx]:.3f}"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_selected_h_full.png", dpi=150)
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
        "n_succ_fit": len(fit_idx),
        "n_fail": n_fail,
        "split_seed": split_seed,
        "ab_split_fraction": ab_split_fraction,
        "alpha_fpr": alpha_fpr,
        "alpha_fnr": alpha_fnr,
        "delta": delta,
        "modulation_alpha": modulation_alpha,
        "modulation_H_kept": int(modulation_H_mask.sum()),
        "modulation_H_total": len(modulation_H_mask),
        "modulation_range": [float(step_size.min()), float(step_size.max())],
        "safe_reference_h": safe_reference_h,
        "safe_conformal_h": safe_conformal_h,
        "safe_fpr": safe_fpr,
        "safe_fnr": safe_fnr,
        "safe_detection_time": safe_detection_time,
        "safe_empirical_p_f_given_e": safe_empirical_p_f_given_e,
        "h_sweep": {"min": h_min, "max": h_max, "num": num_h},
        "fixed_seq_threshold": fixed_seq_threshold,
        "n_starts": len(start_indices),
        "start_h_values": h_sweep[start_indices].tolist(),
        "fixed_seq_valid_range": [float(valid_hs.min()), float(valid_hs.max())] if valid_hs.size else None,
        "fixed_seq_valid_count": int(valid_h_mask.sum()),
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
    # safe_conformal_h rides along too, purely so the unseen-side plots can also draw SAFE's
    # band as a reference line, same as the seen-side plots already do.
    threshold_curve_path = output_dir / f"joint_threshold_{dataset_label}_threshold_curve.npz"
    np.savez(threshold_curve_path, mu_t=mu_t, step_size=step_size, selected_h=selected_h,
              safe_conformal_h=safe_conformal_h)
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
    so there's no h-sweep, no Learn-then-Test p-values, and no fixed-sequence step -- the point is
    to see how far the guarantee established on the calibration dataset holds up (or doesn't)
    on genuinely new data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = np.load(threshold_curve_path)
    mu_t_cal, step_size_cal, selected_h = curve["mu_t"], curve["step_size"], float(curve["selected_h"])
    safe_conformal_h = float(curve["safe_conformal_h"]) if "safe_conformal_h" in curve else None

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
    # length, exactly like run_joint_threshold's own D_all pooling. step_size here is SAFE's
    # Tfunc modulation (time-varying, not constant) -- edge-padding it holds its LAST fitted
    # timestep's value for any t beyond where it was calibrated, same "hold the last value"
    # convention used throughout this project (SAFE's own); mu_t's edge-pad matches too.
    max_len = max(len(mu_t_cal), int(lengths.max()))
    mu_t = edge_pad(mu_t_cal, max_len)
    step_size = edge_pad(step_size_cal, max_len)
    D_all = np.stack([edge_pad(s, max_len) for s in scores])
    eta = mu_t + selected_h * step_size
    eta_safe = mu_t + safe_conformal_h * step_size if safe_conformal_h is not None else None

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

    # SAFE's own frozen h, evaluated on this SAME unseen data -- no separate report if SAFE's
    # h wasn't available (older threshold_curve, or seen calibration didn't compute one).
    safe_fpr = safe_fnr = safe_detection_time = safe_empirical_p_f_given_e = None
    if eta_safe is not None:
        safe_fpr = float(np.mean(np.any(succ_scores > eta_safe, axis=1)))
        safe_fnr = float(np.mean(np.all(fail_scores <= eta_safe, axis=1)))
        safe_detection_mask = fail_scores > eta_safe
        safe_has_detection = safe_detection_mask.any(axis=1)
        safe_first_detection = safe_detection_mask.argmax(axis=1)
        safe_detection_times = np.where(safe_has_detection, safe_first_detection, fail_lengths)
        safe_detection_time = float(np.mean(safe_detection_times / fail_lengths))
        safe_flagged_succ = np.any(succ_scores > eta_safe, axis=1)
        safe_flagged_fail = np.any(fail_scores > eta_safe, axis=1)
        safe_n_flagged = int(safe_flagged_succ.sum() + safe_flagged_fail.sum())
        safe_empirical_p_f_given_e = float(safe_flagged_fail.sum() / safe_n_flagged) if safe_n_flagged > 0 else float("nan")
        logging.info(
            "[%s] SAFE's own fixed h=%.4f (from seen): FPR=%.3f, FNR=%.3f, avg detection time=%.3f, "
            "empirical P(F|E)=%.4f",
            dataset_label, safe_conformal_h, safe_fpr, safe_fnr, safe_detection_time, safe_empirical_p_f_given_e,
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
    if eta_safe is not None:
        ax.plot(t_axis_zoom, eta_safe[:t_max_zoom_eff], color="darkorange", linewidth=2.5, linestyle=":",
                label=f"eta_h(t), SAFE conformal h*={safe_conformal_h:.2f} (from seen calibration)")
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

    # Same plot, full unzoomed trajectories (t=0..max_len) -- see run_joint_threshold's own
    # "_selected_h_full.png" for the seen-side counterpart and why this is useful.
    fig, ax = plt.subplots(figsize=(11, 6))
    success_labeled, failure_labeled = False, False
    for i in range(len(D_all)):
        if success[i]:
            color, label = "cornflowerblue", (None if success_labeled else "success")
            success_labeled = True
        else:
            color, label = "lightcoral", (None if failure_labeled else "failure")
            failure_labeled = True
        ax.plot(np.arange(max_len), D_all[i], color=color, alpha=0.35, linewidth=1, label=label)
    ax.plot(np.arange(max_len), mu_t, color="black", linewidth=2.5, label="mu_t (calibrated elsewhere)")
    ax.plot(np.arange(max_len), eta, color="darkviolet", linewidth=2.5, linestyle="--",
            label=f"eta_h(t), fixed h*={selected_h:.1f} (from seen calibration)")
    if eta_safe is not None:
        ax.plot(np.arange(max_len), eta_safe, color="darkorange", linewidth=2.5, linestyle=":",
                label=f"eta_h(t), SAFE conformal h*={safe_conformal_h:.2f} (from seen calibration)")
    ax.set_xlim(0, max_len)
    ax.set_xlabel("inference call index t")
    ax.set_ylabel("accumulated score")
    ax.set_title(
        f"Transfer check on {dataset_label} -- fixed h*={selected_h:.1f}, full t=0..{max_len}\n"
        f"FPR={fpr:.3f} ({'holds' if guarantee_holds_fpr else 'VIOLATED'} vs alpha_fpr={alpha_fpr}), "
        f"FNR={fnr:.3f} ({'holds' if guarantee_holds_fnr else 'VIOLATED'} vs alpha_fnr={alpha_fnr})"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"joint_threshold_{dataset_label}_transfer_check_full.png", dpi=150)
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
        "safe_conformal_h": safe_conformal_h,
        "safe_fpr": safe_fpr,
        "safe_fnr": safe_fnr,
        "safe_detection_time": safe_detection_time,
        "safe_empirical_p_f_given_e": safe_empirical_p_f_given_e,
    }
    (output_dir / f"joint_threshold_{dataset_label}_transfer_check_result.json").write_text(json.dumps(result, indent=2))
    return result
