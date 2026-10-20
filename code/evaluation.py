"""
evaluation.py
=============

Reviewer-driven evaluation utilities. Imported by `revised_pipeline.ipynb`.

Contents:
    1. LOSO / subject-wise K-fold splitters     (reviewer point 1)
    2. All metrics in one call: MAE, RMSE,
       Pearson r, MAPE, frequency SNR           (reviewer point 6)
    3. Bland-Altman plot                        (reviewer point 6)
    4. Stratified errors by HR bin / motion /
       illumination / skin-tone proxy           (reviewer points 6 and 8)
    5. Pipeline profiling (latency / fps)       (reviewer point 7)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import pearsonr


# --------------------------------------------------------------------------
# 1. Splitters - reviewer point 1
# --------------------------------------------------------------------------

def loso_split(subject_ids):
    """
    Yield (train_subjects, test_subject) tuples for Leave-One-Subject-Out.

    Args:
        subject_ids : iterable of unique subject identifiers, e.g.
                      ['subject1', 'subject2', ...]

    Yields:
        train_subjects : list of subject ids to train on
        test_subject   : a single subject id held out for testing

    Usage:
        for train, test in loso_split(subjects):
            df_train = data[data["subject"].isin(train)]
            df_test  = data[data["subject"] == test]
            ...
    """
    subject_ids = list(dict.fromkeys(subject_ids))   # de-duplicate, keep order
    for s in subject_ids:
        train = [x for x in subject_ids if x != s]
        yield train, s


def subject_kfold_split(subject_ids, k=5, seed=0):
    """
    Subject-wise K-fold. Useful when LOSO is too slow.

    Splits the SUBJECT list into k folds. All frames of a subject go to
    exactly one fold. No frame-level mixing across folds.

    Yields:
        (train_subjects, test_subjects) per fold.
    """
    subject_ids = list(dict.fromkeys(subject_ids))
    rng = np.random.default_rng(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k)
    for i in range(k):
        test = list(folds[i])
        train = [s for s in shuffled if s not in test]
        yield train, test


# --------------------------------------------------------------------------
# 2. Metrics - reviewer point 6
# --------------------------------------------------------------------------

def compute_all_metrics(y_true, y_pred, fs=None, bvp_pred=None):
    """
    Compute every metric the reviewer asked for.

    Args:
        y_true   : (N,) ground-truth HR in BPM
        y_pred   : (N,) predicted HR in BPM
        fs       : sampling rate (Hz). Optional. Used only for SNR if bvp_pred given.
        bvp_pred : (T,) predicted BVP signal. Optional. If given, frequency SNR
                   is computed (reviewer asked for "SNR/PSD-based metrics").

    Returns:
        dict with keys: mae, rmse, mse, pearson_r, mape, snr_db (or NaN).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    assert y_true.shape == y_pred.shape, "shape mismatch"

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    # Pearson r is undefined if either vector has zero variance
    if y_true.std() < 1e-9 or y_pred.std() < 1e-9 or len(y_true) < 2:
        pr = float("nan")
    else:
        pr, _ = pearsonr(y_true, y_pred)
        pr = float(pr)
    # MAPE - skip rows with y_true ~ 0 to avoid div-by-zero
    nonzero = np.abs(y_true) > 1e-6
    if nonzero.any():
        mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0)
    else:
        mape = float("nan")

    snr_db = float("nan")
    if bvp_pred is not None and fs is not None:
        from rppg_baselines import signal_snr_db
        # Use the mean ground-truth HR as the signal frequency for SNR
        hr_for_snr = float(np.nanmean(y_true))
        snr_db = signal_snr_db(np.asarray(bvp_pred), fs, hr_for_snr)

    return {
        "mae":       mae,
        "rmse":      rmse,
        "mse":       mse,
        "pearson_r": pr,
        "mape":      mape,
        "snr_db":    snr_db,
        "n_samples": int(len(y_true)),
    }


def aggregate_metrics(per_subject_df, weight_col=None):
    """
    Mean +/- std of each metric across subjects.

    Args:
        per_subject_df : DataFrame with columns matching compute_all_metrics output
                         and one row per subject.
        weight_col     : optional column name to weight by (e.g. n_samples).

    Returns:
        DataFrame with mean and std rows.
    """
    metric_cols = ["mae", "rmse", "mse", "pearson_r", "mape", "snr_db"]
    metric_cols = [c for c in metric_cols if c in per_subject_df.columns]
    if weight_col is not None and weight_col in per_subject_df.columns:
        w = per_subject_df[weight_col].values.astype(float)
        means = {c: float(np.nansum(per_subject_df[c] * w) / np.nansum(w))
                 for c in metric_cols}
    else:
        means = {c: float(np.nanmean(per_subject_df[c])) for c in metric_cols}
    stds = {c: float(np.nanstd(per_subject_df[c])) for c in metric_cols}
    return pd.DataFrame([means, stds], index=["mean", "std"])


# --------------------------------------------------------------------------
# 3. Bland-Altman plot - reviewer point 6
# --------------------------------------------------------------------------

def bland_altman_plot(y_true, y_pred, title="Bland-Altman: Predicted vs True HR",
                      savepath=None, dpi=300):
    """
    Standard Bland-Altman agreement plot.

    X axis: mean of (predicted, ground truth) in BPM.
    Y axis: predicted - ground truth in BPM.
    Horizontal lines: mean bias and 95% limits of agreement (mean +/- 1.96*SD).

    A good rPPG method should:
      - have bias near zero
      - have narrow limits of agreement (e.g. +/- 5 BPM is excellent;
        +/- 10 BPM is acceptable for rPPG)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mean = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, s=14, alpha=0.45, edgecolor="none")
    ax.axhline(bias, color="navy", lw=1.5, label=f"Bias = {bias:+.2f} BPM")
    ax.axhline(upper, color="crimson", lw=1.0, ls="--",
               label=f"+1.96 SD = {upper:+.2f}")
    ax.axhline(lower, color="crimson", lw=1.0, ls="--",
               label=f"-1.96 SD = {lower:+.2f}")
    ax.set_xlabel("Mean of Predicted and Reference HR (BPM)")
    ax.set_ylabel("Predicted - Reference (BPM)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        # Also save as PDF for vector-quality figures (reviewer point 9)
        if savepath.lower().endswith(".png"):
            fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight")
    return fig, ax, dict(bias=bias, sd=sd, upper=upper, lower=lower)


# --------------------------------------------------------------------------
# 4. Stratified errors - reviewer points 6 and 8
# --------------------------------------------------------------------------

def stratified_errors(y_true, y_pred, strata, bins=None, labels=None):
    """
    Compute MAE / RMSE within each stratum.

    Args:
        y_true, y_pred : arrays of HR in BPM
        strata : array of the same length, the variable to bin on (e.g. true HR
                 in BPM, or per-window motion magnitude).
        bins, labels : passed to pd.cut. If None, uses 4 quantile bins.

    Returns:
        DataFrame indexed by bin, with columns [mae, rmse, n].
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true).flatten(),
        "y_pred": np.asarray(y_pred).flatten(),
        "strat":  np.asarray(strata).flatten(),
    })
    if bins is None:
        # 4 quantile bins by default
        df["bin"] = pd.qcut(df["strat"], q=4, duplicates="drop")
    else:
        df["bin"] = pd.cut(df["strat"], bins=bins, labels=labels)

    out = []
    for b, sub in df.groupby("bin"):
        if len(sub) == 0:
            continue
        err = sub["y_pred"].values - sub["y_true"].values
        out.append({
            "bin":  str(b),
            "n":    len(sub),
            "mae":  float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        })
    return pd.DataFrame(out)


def stratified_error_plot(y_true, y_pred, strata, strata_name="Stratum",
                          savepath=None, dpi=300):
    """
    Bar chart of MAE / RMSE per stratum (e.g. per HR-range bin).
    """
    s = stratified_errors(y_true, y_pred, strata)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(s))
    w = 0.4
    ax.bar(x - w/2, s["mae"],  w, label="MAE",  color="steelblue")
    ax.bar(x + w/2, s["rmse"], w, label="RMSE", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(s["bin"], rotation=20)
    ax.set_xlabel(strata_name)
    ax.set_ylabel("Error (BPM)")
    ax.set_title(f"Error stratified by {strata_name}")
    for xi, n in zip(x, s["n"]):
        ax.text(xi, 0.5, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        if savepath.lower().endswith(".png"):
            fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight")
    return fig, ax, s


def estimate_motion(face_boxes):
    """
    Per-frame motion proxy: Euclidean displacement of the face bounding-box
    center between consecutive frames, averaged into a single value per video.

    Args:
        face_boxes : (T, 4) array of (x, y, w, h) per frame.

    Returns:
        scalar mean motion in pixels.
    """
    fb = np.asarray(face_boxes, dtype=float)
    if len(fb) < 2:
        return 0.0
    cx = fb[:, 0] + fb[:, 2] / 2
    cy = fb[:, 1] + fb[:, 3] / 2
    d = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    return float(d.mean())


def estimate_skin_tone_luma(rgb_means):
    """
    Mean luma of the face ROI as a (very) rough skin-tone proxy.
    Use as a crude bin (light / medium / dark) when ITA isn't available,
    and acknowledge in the paper that this is a proxy.

    Args:
        rgb_means : (T, 3) mean RGB per frame.

    Returns:
        scalar mean luma (0..255).
    """
    rgb_means = np.asarray(rgb_means)
    luma = 0.299 * rgb_means[:, 0] + 0.587 * rgb_means[:, 1] + 0.114 * rgb_means[:, 2]
    return float(np.nanmean(luma))


# --------------------------------------------------------------------------
# 5. Pipeline profiling - reviewer point 7
# --------------------------------------------------------------------------

def time_block(fn, *args, n_runs=10, warmup=2, **kwargs):
    """
    Time a callable. Discard `warmup` runs to avoid Python/CUDA warm-up cost.

    Returns:
        mean_ms, std_ms, results_of_last_run
    """
    last = None
    for _ in range(warmup):
        last = fn(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times)), last


def profile_pipeline(stages, n_runs=10, warmup=2, fps_basis_seconds=None):
    """
    Profile a list of pipeline stages and report ms / call and equivalent fps.

    Args:
        stages : list of (name, callable, args, kwargs) tuples.
                 Each callable should run ONE inference unit, e.g.:
                   - face detection on one frame
                   - ViT forward on one frame
                   - LSTM forward on one window
                   - end-to-end on one second of video
        n_runs, warmup : see time_block.
        fps_basis_seconds : if a stage represents 1 second of video, set this
                            to 1.0 so fps = 1000 / mean_ms is reported.

    Returns:
        DataFrame with columns: stage, mean_ms, std_ms, fps_equiv.
    """
    rows = []
    for name, fn, args, kwargs in stages:
        m, s, _ = time_block(fn, *args, n_runs=n_runs, warmup=warmup, **kwargs)
        fps = 1000.0 / m if m > 0 else float("nan")
        rows.append({"stage": name, "mean_ms": m, "std_ms": s,
                     "fps_equiv": fps})
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.uniform(60, 110, n)
    # Predictions: small noise + slight bias
    y_pred = y_true + rng.normal(loc=1.0, scale=3.0, size=n)
    print("compute_all_metrics:", compute_all_metrics(y_true, y_pred))
    print("\nLOSO splits over 5 fake subjects:")
    for tr, te in loso_split([f"s{i}" for i in range(5)]):
        print("  train:", tr, "test:", te)
    print("\nstratified_errors by HR bin:")
    print(stratified_errors(y_true, y_pred, y_true))
