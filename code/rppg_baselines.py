"""
rppg_baselines.py
=================

Classical rPPG baselines for comparison in your revision.

Implements:
    - GREEN  (mean green channel; the simplest baseline)
    - CHROM  (de Haan & Jeanne, 2013)
    - POS    (Wang et al., 2017) - the field standard
    - ICA    (Poh, McDuff, Picard, 2010)

Each method takes a (T, 3) array of mean RGB values per frame from the face ROI
and the sampling rate `fs`, and returns:
    - the BVP (blood-volume pulse) signal in (T,)
    - an estimated heart rate in BPM (single value over the window, or a
      time-varying series if you call it on overlapping windows)

Reviewer point 4: "Add comparisons against POS, CHROM, ICA/PCA rPPG..."
Reviewer point 3: "Color vs grayscale: Justify or replace with color-aware
                   rPPG features (e.g., green channel emphasis, CHROM/POS)."

Usage example:
    from rppg_baselines import extract_rgb_from_video, pos_method, hr_from_bvp

    rgb, fs = extract_rgb_from_video("subject1/vid.avi")  # (T, 3) float
    bvp = pos_method(rgb, fs)                              # (T,)
    hr_bpm = hr_from_bvp(bvp, fs)                          # scalar
"""

import numpy as np
import cv2
import scipy.signal as signal
from sklearn.decomposition import FastICA, PCA


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def bandpass_filter(x, fs, f_lo=0.7, f_hi=4.0, order=4):
    """
    Zero-phase Butterworth bandpass for the cardiac band.

    Cardiac band 0.7-4.0 Hz covers 42-240 BPM. This is the standard rPPG
    band; do NOT use the 50-60 BPM band that was in the original code.
    """
    nyq = 0.5 * fs
    low, high = f_lo / nyq, f_hi / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    # filtfilt = forward-backward = zero phase delay (important for HR)
    return signal.filtfilt(b, a, x, axis=0)


def hr_from_bvp(bvp, fs, f_lo=0.7, f_hi=3.0, nfft=8192):
    """
    Estimate HR (BPM) from a BVP signal via the dominant frequency of the
    Welch power spectrum, restricted to the cardiac band.

    Two corrections over the naive version:

    1. ZERO-PADDED FFT. A 6 s Welch window at ~30 fps gives ~0.16 Hz
       resolution = ~10 BPM, so HR estimates come out as a 10 BPM staircase.
       Padding the FFT to nfft=8192 interpolates the spectrum and brings
       resolution to ~0.2 BPM without needing more data.

    2. SUB-HARMONIC GUARD. A PPG/BVP waveform is not sinusoidal, so it has a
       strong second harmonic at 2x the true HR. With f_hi=4.0 Hz (240 BPM)
       that harmonic sits inside the search band and sometimes wins, reporting
       double the true rate. We narrow the band to 3.0 Hz (180 BPM) and, if
       half the detected peak frequency also carries substantial power, take
       the lower (fundamental) frequency instead.
    """
    bvp = np.asarray(bvp).flatten()
    if len(bvp) < int(2 * fs):
        return float("nan")
    nperseg = min(len(bvp), int(10 * fs))
    f, p = signal.welch(bvp, fs=fs, nperseg=nperseg, nfft=max(nfft, nperseg))
    band = (f >= f_lo) & (f <= f_hi)
    if not band.any():
        return float("nan")
    f_band, p_band = f[band], p[band]
    f_peak = f_band[np.argmax(p_band)]
    f_half = f_peak / 2.0
    if f_half >= f_lo:
        i = np.argmin(np.abs(f_band - f_half))
        if p_band[i] > 0.25 * p_band.max():
            f_peak = f_band[i]
    return float(f_peak * 60.0)



def signal_snr_db(bvp, fs, hr_bpm, half_width_bpm=6.0):
    """
    Frequency-domain SNR of a BVP, in dB.

    Defined (de Haan & Jeanne, 2013) as the ratio of energy in a narrow band
    around the HR (and its first harmonic) to the energy outside those bands,
    inside the cardiac range.

    A higher SNR (in dB) means a cleaner pulse. Used in reviewer point 6.
    """
    bvp = np.asarray(bvp).flatten()
    nperseg = min(len(bvp), int(6 * fs))
    f, p = signal.welch(bvp, fs=fs, nperseg=nperseg)
    # Convert HR to Hz; build masks for fundamental and first harmonic
    f0 = hr_bpm / 60.0
    half_w = half_width_bpm / 60.0
    band_cardiac = (f >= 0.7) & (f <= 4.0)
    near_f0 = (np.abs(f - f0) <= half_w) & band_cardiac
    near_2f0 = (np.abs(f - 2 * f0) <= half_w) & band_cardiac
    signal_mask = near_f0 | near_2f0
    noise_mask = band_cardiac & ~signal_mask
    if not signal_mask.any() or not noise_mask.any():
        return float("nan")
    signal_power = p[signal_mask].sum()
    noise_power = p[noise_mask].sum()
    if noise_power <= 0:
        return float("nan")
    return float(10.0 * np.log10(signal_power / noise_power))


# --------------------------------------------------------------------------
# Face ROI -> mean RGB time series
# --------------------------------------------------------------------------

def _get_skin_mask(bgr_face):
    """
    Simple YCrCb-range skin mask. Used to drop background pixels inside the
    face bounding box (eyes, hair, etc. are partially excluded).
    """
    ycrcb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return mask > 0


def extract_rgb_from_video(video_path, use_skin_mask=True, face_cascade=None):
    """
    Read a video, detect a face per frame, and return a (T, 3) array of mean
    R, G, B values from the face skin region, plus the video's fps.

    This is the input for ALL rPPG methods (CHROM, POS, ICA, GREEN).
    """
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    cap = cv2.VideoCapture(video_path)
    fs = cap.get(cv2.CAP_PROP_FPS)
    rgbs = []
    last_box = None  # reuse previous box if detection fails on a frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            last_box = max(faces, key=lambda b: b[2] * b[3])  # biggest face
        if last_box is None:
            # No face yet; skip this frame (push a NaN row to keep timing aligned)
            rgbs.append([np.nan, np.nan, np.nan])
            continue
        x, y, w, h = last_box
        face_bgr = frame[y:y + h, x:x + w]
        if use_skin_mask:
            mask = _get_skin_mask(face_bgr)
            if mask.sum() < 50:  # too few skin pixels; fall back to whole box
                mask = np.ones(face_bgr.shape[:2], dtype=bool)
        else:
            mask = np.ones(face_bgr.shape[:2], dtype=bool)
        # cv2 is BGR; reorder to RGB for clarity
        b_mean = face_bgr[..., 0][mask].mean()
        g_mean = face_bgr[..., 1][mask].mean()
        r_mean = face_bgr[..., 2][mask].mean()
        rgbs.append([r_mean, g_mean, b_mean])
    cap.release()

    rgb = np.asarray(rgbs, dtype=np.float64)
    # If any leading frames had no face, drop them
    valid = ~np.isnan(rgb).any(axis=1)
    rgb = rgb[valid]
    return rgb, fs


# --------------------------------------------------------------------------
# Baseline methods. Each returns a 1-D BVP signal of length T.
# --------------------------------------------------------------------------

def green_method(rgb, fs):
    """
    GREEN: bandpass-filtered mean green channel.

    The simplest possible rPPG baseline. Hemoglobin absorbs green light
    most strongly, so the green channel often carries the strongest pulse.
    """
    g = rgb[:, 1]
    g = (g - g.mean()) / (g.std() + 1e-9)
    return bandpass_filter(g, fs)


def chrom_method(rgb, fs, win_seconds=1.6):
    """
    CHROM (de Haan & Jeanne, IEEE TBME 2013).

    Defines two chrominance signals
        Xs = 3*Rn - 2*Gn
        Ys = 1.5*Rn + Gn - 1.5*Bn
    where Rn,Gn,Bn are temporally normalized RGB. The pulse is recovered as
        S = Xf - alpha * Yf,    alpha = std(Xf)/std(Yf)
    inside short overlapping windows. Robust to specular reflection.
    """
    T = rgb.shape[0]
    win = int(win_seconds * fs)
    if win < 8:
        win = 8
    H = np.zeros(T)
    overlap = win // 2
    step = win - overlap
    # Hann window for smoothing
    hann = np.hanning(win)
    for start in range(0, T - win + 1, step):
        end = start + win
        block = rgb[start:end]
        mu = block.mean(axis=0) + 1e-9
        norm = block / mu                         # temporal normalization
        Xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        Ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        Xf = bandpass_filter(Xs, fs)
        Yf = bandpass_filter(Ys, fs)
        alpha = (Xf.std() / (Yf.std() + 1e-9))
        S = Xf - alpha * Yf
        S = S * hann                               # taper edges
        H[start:end] += S                          # overlap-add
    return H


def pos_method(rgb, fs, win_seconds=1.6):
    """
    POS - Plane Orthogonal to Skin (Wang et al., IEEE TBME 2017).

    The current de-facto rPPG baseline. Projects the temporally normalized
    RGB onto the plane orthogonal to the skin-tone direction.

    Algorithm (from the paper):
        for each overlapping window of length L = 1.6*fs:
            Cn = C / mean(C)                    # temporal normalization
            S  = [[ 0, 1, -1], [-2, 1,  1]] @ Cn^T
            h  = S[0] + (std(S[0])/std(S[1])) * S[1]
            h  = h - mean(h)
            H[t..t+L-1] += h                    # overlap-add
    """
    T = rgb.shape[0]
    L = int(win_seconds * fs)
    if L < 8:
        L = 8
    H = np.zeros(T)
    proj = np.array([[0.0, 1.0, -1.0],
                     [-2.0, 1.0, 1.0]])
    for n in range(0, T - L + 1):
        block = rgb[n:n + L]                    # (L, 3)
        mu = block.mean(axis=0) + 1e-9
        Cn = block / mu                         # (L, 3)
        S = proj @ Cn.T                         # (2, L)
        std0 = S[0].std() + 1e-9
        std1 = S[1].std() + 1e-9
        h = S[0] + (std0 / std1) * S[1]
        h = h - h.mean()
        H[n:n + L] += h
    # Optional final bandpass to suppress residual drift
    return bandpass_filter(H, fs)


def ica_method(rgb, fs, n_components=3):
    """
    ICA-based rPPG (Poh, McDuff, Picard, Optics Express 2010).

    Detrend RGB, run FastICA to get 3 sources, pick the source with the
    highest spectral peak in the cardiac band as the pulse.
    """
    # Temporal normalization (zero-mean, unit-variance per channel)
    rgb_norm = (rgb - rgb.mean(axis=0)) / (rgb.std(axis=0) + 1e-9)
    try:
        ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
        sources = ica.fit_transform(rgb_norm)   # (T, 3)
    except Exception:
        # FastICA can fail to converge on very short signals; fall back to PCA.
        pca = PCA(n_components=n_components)
        sources = pca.fit_transform(rgb_norm)

    best_score = -np.inf
    best_idx = 0
    for k in range(sources.shape[1]):
        s = bandpass_filter(sources[:, k], fs)
        nperseg = min(len(s), int(6 * fs))
        f, p = signal.welch(s, fs=fs, nperseg=nperseg)
        band = (f >= 0.7) & (f <= 4.0)
        if not band.any():
            continue
        score = p[band].max()
        if score > best_score:
            best_score = score
            best_idx = k

    chosen = bandpass_filter(sources[:, best_idx], fs)
    return chosen


def pca_method(rgb, fs, n_components=3):
    """
    PCA variant: same idea as ICA but with PCA. Cheaper and more stable but
    usually slightly worse than ICA.
    """
    rgb_norm = (rgb - rgb.mean(axis=0)) / (rgb.std(axis=0) + 1e-9)
    pca = PCA(n_components=n_components)
    sources = pca.fit_transform(rgb_norm)
    # Choose the component with the highest cardiac-band peak
    best_score = -np.inf
    best_idx = 0
    for k in range(sources.shape[1]):
        s = bandpass_filter(sources[:, k], fs)
        nperseg = min(len(s), int(6 * fs))
        f, p = signal.welch(s, fs=fs, nperseg=nperseg)
        band = (f >= 0.7) & (f <= 4.0)
        if not band.any():
            continue
        score = p[band].max()
        if score > best_score:
            best_score = score
            best_idx = k
    return bandpass_filter(sources[:, best_idx], fs)


# --------------------------------------------------------------------------
# Sliding-window HR estimator
# --------------------------------------------------------------------------

def windowed_hr(bvp, fs, win_seconds=10.0, step_seconds=1.0):
    """
    Compute a time-varying HR from a long BVP by sliding a Welch-PSD window.

    Returns:
        timestamps : (K,) center-of-window time in seconds
        hr_bpm     : (K,) HR estimate per window
        snr_db     : (K,) per-window SNR

    Use this when you want a HR-vs-time series for a video, matching the
    UBFC ground-truth time series.
    """
    bvp = np.asarray(bvp).flatten()
    L = int(win_seconds * fs)
    step = max(1, int(step_seconds * fs))
    times, hrs, snrs = [], [], []
    for start in range(0, len(bvp) - L + 1, step):
        seg = bvp[start:start + L]
        hr = hr_from_bvp(seg, fs)
        snr = signal_snr_db(seg, fs, hr) if not np.isnan(hr) else float("nan")
        times.append((start + L / 2) / fs)
        hrs.append(hr)
        snrs.append(snr)
    return np.asarray(times), np.asarray(hrs), np.asarray(snrs)


# --------------------------------------------------------------------------
# Convenience: run all baselines on one video
# --------------------------------------------------------------------------

def run_all_baselines(video_path, win_seconds=10.0, step_seconds=1.0):
    """
    Convenience wrapper for the experiments table. Returns a dict:
        { 'GREEN':  (times, hr, snr),
          'CHROM':  (times, hr, snr),
          'POS':    (times, hr, snr),
          'ICA':    (times, hr, snr),
          'PCA':    (times, hr, snr),
          'fs':     fs }
    """
    rgb, fs = extract_rgb_from_video(video_path)
    methods = {
        "GREEN": green_method,
        "CHROM": chrom_method,
        "POS":   pos_method,
        "ICA":   ica_method,
        "PCA":   pca_method,
    }
    out = {"fs": fs}
    for name, fn in methods.items():
        bvp = fn(rgb, fs)
        times, hr, snr = windowed_hr(bvp, fs,
                                     win_seconds=win_seconds,
                                     step_seconds=step_seconds)
        out[name] = (times, hr, snr)
    return out
