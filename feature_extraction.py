"""
feature_extraction.py — MCU-friendly feature extraction + calibration-based
                         standing gate.

All features use simple arithmetic (mean, std, sum-of-squares, diff).
No FFT, no matrix decomposition — suitable for embedded C implementation.

Input:  X  shape (N, T, 9)  where channels = [FSR×6, ACC×3]
Output: Xf shape (N, 38)

─────────────────────────────────────────────────────────────────────────────
Calibration-based standing gate (runs BEFORE the RF gate)
─────────────────────────────────────────────────────────────────────────────
At device startup the user stands still for 5 s (= CALIB_N_SAMPLES rows at
5 Hz).  calibrate_standing() extracts overlapping windows and stores a
(N_windows, 5) feature reference matrix — one row per window.

Calibration features (7):
    acc_x_mean, acc_y_mean, acc_z_mean, acc_mag_std, acc_diff_energy,
    fsr_lr_asymmetry, fsr_diff_energy

During live inference, for every window w:

    feat(w)  = _calib_window_features(w)             # (7,)
    z(w)     = (feat(w) - calib_mean) / calib_std    # z-scored
    dist     = L1( z(w), centroid )                  # distance to cluster centre

Rejection rule:
    if dist < ref_dist → reject as standing / IGNORE  (skip RF gate)
    else               → pass to RF gate

ref_dist is the maximum pairwise L1 distance between calibration windows in
z-scored space — computed automatically by calibrate_standing(), no manual
threshold tuning needed.

MCU cost: 7 feature computes per inference window + 1 L1 distance to centroid.
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
import config as cfg
from config import FSR_THRESHOLD


# ── Feature names (30) ────────────────────────────────────────────────────────

def feature_names() -> list:
    names = ["fsr_total_mean", "fsr_total_std"]
    names += [f"fsr_ch{i}_mean" for i in range(6)]
    names += [f"fsr_ch{i}_std"  for i in range(6)]
    names += [
        "fsr_energy",
        "fsr_active_mean", "fsr_active_max",
        "fsr_lr_asymmetry",
        "fsr_diff_energy",
        # ── NEW: temporal pattern features (gate: twisting vs walking/standing) ──
        "fsr_total_slope",      # linear trend of total FSR — positive=load increasing
        "fsr_zcr",              # zero-crossing rate of diff(fsr_total) — walking=high
        "fsr_asym_std",         # std of instantaneous L-R asymmetry — twisting=dynamic
    ]
    for ax in ["x", "y", "z"]:
        names += [f"acc_{ax}_mean", f"acc_{ax}_std"]
    names += ["acc_mag_mean", "acc_mag_std", "acc_mag_range",
              "acc_diff_energy", "acc_tilt_deg",
              # ── NEW: temporal pattern features (gate + posture) ──────────────
              "acc_x_slope",      # directional trend x-axis (twisting has slope)
              "acc_y_slope",      # directional trend y-axis
              "acc_z_slope",      # directional trend z-axis
              "acc_zcr",          # zero-crossing rate of diff(acc_mag) — walking=high
              "acc_autocorr",     # autocorr of acc_mag at lag T//2 — walking=positive
              ]
    return names


FEAT_NAMES  = feature_names()
N_FEATURES  = len(FEAT_NAMES)          # 38

# Indices of the 11 ACC-only features (used by posture classifier)
ACC_FEAT_IDX = [i for i, n in enumerate(FEAT_NAMES) if n.startswith("acc_")]
FSR_FEAT_IDX = [i for i, n in enumerate(FEAT_NAMES) if n.startswith("fsr_")]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    X : (N, T, 9)  — 9 channels: 6 FSR + 3 ACC
    Returns feature matrix (N, 38).

    Feature groups:
      FSR basic        [0-13]  : per-channel mean/std, total mean/std
      FSR energy       [14-18] : energy, active count, lr_asymmetry, diff_energy
      FSR temporal     [19-21] : slope, ZCR, asymmetry std  (NEW)
      ACC per-axis     [22-27] : mean/std for x/y/z
      ACC magnitude    [28-32] : mag mean/std/range, diff_energy, tilt_deg
      ACC temporal     [33-37] : slope x/y/z, ZCR, autocorr  (NEW)

    ACC_FEAT_IDX = [22..37] (16 features, used by posture classifier)
    FSR_FEAT_IDX = [0..21]  (22 features)
    """
    N, T, _ = X.shape
    fsr = X[:, :, :6].astype(np.float32)   # (N, T, 6)
    acc = X[:, :, 6:].astype(np.float32)   # (N, T, 3)

    feats = []

    # ── FSR basic ──────────────────────────────────────────────────────────────
    fsr_total = fsr.sum(axis=2)                          # (N, T)
    fsr_total_mean = fsr_total.mean(axis=1)              # (N,)
    feats += [fsr_total_mean, fsr_total.std(axis=1)]

    for i in range(6):
        feats.append(fsr[:, :, i].mean(axis=1))          # fsr_ch{i}_mean

    for i in range(6):
        feats.append(fsr[:, :, i].std(axis=1))           # fsr_ch{i}_std

    # ── FSR energy ─────────────────────────────────────────────────────────────
    feats.append((fsr ** 2).sum(axis=(1, 2)) / T)        # fsr_energy

    # ── FSR active channel count ───────────────────────────────────────────────
    active = (fsr > FSR_THRESHOLD).sum(axis=2).astype(np.float32)  # (N, T)
    feats += [active.mean(axis=1), active.max(axis=1).astype(np.float32)]

    # ── FSR left-right asymmetry ───────────────────────────────────────────────
    left  = fsr[:, :, :3].sum(axis=2).mean(axis=1)
    right = fsr[:, :, 3:].sum(axis=2).mean(axis=1)
    feats.append((left - right) / (left + right + 1e-6))

    # ── FSR temporal diff energy ───────────────────────────────────────────────
    feats.append((np.diff(fsr, axis=1) ** 2).mean(axis=(1, 2)))

    # ── FSR temporal pattern features (NEW) ───────────────────────────────────
    # Linear slope of total FSR using centred time vector (MCU: dot product only)
    t_vec = np.arange(T, dtype=np.float32) - (T - 1) / 2.0   # centred, (T,)
    t_var = float((t_vec ** 2).sum())                          # scalar
    fsr_slope = (fsr_total * t_vec).sum(axis=1) / t_var        # (N,)
    feats.append(fsr_slope)

    # Zero-crossing rate of diff(fsr_total): how often the total pressure
    # changes direction → walking=high (rhythmic), lifting/twisting=low
    fsr_d     = np.diff(fsr_total, axis=1)                     # (N, T-1)
    fsr_signs = np.sign(fsr_d)
    fsr_zcr   = (fsr_signs[:, 1:] != fsr_signs[:, :-1]).mean(axis=1).astype(np.float32)
    feats.append(fsr_zcr)

    # Std of instantaneous L-R asymmetry over time:
    # mean asymmetry already captured; this captures whether asymmetry is
    # dynamic (twisting = varies) or static (standing/walking = stable)
    left_t  = fsr[:, :, :3].sum(axis=2)                        # (N, T)
    right_t = fsr[:, :, 3:].sum(axis=2)                        # (N, T)
    asym_t  = (left_t - right_t) / (left_t + right_t + 1e-6)  # (N, T)
    feats.append(asym_t.std(axis=1))

    # ── ACC per-axis mean + std ────────────────────────────────────────────────
    acc_mean_vec = acc.mean(axis=1)                      # (N, 3)
    for i in range(3):
        feats += [acc_mean_vec[:, i], acc[:, :, i].std(axis=1)]

    # acc_unit needed for acc_tilt_deg only — not a feature itself
    acc_norm = np.linalg.norm(acc_mean_vec, axis=1, keepdims=True) + 1e-6
    acc_unit = acc_mean_vec / acc_norm                   # (N, 3)

    # ── ACC magnitude ──────────────────────────────────────────────────────────
    acc_mag = np.sqrt((acc ** 2).sum(axis=2))            # (N, T)
    feats += [
        acc_mag.mean(axis=1),
        acc_mag.std(axis=1),
        acc_mag.max(axis=1) - acc_mag.min(axis=1),
    ]

    # ── ACC diff energy ────────────────────────────────────────────────────────
    feats.append((np.diff(acc, axis=1) ** 2).mean(axis=(1, 2)))

    # ── ACC tilt from vertical ─────────────────────────────────────────────────
    cos_theta = np.clip(-acc_unit[:, 2], -1.0, 1.0)
    feats.append(np.degrees(np.arccos(cos_theta)))

    # ── ACC temporal pattern features (NEW) ──────────────────────────────────
    # t_vec already defined above in FSR section; reuse it

    # Linear slope per ACC axis: captures directional change over the window.
    # Twisting shows slope in lateral axes; standing is flat; walking has
    # small alternating slopes.
    for i in range(3):
        feats.append((acc[:, :, i] * t_vec).sum(axis=1) / t_var)   # (N,)

    # Zero-crossing rate of diff(acc_mag): walking has regular oscillations
    # (many ZC), twisting has a short burst (few ZC), standing is near zero (few ZC).
    acc_mag_d  = np.diff(acc_mag, axis=1)                           # (N, T-1)
    acc_signs  = np.sign(acc_mag_d)
    acc_zcr    = (acc_signs[:, 1:] != acc_signs[:, :-1]).mean(axis=1).astype(np.float32)
    feats.append(acc_zcr)

    # Normalised autocorrelation of acc_mag at lag = T//2:
    # walking (periodic) → positive value; twisting/lifting (aperiodic) → near 0.
    # MCU cost: two dot products + one divide — trivial.
    lag       = T // 2
    am_dm     = acc_mag - acc_mag.mean(axis=1, keepdims=True)       # de-mean (N, T)
    num       = (am_dm[:, :T - lag] * am_dm[:, lag:]).sum(axis=1)   # (N,)
    den       = (am_dm ** 2).sum(axis=1) + 1e-6                     # (N,)
    feats.append(num / den)

    return np.stack(feats, axis=1).astype(np.float32)    # (N, 38)


# ── Calibration-based standing gate ──────────────────────────────────────────

# 7-feature calibration vector:
#   ACC orientation (3) + ACC motion quality (2) + FSR motion quality (2)
#
#   fsr_diff_energy  — notebook M1 shows 3× ratio standing vs twisting (best FSR feature)
#   fsr_lr_asymmetry — weak alone (~1× ratio) but adds orthogonal information to ACC
#
CALIB_FEAT_NAMES = [
    "acc_x_mean",       # orientation
    "acc_y_mean",       # orientation
    "acc_z_mean",       # orientation
    "acc_mag_std",      # motion magnitude variability
    "acc_diff_energy",  # ACC frame-to-frame energy
    "fsr_lr_asymmetry", # left-right weight distribution
    "fsr_diff_energy",  # FSR frame-to-frame energy (strongest FSR separator)
]
N_CALIB_FEATS = len(CALIB_FEAT_NAMES)   # 7


def _calib_window_features(window: np.ndarray) -> np.ndarray:
    """
    Compute the 7 calibration features from one (T, 9) raw window.

    Features (in order):
        acc_x_mean, acc_y_mean, acc_z_mean,
        acc_mag_std, acc_diff_energy,
        fsr_lr_asymmetry, fsr_diff_energy

    Parameters
    ----------
    window : array, shape (T, 9)
        One raw sensor window.  Channels: [FSR×6, ACC×3].

    Returns
    -------
    feats : ndarray, shape (7,)  — raw (un-normalised) values
    """
    w   = np.asarray(window, dtype=np.float32)   # (T, 9)
    fsr = w[:, :6]                               # (T, 6)
    acc = w[:, 6:]                               # (T, 3)

    # ── ACC features ──────────────────────────────────────────────────────────
    acc_mean = acc.mean(axis=0)                  # (3,)
    acc_mag  = np.sqrt((acc ** 2).sum(axis=1))   # (T,)
    acc_diff = np.diff(acc, axis=0)              # (T-1, 3)
    acc_diff_energy = (acc_diff ** 2).mean()

    # ── FSR features ──────────────────────────────────────────────────────────
    left        = fsr[:, :3].sum(axis=1).mean()           # scalar — mean left load
    right       = fsr[:, 3:].sum(axis=1).mean()           # scalar — mean right load
    fsr_lr_asym = (left - right) / (left + right + 1e-6)  # (-1, 1)
    fsr_diff    = np.diff(fsr, axis=0)                    # (T-1, 6)
    fsr_diff_energy = (fsr_diff ** 2).mean()

    return np.array(
        [
            acc_mean[0], acc_mean[1], acc_mean[2],
            acc_mag.std(), acc_diff_energy,
            fsr_lr_asym, fsr_diff_energy,
        ],
        dtype=np.float32,
    )                                            # (7,)


def calibrate_standing(calib_data: np.ndarray) -> dict:
    """
    Build a personalised standing reference from 5 s of standing-still data.

    Extracts overlapping windows from the calibration recording, computes the
    7-feature vector for each window, then stores z-score normalisation
    statistics so that is_standing() can compare inference windows in a
    scale-invariant way regardless of sensor gain or user body weight.

    Parameters
    ----------
    calib_data : array, shape (N_samples, 9) or (K, T, 9)
        Raw sensor data recorded while the user stands still.
        If 3-D (windowed), the array is flattened to (K*T, 9) first.
        Recommended: N_samples = CALIB_N_SAMPLES (25 rows = 5 s at 5 Hz).

    Returns
    -------
    calib_info : dict with keys:
        "norm_feats"  — ndarray (N_windows, 7) z-score normalised reference
        "feat_mean"   — ndarray (7,)  per-feature mean of calibration windows
        "feat_std"    — ndarray (7,)  per-feature std  of calibration windows
        "n_windows"   — int  number of reference windows extracted

    Example
    -------
    >>> calib_info = calibrate_standing(calib_buf)   # once at startup
    >>> reject     = is_standing(X_batch, calib_info) # every inference step
    """
    data = np.asarray(calib_data, dtype=np.float32)
    if data.ndim == 3:                               # (K, T, 9) → (K*T, 9)
        K, T, C = data.shape
        data = data.reshape(K * T, C)

    W, S   = cfg.WINDOW_SIZE, cfg.STRIDE
    feats  = []
    for start in range(0, len(data) - W + 1, S):
        feats.append(_calib_window_features(data[start : start + W]))
    if not feats:                                    # fallback: too few samples
        feats.append(_calib_window_features(data[:W]))

    raw = np.stack(feats, axis=0)                    # (N_windows, 7)

    # z-score normalisation derived from the calibration windows themselves.
    #
    # IMPORTANT: floor feat_std at a minimum so that a very-still calibration
    # does not amplify tiny sensor noise into huge z-scores.
    # Without the floor, std ≈ 0.001 turns a real variation of 0.009 into
    # z-score = 9.0, which drives spread to ~9 and ref_dist to ~32 — far too
    # large, catching twisting as standing.
    # Minimum per feature group:
    #   ACC features (indices 0-4): 0.05 m/s² — typical sensor noise floor
    #   FSR features (indices 5-6): 5.0 ADC   — typical FSR noise floor
    _std_raw = raw.std(axis=0)                           # (7,)
    _std_min = np.array([0.05, 0.05, 0.05, 0.05, 0.05,  # acc_x/y/z, acc_mag_std, acc_diff
                         5.0,  5.0],                     # fsr_lr_asym, fsr_diff_energy
                        dtype=np.float32)
    feat_std  = np.maximum(_std_raw, _std_min)           # (7,)  floored
    feat_mean = raw.mean(axis=0)                         # (7,)
    norm      = (raw - feat_mean) / feat_std             # (N_windows, 7)

    # centroid of calibration windows in z-scored space (≈ zero vector)
    centroid = norm.mean(axis=0)                         # (7,)

    # ── Self-calibrated standing threshold ───────────────────────────────────
    # Use 90th-percentile distance (not max) so a single noisy outlier window
    # during calibration does not inflate the spread.
    SWAY_MARGIN  = 3.5   # multiplier — overridden at runtime by sidebar slider
    MAX_REF_DIST = 20.0  # hard ceiling — prevents pathological large thresholds
    if len(norm) > 1:
        d2c    = np.sum(np.abs(norm - centroid), axis=1)   # (K,)
        spread = float(np.percentile(d2c, 90))             # 90th pct, not max
    else:
        spread = 0.0
    ref_dist = float(np.clip(
        max(spread * SWAY_MARGIN, cfg.T_STAND_DIST),
        0.0, MAX_REF_DIST
    ))

    return {
        "norm_feats": norm,
        "feat_mean":  feat_mean,
        "feat_std":   feat_std,
        "n_windows":  len(raw),
        "centroid":   centroid,
        "ref_dist":   ref_dist,
        "spread":     spread,        # raw calibration spread (for debug/logging)
    }


def is_standing(
    X: np.ndarray,
    calib_info: dict,
) -> np.ndarray:
    """
    Apply the calibration-based standing gate to a batch of windows.

    Call this BEFORE extract_features / the RF gate.  Windows flagged as
    standing are output as IGNORE without invoking the RF models.

    Parameters
    ----------
    X : array, shape (N, T, 9)
        Batch of inference windows.
    calib_info : dict
        Returned by calibrate_standing() — contains "feat_mean", "feat_std",
        "centroid", and "ref_dist".

    Returns
    -------
    reject : ndarray, shape (N,), dtype bool
        True  → window is standing / at rest → output IGNORE.
        False → pass window to the RF gate.

    How it works
    ------------
    For each inference window w:

        feat(w)  = _calib_window_features(w)            # (7,)  raw
        z(w)     = (feat(w) - calib_mean) / calib_std   # (7,)  z-scored
        dist     = L1( z(w), centroid )                 # distance to cluster centre

        reject   = (dist < ref_dist)

    ref_dist is computed in calibrate_standing() as:
        ref_dist = max(spread × SWAY_MARGIN, T_STAND_DIST)
    where spread is the maximum L1 distance from the calibration centroid in
    z-scored space, SWAY_MARGIN=3 absorbs natural body sway, and T_STAND_DIST
    (from config.py, default 7.5) acts as a safety floor so a perfectly-still
    calibration still gives a sane threshold.
    """
    X = np.asarray(X, dtype=np.float32)             # (N, T, 9)

    feat_mean = np.asarray(calib_info["feat_mean"], dtype=np.float32)  # (7,)
    feat_std  = np.asarray(calib_info["feat_std"],  dtype=np.float32)  # (7,)
    centroid  = np.asarray(calib_info["centroid"],  dtype=np.float32)  # (7,)
    ref_dist  = float(calib_info["ref_dist"])

    # z-score inference windows using calibration statistics
    raw_feats = np.stack(
        [_calib_window_features(X[i]) for i in range(len(X))], axis=0
    )                                                # (N, 7)
    z_feats   = (raw_feats - feat_mean) / feat_std   # (N, 7)

    # L1 distance from each inference window to the calibration centroid
    dists = np.sum(np.abs(z_feats - centroid), axis=1)  # (N,)

    return dists < ref_dist
