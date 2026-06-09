"""
data_loader.py — Load labelled CSVs, remap labels, and extract sliding windows.
"""
import os
import numpy as np
import pandas as pd
from config import (
    FSR_COLS, ACC_COLS, ALL_SENSOR, SENSOR_COL_MAP,
    REMAP, VALID_LABELS_RAW, INVALID_LABELS_RAW,
    WINDOW_SIZE, STRIDE,
)


def _is_5hz_device(dev: str, source_tag: str) -> bool:
    """Return False for dynamic device folders that are not 5 Hz.

    Dynamic device folders carry an explicit Hz marker (e.g. BACKY400@5Hz,
    BACKY400@2Hz, DemoSet@2Hz).  Any folder that has a '@' suffix but does
    NOT end with '@5Hz' is a different sample rate and must be excluded.
    Static device folders have no Hz marker and are always accepted.
    """
    if source_tag == "dynamic" and "@" in dev:
        return dev.endswith("@5Hz")
    return True


def load_directory(root: str, source_tag: str) -> pd.DataFrame:
    """Walk *root* (user → device → *.csv), load every valid CSV.

    For dynamic data, only device folders ending in '@5Hz' are loaded.
    Static device folders have no Hz marker and are all loaded.
    Label 4 is remapped to 3 downstream by apply_label_remap (via REMAP in config).
    """
    records = []
    if not os.path.isdir(root):
        print(f"  [WARN] Directory not found: {root}")
        return pd.DataFrame()

    for user in sorted(os.listdir(root)):
        user_path = os.path.join(root, user)
        if not os.path.isdir(user_path):
            continue
        for dev in sorted(os.listdir(user_path)):
            dev_path = os.path.join(user_path, dev)
            if not os.path.isdir(dev_path):
                continue
            if not _is_5hz_device(dev, source_tag):
                print(f"  [SKIP] {source_tag}/{user}/{dev}  (not 5 Hz)")
                continue
            for fname in sorted(os.listdir(dev_path)):
                if not fname.endswith(".csv"):
                    continue
                df = _load_csv(os.path.join(dev_path, fname))
                if df is None:
                    continue
                df["user"]   = user
                df["device"] = dev
                df["source"] = source_tag
                records.append(df)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _load_csv(fpath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(fpath)}: {e}")
        return None
    df = df.rename(columns=SENSOR_COL_MAP)
    needed = ALL_SENSOR + ["label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None
    return df[needed].copy()


def apply_label_remap(raw: pd.DataFrame) -> pd.DataFrame:
    """Add gate_label and posture_label columns; drop unknown labels."""
    raw = raw.copy()
    raw["label"] = raw["label"].astype(int)

    def _remap(x):
        if x in REMAP:
            return REMAP[x]
        if x in INVALID_LABELS_RAW:
            return "invalid"
        return None

    raw["label_remapped"] = raw["label"].apply(_remap)
    raw = raw[raw["label_remapped"].notna()].copy()
    raw["gate_label"]    = (raw["label_remapped"] == "invalid").astype(int)
    raw["posture_label"] = raw["label_remapped"].apply(
        lambda x: int(x) if isinstance(x, int) else np.nan
    )
    return raw


def extract_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Slide a window over contiguous same-label blocks.
    Returns:
        X     — (N, window_size, 9) float32
        meta  — (N,) DataFrame with gate_label, posture_label, user, device, etc.
    """
    X_list, meta_list = [], []
    df = df.reset_index(drop=True)

    # Use raw label (not label_remapped) so walking/sitting/standing stay as
    # separate blocks even when consecutive — otherwise all three collapse to
    # "invalid" and orig_label only reflects the first activity in the run.
    group_key = df[["user", "device", "label"]].astype(str).agg("|".join, axis=1)
    block_id = (group_key != group_key.shift()).cumsum()

    for _, grp in df.groupby(block_id, sort=False):
        arr = grp[ALL_SENSOR].values.astype(np.float32)
        T = len(arr)
        if T < window_size:
            continue
        m = {
            "user":           grp["user"].iloc[0],
            "device":         grp["device"].iloc[0],
            "source":         grp["source"].iloc[0],
            "label_remapped": grp["label_remapped"].iloc[0],
            "gate_label":     int(grp["gate_label"].iloc[0]),
            "posture_label":  grp["posture_label"].iloc[0],
            "orig_label":     int(grp["label"].iloc[0]),
        }
        for start in range(0, T - window_size + 1, stride):
            X_list.append(arr[start : start + window_size])
            meta_list.append(m)

    if not X_list:
        return np.empty((0, window_size, 9), dtype=np.float32), pd.DataFrame()

    return np.stack(X_list), pd.DataFrame(meta_list).reset_index(drop=True)


def load_unlabelled_file(
    fpath: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a raw (unlabelled) CSV, apply column aliases, drop zero-accel rows,
    then extract sequential sliding windows.

    Returns:
        X            — (N, window_size, 9) float32
        win_centers  — (N,) sample index of each window centre
    """
    df = pd.read_csv(fpath).rename(columns=SENSOR_COL_MAP)
    missing = [c for c in ALL_SENSOR if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {fpath}: {missing}")

    df = df[ALL_SENSOR].copy().astype(np.float32)
    df = df[df["az"] != 0].reset_index(drop=True)   # drop startup rows

    arr = df.values
    T = len(arr)
    X_list, centers = [], []
    for start in range(0, T - window_size + 1, stride):
        X_list.append(arr[start : start + window_size])
        centers.append(start + window_size // 2)

    if not X_list:
        return np.empty((0, window_size, 9), dtype=np.float32), np.array([])

    return np.stack(X_list).astype(np.float32), np.array(centers)
