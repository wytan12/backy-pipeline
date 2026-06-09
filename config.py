"""
config.py — Central configuration for the BACKY MCU pipeline.
All paths, constants, and label maps live here.
"""
import os

# ── Data paths ─────────────────────────────────────────────────────────────────
DATA_BASE = os.path.join(
    os.path.dirname(__file__),
    "Labelled_Data"
)
DYN_DIR  = os.path.join(DATA_BASE, "Organized Dynamic Data by User")
STAT_DIR = os.path.join(DATA_BASE, "Organized Static Data by User")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

GATE_MODEL_PATH    = os.path.join(MODELS_DIR, "gate_rf.pkl")
POSTURE_MODEL_PATH = os.path.join(MODELS_DIR, "posture_rf.pkl")
METADATA_PATH      = os.path.join(MODELS_DIR, "pipeline_metadata.pkl")

# ── Sensor columns ─────────────────────────────────────────────────────────────
FSR_COLS    = ["sensor1", "sensor6", "sensor2", "sensor5", "sensor3", "sensor4"]
ACC_COLS    = ["ax", "ay", "az"]
ALL_SENSOR  = FSR_COLS + ACC_COLS

# Canonical ordered sensor columns (FSR in index order, then ACC)
SENSOR_COLS = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6",
               "ax", "ay", "az"]

# Unified column rename map — covers every naming variant seen across devices:
#   s1-s6        : short FSR names
#   accel_x/y/z  : accelerometer long names
#   lael         : typo in some CSV headers (should be "label")
SENSOR_COL_MAP = {
    "s1": "sensor1", "s2": "sensor2", "s3": "sensor3",
    "s4": "sensor4", "s5": "sensor5", "s6": "sensor6",
    "accel_x": "ax", "accel_y": "ay", "accel_z": "az",
    "lael": "label",
}

# Backward-compatible alias used by realtime_ui.py
ALT_COL_MAP = SENSOR_COL_MAP

# ── Sampling rate ──────────────────────────────────────────────────────────────
TARGET_HZ        = 5                    # training data sample rate
TARGET_PERIOD_MS = 1000 / TARGET_HZ    # 200 ms per sample

# ── Windowing ──────────────────────────────────────────────────────────────────
WINDOW_SIZE = 10    # samples  (2 s at 5 Hz)
STRIDE      = 5     # 50 % overlap

# ── Label mapping ──────────────────────────────────────────────────────────────
# Original labels 3 and 4 are both twisting — merge to 3
REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
VALID_LABELS_RAW   = {0, 1, 2, 3, 4}
INVALID_LABELS_RAW = {5, 6, 7}

POSTURE_NAMES = {
    0: "good_pickup",
    1: "forward_bend",
    2: "backward_bend",
    3: "twisting",
}
INVALID_NAMES = {5: "walking", 6: "sitting", 7: "standing"}

# Pipeline output codes
OUTPUT_IGNORE    = -1
OUTPUT_LABELS    = {**POSTURE_NAMES, -1: "ignore"}

# ── Gate ───────────────────────────────────────────────────────────────────────
FSR_THRESHOLD  = 50         # ADC counts — channel considered "active"

# Minimum probability that the gate must assign to "valid" before the posture
# classifier is invoked.  0.5 = standard majority vote; 0.75 = stricter
# (false-accept is dangerous, so default to the higher threshold).
GATE_THRESHOLD = 0.6

# ── Calibration-based standing gate ───────────────────────────────────────────
# Applied BEFORE the RF gate.  User stands still for 5 s at startup;
# calibrate_standing() records CALIB_N_SAMPLES raw rows, extracts overlapping
# windows, z-scores them, and stores the cluster centroid + max pairwise L1
# distance (ref_dist) as the self-calibrated standing threshold.
#
# At inference, each window's 7-feature vector is z-scored using the calibration
# statistics, then its L1 distance to the centroid is compared against ref_dist.
# If the distance is below ref_dist the window is output as IGNORE without
# invoking the RF gate.
#
# Features (7): acc_x_mean, acc_y_mean, acc_z_mean, acc_mag_std, acc_diff_energy,
#               fsr_lr_asymmetry, fsr_diff_energy
#
# ref_dist is computed automatically from the calibration data — no manual
# threshold tuning required.  It equals the diameter (max pairwise L1 distance)
# of the standing cluster in z-scored 7-D space.
#
# NOTE: "Stand vs All Valid" shows heavy overlap in this feature space because
# slow-start lifting (good_pickup) resembles standing.  The RF gate provides the
# second layer of classification for windows that pass this gate.
CALIB_N_SAMPLES = 25    # raw 5 Hz samples to collect during calibration (= 5 s)

# Safety floor for ref_dist (used by calibrate_standing).  ref_dist is computed
# as max(calibration_spread × SWAY_MARGIN, T_STAND_DIST), so even a perfectly-
# still calibration gives a sane threshold.  7.5 is the M4-derived F1-optimal
# L1 threshold for Stand-vs-Twisting in the z-scored 7-D feature space.
T_STAND_DIST    = 7.5

# ── Gate model hyperparameters ─────────────────────────────────────────────────
GATE_N_TREES   = 20
GATE_MAX_DEPTH = 5
GATE_MIN_LEAF  = 5

# ── Posture model hyperparameters ──────────────────────────────────────────────
POSTURE_N_TREES   = 30
POSTURE_MAX_DEPTH = 8
POSTURE_MIN_LEAF  = 5
