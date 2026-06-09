"""
realtime_ui.py  BACKY MCU  Real-Time Inference Interface

Usage:
    uv run streamlit run realtime_ui.py

Two modes:
  Live BLE  -- connects to the physical BACKY device via Bluetooth
  CSV Replay -- feeds a saved sensor CSV row-by-row for testing
"""
import os, sys, time, collections, pickle, queue, logging, atexit, json
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from src.feature_extraction import (
    extract_features, calibrate_standing, is_standing, CALIB_FEAT_NAMES,
)

try:
    from device import BLEWorker
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False

# ── logger setup ───────────────────────────────────────────────────────────────
_LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "realtime.log")
os.makedirs(_LOG_DIR, exist_ok=True)

log = logging.getLogger("backy_ui")
if not log.handlers:                       # avoid duplicate handlers on Streamlit reruns
    log.setLevel(logging.DEBUG)
    _fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # rotating file — max 5 MB × 3 backup files
    _fh = RotatingFileHandler(_LOG_FILE, maxBytes=5 * 1024 * 1024,
                               backupCount=3, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    # console (stderr — visible in the terminal running streamlit)
    _ch = logging.StreamHandler(sys.stderr)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    log.addHandler(_fh)
    log.addHandler(_ch)

@st.cache_resource
def _logged_startup_once() -> bool:
    log.info("[startup] realtime_ui loaded  (BLE_AVAILABLE=%s)", BLE_AVAILABLE)
    return True
_logged_startup_once()

# ── module-level BLE state (survives Streamlit reruns + session resets) ────────
# `@st.cache_resource` guarantees the dict is created exactly ONCE for the lifetime
# of the Python process — not on every script rerun. Without this, Streamlit
# re-executes module-level code every tick and resets the status back to "idle"
# even while the BLE daemon thread is still running and producing readings.
@st.cache_resource
def _get_ble_state() -> dict:
    return {
        "status":     "idle",   # "idle" | "searching" | "connected" | "error:<msg>"
        "last_error": "",
        "queue":      queue.Queue(),
        "worker":     None,     # BLEWorker instance
    }

_BLE = _get_ble_state()

# Register a one-shot atexit hook to stop the BLE worker on process exit.
# Cached so re-runs of the script don't register the hook multiple times.
@st.cache_resource
def _register_atexit_once() -> bool:
    def _on_exit():
        w = _BLE.get("worker")
        if w and w.is_alive:
            log.info("[atexit] stopping BLE worker on process exit")
            try:
                w.stop()
                th = getattr(w, "_thread", None)
                if th is not None:
                    th.join(timeout=2.0)
            except Exception as exc:
                log.warning("[atexit] cleanup error: %s", exc)
    atexit.register(_on_exit)
    return True
_register_atexit_once()

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BACKY",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .status-badge {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:13px; font-weight:600; letter-spacing:.5px;
  }
  .badge-connected  { background:#E8F5E9; color:#2E7D32; border:1px solid #A5D6A7; }
  .badge-searching  { background:#FFF8E1; color:#F57F17; border:1px solid #FFE082; }
  .badge-idle       { background:#ECEFF1; color:#546E7A; border:1px solid #CFD8DC; }
  .badge-error      { background:#FFEBEE; color:#C62828; border:1px solid #EF9A9A; }

  .result-card {
    border-radius:14px; padding:22px 28px; text-align:center;
    display:flex; flex-direction:column; justify-content:center;
  }
  .card-label   { font-size:11px; letter-spacing:1.5px; opacity:.75; margin-bottom:4px; }
  .card-main    { font-size:32px; font-weight:700; line-height:1.1; }
  .card-sub     { font-size:13px; opacity:.8; margin-top:6px; }

  div[data-testid="metric-container"] {
    background:#F8F9FA; border:1px solid #E9ECEF;
    border-radius:10px; padding:12px 16px;
  }
  .section-title {
    font-size:14px; font-weight:600; color:#546E7A;
    letter-spacing:.8px; text-transform:uppercase;
    margin-bottom:8px; margin-top:4px;
  }
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────
_WINDOW_COLS = cfg.ALL_SENSOR   # ["sensor1"..."sensor6", "ax", "ay", "az"]

def _window_to_records(window: np.ndarray) -> list:
    """Convert (T, 9) array to list of labelled dicts for session file."""
    return [
        {col: round(float(val), 4) for col, val in zip(_WINDOW_COLS, row)}
        for row in window
    ]

POSTURE = {
    0: {"name": "Good Pickup",    "color": "#43A047", "icon": "checkmark"},
    1: {"name": "Forward Bend",   "color": "#1E88E5", "icon": "arrow-down"},
    2: {"name": "Backward Bend",  "color": "#FB8C00", "icon": "arrow-up"},
    3: {"name": "Twisting",       "color": "#8E24AA", "icon": "refresh"},
}
INVALID_COLOR  = "#E53935"
CHART_BUF      = 200
CALIB_SAMPLES  = cfg.CALIB_N_SAMPLES   # raw 5 Hz rows to collect during calibration

_DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(_DATA_DIR, exist_ok=True)
ANNOTATION_FILE   = os.path.join(_DATA_DIR, "annotations.jsonl")

POSTURE_OPTIONS = {
    "Good Pickup (0)":   0,
    "Forward Bend (1)":  1,
    "Backward Bend (2)": 2,
    "Twisting (3)":      3,
    "Invalid / Walking": -1,
    "Standing":          -2,
}

KNOWN_FILES = {
    "WY  7-pose dynamic":   r"C:\Users\tanwe\Downloads\BKY00507_WY_dynamic_7pose.csv",
    "Hari  7-pose dynamic": r"C:\Users\tanwe\Downloads\BKY00507_Hari_7Pose_dynamic_raw.csv",
    "Hari  GP (good pickup)":    r"C:\Users\tanwe\Downloads\BKY00507_Hari_GP.csv",
    "Hari  FB (forward bend)":   r"C:\Users\tanwe\Downloads\BKY00507_Hari_FB.csv",
    "Hari  BB (backward bend)":  r"C:\Users\tanwe\Downloads\BKY00507_Hari_BB.csv",
    "Hari  LT (left twist)":     r"C:\Users\tanwe\Downloads\BKY00507_Hari_LT.csv",
    "Hari  RT (right twist)":    r"C:\Users\tanwe\Downloads\BKY00507_Hari_RT.csv",
}

# ── load models ────────────────────────────────────────────────────────────────
def _patch_rf(clf):
    """Patch sklearn RF/DT objects pickled with older sklearn versions."""
    estimators = getattr(clf, "estimators_", [])
    for tree in estimators:
        for attr, default in [
            ("monotonic_cst", None),
            ("missing_go_to_left", None),
        ]:
            if not hasattr(tree, attr):
                setattr(tree, attr, default)
    return clf

@st.cache_resource
def load_pipeline():
    paths = [cfg.METADATA_PATH, cfg.GATE_MODEL_PATH, cfg.POSTURE_MODEL_PATH]
    if not all(os.path.exists(p) for p in paths):
        log.error("[startup] model files not found — checked: %s", paths)
        return None, None, None
    with open(cfg.GATE_MODEL_PATH,    "rb") as f: gate    = pickle.load(f)
    with open(cfg.POSTURE_MODEL_PATH, "rb") as f: posture = pickle.load(f)
    with open(cfg.METADATA_PATH,      "rb") as f: meta    = pickle.load(f)
    log.info(
        "[startup] pipeline loaded  gate=%dtrees/d%d  posture=%dtrees/d%d  window=%d  stride=%d",
        cfg.GATE_N_TREES, cfg.GATE_MAX_DEPTH,
        cfg.POSTURE_N_TREES, cfg.POSTURE_MAX_DEPTH,
        cfg.WINDOW_SIZE, cfg.STRIDE,
    )
    return _patch_rf(gate), _patch_rf(posture), meta

gate_clf, posture_clf, pipeline_meta = load_pipeline()

# ── session state init ─────────────────────────────────────────────────────────
def _init():
    defaults = {
        "mode":         "CSV Replay",   # "Live BLE" | "CSV Replay"
        # shared signal / inference state
        "sensor_buf":   collections.deque(maxlen=CHART_BUF),
        "pred_history": [],
        "gate_result":   None,
        "gate_p_valid":  None,   # raw gate probability of valid (0–1)
        "posture_int":   None,
        "confidence":    None,
        "proba":         None,
        "n_win": 0, "n_acc": 0, "n_rej": 0,
        # per-prediction counts
        "n_pred": {0: 0, 1: 0, 2: 0, 3: 0, "rejected": 0, "standing": 0},
        "gate_threshold":    cfg.GATE_THRESHOLD,
        "posture_threshold": 0.0,   # min confidence to show posture (0.0 = off)
        "sway_margin":       3.5,   # multiplier on calib spread for standing gate
        # BLE 5 Hz downsampler
        "ble_accum":        [],    # raw rows collected in current 200 ms bucket
        "ble_last_emit_ms": 0.0,  # wall-clock ms when last 5 Hz sample was emitted
        "ble_total_emitted": 0,   # cumulative 5 Hz samples emitted (used for stride gating)
        # CSV replay
        "csv_running":  False,
        "csv_arr":      None,
        "csv_labels":   None,   # (N,) int array of ground-truth labels, or None
        "csv_idx":      0,
        "csv_total":    0,
        "csv_label":    "",
        # Ground-truth evaluation (CSV replay with label column)
        "eval_records": [],     # list of {win, true_label, pred_gate, pred_posture}
        # Session recording — manual start/stop
        "session_recording": False,  # True while actively writing to CSV
        "session_file":      None,   # path to current session CSV
        "session_n_rec":     0,      # inference rows written this session
        "session_flagged":   set(),  # set of win numbers flagged in this session
        "flag_history":    collections.deque(maxlen=8),  # recent windows for quick-flag dropdown
        # BLE live  (UI-only flags; the actual queue/status/worker live in _BLE)
        "ble_device":      "BACKY",
        "ble_running":     False,
        "ble_reconnect":   True,   # auto-reconnect on unexpected drop
        # ── calibration-based standing gate ───────────────────────────────────
        "calib_user_rest":   None,  # dict from calibrate_standing() after calibration, else None
        "calib_active":      False, # whether standing gate is enabled
        "calib_collecting":  False, # True while collecting calibration samples
        "calib_collect_buf": [],    # list of (9,) rows accumulated during collection
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()
s = st.session_state

# ── helpers ────────────────────────────────────────────────────────────────────
def _load_csv(src) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (sensor_arr, labels_arr).  labels_arr is None when no label column exists."""
    try:
        df = pd.read_csv(src).rename(columns=cfg.ALT_COL_MAP)
    except Exception as e:
        log.error("[csv] load error: %s", e)
        st.sidebar.error(f"Read error: {e}"); return None, None
    missing = [c for c in cfg.ALL_SENSOR if c not in df.columns]
    if missing:
        log.error("[csv] missing columns: %s", missing)
        st.sidebar.error(f"Missing columns: {missing}"); return None, None

    valid_mask = df["az"].values != 0
    df = df[valid_mask].reset_index(drop=True)

    arr = df[cfg.ALL_SENSOR].values.astype(np.float32)

    # extract labels if column present (handles "label" and the "lael" typo, already renamed)
    labels = None
    if "label" in df.columns:
        raw_lbl = df["label"].values.astype(int)
        # apply the same REMAP as training (merge label 4 → 3)
        labels = np.vectorize(lambda x: cfg.REMAP.get(x, x))(raw_lbl)
        log.info("[csv] ground-truth labels found  unique=%s", sorted(set(labels.tolist())))

    log.info("[csv] loaded  rows=%d  has_labels=%s  src=%s", len(arr), labels is not None,
             getattr(src, "name", src) if not isinstance(src, str) else os.path.basename(src))
    return arr, labels

# ── session CSV column layout ──────────────────────────────────────────────────
# Metadata (14 cols): type, win, timestamp, gate, gate_p, pred, pred_name,
#                     proba_gp, proba_fb, proba_bb, proba_tw, confidence, flagged, label
# Window  (90 cols): t{0..9}_{sensor1..sensor6,ax,ay,az}
#
# type = "calib" for calibration rows (25 raw samples, no gate/pred/proba)
# type = "infer" for inference windows
#
# Load in pandas: pd.read_csv("session_*.csv")
# Get window for row i: df.loc[i, 't0_sensor1':'t9_az'].values.reshape(10,9)
# Filter wrong: df[df.flagged]
# Check accuracy: (df.pred == df.label).mean()
# ──────────────────────────────────────────────────────────────────────────────

def _session_header() -> str:
    meta_cols = ("type,win,timestamp,gate,gate_p,pred,pred_name,"
                 "proba_gp,proba_fb,proba_bb,proba_tw,confidence,flagged,label")
    win_cols  = ",".join(
        f"t{t}_{ch}"
        for t in range(cfg.WINDOW_SIZE)
        for ch in cfg.ALL_SENSOR
    )
    return meta_cols + "," + win_cols + "\n"


def _new_session_file() -> str:
    """Create a new session CSV with full inline window data. Return its path."""
    ts       = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(_DATA_DIR, f"session_{ts}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(_session_header())
    log.info("[session] recording started  csv=%s", csv_path)
    return csv_path


def _fmt(v) -> str:
    """Format a value for CSV — empty string for None."""
    if v is None: return ""
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)


def _write_infer(record: dict) -> None:
    """Append one inference row (metadata + full 10×9 window) to session CSV."""
    if not s.session_recording or s.session_file is None:
        return
    try:
        proba = record.get("proba") or [None] * 4
        # flatten window: (T, 9) → 90 values in t0_s1, t0_s2 … t9_az order
        win_arr = record["window_arr"]   # (T, 9) float32
        win_flat = ",".join(f"{float(v):.2f}" for v in win_arr.flatten())
        row = (
            f"infer,"
            f"{record['win']},"
            f"{record['timestamp']},"
            f"{record['gate']},"
            f"{_fmt(record.get('gate_p'))},"
            f"{record['pred']},"
            f"{record.get('pred_name','')},"
            f"{_fmt(proba[0])},{_fmt(proba[1])},{_fmt(proba[2])},{_fmt(proba[3])},"
            f"{_fmt(record.get('confidence'))},"
            f"False,"   # flagged
            f","        # label — empty until filled in
            f"{win_flat}\n"
        )
        with open(s.session_file, "a", encoding="utf-8") as f:
            f.write(row)
        s.session_n_rec += 1
    except Exception as exc:
        log.error("[session] write_infer failed: %s", exc)


def _write_calib(calib_arr: np.ndarray) -> None:
    """Append calibration rows to session CSV (one row per raw sample)."""
    if not s.session_recording or s.session_file is None:
        return
    try:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        # metadata columns have no gate/pred/proba for calib rows
        meta_empty = "," * 10   # gate_p, pred, pred_name, 4×proba, confidence, flagged, label
        with open(s.session_file, "a", encoding="utf-8") as f:
            for i, row_9 in enumerate(calib_arr):
                # calib samples are single rows, not 10-sample windows
                # store in t0_* columns; remaining t1..t9 columns are empty
                t0_vals   = ",".join(f"{float(v):.2f}" for v in row_9)
                t1_9_empty = "," * (cfg.WINDOW_SIZE - 1) * len(cfg.ALL_SENSOR)
                f.write(
                    f"calib,{i},{ts},calib{meta_empty}"
                    f"{t0_vals}{t1_9_empty}\n"
                )
    except Exception as exc:
        log.error("[session] write_calib failed: %s", exc)


# keep _write_record as thin alias so the old call sites still work
def _write_record(record: dict) -> None:
    _write_infer(record)


def _reset_shared():
    s.sensor_buf.clear()
    s.pred_history.clear()
    s.eval_records      = []
    s.session_recording = False
    s.session_file      = None
    s.session_n_rec     = 0
    s.session_flagged   = set()
    s.flag_history      = collections.deque(maxlen=8)
    s.gate_result      = None
    s.gate_p_valid     = None
    s.posture_int      = None
    s.confidence       = None
    s.proba            = None
    s.n_win = s.n_acc = s.n_rej = 0
    s.n_pred = {0: 0, 1: 0, 2: 0, 3: 0, "rejected": 0, "standing": 0}
    s.ble_accum         = []
    s.ble_last_emit_ms  = 0.0
    s.ble_total_emitted = 0
    log.debug("[state] shared state reset")

_POSTURE_NAMES = {0: "good_pickup", 1: "forward_bend", 2: "backward_bend", 3: "twisting"}

def _infer(window: np.ndarray, true_label: int | None = None):
    """Run gate → posture classifier on one window.

    Gate uses predict_proba so a configurable threshold can be applied.
    The default threshold (cfg.GATE_THRESHOLD = 0.75) is stricter than the
    standard 0.5 majority vote because false-accepts are dangerous — a bad
    posture classified as a valid lift gives the user false confidence.

    The threshold is read from st.session_state["gate_threshold"] at runtime
    so the sidebar slider takes effect immediately without restarting.

    true_label: raw (remapped) label from CSV, or None in live/unlabelled mode.
    """
    s.n_win += 1
    # snapshot stored before inference so gate/posture results are added below
    _snap = {"win": s.n_win, "window": window.copy()}

    # ── Stage 0: calibration-based standing gate (before RF) ──────────────────
    # Runs only when the user has completed a standing calibration.
    # calib_user_rest holds the dict returned by calibrate_standing().
    # is_standing() z-scores the inference window using calibration stats,
    # then computes min L2 distance to any of the N_ref normalised reference
    # windows in the 7-D feature space (ACC orient + ACC motion + FSR motion).
    # Windows closer than T_STAND_DIST are output as IGNORE — no RF call needed.
    if s.calib_active and s.calib_user_rest is not None:
        # Compute distance to centroid so we can log it whether or not the gate
        # fires — makes it easy to tell from realtime.log when standing windows
        # are drifting beyond ref_dist (e.g. natural sway accumulating).
        from src.feature_extraction import _calib_window_features
        _ci = s.calib_user_rest
        _z  = (_calib_window_features(window) - _ci["feat_mean"]) / _ci["feat_std"]
        _dist = float(np.sum(np.abs(_z - _ci["centroid"])))
        # recompute ref_dist live so the sway_margin slider takes effect instantly
        _sway  = float(s.get("sway_margin", 3.5))
        _ref   = max(_ci.get("spread", 0.0) * _sway, float(cfg.T_STAND_DIST))
        if _dist < _ref:
            s.gate_result  = "standing"
            s.gate_p_valid = None
            s.posture_int  = None
            s.confidence   = None
            s.proba        = None
            s.n_rej       += 1
            s.n_pred["standing"] += 1
            s.pred_history.append({
                "w": s.n_win, "gate": "standing", "p": None, "conf": None,
            })
            log.debug("[infer] win=%d  STANDING  dist=%.2f < ref=%.2f",
                      s.n_win, _dist, _ref)
            return
        # Not standing — log the miss so we can see how far above ref_dist we are
        log.debug("[infer] win=%d  not-standing  dist=%.2f >= ref=%.2f",
                  s.n_win, _dist, _ref)

    Xf = extract_features(window[np.newaxis])

    # ── gate: use probability, not hard predict ────────────────────────────────
    gate_raw   = gate_clf.predict_proba(Xf)[0]          # [p(valid), p(invalid)]
    # normalise — guards against sklearn version mismatch returning raw vote counts
    gate_raw   = gate_raw / (gate_raw.sum() + 1e-9)
    gate_cls   = {c: i for i, c in enumerate(gate_clf.classes_)}
    p_valid    = float(gate_raw[gate_cls[0]])            # probability of class 0 = valid (0–1)
    threshold  = float(st.session_state.get("gate_threshold", cfg.GATE_THRESHOLD))
    gate_pass  = p_valid >= threshold

    s.gate_result  = "accept" if gate_pass else "reject"
    s.gate_p_valid = p_valid                             # expose to UI

    if gate_pass:
        acc_idx   = pipeline_meta["acc_feat_idx"]
        Xf_acc    = Xf[:, acc_idx]
        p_int     = posture_clf.predict(Xf_acc)[0]
        raw_proba = posture_clf.predict_proba(Xf_acc)[0]
        # normalise — guards against sklearn version mismatch returning raw counts
        total = raw_proba.sum()
        proba = raw_proba / total if total > 0 else raw_proba
        cls_map   = {c: i for i, c in enumerate(posture_clf.classes_)}
        conf      = float(proba[cls_map[p_int]])
        full      = np.zeros(4)
        for c, i in cls_map.items():
            if 0 <= c <= 3: full[c] = proba[i]
        pos_thr = float(st.session_state.get("posture_threshold", 0.0))
        if pos_thr > 0.0 and conf < pos_thr:
            # gate accepted but posture confidence too low — show as uncertain
            s.gate_result = "low_conf"
            s.posture_int = int(p_int)   # still store so UI can show it dimmed
            s.confidence  = conf
            s.proba       = full
            s.n_rej += 1
            s.n_pred["rejected"] += 1
            log.debug(
                "[infer] win=%d  LOW_CONF (p_valid=%.2f >= gate_thr=%.2f, conf=%.2f < pos_thr=%.2f) → %s",
                s.n_win, p_valid, threshold, conf, pos_thr,
                _POSTURE_NAMES.get(int(p_int), "?"),
            )
        else:
            s.posture_int = int(p_int)
            s.confidence  = conf          # stored as 0–1
            s.proba       = full          # stored as 0–1 per class
            s.n_acc += 1
            s.n_pred[int(p_int)] += 1
        log.debug(
            "[infer] win=%d  ACCEPT (p_valid=%.2f >= thr=%.2f) → %s  conf=%.1f%%"
            "  proba=[GP:%.1f%% FB:%.1f%% BB:%.1f%% TW:%.1f%%]",
            s.n_win, p_valid, threshold,
            _POSTURE_NAMES.get(int(p_int), "?"), conf * 100,
            full[0]*100, full[1]*100, full[2]*100, full[3]*100,
        )
    else:
        s.posture_int = None
        s.confidence  = None
        s.proba       = None
        s.n_rej += 1
        s.n_pred["rejected"] += 1
        log.debug(
            "[infer] win=%d  REJECT (p_valid=%.2f < thr=%.2f)",
            s.n_win, p_valid, threshold,
        )

    s.pred_history.append({
        "w": s.n_win, "gate": s.gate_result,
        "p": s.posture_int, "conf": s.confidence,
    })

    # ── session recording ─────────────────────────────────────────────────────
    _pred_int  = (s.posture_int if s.posture_int is not None
                  else (-2 if s.gate_result == "standing" else -1))
    _pred_name = _POSTURE_NAMES.get(_pred_int, {
        -1: "invalid_rejected", -2: "standing_calib"}.get(_pred_int, ""))
    _write_record({
        "win":        s.n_win,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gate":       s.gate_result,
        "gate_p":     round(s.gate_p_valid, 4) if s.gate_p_valid is not None else None,
        "pred":       _pred_int,
        "pred_name":  _pred_name,
        "proba":      [round(float(v), 4) for v in s.proba] if s.proba is not None else None,
        "confidence": round(s.confidence, 4) if s.confidence is not None else None,
        "window_arr": _snap["window"],   # stored in session_windows list as NPY
    })
    # ── flag history (rolling buffer for the quick-flag panel) ────────────────
    _snap["gate"]       = s.gate_result
    _snap["gate_p"]     = s.gate_p_valid
    _snap["pred"]       = _pred_int
    _snap["confidence"] = s.confidence
    s.flag_history.append(_snap)

    # ── ground-truth evaluation record ────────────────────────────────────────
    if true_label is not None:
        s.eval_records.append({
            "win":         s.n_win,
            "true_label":  int(true_label),
            "gate":        s.gate_result,          # "accept" | "reject" | "standing"
            "pred":        s.posture_int,          # int 0-3 or None
        })

def _ble_row_from_reading(r: dict) -> np.ndarray:
    """Convert BLE reading dict to numpy row in ALL_SENSOR order.

    Axis remap (raw device frame → training/model frame):
        tx =  ry,  ty =  rz,  tz = -rx
    Device packet field names are rx→"ax", ry→"ay", rz→"az".

    Training CSVs store tx/ty/tz values under the names ax/ay/az, so the
    live BLE readings must go through this same transform before the
    rolling buffer, feature extraction and the model see them.
    """
    rx, ry, rz = r["ax"], r["ay"], r["az"]
    tx, ty, tz = ry, rz, -rx
    return np.array([
        r["sensor1"], r["sensor2"], r["sensor3"],
        r["sensor4"], r["sensor5"], r["sensor6"],
        tx, ty, tz,
    ], dtype=np.float32)

def _stop_ble_worker(timeout: float = 2.0) -> None:
    """Stop the current BLE worker and wait for the thread to actually exit.

    `worker.stop()` is non-blocking — it only signals the cancel event. We must
    join the underlying thread before starting a new worker, otherwise two
    threads can run simultaneously and both keep producing log lines.
    """
    worker = _BLE.get("worker")
    if not worker:
        return
    log.info("[ble] stopping current worker (timeout=%.1fs)…", timeout)
    try:
        worker.stop()
    except Exception as exc:
        log.warning("[ble] worker.stop() raised: %s", exc)
    # Wait for the underlying daemon thread to exit
    th = getattr(worker, "_thread", None)
    if th is not None:
        th.join(timeout=timeout)
        if th.is_alive():
            log.warning("[ble] worker thread did NOT exit within %.1fs — abandoning it",
                        timeout)
        else:
            log.info("[ble] worker thread exited cleanly")
    _BLE["worker"] = None
    _BLE["status"] = "idle"


def _start_ble_worker(dev_name: str) -> None:
    """(Re-)create and start a BLEWorker.

    All worker callbacks write to module-level `_BLE` — NEVER to st.session_state.
    Session state can be wiped on browser refresh while the daemon thread is
    still alive, so the thread cannot rely on it existing.
    """
    # Always tear down any existing worker before spawning a new one so we
    # never end up with two threads racing into _BLE["queue"].
    if _BLE.get("worker") is not None:
        _stop_ble_worker()

    log.info("[ble] starting worker  device=%s", dev_name)

    def _on_status(st_str: str) -> None:
        _BLE["status"] = st_str
        if st_str.startswith("error:"):
            msg = st_str[6:]
            _BLE["last_error"] = msg
            log.error("[ble] error → %s", msg)
        elif st_str == "connected":
            log.info("[ble] status → connected")
        elif st_str == "searching":
            log.info("[ble] status → scanning for '%s'", dev_name)
        elif st_str == "idle":
            log.info("[ble] status → idle (disconnected)")
        else:
            log.debug("[ble] status → %s", st_str)

    _reading_count = [0]   # mutable counter captured by closure

    def _on_reading(reading: dict) -> None:
        try:
            _BLE["queue"].put(_ble_row_from_reading(reading))
        except Exception as exc:
            log.exception("[ble] failed to enqueue reading: %s", exc)
            return
        _reading_count[0] += 1
        # log every 100th packet to show the stream is alive without flooding the log
        if _reading_count[0] % 100 == 0:
            log.debug(
                "[ble] %d packets received  ax=%.2f ay=%.2f az=%.2f  "
                "fsr=[%d,%d,%d,%d,%d,%d]",
                _reading_count[0],
                reading["ax"], reading["ay"], reading["az"],
                reading["sensor1"], reading["sensor2"], reading["sensor3"],
                reading["sensor4"], reading["sensor5"], reading["sensor6"],
            )

    worker = BLEWorker(_on_status, _on_reading)
    worker.start(dev_name)
    _BLE["worker"] = worker

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BACKY")

    if gate_clf is None:
        st.error("Model not found. Run train.py first.")
        st.stop()

    st.markdown(
        f'<div style="margin-bottom:12px">'
        f'<div style="font-size:12px;color:#888;margin-bottom:4px">Models loaded</div>'
        f'<div style="font-size:12px">Gate &nbsp; {cfg.GATE_N_TREES} trees / depth {cfg.GATE_MAX_DEPTH}</div>'
        f'<div style="font-size:12px">Posture  {cfg.POSTURE_N_TREES} trees / depth {cfg.POSTURE_MAX_DEPTH} (ACC-only)</div>'
        f'<div style="font-size:12px">Window &nbsp; {cfg.WINDOW_SIZE} samples &bull; stride {cfg.STRIDE}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── mode selector ─────────────────────────────────────────────────────────
    mode = st.radio("Mode", ["CSV Replay", "Live BLE"],
                    index=0 if s.mode == "CSV Replay" else 1,
                    horizontal=True)
    s.mode = mode
    st.divider()

    # ── CSV REPLAY controls ───────────────────────────────────────────────────
    if mode == "CSV Replay":
        st.markdown('<div class="section-title">Data File</div>', unsafe_allow_html=True)

        src = st.radio("Source", ["Known file", "Upload CSV"], horizontal=True)

        if src == "Known file":
            sel   = st.selectbox("Select file", list(KNOWN_FILES.keys()))
            fpath = KNOWN_FILES[sel]
            if not os.path.exists(fpath):
                st.warning("File not found on disk.")
            custom = st.text_input("Custom path", "", placeholder="C:\\...\\file.csv")
            if custom: fpath = custom
            uploaded = None
        else:
            uploaded = st.file_uploader("Drop CSV here", type="csv")
            fpath    = None

        st.markdown('<div class="section-title" style="margin-top:12px">Speed</div>',
                    unsafe_allow_html=True)
        speed    = st.select_slider("Rows per tick", [1, 2, 5, 10, 20], value=5)
        delay_ms = st.slider("Delay (ms)", 50, 800, 100)

        st.markdown("")
        c1, c2 = st.columns(2)
        start_csv = c1.button("Start", use_container_width=True, type="primary",
                              disabled=s.csv_running)
        stop_csv  = c2.button("Stop",  use_container_width=True,
                              disabled=not s.csv_running)
        reset_btn = st.button("Reset", use_container_width=True)

        # CSV control handling
        if reset_btn:
            s.csv_running = False
            _reset_shared()
            s.csv_arr = None; s.csv_labels = None
            s.csv_idx = 0; s.csv_total = 0; s.csv_label = ""
            st.rerun()

        if stop_csv:
            log.info("[csv] stopped at row %d / %d", s.csv_idx, s.csv_total)
            s.csv_running = False

        if start_csv:
            src = uploaded if uploaded else (fpath if fpath else None)
            arr, lbl_arr = _load_csv(src) if src else (None, None)
            lbl = getattr(uploaded, "name", None) or (os.path.basename(fpath) if fpath else "")
            if arr is not None and len(arr) >= cfg.WINDOW_SIZE:
                _reset_shared()
                s.csv_arr    = arr
                s.csv_labels = lbl_arr   # None when CSV has no label column
                s.csv_total  = len(arr)
                s.csv_idx    = 0
                s.csv_label  = lbl
                s.csv_running = True
                log.info("[csv] replay started  file=%s  rows=%d  has_labels=%s  speed=%d  delay=%dms",
                         lbl, len(arr), lbl_arr is not None, speed, delay_ms)
                st.rerun()
            elif arr is not None:
                log.warning("[csv] file too short  rows=%d  need>=%d", len(arr), cfg.WINDOW_SIZE)
                st.error(f"Too short ({len(arr)} rows, need >= {cfg.WINDOW_SIZE})")

    # ── LIVE BLE controls ─────────────────────────────────────────────────────
    else:
        if not BLE_AVAILABLE:
            st.error("bleak not installed. Run: uv sync")
        else:
            st.markdown('<div class="section-title">Device</div>', unsafe_allow_html=True)
            dev_name = st.text_input("Device name", value=s.ble_device,
                                     placeholder="e.g. BACKY")
            s.ble_device = dev_name

            status     = _BLE["status"]
            status_key = status.split(":")[0]   # "idle" | "searching" | "connected" | "error"
            badge_cls  = {
                "connected": "badge-connected",
                "searching": "badge-searching",
                "idle":      "badge-idle",
            }.get(status_key, "badge-error")
            badge_label = {
                "connected": "Connected",
                "searching": "Scanning…",
                "idle":      "Disconnected",
            }.get(status_key, f"Error: {status[6:]}")
            st.markdown(
                f'<div style="margin:8px 0 4px">'
                f'<span class="status-badge {badge_cls}">{badge_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # show persistent last-error so it survives the "idle" that follows a drop
            if _BLE["last_error"]:
                st.caption(f"⚠️ Last error: {_BLE['last_error']}")

            auto_reconnect = st.checkbox("Auto-reconnect on drop",
                                         value=s.ble_reconnect, key="ble_reconnect_cb")
            s.ble_reconnect = auto_reconnect

            c1, c2 = st.columns(2)
            # Connect is available when idle OR after an error drop
            connect_btn    = c1.button("Connect",    use_container_width=True,
                                       type="primary",
                                       disabled=(status_key not in ("idle", "error")))
            disconnect_btn = c2.button("Disconnect", use_container_width=True,
                                       disabled=(status_key == "idle"))
            reset_ble = st.button("Reset readings", use_container_width=True)

            if reset_ble:
                _reset_shared()
                st.rerun()

            if connect_btn and dev_name:
                log.info("[ble] connect requested  device=%s", dev_name)
                _reset_shared()
                s.ble_running        = True
                _BLE["last_error"]   = ""
                _start_ble_worker(dev_name)
                st.rerun()

            if disconnect_btn:
                log.info("[ble] disconnect requested by user")
                s.ble_running  = False     # stop the inference loop BEFORE killing worker
                _stop_ble_worker()         # blocks until thread exits
                st.rerun()

# ── Gate threshold slider (shared between both modes) ────────────────────────
with st.sidebar:
    st.divider()
    st.markdown('<div class="section-title">Gate Sensitivity</div>', unsafe_allow_html=True)
    gate_thr = st.slider(
        "Valid probability threshold",
        min_value=0.50, max_value=0.95, step=0.05,
        value=float(s.gate_threshold),
        help=(
            "The gate only passes a window to the posture classifier if it assigns "
            "at least this probability to 'valid'.  "
            "0.50 = standard majority vote (more permissive).  "
            "0.75 = stricter — recommended because a false-accept is dangerous.  "
            "0.90 = very strict, may reject some genuine postures."
        ),
    )
    s.gate_threshold = gate_thr
    st.caption(
        f"Current: **{gate_thr:.2f}**  |  default: {cfg.GATE_THRESHOLD:.2f}  "
        f"|  windows accepted so far: **{s.n_acc}** / {s.n_win}"
    )

    st.markdown('<div class="section-title" style="margin-top:12px">Posture Confidence</div>',
                unsafe_allow_html=True)
    pos_thr = st.slider(
        "Min posture confidence",
        min_value=0.0, max_value=0.95, step=0.05,
        value=float(s.posture_threshold),
        help=(
            "Minimum confidence the posture classifier must have before showing a result.  "
            "0.0 = off (always show).  "
            "0.50 = only show when majority class probability ≥ 50%.  "
            "0.70 = recommended for reliable output — suppresses uncertain low-confidence windows."
        ),
    )
    s.posture_threshold = pos_thr
    st.caption(
        f"Current: **{pos_thr:.2f}**  "
        + (f"|  off (show all)" if pos_thr == 0.0 else f"|  hide if conf < {pos_thr*100:.0f}%")
    )

    # ── Sway margin slider ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Sway Margin</div>', unsafe_allow_html=True)
    sway = st.slider(
        "Standing gate sway margin",
        min_value=1.0, max_value=10.0, step=0.5,
        value=float(s.sway_margin),
        help=(
            "Multiplier on calibration spread to set the standing tolerance zone.  "
            "ref_dist = max(spread × margin, 7.5).  "
            "**Low (1–3):** tight — only exactly your calibration pose counts as standing.  "
            "**Mid (3–5):** covers natural weight-shift.  "
            "**High (6+):** loose — risk of catching twisting as standing."
        ),
    )
    s.sway_margin = sway
    if s.calib_user_rest is not None:
        _spread  = s.calib_user_rest.get("spread", 0.0)
        _ref_live = max(_spread * sway, cfg.T_STAND_DIST)
        st.caption(
            f"Current: **{sway:.1f}×**  |  spread: {_spread:.2f}  "
            f"→  ref_dist: **{_ref_live:.2f}**"
        )
    else:
        st.caption(f"Current: **{sway:.1f}×**  |  calibrate first to see ref_dist")

    # ── Calibration-based standing gate ──────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">Standing Calibration</div>',
                unsafe_allow_html=True)

    _calib_duration_s = CALIB_SAMPLES // cfg.TARGET_HZ   # 5 s

    if s.calib_collecting:
        # ── COLLECTING state ─────────────────────────────────────────────────
        n_so_far = len(s.calib_collect_buf)
        pct      = min(n_so_far / CALIB_SAMPLES, 1.0)
        st.info(
            f"**Stand naturally, shift weight a little…**  {n_so_far} / {CALIB_SAMPLES} samples",
            icon="🧍",
        )
        st.progress(pct)
        if st.button("Cancel", use_container_width=True):
            s.calib_collecting  = False
            s.calib_collect_buf = []
            st.rerun()

    elif s.calib_user_rest is not None:
        # ── DONE state ───────────────────────────────────────────────────────
        calib_info = s.calib_user_rest             # dict from calibrate_standing()
        n_ref      = calib_info["n_windows"]
        raw_means  = calib_info["feat_mean"]       # (7,) raw feature means
        st.success(f"Calibrated ✓  ({n_ref} reference windows, 7 features)")
        # show raw feature means from calibration phase
        feat_lines = "  ".join(
            f"**{n}**: {raw_means[i]:.3f}"
            for i, n in enumerate(CALIB_FEAT_NAMES)
        )
        ref_dist = calib_info.get("ref_dist", 0.0)
        st.caption(feat_lines + f"\n\ndist threshold: **{ref_dist:.3f}** (max pairwise L1, z-scored)")
        calib_on = st.checkbox("Enable standing gate", value=s.calib_active,
                               key="calib_active_cb")
        s.calib_active = calib_on
        if st.button("Recalibrate", use_container_width=True):
            s.calib_user_rest   = None
            s.calib_active      = False
            s.calib_collecting  = False
            s.calib_collect_buf = []
            st.rerun()

    else:
        # ── IDLE state ───────────────────────────────────────────────────────
        st.info(
            f"Stand in your **natural resting position** for **{_calib_duration_s} s** "
            f"then click **Start Calibration**.\n\n"
            f"💡 Shift your weight slightly and relax one leg as you normally would — "
            f"don't stand perfectly rigid. This makes the gate recognise all your natural "
            f"standing variations, not just one fixed pose.",
            icon="🧍",
        )
        if st.button("Start Calibration", use_container_width=True,
                     type="primary"):
            s.calib_collecting  = True
            s.calib_collect_buf = []
            log.info("[calib] collection started  target=%d samples", CALIB_SAMPLES)
            st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────────────────

# ── Header row ────────────────────────────────────────────────────────────────
title_col, status_col = st.columns([3, 1])
with title_col:
    st.markdown("## BACKY Real-Time Posture Inference")

# file / connection context line
if s.mode == "CSV Replay" and s.csv_label:
    pct = s.csv_idx / s.csv_total * 100 if s.csv_total else 0
    st.caption(f"**{s.csv_label}**  |  row {s.csv_idx} / {s.csv_total}  ({pct:.0f}%)")
    st.progress(min(s.csv_idx / s.csv_total, 1.0) if s.csv_total else 0.0)
elif s.mode == "Live BLE":
    status     = _BLE["status"]
    status_key = status.split(":")[0]
    badge_cls  = {
        "connected": "badge-connected",
        "searching": "badge-searching",
        "idle":      "badge-idle",
    }.get(status_key, "badge-error")
    label = {
        "connected": "Connected",
        "searching": "Scanning for device…",
        "idle":      "Not connected",
    }.get(status_key, f"Error — {status[6:]}")
    st.markdown(
        f'<span class="status-badge {badge_cls}" style="font-size:14px">'
        f'BLE &nbsp; {label}</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Signal charts ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Live Sensor Signals</div>', unsafe_allow_html=True)
col_fsr, col_acc = st.columns(2)
ph_fsr = col_fsr.empty()
ph_acc = col_acc.empty()

def _theme() -> dict:
    """Return a colour palette dict that matches the active Streamlit theme."""
    base = st.get_option("theme.base") or "light"
    if base == "dark":
        return dict(
            bg         = "#1A1A2E",
            paper_bg   = "rgba(0,0,0,0)",
            grid       = "rgba(255,255,255,0.07)",
            title_clr  = "#A0AEC0",
            tick_clr   = "#A0AEC0",
            legend_clr = "#E2E8F0",
            zero_line  = "rgba(255,255,255,0.2)",
            hover_bg   = "#2D3748",
            hover_fg   = "white",
            hover_bd   = "#4A5568",
            axis_line  = "rgba(255,255,255,0.1)",
            bar_text   = "white",
            fsr_colors = ["#FF6B6B","#FFA94D","#FFD43B","#69DB7C","#4DABF7","#DA77F2"],
            acc_colors = ["#FF6B6B","#69DB7C","#4DABF7"],
            fsr_total  = "#FFFFFF",
            fsr_fill   = "rgba(255,255,255,0.04)",
            prob_bg    = "#1E1E1E",
            ph_border  = "#2D3748",
            ph_text    = "#4A5568",
        )
    else:
        return dict(
            bg         = "#EEF2F7",
            paper_bg   = "rgba(0,0,0,0)",
            grid       = "rgba(0,0,0,0.07)",
            title_clr  = "#4A5568",
            tick_clr   = "#4A5568",
            legend_clr = "#2D3748",
            zero_line  = "rgba(0,0,0,0.20)",
            hover_bg   = "#FFFFFF",
            hover_fg   = "#2D3748",
            hover_bd   = "#CBD5E0",
            axis_line  = "rgba(0,0,0,0.12)",
            bar_text   = "#2D3748",
            fsr_colors = ["#E53935","#F57C00","#F9A825","#2E7D32","#1565C0","#6A1B9A"],
            acc_colors = ["#E53935","#2E7D32","#1565C0"],
            fsr_total  = "#263238",
            fsr_fill   = "rgba(0,0,0,0.05)",
            prob_bg    = "#F8FAFC",
            ph_border  = "#CBD5E0",
            ph_text    = "#94A3B8",
        )

def _chart_layout(t: dict, title_text: str, y_title: str, height: int = 240) -> dict:
    return dict(
        height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=title_text, font=dict(size=13, color=t["title_clr"]), x=0.01),
        legend=dict(
            orientation="h", font=dict(size=11, color=t["legend_clr"]),
            y=-0.28, x=0, bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(showgrid=False, showticklabels=False,
                   zeroline=False, linecolor=t["axis_line"]),
        yaxis=dict(
            title=y_title, title_font=dict(size=11, color=t["title_clr"]),
            tickfont=dict(color=t["tick_clr"]),
            gridcolor=t["grid"], zeroline=False,
        ),
        plot_bgcolor=t["bg"],
        paper_bgcolor=t["paper_bg"],
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["hover_bg"], font_color=t["hover_fg"],
                        bordercolor=t["hover_bd"]),
    )

def draw_signals():
    buf = list(s.sensor_buf)
    t   = _theme()
    placeholder_style = (
        f'background:{t["bg"]};border:1px solid {t["ph_border"]};border-radius:10px;'
        f'padding:50px;text-align:center;color:{t["ph_text"]};font-size:14px'
    )
    if len(buf) < 2:
        ph_fsr.markdown(f'<div style="{placeholder_style}">FSR signal will appear here</div>',
                        unsafe_allow_html=True)
        ph_acc.markdown(f'<div style="{placeholder_style}">Accelerometer signal will appear here</div>',
                        unsafe_allow_html=True)
        return

    arr = np.array(buf)
    xs  = list(range(len(arr)))

    # ── FSR chart ──────────────────────────────────────────────────────────────
    fig_fsr = go.Figure()

    for i, color in enumerate(t["fsr_colors"]):
        fig_fsr.add_trace(go.Scatter(
            x=xs, y=arr[:, i], name=f"S{i+1}",
            line=dict(color=color, width=1.5),
            opacity=0.80,
            hovertemplate=f"S{i+1}: %{{y:.0f}}<extra></extra>",
        ))

    fsr_total = arr[:, :6].sum(axis=1)
    fig_fsr.add_trace(go.Scatter(
        x=xs, y=fsr_total, name="Total",
        line=dict(color=t["fsr_total"], width=2.5),
        fill="tozeroy", fillcolor=t["fsr_fill"],
        hovertemplate="Total: %{y:.0f}<extra></extra>",
    ))

    fig_fsr.update_layout(**_chart_layout(t, "FSR Pressure  (ADC counts)", "ADC"))
    ph_fsr.plotly_chart(fig_fsr, use_container_width=True,
                        config={"displayModeBar": False})

    # ── ACC chart ──────────────────────────────────────────────────────────────
    fig_acc = go.Figure()
    for i, (label, color) in enumerate(zip(["ax", "ay", "az"], t["acc_colors"])):
        fig_acc.add_trace(go.Scatter(
            x=xs, y=arr[:, 6+i], name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
        ))

    fig_acc.add_hline(y=0, line=dict(color=t["zero_line"], width=1, dash="dot"))

    fig_acc.update_layout(**_chart_layout(t, "Accelerometer  (m/s²)", "m/s²"))
    ph_acc.plotly_chart(fig_acc, use_container_width=True,
                        config={"displayModeBar": False})

draw_signals()

# ── Classification output ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Classification Output</div>', unsafe_allow_html=True)
ph_result = st.empty()

def draw_result():
    gate = s.gate_result

    if gate is None:
        ph_result.markdown(
            '<div style="background:#F8F9FA;border:1px solid #E9ECEF;border-radius:14px;'
            'padding:36px;text-align:center;color:#9E9E9E;font-size:15px">'
            'Waiting for first window  &nbsp;'
            f'(needs {cfg.WINDOW_SIZE} samples)</div>',
            unsafe_allow_html=True)
        return

    gate_pv    = s.gate_p_valid        # float 0–1, or None on first window
    thr        = float(s.gate_threshold)
    pv_pct     = f"{gate_pv*100:.0f}%" if gate_pv is not None else "—"

    if gate == "standing":
        ph_result.markdown(
            '<div class="result-card" style="background:#546E7A;color:white">'
            '<div class="card-label">STANDING GATE  &bull;  calibration rule</div>'
            '<div class="card-main">STANDING / AT REST</div>'
            '<div class="card-sub">'
            'dist to centroid &lt; ref_dist &nbsp;&mdash;&nbsp; RF gate skipped'
            '</div>'
            '</div>',
            unsafe_allow_html=True)
        return

    if gate == "reject":
        ph_result.markdown(
            f'<div class="result-card" style="background:{INVALID_COLOR};color:white">'
            f'<div class="card-label">RF GATE  REJECTED  &bull;  '
            f'p(valid)={pv_pct} &lt; threshold {thr:.2f}</div>'
            f'<div class="card-main">INVALID POSTURE</div>'
            f'<div class="card-sub">Walking  &bull;  Sitting  &bull;  Other non-lift activity</div>'
            f'</div>',
            unsafe_allow_html=True)
        return

    if gate == "low_conf":
        p    = s.posture_int
        info = POSTURE.get(p, {"name": "Unknown", "color": "#78909C"})
        conf_pct = s.confidence * 100 if s.confidence is not None else 0
        pos_thr  = float(st.session_state.get("posture_threshold", 0.0))
        ph_result.markdown(
            f'<div class="result-card" style="background:#B0BEC5;color:white">'
            f'<div class="card-label">LOW CONFIDENCE  &bull;  '
            f'p(valid)={pv_pct} &ge; {thr:.2f}  &bull;  conf={conf_pct:.0f}% &lt; {pos_thr*100:.0f}%</div>'
            f'<div class="card-main" style="opacity:0.7">{info["name"]}</div>'
            f'<div class="card-sub">Confidence below threshold — posture uncertain</div>'
            f'</div>',
            unsafe_allow_html=True)
        return

    # Gate accepted
    p    = s.posture_int
    info = POSTURE.get(p, {"name": "Unknown", "color": "#78909C"})
    col  = info["color"]
    name = info["name"]
    conf_pct   = min(s.confidence * 100, 100.0) if s.confidence is not None else 0
    conf_color = col if conf_pct >= 70 else ("#FF9800" if conf_pct >= 50 else "#E53935")

    left, mid, right = st.columns([2, 1, 3], gap="large")

    with left:
        ph_result.empty()
        st.markdown(
            f'<div class="result-card" style="background:{col};color:white;height:150px">'
            f'<div class="card-label">GATE  ACCEPTED  &bull;  '
            f'p(valid)={pv_pct} &ge; {thr:.2f}  &bull;  Label {p}</div>'
            f'<div class="card-main">{name}</div>'
            f'<div class="card-sub">'
            f'0&nbsp;Good Pickup &nbsp;&bull;&nbsp; 1&nbsp;Forward Bend'
            f' &nbsp;&bull;&nbsp; 2&nbsp;Backward Bend &nbsp;&bull;&nbsp; 3&nbsp;Twisting'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True)

    with mid:
        st.markdown(
            f'<div style="border:2.5px solid {conf_color};border-radius:14px;'
            f'height:150px;display:flex;flex-direction:column;'
            f'justify-content:center;align-items:center">'
            f'<div style="font-size:11px;color:#aaa;letter-spacing:1.5px;margin-bottom:6px">'
            f'CONFIDENCE</div>'
            f'<div style="font-size:46px;font-weight:700;color:{conf_color};line-height:1">'
            f'{conf_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True)

    with right:
        if s.proba is not None:
            t      = _theme()
            labels = [POSTURE[i]["name"] for i in range(4)]
            colors = [POSTURE[i]["color"] for i in range(4)]
            vals   = [float(s.proba[i]) * 100 for i in range(4)]

            fig_prob = go.Figure()
            for i in range(4):
                is_pred = (i == p)
                fig_prob.add_trace(go.Bar(
                    x=[vals[i]], y=[labels[i]], orientation="h",
                    marker_color=colors[i],
                    marker_line=dict(color=t["bar_text"], width=1.5 if is_pred else 0),
                    opacity=1.0 if is_pred else 0.4,
                    text=f"  {vals[i]:.1f}%",
                    textposition="outside",
                    textfont=dict(size=13, color=t["bar_text"],
                                  family="Arial Black" if is_pred else "Arial"),
                    showlegend=False, cliponaxis=False,
                ))
            fig_prob.update_layout(
                height=150,
                margin=dict(l=0, r=70, t=8, b=4),
                xaxis=dict(range=[0, 125], showticklabels=False,
                           showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False,
                           tickfont=dict(size=12, color=t["tick_clr"])),
                plot_bgcolor=t["prob_bg"],
                paper_bgcolor="rgba(0,0,0,0)",
                barmode="overlay", bargap=0.25,
            )
            st.plotly_chart(fig_prob, use_container_width=True,
                            config={"displayModeBar": False})

draw_result()

# ── Prediction distribution row ────────────────────────────────────────────────
st.markdown("---")
_pc = s.n_pred
c_tot, c_gp, c_fb, c_bb, c_tw, c_rej, c_stand = st.columns([1.4, 1, 1, 1, 1, 1, 1])
c_tot.metric("Total Windows", s.n_win)
c_gp.metric("🟢 Good Pickup",    _pc[0])
c_fb.metric("🔵 Forward Bend",   _pc[1])
c_bb.metric("🟠 Backward Bend",  _pc[2])
c_tw.metric("🟣 Twisting",       _pc[3])
c_rej.metric("❌ Gate Rejected",  _pc["rejected"])
c_stand.metric("🧍 Standing",    _pc["standing"])

# ── Ground-truth evaluation panel (CSV replay with label column) ───────────────
def draw_eval():
    records = s.eval_records
    if not records:
        return

    st.markdown("---")
    st.markdown('<div class="section-title">Ground-Truth Evaluation</div>',
                unsafe_allow_html=True)

    df_ev = pd.DataFrame(records)   # cols: win, true_label, gate, pred

    # ── separate valid-lift windows from invalid (true_label >= 5 or 7/6/5)
    VALID_TRUE   = df_ev["true_label"].isin([0, 1, 2, 3])
    INVALID_TRUE = ~VALID_TRUE

    # For valid-lift windows: correct = gate accepted AND pred == true_label
    valid_df = df_ev[VALID_TRUE]
    n_valid  = len(valid_df)
    if n_valid:
        correct_posture = (
            (valid_df["gate"] == "accept") &
            (valid_df["pred"] == valid_df["true_label"])
        ).sum()
        false_reject    = (valid_df["gate"] != "accept").sum()   # missed lifts
    else:
        correct_posture = false_reject = 0

    # For invalid windows (walk/sit/stand): correct = gate rejected
    invalid_df = df_ev[INVALID_TRUE]
    n_invalid  = len(invalid_df)
    if n_invalid:
        correct_reject = (invalid_df["gate"] != "accept").sum()
        false_accept   = (invalid_df["gate"] == "accept").sum()
    else:
        correct_reject = false_accept = 0

    n_total  = len(df_ev)
    n_correct = correct_posture + correct_reject
    overall_acc = n_correct / n_total * 100 if n_total else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall acc",    f"{overall_acc:.1f}%")
    m2.metric("Posture correct", f"{correct_posture}/{n_valid}" if n_valid else "—")
    m3.metric("False reject",    f"{false_reject}/{n_valid}"    if n_valid else "—",
              delta=f"-{false_reject/n_valid*100:.0f}%" if n_valid else None,
              delta_color="inverse")
    m4.metric("False accept",    f"{false_accept}/{n_invalid}"  if n_invalid else "—",
              delta=f"+{false_accept/n_invalid*100:.0f}%" if n_invalid else None,
              delta_color="inverse")
    m5.metric("Windows scored",  n_total)

    # ── confusion matrix for valid-lift postures ───────────────────────────────
    if n_valid >= 4:
        POSTURE_LABELS = ["Good Pickup", "Fwd Bend", "Bwd Bend", "Twisting"]
        cm = np.zeros((4, 4), dtype=int)
        for _, row in valid_df[valid_df["gate"] == "accept"].iterrows():
            tl = int(row["true_label"])
            pl = int(row["pred"])
            if 0 <= tl <= 3 and 0 <= pl <= 3:
                cm[tl, pl] += 1

        # normalise per row (true class)
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct   = cm / row_sums * 100

        fig_cm = go.Figure(go.Heatmap(
            z=cm_pct[::-1],
            x=POSTURE_LABELS,
            y=POSTURE_LABELS[::-1],
            text=[[f"{cm[::-1][i,j]}<br>({cm_pct[::-1][i,j]:.0f}%)"
                   for j in range(4)] for i in range(4)],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
        ))
        t = _theme()
        fig_cm.update_layout(
            height=260, margin=dict(l=0, r=0, t=28, b=0),
            title=dict(text="Posture Confusion Matrix  (accepted windows only)",
                       font=dict(size=13, color=t["title_clr"]), x=0.01),
            xaxis=dict(title="Predicted", tickfont=dict(size=11)),
            yaxis=dict(title="True",      tickfont=dict(size=11)),
            plot_bgcolor=t["bg"], paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    # ── export button ──────────────────────────────────────────────────────────
    if st.button("Export eval CSV", use_container_width=False):
        csv_bytes = df_ev.to_csv(index=False).encode()
        st.download_button("Download", csv_bytes, "eval_records.csv", "text/csv")

draw_eval()

# ── Session recording panel ────────────────────────────────────────────────────
def _pred_label(pred_int):
    if pred_int == -2: return "Standing (calib)"
    if pred_int == -1: return "Invalid / Rejected"
    return POSTURE.get(pred_int, {}).get("name", f"Label {pred_int}")


def _flag_win(win: int) -> None:
    """Mark a window as flagged by rewriting that CSV row's flagged column."""
    if s.session_file is None or not os.path.exists(s.session_file):
        return
    try:
        lines = open(s.session_file, encoding="utf-8").readlines()
        with open(s.session_file, "w", encoding="utf-8") as f:
            for i, line in enumerate(lines):
                if i == 0:          # header row — keep as-is
                    f.write(line)
                    continue
                cols = line.rstrip("\n").split(",")
                if cols and cols[0] == str(win):
                    cols[11] = "True"   # flagged column (index 11)
                    f.write(",".join(cols) + "\n")
                else:
                    f.write(line)
        s.session_flagged.add(win)
        log.info("[flag] marked win=%d as flagged in CSV", win)
    except Exception as exc:
        log.error("[flag] could not flag win %d: %s", win, exc)


def draw_session_panel():
    st.markdown("---")
    st.markdown('<div class="section-title">Session Recording</div>',
                unsafe_allow_html=True)

    hist = list(s.flag_history)

    # ── row 1: [Start] [Stop] | status ───────────────────────────────────────
    c_start, c_stop, c_status = st.columns([1, 1, 3])
    with c_start:
        btn_start = st.button("⏺ Start", type="primary",
                              disabled=s.session_recording,
                              use_container_width=True)
    with c_stop:
        btn_stop = st.button("⏹ Stop",
                             disabled=not s.session_recording,
                             use_container_width=True)
    with c_status:
        if s.session_recording and s.session_file:
            fname = os.path.basename(s.session_file)
            st.success(f"⏺ **{fname}**  ·  {s.session_n_rec} windows  ·  {len(s.session_flagged)} flagged")
        elif s.session_file:
            fname = os.path.basename(s.session_file)
            st.info(f"⏹ **{fname}**  ·  {s.session_n_rec} windows")
        else:
            st.caption("Press ⏺ Start to begin recording.")

    if btn_start:
        s.session_file      = _new_session_file()
        s.session_n_rec     = 0
        s.session_flagged   = set()
        s.session_recording = True
        log.info("[session] recording started by user")
        st.rerun()

    if btn_stop:
        s.session_recording = False
        log.info("[session] recording stopped  rows=%d  file=%s",
                 s.session_n_rec, s.session_file)
        st.rerun()

    # ── row 2: Download CSV (only when file exists) ───────────────────────────
    if s.session_file and os.path.exists(s.session_file):
        try:
            st.download_button(
                "Download CSV",
                open(s.session_file, "rb").read(),
                file_name=os.path.basename(s.session_file),
                mime="text/csv",
                use_container_width=False,
            )
        except Exception:
            pass

    # ── row 3: Flag form (only when windows exist) ────────────────────────────
    if hist:
        hist_rev = list(reversed(hist))
        def _lbl(h):
            conf  = f" {h['confidence']*100:.0f}%" if h["confidence"] else ""
            check = " ✓" if h["win"] in s.session_flagged else ""
            return f"Win #{h['win']}  {_pred_label(h['pred'])}{conf}{check}"
        labels = [_lbl(h) for h in hist_rev]
        with st.form("flag_form", clear_on_submit=True):
            f_sel, f_btn = st.columns([5, 1])
            with f_sel:
                sel = st.selectbox("Flag window as wrong", range(len(labels)),
                                   format_func=lambda i: labels[i],
                                   label_visibility="collapsed")
            with f_btn:
                submitted = st.form_submit_button("Flag", type="primary",
                                                  use_container_width=True)
        if submitted:
            win = hist_rev[sel]["win"]
            _flag_win(win)
            st.success(f"Win #{win} flagged.")

    # ── list of all past session CSV files ────────────────────────────────────
    session_files = sorted(
        [f for f in os.listdir(_DATA_DIR) if f.startswith("session_") and f.endswith(".csv")],
        reverse=True
    )
    if session_files:
        with st.expander(f"Past sessions ({len(session_files)})", expanded=False):
            for fname in session_files[:10]:
                fpath = os.path.join(_DATA_DIR, fname)
                try:
                    n = max(0, sum(1 for _ in open(fpath, encoding="utf-8")) - 1)
                    st.download_button(
                        f"{fname}  ({n} windows)",
                        open(fpath, "rb").read(),
                        file_name=fname,
                        mime="text/csv",
                        key=f"dl_{fname}",
                    )
                except Exception:
                    pass

draw_session_panel()

# ── Prediction timeline ────────────────────────────────────────────────────────
def draw_timeline():
    hist = s.pred_history
    if not hist:
        return

    t   = _theme()
    fig = go.Figure()
    STANDING_COLOR = "#546E7A"
    for entry in hist:
        if entry["gate"] == "standing":
            color = STANDING_COLOR
            tip   = f"Window {entry['w']}<br>STANDING (calibration gate)"
        elif entry["gate"] == "reject":
            color = INVALID_COLOR
            tip   = f"Window {entry['w']}<br>REJECTED — Invalid Posture"
        else:
            pinfo = POSTURE.get(entry["p"], {"name": "?", "color": "#78909C"})
            color = pinfo["color"]
            conf  = f"{entry['conf']*100:.0f}%" if entry["conf"] is not None else "—"
            tip   = (f"Window {entry['w']}<br>ACCEPTED<br>"
                     f"<b>{pinfo['name']}</b><br>Confidence: {conf}")
        fig.add_trace(go.Bar(
            x=[entry["w"]], y=[1],
            marker_color=color,
            marker_line=dict(color=t["bg"], width=1),
            showlegend=False,
            hovertemplate=tip + "<extra></extra>",
        ))

    # legend swatches only (no duplicate bars)
    for i, info in POSTURE.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=info["color"],
                             name=info["name"], showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=INVALID_COLOR,
                         name="Invalid / Rejected", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=STANDING_COLOR,
                         name="Standing (calib gate)", showlegend=True))

    fig.update_layout(
        height=130,
        barmode="overlay",
        bargap=0.06,
        margin=dict(l=0, r=0, t=24, b=56),
        title=dict(
            text="Prediction Timeline  (hover for details)",
            font=dict(size=13, color=t["title_clr"]), x=0.01,
        ),
        xaxis=dict(
            title="window #",
            title_font=dict(color=t["title_clr"], size=11),
            tickfont=dict(color=t["tick_clr"]),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(visible=False, range=[0, 1.4]),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.30,
            xanchor="left", x=0,
            font=dict(size=12, color=t["legend_clr"]),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        plot_bgcolor=t["bg"],
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=t["hover_bg"], font_color=t["hover_fg"],
                        bordercolor=t["hover_bd"]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

draw_timeline()

# ── Inference loop — CSV replay ────────────────────────────────────────────────
if s.mode == "CSV Replay" and s.csv_running and s.csv_arr is not None:
    T   = s.csv_total
    idx = s.csv_idx
    if idx >= T:
        s.csv_running = False
        accept_rate = s.n_acc / s.n_win * 100 if s.n_win else 0
        log.info(
            "[csv] replay complete  file=%s  windows=%d  accepted=%d  rejected=%d  accept_rate=%.1f%%",
            s.csv_label, s.n_win, s.n_acc, s.n_rej, accept_rate,
        )
        st.success(f"Replay complete  {s.n_win} windows processed.")
    else:
        end = min(idx + speed, T)
        for r in range(idx, end):
            row = s.csv_arr[r]
            s.sensor_buf.append(row)

            # ── calibration collection ───────────────────────────────────────
            if s.calib_collecting:
                s.calib_collect_buf.append(row)
                if len(s.calib_collect_buf) >= CALIB_SAMPLES:
                    calib_arr         = np.array(s.calib_collect_buf, dtype=np.float32)
                    s.calib_user_rest = calibrate_standing(calib_arr)
                    s.calib_active    = True
                    s.calib_collecting  = False
                    _write_calib(calib_arr)
                    s.calib_collect_buf = []
                    log.info(
                        "[calib] complete  ref_windows=%d  ref_dist=%.2f",
                        s.calib_user_rest["n_windows"], s.calib_user_rest["ref_dist"],
                    )
                continue                # skip inference during calibration

        s.csv_idx = end

        if not s.calib_collecting:          # skip inference while collecting calibration
            buf = np.array(s.sensor_buf)
            if len(buf) >= cfg.WINDOW_SIZE:
                # ground-truth label = label of the last row in the current window
                gt = None
                if s.csv_labels is not None and s.csv_idx > 0:
                    gt = int(s.csv_labels[min(s.csv_idx - 1, len(s.csv_labels) - 1)])
                _infer(buf[-cfg.WINDOW_SIZE:], true_label=gt)

        time.sleep(delay_ms / 1000)
        st.rerun()

# ── Inference loop — Live BLE ──────────────────────────────────────────────────
elif s.mode == "Live BLE" and s.ble_running:
    # ── drain all waiting packets into the accumulator ─────────────────────────
    raw_drained = 0
    try:
        while True:
            row = _BLE["queue"].get_nowait()
            s.ble_accum.append(row)
            raw_drained += 1
    except queue.Empty:
        pass

    # anchor the clock on the very first packet so the first bucket starts now
    if raw_drained > 0 and s.ble_last_emit_ms == 0.0:
        s.ble_last_emit_ms = time.time() * 1000

    # ── 5 Hz downsampler: emit one averaged sample every TARGET_PERIOD_MS ──────
    # BLE may send 10–100 Hz raw, often in bursts.  We average all packets in
    # the current accumulator into one sample per 200 ms slot.  When a burst
    # covers several elapsed slots (e.g. a 1 s burst → 5 slots), we reuse the
    # same averaged row for the remaining slots rather than emitting nothing.
    # Previously ble_accum was cleared inside the loop, so burst arrivals only
    # ever produced 1 sample instead of the correct 5 — inference ran at ~1 Hz.
    now_ms      = time.time() * 1000
    period_ms   = cfg.TARGET_PERIOD_MS          # 200 ms
    emitted     = 0
    last_avg    = None                          # last computed average (reused for gap-fill)

    # emit as many 200 ms slots as have elapsed since the last emission
    while (now_ms - s.ble_last_emit_ms) >= period_ms:
        if s.ble_accum:
            # fresh data available — compute new average and clear accumulator
            batch    = np.array(s.ble_accum, dtype=np.float32)
            last_avg = batch.mean(axis=0)       # (9,)
            s.ble_accum = []
        if last_avg is None:
            break                               # no data ever arrived — nothing to emit

        avg_row = last_avg                      # hold-last-value for gap slots
        s.sensor_buf.append(avg_row)
        s.ble_last_emit_ms += period_ms         # advance clock by exactly one slot
        s.ble_total_emitted += 1
        emitted += 1

        # ── calibration collection ───────────────────────────────────────────
        if s.calib_collecting:
            s.calib_collect_buf.append(avg_row)
            if len(s.calib_collect_buf) >= CALIB_SAMPLES:
                calib_arr         = np.array(s.calib_collect_buf, dtype=np.float32)
                s.calib_user_rest = calibrate_standing(calib_arr)
                s.calib_active    = True
                s.calib_collecting  = False
                _write_calib(calib_arr)
                s.calib_collect_buf = []
                log.info(
                    "[calib] complete  ref_windows=%d  spread=%.2f  ref_dist=%.2f  "
                    "(spread×3=%.2f, floor=%.2f)",
                    s.calib_user_rest["n_windows"],
                    s.calib_user_rest.get("spread", 0.0),
                    s.calib_user_rest["ref_dist"],
                    s.calib_user_rest.get("spread", 0.0) * 3.0,
                    cfg.T_STAND_DIST,
                )

    if raw_drained > 0 or emitted > 0:
        log.debug(
            "[ble] drained=%d raw  emitted=%d 5Hz samples  buf=%d  total_emitted=%d",
            raw_drained, emitted, len(s.sensor_buf), s.ble_total_emitted,
        )

    # ── run inference once (on the latest window) ────────────────────────────
    # sensor_buf already holds all emitted samples from this rerun.
    # buf[-WINDOW_SIZE:] is always the freshest window — calling _infer N times
    # on the identical slice would inflate n_win without producing new information.
    if (emitted > 0
            and not s.calib_collecting
            and s.ble_total_emitted >= cfg.WINDOW_SIZE
            and len(s.sensor_buf) >= cfg.WINDOW_SIZE):
        buf = np.array(s.sensor_buf)
        _infer(buf[-cfg.WINDOW_SIZE:])

    # detect unexpected worker exit and auto-reconnect if enabled
    worker = _BLE["worker"]
    if worker and not worker.is_alive:
        if s.ble_reconnect and s.ble_device:
            log.warning("[ble] worker exited unexpectedly — auto-reconnecting to '%s'",
                        s.ble_device)
            _BLE["status"] = "searching"
            _start_ble_worker(s.ble_device)
        else:
            log.info("[ble] worker exited  auto-reconnect disabled  stopping loop")
            s.ble_running  = False
            _BLE["status"] = "idle"

    # Rerun at ~20 Hz (50 ms) — fast enough to pick up every 200 ms 5Hz slot
    # within the same rerun cycle, reducing display lag to ≤50 ms.
    time.sleep(0.050)
    st.rerun()
