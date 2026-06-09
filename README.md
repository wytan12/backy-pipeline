# BACKY Pipeline

BACKY is a wearable device with **6 force-sensitive resistors (FSR)** and a **3-axis
accelerometer (IMU)**. It classifies lifting postures in real time, including good pickup, forward bend, backward bend, and twisting, while distinguishing them from non-lifting activities such as walking, sitting, and standing.

It runs a three-stage inference pipeline:
**Standing Calibration Gate → RF Validity Gate → RF Posture Classifier**

---

## Requirements

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** for environment management (recommended), or conda
- **OS:** Windows / macOS / Linux (developed on Windows 11, PowerShell)
- **For Live BLE mode only:** Bluetooth adapter + the physical BACKY device in range (`bleak`)

Core dependencies (installed by `uv sync`): `numpy`, `pandas`, `scikit-learn`,
`streamlit`, `plotly`, `bleak`, `matplotlib`, `seaborn`.

### Training data (optional)

You only need raw data if you want to **retrain** the models. The shipped
`models/*.pkl` checkpoints let you run inference without it. Expected layout:

```
Labelled_Data/
  Organized Dynamic Data by User/
  Organized Static Data by User/
```

Data is sampled at **5 Hz** (2 Hz recordings are skipped). Users in the LOUO folds:
Aashish, LH, LW, Melvin.

---

## Quick start

```bash
# from the repo root
cd backy_mcu_pipeline
uv sync                              # create .venv and install everything
uv run streamlit run realtime_ui.py  # launch the dashboard
```

Pre-trained models are already in `models/`, so the dashboard and `inference.py`
work out of the box — no training step required.

---

## What to run

### Real-time interface (main thing to use)
```bash
uv run streamlit run realtime_ui.py
```
Opens a browser dashboard. Two modes:
- **CSV Replay** — feed a saved sensor CSV row-by-row (use for testing accuracy)
- **Live BLE** — connect to the physical BACKY device over Bluetooth

### Train models
```bash
uv run python train.py              # full LOUO cross-validation + save models
uv run python train.py --skip-louo  # faster, no cross-validation
```
Saves `models/gate_rf.pkl`, `models/posture_rf.pkl`, `models/pipeline_metadata.pkl`

### Batch prediction on a CSV file
```bash
uv run python inference.py path/to/sensor_data.csv
uv run python inference.py data.csv --output results.csv
uv run python inference.py data.csv --bench-iters 100   # latency benchmark
```

---

## How the pipeline works

Raw sensor data flows through pre-processing into three sequential stages. Each stage
can reject a window (emit **IGNORE**, code `-1`) — only windows that pass every gate
get a posture label.

```
Device / CSV  →  9 channels (sensor1-6, ax, ay, az) @ 5 Hz
      │
      ▼  column rename + sliding window  (T=10 samples, stride=5  →  1 decision / 1.0 s)
      │
 ┌────┴─────────────────────────────────────────────────────────┐
 │ STAGE 0 — Standing Calibration Gate  (feature_extraction.py) │
 │  • One-time: user stands still 5 s → personalised reference  │
 │  • Per window: 7 calib features → z-score → L1 distance      │
 │  • dist < ref_dist  →  IGNORE (it's standing, skip the RFs)  │
 └────┬─────────────────────────────────────────────────────────┘
      ▼  extract_features()  →  38 hand-crafted features
 ┌────┴─────────────────────────────────────────────────────────┐
 │ STAGE 1 — Validity Gate RF   (20 trees, depth 5, all 38 feat)│
 │  • p_valid = predict_proba()                                 │
 │  • p_valid <  GATE_THRESHOLD (0.60)  →  IGNORE               │
 │  • p_valid >= GATE_THRESHOLD         →  accept, go to Stage 2│
 └────┬─────────────────────────────────────────────────────────┘
      ▼
 ┌────┴─────────────────────────────────────────────────────────┐
 │ STAGE 2 — Posture RF   (30 trees, depth 8, 16 ACC-only feat) │
 │  → 0 good_pickup  1 forward_bend  2 backward_bend  3 twisting│
 └────┬─────────────────────────────────────────────────────────┘
      ▼  
  final label + confidence  (logged to logs/realtime.log)
```

### Why three stages

False-accepts (a non-lift scored as a posture → wrong ergonomic warning) are
**dangerous**; false-rejects (a real lift gated out) are **benign** because the next
window usually catches it. Splitting the pipeline gives one tunable safety dial (the
gate threshold) and lets the gate and posture model improve independently.

1. **Calibration gate** removes between-person variance cheaply (body weight shifts the
   FSR baseline; resting posture shifts the ACC tilt baseline) before any learned model
   runs. Standing is the most common — and most dangerous-to-confuse — invalid state.
2. **Validity gate** is a binary RF separating valid lifts from invalid activity
   (walking/sitting/standing). Threshold is tunable (sidebar slider, 0.50–0.95).
3. **Posture classifier** uses **ACC-only** features — trunk tilt is the robust
   discriminant for posture type; FSR readings drift with body weight and belt tightness.

### Features (38 hand-crafted, `feature_extraction.py`)

- **FSR (indices 0–21):** per-channel + total mean/std, energy, L/R asymmetry, and
  temporal cues (slope, zero-crossing rate, dynamic asymmetry).
- **ACC (indices 22–37, used by Stage 2):** per-axis mean/std, magnitude stats,
  `tilt_deg` (the single most informative posture cue), plus temporal slope/ZCR/autocorr.

All features are MCU-friendly (means, sums, dot products, one `arccos` — no FFT, no
sorting).

### Label mapping

| Raw CSV label | Meaning | Output code |
|---|---|---|
| 0 | good pickup | 0 |
| 1 | forward bend | 1 |
| 2 | backward bend | 2 |
| 3 / 4 | twisting (L / R, merged) | 3 |
| 5 / 6 / 7 | walking / sitting / standing | -1 (gate rejects) |

---

## Folder structure

```
backy_mcu_pipeline/
  realtime_ui.py        ← real-time Streamlit interface
  train.py              ← train gate + posture models
  inference.py          ← batch prediction on CSV
  config.py             ← all constants and paths (imported everywhere)
  data_loader.py        ← CSV loading, windowing, BLE stream parsing
  feature_extraction.py ← 38 hand-crafted features + calibration gate
  evaluate.py           ← LOUO split helpers, metrics
  device.py             ← BLE driver (BLESession + BLEWorker)
  
  models/               ← trained model checkpoints (gate_rf, posture_rf)
```

---

## Best model (recommended)

**HC-RF** — Hand-crafted features + Random Forest (current default)
- Gate: 20 trees / depth 5 (all 38 features)
- Posture: 30 trees / depth 8 (16 ACC-only features)
- End-to-end accuracy: 0.924 (T=64 LOUO), 0.809 (T=10 LOUO)
- Latency: ~6.6 ms CPU, ~25 KB RAM

---

## Testing sessions

1. Run `realtime_ui.py` in **CSV Replay** mode
2. Select a labelled file (Hari GP / FB / BB / LT / RT)
3. Watch the **Ground-Truth Evaluation** panel for live accuracy + confusion matrix
4. If you see a wrong classification → use the **Flag Wrong Classification** panel to record it
5. Flags are saved to `data/annotations.jsonl` with the raw sensor window

---

## Window size and classification lag

- Window: **10 samples × 200 ms = 2 seconds**
- First classification fires **2 s after** a posture starts
- Subsequent updates every **200 ms** (BLE) or per-tick (CSV replay)
- Hold each posture at least **3–4 s** for reliable evaluation


