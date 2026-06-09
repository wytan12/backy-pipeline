"""
inference.py — Load saved pipeline checkpoints and run inference on a new CSV file.

Usage:
    uv run python inference.py path/to/file.csv
    uv run python inference.py path/to/file.csv --output results.csv
    uv run python inference.py path/to/file.csv --bench-iters 100

Timing breakdown reported:
    - preprocessing (column rename, zero-row removal)
    - windowing (sliding window extraction)
    - feature extraction (30 MCU-friendly features)
    - gate inference (RF predict)
    - posture inference (RF predict on accepted windows)
    - total end-to-end
    - per-window average
"""
import argparse
import pickle
import time
import sys
import os

import numpy as np
import pandas as pd

import config as cfg
from src.data_loader import load_unlabelled_file
from src.feature_extraction import (
    extract_features, calibrate_standing, is_standing, CALIB_FEAT_NAMES,
)


def load_pipeline(metadata_path=cfg.METADATA_PATH,
                  gate_path=cfg.GATE_MODEL_PATH,
                  posture_path=cfg.POSTURE_MODEL_PATH):
    if not all(os.path.exists(p) for p in [metadata_path, gate_path, posture_path]):
        print("ERROR: Model checkpoints not found. Run train.py first.")
        sys.exit(1)
    with open(metadata_path,  "rb") as f: meta      = pickle.load(f)
    with open(gate_path,      "rb") as f: gate_clf   = pickle.load(f)
    with open(posture_path,   "rb") as f: posture_clf= pickle.load(f)
    return gate_clf, posture_clf, meta


def run_inference(fpath: str, gate_clf, posture_clf, pipeline_meta: dict,
                  bench_iters: int = 1,
                  calib_window: np.ndarray | None = None) -> dict:
    """
    Full pipeline inference on one CSV file.

    Parameters
    ----------
    calib_window : ndarray, shape (N_samples, 9) or (K, T, 9), optional
        If provided, the calibration-based standing gate runs BEFORE the RF gate.
        Pass raw sensor rows (or windows) from a standing-calibration recording.
        calib_info is computed with calibrate_standing(calib_window) and is
        a dict with "norm_feats" (N_windows, 7), "feat_mean" (7,), "feat_std" (7,).

    Returns a dict with:
        predictions   — pd.DataFrame (one row per window)
        timing        — dict of stage timings in ms (averaged over bench_iters)
        summary       — dict with counts per output label
    """
    acc_idx         = pipeline_meta["acc_feat_idx"]
    W               = pipeline_meta["window_size"]
    S               = pipeline_meta["stride"]
    calib_feat_vecs = calibrate_standing(calib_window) if calib_window is not None else None

    timings = {k: 0.0 for k in ["preprocess", "windowing", "features", "gate", "posture"]}

    # ── warm-up pass (do not count toward timing) ─────────────────────────────
    X_raw, win_centers = load_unlabelled_file(fpath, W, S)
    if len(X_raw) == 0:
        print(f"WARNING: No windows extracted from {fpath}")
        return {"predictions": pd.DataFrame(), "timing": timings, "summary": {}}

    Xf_warm = extract_features(X_raw)
    _ = gate_clf.predict(Xf_warm)
    valid_warm = _ == 0
    if valid_warm.any():
        _ = posture_clf.predict(Xf_warm[valid_warm][:, acc_idx])

    # ── timed passes ──────────────────────────────────────────────────────────
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        X_raw, win_centers = load_unlabelled_file(fpath, W, S)
        timings["preprocess"] += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        # windowing already done in load_unlabelled_file; log separately below
        timings["windowing"] += 0.0   # included in preprocess above; split below

        t0 = time.perf_counter()
        Xf = extract_features(X_raw)
        timings["features"] += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        # Stage 0: calibration standing gate (skips RF for standing windows)
        if calib_feat_vecs is not None:
            standing_mask = is_standing(X_raw, calib_feat_vecs)  # (N,) bool
        else:
            standing_mask = np.zeros(len(Xf), dtype=bool)
        gate_pred = np.ones(len(Xf), dtype=int)            # default: invalid
        rf_mask   = ~standing_mask                         # windows that reach RF
        gate_pred[rf_mask] = gate_clf.predict(Xf[rf_mask])
        timings["gate"] += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        final_pred = np.full(len(Xf), cfg.OUTPUT_IGNORE, dtype=int)
        valid_mask = gate_pred == 0
        if valid_mask.any():
            final_pred[valid_mask] = posture_clf.predict(Xf[valid_mask][:, acc_idx])
        timings["posture"] += (time.perf_counter() - t0) * 1000

    # ── average over iterations ────────────────────────────────────────────────
    for k in timings:
        timings[k] /= bench_iters

    n_windows = len(Xf)
    total_ms  = sum(timings.values())
    timings["total"]        = total_ms
    timings["per_window"]   = total_ms / max(n_windows, 1)

    # ── final prediction for reporting ────────────────────────────────────────
    if calib_feat_vecs is not None:
        standing_mask = is_standing(X_raw, calib_feat_vecs)
    else:
        standing_mask = np.zeros(len(Xf), dtype=bool)
    gate_pred  = np.ones(len(Xf), dtype=int)
    rf_mask    = ~standing_mask
    gate_pred[rf_mask] = gate_clf.predict(Xf[rf_mask])
    final_pred = np.full(len(Xf), cfg.OUTPUT_IGNORE, dtype=int)
    valid_mask = gate_pred == 0
    if valid_mask.any():
        final_pred[valid_mask] = posture_clf.predict(Xf[valid_mask][:, acc_idx])

    posture_names = pipeline_meta["posture_names"]
    label_map = {**{k: v for k, v in posture_names.items()}, cfg.OUTPUT_IGNORE: "ignore"}

    def _gate_label(i):
        if standing_mask[i]: return "standing"
        return "accept" if gate_pred[i] == 0 else "reject"

    preds_df = pd.DataFrame({
        "window_idx":        np.arange(n_windows),
        "center_sample":     win_centers,
        "gate_decision":     [_gate_label(i) for i in range(n_windows)],
        "posture_label_int": final_pred,
        "posture_label":     [label_map.get(p, "unknown") for p in final_pred],
    })

    summary = {
        "total_windows":    n_windows,
        "gate_accepted":    int(valid_mask.sum()),
        "gate_rejected":    int((~valid_mask).sum()),
        "calib_standing":   int(standing_mask.sum()),
    }
    for pl, pn in posture_names.items():
        summary[pn] = int((final_pred == pl).sum())
    summary["ignore"] = int((final_pred == cfg.OUTPUT_IGNORE).sum())

    return {"predictions": preds_df, "timing": timings, "summary": summary}


def print_report(fpath: str, result: dict, bench_iters: int):
    t = result["timing"]
    s = result["summary"]

    print("=" * 60)
    print(f"  BACKY MCU Pipeline — Inference Report")
    print(f"  File: {os.path.basename(fpath)}")
    print("=" * 60)
    print()
    print(f"  Windows processed  : {s['total_windows']}")
    if s.get('calib_standing', 0) > 0:
        print(f"  Standing (calib)   : {s['calib_standing']}  "
              f"({s['calib_standing']/max(s['total_windows'],1)*100:.1f}%)  ← skipped RF gate")
    print(f"  RF gate accepted   : {s['gate_accepted']}  "
          f"({s['gate_accepted']/max(s['total_windows'],1)*100:.1f}%)")
    print(f"  RF gate rejected   : {s['gate_rejected']}  "
          f"({s['gate_rejected']/max(s['total_windows'],1)*100:.1f}%)")
    print()
    print("  Predicted posture distribution:")
    for pn in cfg.POSTURE_NAMES.values():
        n = s.get(pn, 0)
        pct = n / max(s['gate_accepted'], 1) * 100
        print(f"    {pn:<18}: {n:4d} windows  ({pct:.1f}% of accepted)")
    print()
    print(f"  -- Timing (avg over {bench_iters} iteration(s)) --")
    print(f"    Preprocessing     : {t['preprocess']:.2f} ms")
    print(f"    Feature extraction: {t['features']:.2f} ms")
    print(f"    Gate inference    : {t['gate']:.2f} ms")
    print(f"    Posture inference : {t['posture']:.2f} ms")
    print(f"    ---------------------------------")
    print(f"    Total             : {t['total']:.2f} ms  ({s['total_windows']} windows)")
    print(f"    Per window        : {t['per_window']:.3f} ms")
    print("=" * 60)
    print()

    print("  Window-level predictions (first 20):")
    print(result["predictions"].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="BACKY MCU pipeline inference")
    parser.add_argument("input_file", help="Path to sensor CSV file")
    parser.add_argument("--output",       default=None,  help="Save predictions to CSV")
    parser.add_argument("--bench-iters",  type=int, default=50,
                        help="Timing iterations for stable measurement (default: 50)")
    parser.add_argument("--calib-csv", default=None,
                        help="CSV file recorded while standing still (first WINDOW_SIZE "
                             "rows used as calibration for the standing gate)")
    args = parser.parse_args()

    gate_clf, posture_clf, pipeline_meta = load_pipeline()
    print(f"Loaded pipeline checkpoints.")
    print(f"  Gate    : {cfg.GATE_N_TREES} trees, depth {cfg.GATE_MAX_DEPTH}")
    print(f"  Posture : {cfg.POSTURE_N_TREES} trees, depth {cfg.POSTURE_MAX_DEPTH}, "
          f"{len(pipeline_meta['acc_feat_idx'])} ACC features")

    calib_window = None
    if args.calib_csv:
        from src.data_loader import load_unlabelled_file
        W = pipeline_meta["window_size"]
        # Load all windows from the calib CSV — calibrate_standing handles (K, T, 9)
        calib_raw, _ = load_unlabelled_file(args.calib_csv, W, cfg.STRIDE)
        if len(calib_raw) > 0:
            calib_window   = calib_raw                  # (K, W, 9) — all windows
            calib_preview  = calibrate_standing(calib_window)
            n_ref          = calib_preview["n_windows"]
            raw_means      = calib_preview["feat_mean"]
            feat_str = "  ".join(
                f"{n}={raw_means[i]:.3f}"
                for i, n in enumerate(CALIB_FEAT_NAMES)
            )
            print(f"  Standing calib: {n_ref} reference windows  |  "
                  f"ref_dist={calib_preview['ref_dist']:.3f}  (max pairwise L1, z-scored 7-D space)")
            print(f"    calib means: {feat_str}")
        else:
            print("  WARNING: calib CSV too short — standing gate disabled")
    print()

    result = run_inference(
        args.input_file, gate_clf, posture_clf, pipeline_meta,
        bench_iters=args.bench_iters,
        calib_window=calib_window,
    )
    print_report(args.input_file, result, args.bench_iters)

    if args.output:
        result["predictions"].to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
