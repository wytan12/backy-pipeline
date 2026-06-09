"""
evaluate.py — LOUO and LODO evaluation of the trained pipeline.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --mode louo
    uv run python evaluate.py --mode lodo
    uv run python evaluate.py --mode both
"""
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

import config as cfg
from data_loader import load_directory, apply_label_remap, extract_windows
from feature_extraction import extract_features, ACC_FEAT_IDX


def load_and_prepare():
    from data_loader import load_directory
    import pandas as pd
    dyn  = load_directory(cfg.DYN_DIR,  "dynamic")
    stat = load_directory(cfg.STAT_DIR, "static")
    raw  = pd.concat([d for d in [dyn, stat] if len(d) > 0], ignore_index=True)
    raw  = apply_label_remap(raw)
    X_raw, meta = extract_windows(raw)
    Xf   = extract_features(X_raw)
    gate_y = meta["gate_label"].values.astype(int)
    return Xf, gate_y, meta


def gate_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "valid_accept_rate":   tn / (tn + fp) if (tn + fp) else 0,
        "invalid_reject_rate": tp / (tp + fn) if (tp + fn) else 0,
        "false_accept_rate":   fp / (fp + tn) if (fp + tn) else 0,
        "false_reject_rate":   fn / (fn + tp) if (fn + tp) else 0,
        "accuracy":            accuracy_score(y_true, y_pred),
    }


def run_fold(Xf, gate_y, meta, train_m, test_m):
    # gate
    g = RandomForestClassifier(n_estimators=cfg.GATE_N_TREES, max_depth=cfg.GATE_MAX_DEPTH,
                               min_samples_leaf=cfg.GATE_MIN_LEAF, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    g.fit(Xf[train_m], gate_y[train_m])
    gate_pred = g.predict(Xf[test_m])
    gm = gate_metrics(gate_y[test_m], gate_pred)

    # posture
    valid_train = train_m & (gate_y == 0)
    p = RandomForestClassifier(n_estimators=cfg.POSTURE_N_TREES, max_depth=cfg.POSTURE_MAX_DEPTH,
                               min_samples_leaf=cfg.POSTURE_MIN_LEAF, random_state=42, n_jobs=-1)
    p.fit(Xf[valid_train][:, ACC_FEAT_IDX],
          meta.loc[valid_train, "posture_label"].astype(int).values)

    valid_test = test_m & (gate_y == 0)
    p_pred = p.predict(Xf[valid_test][:, ACC_FEAT_IDX])
    p_true = meta.loc[valid_test, "posture_label"].astype(int).values

    pm = {
        "posture_accuracy": accuracy_score(p_true, p_pred),
        "posture_macro_f1": f1_score(p_true, p_pred, average="macro", zero_division=0),
    }
    for pl, pn in cfg.POSTURE_NAMES.items():
        mask = p_true == pl
        pm[f"recall_{pn[:4]}"] = (p_pred[mask] == pl).mean() if mask.sum() else float("nan")

    return {**gm, **pm}


def evaluate_louo(Xf, gate_y, meta):
    print("=" * 56)
    print("  Leave-One-User-Out (LOUO) Evaluation")
    print("=" * 56)
    rows = []
    for u in sorted(meta["user"].unique()):
        test_m  = (meta["user"] == u).values
        train_m = ~test_m
        r = run_fold(Xf, gate_y, meta, train_m, test_m)
        r["held_out"] = u
        rows.append(r)
        print(f"  {u:<10}  gate_acc={r['accuracy']:.3f}  "
              f"valid_accept={r['valid_accept_rate']:.3f}  "
              f"inv_reject={r['invalid_reject_rate']:.3f}  "
              f"posture_acc={r['posture_accuracy']:.3f}  "
              f"posture_f1={r['posture_macro_f1']:.3f}")

    df = pd.DataFrame(rows).set_index("held_out")
    print()
    print("  Mean across users:")
    print(df.mean().round(3).to_string())
    return df


def evaluate_lodo(Xf, gate_y, meta):
    print("=" * 56)
    print("  Leave-One-Device-Out (LODO) Evaluation")
    print("=" * 56)
    rows = []
    for d in sorted(meta["device"].unique()):
        test_m  = (meta["device"] == d).values
        train_m = ~test_m
        if train_m.sum() < 10 or test_m.sum() < 10:
            continue
        r = run_fold(Xf, gate_y, meta, train_m, test_m)
        r["held_out"] = d
        rows.append(r)
        print(f"  {d:<20}  gate_acc={r['accuracy']:.3f}  "
              f"posture_acc={r['posture_accuracy']:.3f}  "
              f"posture_f1={r['posture_macro_f1']:.3f}")

    df = pd.DataFrame(rows).set_index("held_out")
    print()
    print("  Mean across devices:")
    print(df.mean().round(3).to_string())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["louo", "lodo", "both"], default="both")
    args = parser.parse_args()

    print("Loading data and extracting features...")
    Xf, gate_y, meta = load_and_prepare()
    print(f"  {len(Xf):,} windows, {Xf.shape[1]} features\n")

    if args.mode in ("louo", "both"):
        louo_df = evaluate_louo(Xf, gate_y, meta)
        print()
    if args.mode in ("lodo", "both"):
        lodo_df = evaluate_lodo(Xf, gate_y, meta)


if __name__ == "__main__":
    main()
