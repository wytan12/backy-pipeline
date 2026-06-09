"""
train.py — Train the BACKY MCU two-stage pipeline and save model checkpoints.

Usage:
    uv run python train.py
    uv run python train.py --dyn-dir path/to/dynamic --stat-dir path/to/static

Saved files:
    models/gate_rf.pkl         — RF gate (valid vs invalid)
    models/posture_rf.pkl      — RF posture classifier (acc-only, valid windows)
    models/pipeline_metadata.pkl — feature names, window config, label maps
"""
import argparse
import time
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

import config as cfg
from src.data_loader import load_directory, apply_label_remap, extract_windows
from src.feature_extraction import extract_features, FEAT_NAMES, ACC_FEAT_IDX


def load_all_data(dyn_dir: str, stat_dir: str) -> tuple:
    print("Loading data...")
    dyn  = load_directory(dyn_dir,  "dynamic")
    stat = load_directory(stat_dir, "static")
    raw  = _concat(dyn, stat)
    raw  = apply_label_remap(raw)

    print(f"  Dynamic rows : {len(dyn):,}")
    print(f"  Static  rows : {len(stat):,}")
    print(f"  Total   rows : {len(raw):,}")
    print(f"  Users        : {sorted(raw['user'].unique())}")
    print(f"  Devices      : {len(raw['device'].unique())} unique devices")
    print()
    return raw


def _concat(*dfs):
    import pandas as pd
    return pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)


def build_features(raw) -> tuple:
    print("Extracting windows and features...")
    X_raw, meta = extract_windows(raw)
    Xf = extract_features(X_raw)
    gate_y = meta["gate_label"].values.astype(int)

    print(f"  Windows  : {len(Xf):,}  ({Xf.shape[1]} features each)")
    print(f"  Valid    : {(gate_y==0).sum():,}")
    print(f"  Invalid  : {(gate_y==1).sum():,}")
    print()
    return Xf, gate_y, meta


def train_gate(Xf: np.ndarray, gate_y: np.ndarray, meta) -> RandomForestClassifier:
    # To exclude standing from gate training (improves twisting acceptance +8.9%
    # but standing RF rejection drops −12.8% — only viable if user always calibrates):
    # no_stand  = meta["orig_label"].values != 7
    # n_excluded = int((~no_stand).sum())
    # Xf_tr   = Xf[no_stand]
    # gate_tr = gate_y[no_stand]
    # print(f"Training gate (all {Xf.shape[1]} features, standing excluded={n_excluded} windows)...")

    # Current: train WITH standing (RF gate acts as fallback if user skips calibration)
    Xf_tr   = Xf
    gate_tr = gate_y
    print(f"Training gate (all {Xf.shape[1]} features, standing included)...")
    clf = RandomForestClassifier(
        n_estimators = cfg.GATE_N_TREES,
        max_depth    = cfg.GATE_MAX_DEPTH,
        min_samples_leaf = cfg.GATE_MIN_LEAF,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )
    t0 = time.perf_counter()
    clf.fit(Xf_tr, gate_tr)
    elapsed = time.perf_counter() - t0

    pred = clf.predict(Xf_tr)
    acc  = accuracy_score(gate_tr, pred)
    print(f"  Train time : {elapsed:.2f}s")
    print(f"  Train acc  : {acc:.4f}  (in-sample, not generalisation)")
    print(f"  Trees      : {cfg.GATE_N_TREES},  depth={cfg.GATE_MAX_DEPTH}")
    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
    print(f"  Total nodes: {total_nodes}")
    print()
    return clf


def train_posture_classifier(Xf: np.ndarray, gate_y: np.ndarray, meta) -> RandomForestClassifier:
    print("Training posture classifier (ACC-only features, valid windows)...")
    valid_mask = gate_y == 0
    Xf_valid   = Xf[valid_mask][:, ACC_FEAT_IDX]
    y_posture  = meta.loc[valid_mask, "posture_label"].astype(int).values

    print(f"  Valid windows per class:")
    for pl, pn in cfg.POSTURE_NAMES.items():
        print(f"    {pn:<15}: {(y_posture==pl).sum():,}")

    clf = RandomForestClassifier(
        n_estimators = cfg.POSTURE_N_TREES,
        max_depth    = cfg.POSTURE_MAX_DEPTH,
        min_samples_leaf = cfg.POSTURE_MIN_LEAF,
        random_state = 42,
        n_jobs       = -1,
    )
    t0 = time.perf_counter()
    clf.fit(Xf_valid, y_posture)
    elapsed = time.perf_counter() - t0

    pred = clf.predict(Xf_valid)
    acc  = accuracy_score(y_posture, pred)
    mf1  = f1_score(y_posture, pred, average="macro")
    print(f"  Train time : {elapsed:.2f}s")
    print(f"  Train acc  : {acc:.4f}  macro-F1={mf1:.4f}  (in-sample)")
    print(f"  Trees      : {cfg.POSTURE_N_TREES},  depth={cfg.POSTURE_MAX_DEPTH}")
    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
    print(f"  Total nodes: {total_nodes}")
    print()
    return clf


def save_models(gate_clf, posture_clf):
    metadata = {
        "feat_names":    FEAT_NAMES,
        "acc_feat_idx":  ACC_FEAT_IDX,
        "window_size":   cfg.WINDOW_SIZE,
        "stride":        cfg.STRIDE,
        "posture_names": cfg.POSTURE_NAMES,
        "invalid_names": cfg.INVALID_NAMES,
        "remap":         cfg.REMAP,
        "fsr_threshold": cfg.FSR_THRESHOLD,
        "gate_params":   dict(n_trees=cfg.GATE_N_TREES, max_depth=cfg.GATE_MAX_DEPTH),
        "posture_params":dict(n_trees=cfg.POSTURE_N_TREES, max_depth=cfg.POSTURE_MAX_DEPTH),
    }
    import os
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    with open(cfg.GATE_MODEL_PATH,    "wb") as f: pickle.dump(gate_clf,    f)
    with open(cfg.POSTURE_MODEL_PATH, "wb") as f: pickle.dump(posture_clf, f)
    with open(cfg.METADATA_PATH,      "wb") as f: pickle.dump(metadata,    f)

    print(f"Saved:")
    print(f"  {cfg.GATE_MODEL_PATH}")
    print(f"  {cfg.POSTURE_MODEL_PATH}")
    print(f"  {cfg.METADATA_PATH}")
    print()


def quick_louo_report(Xf: np.ndarray, gate_y: np.ndarray, meta):
    """One-pass LOUO on gate + posture to give generalisation numbers."""
    from sklearn.metrics import confusion_matrix

    print("Quick LOUO evaluation...")
    users = sorted(meta["user"].unique())
    gate_rows, posture_rows = [], []

    for u in users:
        test_m  = (meta["user"] == u).values
        train_m = ~test_m

        # To exclude standing (match train_gate() no-standing variant):
        # no_stand_tr = train_m & (meta["orig_label"].values != 7)
        # g.fit(Xf[no_stand_tr], gate_y[no_stand_tr])
        g = RandomForestClassifier(n_estimators=cfg.GATE_N_TREES, max_depth=cfg.GATE_MAX_DEPTH,
                                   min_samples_leaf=cfg.GATE_MIN_LEAF, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
        g.fit(Xf[train_m], gate_y[train_m])
        g_pred = g.predict(Xf[test_m])
        g_true = gate_y[test_m]

        tn, fp, fn, tp = confusion_matrix(g_true, g_pred, labels=[0,1]).ravel()
        gate_rows.append({
            "user": u,
            "valid_accept":   tn/(tn+fp) if (tn+fp) else 0,
            "invalid_reject": tp/(tp+fn) if (tp+fn) else 0,
            "accuracy":       accuracy_score(g_true, g_pred),
        })

        valid_train = train_m & (gate_y == 0)
        p = RandomForestClassifier(n_estimators=cfg.POSTURE_N_TREES, max_depth=cfg.POSTURE_MAX_DEPTH,
                                   min_samples_leaf=cfg.POSTURE_MIN_LEAF, random_state=42, n_jobs=-1)
        p.fit(Xf[valid_train][:, ACC_FEAT_IDX],
              meta.loc[valid_train, "posture_label"].astype(int).values)

        valid_test = test_m & (gate_y == 0)
        p_pred = p.predict(Xf[valid_test][:, ACC_FEAT_IDX])
        p_true = meta.loc[valid_test, "posture_label"].astype(int).values
        posture_rows.append({
            "user":     u,
            "accuracy": accuracy_score(p_true, p_pred),
            "macro_f1": f1_score(p_true, p_pred, average="macro", zero_division=0),
        })

    import pandas as pd
    gdf = pd.DataFrame(gate_rows).set_index("user")
    pdf = pd.DataFrame(posture_rows).set_index("user")

    print("\n  Gate LOUO:")
    print(gdf.round(3).to_string())
    print(f"  Mean: {gdf.mean().round(3).to_dict()}")
    print("\n  Posture LOUO (ACC-only):")
    print(pdf.round(3).to_string())
    print(f"  Mean: {pdf.mean().round(3).to_dict()}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train BACKY MCU pipeline")
    parser.add_argument("--dyn-dir",  default=cfg.DYN_DIR)
    parser.add_argument("--stat-dir", default=cfg.STAT_DIR)
    parser.add_argument("--skip-louo", action="store_true",
                        help="Skip LOUO evaluation (faster)")
    args = parser.parse_args()

    raw        = load_all_data(args.dyn_dir, args.stat_dir)
    Xf, gate_y, meta = build_features(raw)
    gate_clf   = train_gate(Xf, gate_y, meta)
    posture_clf= train_posture_classifier(Xf, gate_y, meta)
    save_models(gate_clf, posture_clf)

    if not args.skip_louo:
        quick_louo_report(Xf, gate_y, meta)

    print("Training complete.")


if __name__ == "__main__":
    main()
