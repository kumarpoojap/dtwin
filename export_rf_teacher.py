<<<<<<< HEAD
#!/usr/bin/env python3
"""
export_rf_teacher.py

Bundle a trained RandomForest teacher with its metadata for the Hybrid PINN trainer.

Outputs a single joblib file (a plain dict) containing:
- model (sklearn estimator)
- feature_columns (ordered)
- target_columns (ordered)
- target_normalization_stats (TRAIN-only stats), optional
- k_ahead (steps), cadence_seconds
- versions, timestamp, fingerprints

Usage:
  python export_rf_teacher.py \
    --rf-model ./artifacts/model_random_forest.pkl \
    --feature-cols ./artifacts/feature_columns.json \
    --target-cols ./artifacts/targets_used.json \
    --target-stats ./artifacts/target_normalization_stats.json \
    --k-ahead 12 \
    --cadence-s 10 \
    --out ./artifacts_rf/model_random_forest.pkl
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from src.models.feature_builder import materialize_features_from_feature_list

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def fingerprint_list(items) -> str:
    """Deterministic fingerprint over an ordered list of strings."""
    return sha256_str("|".join([str(x) for x in items]))

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="Export a RandomForest teacher bundle for Hybrid PINN.")
    ap.add_argument("--rf-model", required=True, help="Path to trained RF model (joblib/pkl).")
    ap.add_argument("--feature-cols", required=True, help="Path to JSON list of feature columns used at train time.")
    ap.add_argument("--target-cols", required=True, help="Path to JSON list of target columns used at train time.")
    ap.add_argument("--target-stats", default=None, help="Path to target normalization stats JSON (TRAIN-only).")
    ap.add_argument("--feature-winsor-bounds", default=None, help="Optional path to feature winsor bounds JSON (TRAIN-only).")
    ap.add_argument("--k-ahead", type=int, default=12, help="Horizon steps (k). With 10s cadence, 12 ≈ 120s.")
    ap.add_argument("--cadence-s", type=float, default=10.0, help="Cadence in seconds.")
    ap.add_argument("--out", default="artifacts/model_random_forest.pkl", help="Output teacher bundle path.")
    ap.add_argument("--sample-parquet", default=None, help="Optional parquet to sanity-check predict() shapes/columns.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # --- Load model and metadata ---
    print(f"[INFO] Loading RF model: {args.rf_model}")
    rf_model = joblib.load(args.rf_model)

    print(f"[INFO] Loading feature columns: {args.feature_cols}")
    feature_cols = load_json(args.feature_cols)
    print(f"[INFO] Loading target columns: {args.target_cols}")
    target_cols = load_json(args.target_cols)

    target_stats = None
    if args.target_stats and os.path.exists(args.target_stats):
        print(f"[INFO] Loading target normalization stats: {args.target_stats}")
        target_stats = load_json(args.target_stats)

    feature_winsor_bounds = None
    if args.feature_winsor_bounds and os.path.exists(args.feature_winsor_bounds):
        print(f"[INFO] Loading feature winsor bounds: {args.feature_winsor_bounds}")
        feature_winsor_bounds = load_json(args.feature_winsor_bounds)

    # --- Version stamping ---
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"

    bundle = {
        "type": "rf_teacher_bundle",
        "framework": "sklearn",
        "sklearn_version": sklearn_version,
        "python_version": sys.version.split()[0],
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "created_at_epoch": int(time.time()),
        "k_ahead": int(args.k_ahead),
        "cadence_seconds": float(args.cadence_s),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "target_normalization_stats": target_stats,  # may be None
        "feature_winsor_bounds": feature_winsor_bounds,  # may be None
        "feature_fingerprint": fingerprint_list(feature_cols),
        "target_fingerprint": fingerprint_list(target_cols),
        # Save the actual estimator
        "model": rf_model,
    }

    # --- Optional sanity check with a sample parquet ---
    if args.sample_parquet:
        try:
            print(f"[INFO] Running sanity check with: {args.sample_parquet}")
            df = pd.read_parquet(args.sample_parquet)

            # Find timestamp index if present
            ts_candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "datetime")]
            if ts_candidates:
                df[ts_candidates[0]] = pd.to_datetime(df[ts_candidates[0]], errors="coerce")
                df = df.set_index(ts_candidates[0]).sort_index()

            # Materialize engineered features (e.g., *_lagN) on-the-fly based on feature_cols.
            X_feat = materialize_features_from_feature_list(df, feature_cols=feature_cols)

            missing = [c for c in feature_cols if c not in X_feat.columns]
            if missing:
                print(f"[WARN] Could not materialize {len(missing)} feature cols from sample parquet; skipping predict sanity check.")
            else:
                X = X_feat.dropna().head(8)
                if len(X) >= 1:
                    y_hat = rf_model.predict(X)
                    assert y_hat.shape[1] == len(target_cols), \
                        f"Pred shape mismatch: got {y_hat.shape}, expected target count {len(target_cols)}"
                    print(f"[OK] Predict sanity check passed. y_hat shape: {y_hat.shape}")
        except Exception as e:
            print(f"[WARN] Sanity check failed (continuing): {e}")

    # --- Save bundle ---
    joblib.dump(bundle, args.out)
    print(f"[INFO] Saved teacher bundle: {args.out}")

    # --- Print quick summary ---
    print("----- Teacher Bundle Summary -----")
    print(f"type           : {bundle['type']}")
    print(f"framework      : {bundle['framework']} (sklearn {bundle['sklearn_version']})")
    print(f"k_ahead        : {bundle['k_ahead']} steps  (~{bundle['k_ahead']*bundle['cadence_seconds']} s)")
    print(f"cadence_seconds: {bundle['cadence_seconds']}")
    print(f"#features      : {len(feature_cols)}")
    print(f"#targets       : {len(target_cols)}")
    print(f"feature_fp     : {bundle['feature_fingerprint'][:12]}...")
    print(f"target_fp      : {bundle['target_fingerprint'][:12]}...")
    print("----------------------------------")

if __name__ == "__main__":
=======
#!/usr/bin/env python3
"""
export_rf_teacher.py

Bundle a trained RandomForest teacher with its metadata for the Hybrid PINN trainer.

Outputs a single joblib file (a plain dict) containing:
- model (sklearn estimator)
- feature_columns (ordered)
- target_columns (ordered)
- target_normalization_stats (TRAIN-only stats), optional
- k_ahead (steps), cadence_seconds
- versions, timestamp, fingerprints

Usage:
  python export_rf_teacher.py \
    --rf-model ./artifacts/model_random_forest.pkl \
    --feature-cols ./artifacts/feature_columns.json \
    --target-cols ./artifacts/targets_used.json \
    --target-stats ./artifacts/target_normalization_stats.json \
    --k-ahead 12 \
    --cadence-s 10 \
    --out ./artifacts_rf/model_random_forest.pkl
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from src.models.feature_builder import materialize_features_from_feature_list

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def fingerprint_list(items) -> str:
    """Deterministic fingerprint over an ordered list of strings."""
    return sha256_str("|".join([str(x) for x in items]))

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="Export a RandomForest teacher bundle for Hybrid PINN.")
    ap.add_argument("--rf-model", required=True, help="Path to trained RF model (joblib/pkl).")
    ap.add_argument("--feature-cols", required=True, help="Path to JSON list of feature columns used at train time.")
    ap.add_argument("--target-cols", required=True, help="Path to JSON list of target columns used at train time.")
    ap.add_argument("--target-stats", default=None, help="Path to target normalization stats JSON (TRAIN-only).")
    ap.add_argument("--feature-winsor-bounds", default=None, help="Optional path to feature winsor bounds JSON (TRAIN-only).")
    ap.add_argument("--k-ahead", type=int, default=12, help="Horizon steps (k). With 10s cadence, 12 ≈ 120s.")
    ap.add_argument("--cadence-s", type=float, default=10.0, help="Cadence in seconds.")
    ap.add_argument("--out", default="artifacts/model_random_forest.pkl", help="Output teacher bundle path.")
    ap.add_argument("--sample-parquet", default=None, help="Optional parquet to sanity-check predict() shapes/columns.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # --- Load model and metadata ---
    print(f"[INFO] Loading RF model: {args.rf_model}")
    rf_model = joblib.load(args.rf_model)

    print(f"[INFO] Loading feature columns: {args.feature_cols}")
    feature_cols = load_json(args.feature_cols)
    print(f"[INFO] Loading target columns: {args.target_cols}")
    target_cols = load_json(args.target_cols)

    target_stats = None
    if args.target_stats and os.path.exists(args.target_stats):
        print(f"[INFO] Loading target normalization stats: {args.target_stats}")
        target_stats = load_json(args.target_stats)

    feature_winsor_bounds = None
    if args.feature_winsor_bounds and os.path.exists(args.feature_winsor_bounds):
        print(f"[INFO] Loading feature winsor bounds: {args.feature_winsor_bounds}")
        feature_winsor_bounds = load_json(args.feature_winsor_bounds)

    # --- Version stamping ---
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"

    bundle = {
        "type": "rf_teacher_bundle",
        "framework": "sklearn",
        "sklearn_version": sklearn_version,
        "python_version": sys.version.split()[0],
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "created_at_epoch": int(time.time()),
        "k_ahead": int(args.k_ahead),
        "cadence_seconds": float(args.cadence_s),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "target_normalization_stats": target_stats,  # may be None
        "feature_winsor_bounds": feature_winsor_bounds,  # may be None
        "feature_fingerprint": fingerprint_list(feature_cols),
        "target_fingerprint": fingerprint_list(target_cols),
        # Save the actual estimator
        "model": rf_model,
    }

    # --- Optional sanity check with a sample parquet ---
    if args.sample_parquet:
        try:
            print(f"[INFO] Running sanity check with: {args.sample_parquet}")
            df = pd.read_parquet(args.sample_parquet)

            # Find timestamp index if present
            ts_candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "datetime")]
            if ts_candidates:
                df[ts_candidates[0]] = pd.to_datetime(df[ts_candidates[0]], errors="coerce")
                df = df.set_index(ts_candidates[0]).sort_index()

            # Materialize engineered features (e.g., *_lagN) on-the-fly based on feature_cols.
            X_feat = materialize_features_from_feature_list(df, feature_cols=feature_cols)

            missing = [c for c in feature_cols if c not in X_feat.columns]
            if missing:
                print(f"[WARN] Could not materialize {len(missing)} feature cols from sample parquet; skipping predict sanity check.")
            else:
                X = X_feat.dropna().head(8)
                if len(X) >= 1:
                    y_hat = rf_model.predict(X)
                    assert y_hat.shape[1] == len(target_cols), \
                        f"Pred shape mismatch: got {y_hat.shape}, expected target count {len(target_cols)}"
                    print(f"[OK] Predict sanity check passed. y_hat shape: {y_hat.shape}")
        except Exception as e:
            print(f"[WARN] Sanity check failed (continuing): {e}")

    # --- Save bundle ---
    joblib.dump(bundle, args.out)
    print(f"[INFO] Saved teacher bundle: {args.out}")

    # --- Print quick summary ---
    print("----- Teacher Bundle Summary -----")
    print(f"type           : {bundle['type']}")
    print(f"framework      : {bundle['framework']} (sklearn {bundle['sklearn_version']})")
    print(f"k_ahead        : {bundle['k_ahead']} steps  (~{bundle['k_ahead']*bundle['cadence_seconds']} s)")
    print(f"cadence_seconds: {bundle['cadence_seconds']}")
    print(f"#features      : {len(feature_cols)}")
    print(f"#targets       : {len(target_cols)}")
    print(f"feature_fp     : {bundle['feature_fingerprint'][:12]}...")
    print(f"target_fp      : {bundle['target_fingerprint'][:12]}...")
    print("----------------------------------")

if __name__ == "__main__":
>>>>>>> ecf8702 (feat: Consolidate shared utilities into src/common package)
    main()