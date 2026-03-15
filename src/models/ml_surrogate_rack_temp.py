# surrogate_rack_temp.py
# First ML surrogate to predict rack temperatures from in-row cooler telemetry.
# Author: Pooja-friendly baseline (multi-output RF + lag features)
#
# Usage:
#   python surrogate_rack_temp.py --parquet merged_datacenter.parquet --spec feature_target_spec.json --outdir artifacts
#
# Notes:
# - Avoids leakage by computing normalization stats on TRAIN only (if --target-mode normalized).
# - Uses lag features on inputs (not on targets) to capture short-term dynamics.
# - Time-based splitting (70/15/15).
# - Saves model + metrics + plots.

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.validation import check_is_fitted

from feature_builder import (
    apply_winsorization,
    build_supervised_dataset,
    compute_winsor_bounds,
    drop_near_constant_features,
)

plt.style.use("seaborn-v0_8")

@dataclass
class Spec:
    resample_rule: str
    feature_cols: List[str]
    target_cols_raw: List[str]
    target_cols_normalized: List[str]
    notes: List[str]

def load_spec(path: str) -> Spec:
    with open(path, "r") as f:
        obj = json.load(f)
    # Basic validation
    required = ["resample_rule", "feature_cols", "target_cols_raw", "target_cols_normalized"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing '{k}' in spec.")
    return Spec(
        resample_rule=obj["resample_rule"],
        feature_cols=obj["feature_cols"],
        target_cols_raw=obj["target_cols_raw"],
        target_cols_normalized=obj["target_cols_normalized"],
        notes=obj.get("notes", []),
    )

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # If the DataFrame already has a DatetimeIndex, keep it as-is.
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    # Detect timestamp col
    ts_candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "datetime")]
    if len(ts_candidates) == 0:
        # If the first column looks like a datetime, try parsing that
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col])
            df = df.set_index(first_col).sort_index()
            return df
        except Exception:
            pass
        raise ValueError("No timestamp-like column found (expected 'timestamp' or similar).")
    ts_col = ts_candidates[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    return df

def resample_and_clean(
    df: pd.DataFrame,
    rule: str,
    strict: bool = False,
    max_ffill_steps: int = 6,
    max_interp_seconds: int = 60,
) -> pd.DataFrame:
    """
    Resample to uniform cadence and handle small gaps.
    If strict=True: use mean() and forward-fill only (no interpolation).
    Else: also perform short linear interpolation for small gaps.
    """
    # Resample to regular grid
    df_res = df.resample(rule).mean()

    # Forward fill small gaps (sensor dropouts), limit in steps (e.g., 6 * 10s = 60s)
    df_res = df_res.ffill(limit=max_ffill_steps)

    if not strict:
        # Interpolate remaining small gaps limited by time window
        try:
            df_res = df_res.interpolate(method="time", limit=int(max_interp_seconds / 10), limit_direction="forward")
        except Exception:
            # Fallback if index not strictly time-like
            df_res = df_res.interpolate(limit=int(max_interp_seconds / 10), limit_direction="forward")

    # Drop rows still having all-NaNs
    df_res = df_res.dropna(how="all")
    return df_res

def time_split_index(n: int, train_frac=0.7, val_frac=0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    idx_all = np.arange(n)
    return idx_all[:train_end], idx_all[train_end:val_end], idx_all[val_end:]

def compute_train_target_scaler(y_train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for col in y_train_df.columns:
        mu = float(y_train_df[col].mean())
        sd = float(y_train_df[col].std(ddof=0)) or 1.0
        stats[col] = {"mean": mu, "std": sd}
    return stats

def apply_target_normalization(y_df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    z = {}
    for col in y_df.columns:
        mu = stats[col]["mean"]
        sd = stats[col]["std"]
        z[col] = (y_df[col] - mu) / sd
    return pd.DataFrame(z, index=y_df.index)

def invert_target_normalization(y_pred_z: np.ndarray, y_stats: Dict[str, Dict[str, float]], cols: List[str]) -> pd.DataFrame:
    inv = {}
    for i, col in enumerate(cols):
        mu = y_stats[col]["mean"]
        sd = y_stats[col]["std"]
        inv[col] = y_pred_z[:, i] * sd + mu
    return pd.DataFrame(inv)

def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for col in y_true.columns:
        mae = mean_absolute_error(y_true[col], y_pred[col])
        # Older sklearn may not support squared=False; compute RMSE manually.
        rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
        metrics.append({"target": col, "MAE": mae, "RMSE": rmse})
    # Overall aggregates
    mae_overall = mean_absolute_error(y_true.values.ravel(), y_pred.values.ravel())
    rmse_overall = np.sqrt(mean_squared_error(y_true.values.ravel(), y_pred.values.ravel()))
    metrics.append({"target": "__overall__", "MAE": mae_overall, "RMSE": rmse_overall})
    return pd.DataFrame(metrics)

def persistence_baseline_metrics(y_true_df: pd.DataFrame) -> pd.DataFrame:
    # Predict previous value for each column
    yhat = y_true_df.shift(1).dropna()
    aligned_true = y_true_df.loc[yhat.index]
    rows = []
    for col in y_true_df.columns:
        mae = mean_absolute_error(aligned_true[col], yhat[col])
        rmse = np.sqrt(mean_squared_error(aligned_true[col], yhat[col]))
        rows.append({"target": col, "MAE": mae, "RMSE": rmse})
    rows.append({
        "target": "__overall__",
        "MAE": mean_absolute_error(aligned_true.values.ravel(), yhat.values.ravel()),
        "RMSE": np.sqrt(mean_squared_error(aligned_true.values.ravel(), yhat.values.ravel())),
    })
    return pd.DataFrame(rows)

def persistence_k_ahead_baseline_metrics(y_df: pd.DataFrame, k: int):
    # Predict y(t+k) ≈ y(t)
    y_true_future = y_df.shift(-k).dropna()
    y_hat = y_df.loc[y_true_future.index]
    rows = []
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    for col in y_df.columns:
        mae = mean_absolute_error(y_true_future[col], y_hat[col])
        rmse = np.sqrt(mean_squared_error(y_true_future[col], y_hat[col]))
        rows.append({"target": col, "MAE": mae, "RMSE": rmse})
    rows.append({
        "target": "__overall__",
        "MAE": mean_absolute_error(y_true_future.values.ravel(), y_hat.values.ravel()),
        "RMSE": np.sqrt(mean_squared_error(y_true_future.values.ravel(), y_hat.values.ravel())),
    })
    return pd.DataFrame(rows)

def plot_feature_importance(model: RandomForestRegressor, feature_names: List[str], outpath: str, top_k: int = 25):
    check_is_fitted(model)
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color="#2a9d8f")
    plt.title("Global Feature Importance (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series, outpath: str, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(y_true.index, y_true.values, label="Actual", lw=1.5)
    plt.plot(y_true.index, y_pred.values, label="Predicted", lw=1.2, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_residuals(y_true: pd.Series, y_pred: pd.Series, outpath: str, title: str):
    res = y_true.values - y_pred.values
    plt.figure(figsize=(10, 4))
    plt.plot(y_true.index, res, lw=1)
    plt.axhline(0.0, color="k", ls="--", lw=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Residual (°C)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def validate_cadence(df, expected_seconds=10, tolerance=1):
    dt = df.index.to_series().diff().dt.total_seconds().dropna()
    off = dt[(dt < expected_seconds - tolerance) | (dt > expected_seconds + tolerance)]
    if len(off) > 0:
        print(f"[WARN] Found {len(off)} intervals off expected {expected_seconds}s cadence.")
        print(off.value_counts().head())
    else:
        print("[OK] Cadence looks uniform.")

def main():
    parser = argparse.ArgumentParser(description="First ML surrogate for rack temperature prediction.")
    parser.add_argument("--parquet", required=True, help="Path to merged parquet file.")
    parser.add_argument("--spec", required=True, help="Path to feature_target_spec.json")
    parser.add_argument("--outdir", default="artifacts", help="Directory to write artifacts.")
    parser.add_argument("--resample-strict", default="false", choices=["true", "false"], help="Strict resampling (no interpolation).")
    parser.add_argument("--target-mode", default="normalized", choices=["normalized", "raw"], help="Train on normalized targets or raw °C.")
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 3, 6, 12], help="Lag steps to add on features.")
    parser.add_argument("--n-estimators", type=int, default=400, help="RandomForest n_estimators.")
    parser.add_argument("--max-depth", type=int, default=None, help="RandomForest max_depth.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--k_ahead", type=int, default=1, help="Number of steps ahead for prediction.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    spec = load_spec(args.spec)

    print("Loading parquet...")
    df = pd.read_parquet(args.parquet)
    print(df.columns)
    df = ensure_datetime_index(df)

    validate_cadence(df)

    # Validate required columns
    missing_feats = [c for c in spec.feature_cols if c not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing feature columns in DataFrame: {missing_feats}")

    # Determine target columns to use
    # We'll always TRAIN on RAW targets internally; normalization is applied from TRAIN stats if target-mode=normalized.
    missing_raw = [c for c in spec.target_cols_raw if c not in df.columns]
    if missing_raw:
        raise KeyError(f"Missing raw target columns in DataFrame: {missing_raw}")
    target_cols_raw = spec.target_cols_raw

    # Build supervised dataset with shared feature builder
    ds = build_supervised_dataset(
        df,
        base_feature_cols=spec.feature_cols,
        target_cols=spec.target_cols_raw,
        exog_lags=args.lags,
        include_target_lags=False,
        k_ahead=args.k_ahead,
        dropna=True,
    )
    all_feat_cols = ds.feature_cols
    X = ds.X
    y = ds.y

    # Time-based split
    train_idx, val_idx, test_idx = time_split_index(len(X), train_frac=0.7, val_frac=0.15)
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train_raw, y_val_raw, y_test_raw = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

    winsor_bounds = compute_winsor_bounds(X_train, q_low=0.01, q_high=0.99)
    X_train = apply_winsorization(X_train, winsor_bounds)
    X_val = apply_winsorization(X_val, winsor_bounds)
    X_test = apply_winsorization(X_test, winsor_bounds)

    keep_cols = drop_near_constant_features(X_train, threshold=1e-8)
    X_train = X_train[keep_cols]
    X_val = X_val[keep_cols]
    X_test = X_test[keep_cols]
    all_feat_cols = keep_cols

    # Target normalization (from TRAIN only), if chosen
    if args.target_mode == "normalized":
        print("Computing target normalization from TRAIN only...")
        y_stats = compute_train_target_scaler(y_train_raw)
        y_train = apply_target_normalization(y_train_raw, y_stats)
        y_val = apply_target_normalization(y_val_raw, y_stats)
        y_test = apply_target_normalization(y_test_raw, y_stats)
    else:
        y_stats = None
        y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw

    # Fit multi-output RandomForest
    print("Fitting RandomForestRegressor (multi-output)...")
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=args.random_state,
        oob_score=False,
        min_samples_leaf=1,
    )

    rf.fit(X_train, y_train)

    # Predict on splits
    y_pred_train = pd.DataFrame(rf.predict(X_train), index=y_train.index, columns=y_train.columns)
    y_pred_val = pd.DataFrame(rf.predict(X_val), index=y_val.index, columns=y_val.columns)
    y_pred_test = pd.DataFrame(rf.predict(X_test), index=y_test.index, columns=y_test.columns)

    # If normalized mode, invert for human-readable metrics (°C)
    if args.target_mode == "normalized":
        print("Inverting normalized predictions to raw °C for metrics...")
        y_pred_train_c = invert_target_normalization(y_pred_train.values, y_stats, target_cols_raw)
        y_pred_val_c = invert_target_normalization(y_pred_val.values, y_stats, target_cols_raw)
        y_pred_test_c = invert_target_normalization(y_pred_test.values, y_stats, target_cols_raw)

        y_pred_train_c.index = y_train.index
        y_pred_val_c.index = y_val.index
        y_pred_test_c.index = y_test.index

        y_train_c = y_train_raw
        y_val_c = y_val_raw
        y_test_c = y_test_raw
    else:
        y_train_c, y_val_c, y_test_c = y_train_raw, y_val_raw, y_test_raw
        y_pred_train_c, y_pred_val_c, y_pred_test_c = y_pred_train, y_pred_val, y_pred_test

    # Evaluate
    print("Evaluating...")
    m_train = evaluate_predictions(y_train_c, y_pred_train_c)
    m_val = evaluate_predictions(y_val_c, y_pred_val_c)
    m_test = evaluate_predictions(y_test_c, y_pred_test_c)

    metrics_summary = pd.concat(
        [m_train.assign(split="train"), m_val.assign(split="val"), m_test.assign(split="test")],
        ignore_index=True
    )
    metrics_path = os.path.join(args.outdir, "metrics_summary.csv")
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")
    print(metrics_summary.sort_values(["split", "target"]))

    # --- Persistence baseline on the SAME test set ---
    baseline_test = persistence_k_ahead_baseline_metrics(y_test_c, k=args.k_ahead)
    baseline_test["split"] = "test_persistence"
    baseline_path = os.path.join(args.outdir, "metrics_persistence_baseline_test.csv")
    baseline_test.to_csv(baseline_path, index=False)
    print(f"Saved persistence baseline metrics: {baseline_path}")

    # Optional: print a quick comparison (overall)
    overall_model = metrics_summary[(metrics_summary["split"] == "test") & (metrics_summary["target"] == "__overall__")]
    overall_base  = baseline_test[baseline_test["target"] == "__overall__"]
    print("\n=== Test Overall Comparison ===")
    print("Model:", overall_model[["MAE","RMSE"]].to_dict("records"))
    print("Baseline:", overall_base[["MAE","RMSE"]].to_dict("records"))

    # Save artifacts
    joblib.dump(rf, os.path.join(args.outdir, "model_random_forest.pkl"))
    with open(os.path.join(args.outdir, "feature_columns.json"), "w") as f:
        json.dump(all_feat_cols, f, indent=2)
    with open(os.path.join(args.outdir, "feature_winsor_bounds.json"), "w") as f:
        json.dump(winsor_bounds, f, indent=2)
    with open(os.path.join(args.outdir, "targets_used.json"), "w") as f:
        json.dump(target_cols_raw, f, indent=2)
    with open(os.path.join(args.outdir, "train_val_test_split.json"), "w") as f:
        json.dump({
            "n_rows": int(len(X)),
            "train_idx": [int(train_idx[0]), int(train_idx[-1])],
            "val_idx": [int(val_idx[0]) if len(val_idx) else None, int(val_idx[-1]) if len(val_idx) else None],
            "test_idx": [int(test_idx[0]), int(test_idx[-1])],
            "target_mode": args.target_mode
        }, f, indent=2)
    if y_stats is not None:
        with open(os.path.join(args.outdir, "target_normalization_stats.json"), "w") as f:
            json.dump(y_stats, f, indent=2)

    # Plots
    print("Plotting feature importance...")
    plot_feature_importance(
        rf,
        feature_names=all_feat_cols,
        outpath=os.path.join(args.outdir, "feature_importance.png"),
        top_k=min(40, len(all_feat_cols))
    )

    # One illustrative target
    one_target = target_cols_raw[0]
    print(f"Plotting time series for {one_target} (test split)...")
    plot_actual_vs_pred(
        y_true=y_test_c[one_target],
        y_pred=y_pred_test_c[one_target],
        outpath=os.path.join(args.outdir, f"actual_vs_pred_{one_target}.png"),
        title=f"Actual vs Predicted (Test) — {one_target}"
    )
    plot_residuals(
        y_true=y_test_c[one_target],
        y_pred=y_pred_test_c[one_target],
        outpath=os.path.join(args.outdir, f"residuals_{one_target}.png"),
        title=f"Residuals (Test) — {one_target}"
    )

    print("Done. Artifacts saved in:", args.outdir)

if __name__ == "__main__":
    main()