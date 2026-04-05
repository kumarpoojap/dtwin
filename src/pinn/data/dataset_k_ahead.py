"""
K-ahead dataset loader with time-based splits and strict no-leakage guarantees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ...common.data_utils import time_split_indices, validate_cadence
from ...common.features import (
    add_lag_features,
    add_rolling_features,
    apply_winsorization,
    build_feature_column_names,
    compute_winsor_bounds,
    drop_low_variance_features,
    ensure_datetime_index,
    validate_feature_columns,
)
from ...common.scalers import TargetScaler


@dataclass
class DatasetSpec:
    """Specification loaded from feature_target_spec.json."""
    resample_rule: str
    feature_cols: List[str]
    target_cols_raw: List[str]
    target_cols_normalized: List[str]
    notes: List[str]


def load_spec(path: Path) -> DatasetSpec:
    """Load dataset specification from JSON."""
    with open(path, "r") as f:
        obj = json.load(f)
    
    required = ["resample_rule", "feature_cols", "target_cols_raw", "target_cols_normalized"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing '{k}' in spec: {path}")
    
    return DatasetSpec(
        resample_rule=obj["resample_rule"],
        feature_cols=obj["feature_cols"],
        target_cols_raw=obj["target_cols_raw"],
        target_cols_normalized=obj["target_cols_normalized"],
        notes=obj.get("notes", [])
    )


class KAheadDataset(Dataset):
    """PyTorch Dataset for k-ahead prediction."""
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        timestamps: pd.DatetimeIndex
    ):
        """
        Args:
            X: Features (already engineered, normalized if needed)
            y: Targets (already shifted by -k, normalized if needed)
            timestamps: DatetimeIndex for each sample
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.timestamps = timestamps
        self.feature_cols = list(X.columns)
        self.target_cols = list(y.columns)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (X, y, timestamp_idx)."""
        return self.X[idx], self.y[idx], idx


def prepare_k_ahead_data(
    parquet_path: Path,
    spec_path: Path,
    feature_columns_path: Path,
    base_cols: List[str],
    lags: List[int],
    roll_windows: List[int],
    k_ahead: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    normalize_targets: bool = True,
    winsorize: bool = True,
    winsor_quantiles: Tuple[float, float] = (0.01, 0.99),
    low_var_threshold: float = 1e-8,
    cadence_seconds: float = 10.0,
    dev_run: bool = False,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Prepare k-ahead dataset with strict no-leakage guarantees.
    
    Returns a dict with:
        - train_dataset, val_dataset, test_dataset (KAheadDataset)
        - scaler (TargetScaler or None)
        - feature_cols (final list after variance filtering)
        - target_cols
        - winsor_bounds
        - metadata
    """
    print(f"[INFO] Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = ensure_datetime_index(df)
    
    # Dev run: subsample
    if dev_run and max_samples is not None:
        print(f"[DEV] Subsampling to {max_samples} rows")
        df = df.iloc[:max_samples]
    
    # Validate cadence
    validate_cadence(df, expected_seconds=cadence_seconds)
    
    # Load spec
    spec = load_spec(spec_path)
    
    # Validate base columns exist
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing base columns in parquet: {missing}")
    
    # Validate target columns exist
    missing_targets = [c for c in spec.target_cols_raw if c not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns in parquet: {missing_targets}")
    
    target_cols = spec.target_cols_raw
    
    print(f"[INFO] Building official features (exogenous-only)...")
    # Build features WITHOUT winsorization/filtering first
    from .features import (
        add_lag_features, add_rolling_features, build_feature_column_names,
        compute_winsor_bounds, apply_winsorization, drop_low_variance_features
    )
    
    df_feat = add_lag_features(df, base_cols=base_cols, lags=lags)
    df_feat = add_rolling_features(df_feat, base_cols=base_cols, windows=roll_windows)
    
    expected_cols = build_feature_column_names(base_cols, lags, roll_windows)
    X_all = df_feat[expected_cols].copy()
    
    # Create k-ahead labels: y(t+k)
    print(f"[INFO] Creating k-ahead labels (k={k_ahead})...")
    y_all = df_feat[target_cols].shift(-k_ahead).copy()
    
    # Align and drop NaNs
    idx_valid = X_all.index.intersection(y_all.index)
    df_xy = pd.concat([X_all.loc[idx_valid], y_all.loc[idx_valid]], axis=1).dropna()
    
    X_all = df_xy[expected_cols]
    y_all = df_xy[target_cols]
    timestamps_all = X_all.index
    
    print(f"[INFO] Valid samples after k-ahead shift and dropna: {len(X_all)}")
    
    # Time-based split
    train_idx, val_idx, test_idx = time_split_indices(
        len(X_all), train_frac=train_frac, val_frac=val_frac
    )
    
    X_train = X_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    X_test = X_all.iloc[test_idx]
    
    y_train = y_all.iloc[train_idx]
    y_val = y_all.iloc[val_idx]
    y_test = y_all.iloc[test_idx]
    
    ts_train = timestamps_all[train_idx]
    ts_val = timestamps_all[val_idx]
    ts_test = timestamps_all[test_idx]
    
    # Winsorization (TRAIN-only bounds)
    winsor_bounds = None
    if winsorize:
        print(f"[INFO] Computing winsorization bounds from TRAIN (q={winsor_quantiles})...")
        winsor_bounds = compute_winsor_bounds(X_train, q_low=winsor_quantiles[0], q_high=winsor_quantiles[1])
        X_train = apply_winsorization(X_train, winsor_bounds)
        X_val = apply_winsorization(X_val, winsor_bounds)
        X_test = apply_winsorization(X_test, winsor_bounds)
    
    # Drop low-variance features (TRAIN-only variance)
    print(f"[INFO] Dropping low-variance features (threshold={low_var_threshold})...")
    keep_cols = drop_low_variance_features(X_train, threshold=low_var_threshold)
    print(f"[INFO] Kept {len(keep_cols)}/{len(expected_cols)} features after variance filter")
    
    X_train = X_train[keep_cols]
    X_val = X_val[keep_cols]
    X_test = X_test[keep_cols]
    
    # Validate against feature_columns.json
    print(f"[INFO] Validating feature columns against {feature_columns_path}...")
    is_valid, diff_msgs = validate_feature_columns(keep_cols, feature_columns_path)
    if not is_valid:
        print("[ERROR] Feature column mismatch!")
        for msg in diff_msgs:
            print(f"  - {msg}")
        raise ValueError(
            f"Feature columns do not match {feature_columns_path}. "
            "Regenerate the RF model or update feature_columns.json."
        )
    print("[OK] Feature columns match expected list.")
    
    # Target normalization (TRAIN-only stats)
    scaler = None
    if normalize_targets:
        print("[INFO] Normalizing targets from TRAIN stats...")
        scaler = TargetScaler()
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_val = scaler.transform(y_val)
        y_test = scaler.transform(y_test)
    
    # Create PyTorch datasets
    train_dataset = KAheadDataset(X_train, y_train, ts_train)
    val_dataset = KAheadDataset(X_val, y_val, ts_val)
    test_dataset = KAheadDataset(X_test, y_test, ts_test)
    
    print(f"[INFO] Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "scaler": scaler,
        "feature_cols": keep_cols,
        "target_cols": target_cols,
        "winsor_bounds": winsor_bounds,
        "metadata": {
            "k_ahead": k_ahead,
            "cadence_seconds": cadence_seconds,
            "n_features": len(keep_cols),
            "n_targets": len(target_cols),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset)
        }
    }
