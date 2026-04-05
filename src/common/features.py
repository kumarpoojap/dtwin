"""
Official feature engineering utilities (shared by RF and PINN).

SINGLE SOURCE OF TRUTH for all feature engineering operations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>\d+)$")
_ROLL_RE = re.compile(r"^(?P<base>.+)_roll(?P<w>\d+)_(?P<kind>mean|std|delta)$")


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a sorted DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    
    ts_candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "datetime")]
    if not ts_candidates:
        raise ValueError("No DatetimeIndex or timestamp column found.")
    
    ts_col = ts_candidates[0]
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.set_index(ts_col).sort_index()
    return df


def add_lag_features(
    df: pd.DataFrame, 
    base_cols: Sequence[str], 
    lags: Sequence[int]
) -> pd.DataFrame:
    """Add lagged features for specified columns."""
    df_out = df.copy()
    for col in base_cols:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found for lagging")
        for lag in lags:
            df_out[f"{col}_lag{int(lag)}"] = df_out[col].shift(int(lag))
    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    base_cols: Sequence[str],
    windows: Sequence[int]
) -> pd.DataFrame:
    """Add past-looking rolling window features (mean, std, delta)."""
    df_out = df.copy()
    for col in base_cols:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found for rolling features")
        s = df_out[col]
        for w in windows:
            w_int = int(w)
            df_out[f"{col}_roll{w_int}_mean"] = s.rolling(window=w_int, min_periods=w_int).mean()
            df_out[f"{col}_roll{w_int}_std"] = s.rolling(window=w_int, min_periods=w_int).std(ddof=0)
            df_out[f"{col}_roll{w_int}_delta"] = s - s.shift(w_int)
    return df_out


def compute_winsor_bounds(
    X_train: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.99
) -> Dict[str, Dict[str, float]]:
    """Compute winsorization bounds from training data only."""
    bounds: Dict[str, Dict[str, float]] = {}
    for col in X_train.columns:
        s = X_train[col].astype(float)
        lo = float(np.nanquantile(s.values, q_low))
        hi = float(np.nanquantile(s.values, q_high))
        if not np.isfinite(lo):
            lo = float(np.nanmin(s.values))
        if not np.isfinite(hi):
            hi = float(np.nanmax(s.values))
        if lo > hi:
            lo, hi = hi, lo
        bounds[col] = {"low": lo, "high": hi}
    return bounds


def apply_winsorization(
    X: pd.DataFrame,
    bounds: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Apply winsorization bounds (clip outliers)."""
    X_clipped = X.copy()
    for col, b in bounds.items():
        if col not in X_clipped.columns:
            continue
        X_clipped[col] = X_clipped[col].clip(lower=b["low"], upper=b["high"])
    return X_clipped


def drop_low_variance_features(
    X_train: pd.DataFrame,
    threshold: float = 1e-8
) -> List[str]:
    """Return list of feature columns with variance >= threshold (computed on train only)."""
    variances = X_train.var(axis=0, ddof=0)
    keep = variances[variances >= threshold].index.tolist()
    return keep


def build_feature_column_names(
    base_cols: Sequence[str],
    lags: Sequence[int],
    roll_windows: Sequence[int]
) -> List[str]:
    """Build the expected feature column names in canonical order."""
    feat_cols: List[str] = []
    
    # Base columns
    feat_cols.extend(list(base_cols))
    
    # Lagged base columns
    for col in base_cols:
        for lag in lags:
            feat_cols.append(f"{col}_lag{int(lag)}")
    
    # Rolling window features
    for col in base_cols:
        for w in roll_windows:
            w_int = int(w)
            feat_cols.append(f"{col}_roll{w_int}_mean")
            feat_cols.append(f"{col}_roll{w_int}_std")
            feat_cols.append(f"{col}_roll{w_int}_delta")
    
    # De-duplicate but preserve order
    return list(dict.fromkeys(feat_cols))


def build_official_features(
    df: pd.DataFrame,
    base_cols: Sequence[str],
    lags: Sequence[int] = (1, 3, 6, 12),
    roll_windows: Sequence[int] = (3, 6, 12),
    winsorize: bool = False,
    winsor_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    low_var_cols: Optional[List[str]] = None,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Build the official exogenous-only feature set.
    
    This is the SINGLE source of truth for feature engineering.
    
    Args:
        df: Base DataFrame with DatetimeIndex
        base_cols: Base feature columns (exogenous sensors/actuators)
        lags: Lag steps to add
        roll_windows: Rolling window sizes
        winsorize: Whether to apply winsorization
        winsor_bounds: Pre-computed winsor bounds (if None, no clipping)
        low_var_cols: Pre-filtered column list (if None, keep all)
        dropna: Whether to drop rows with NaNs
    
    Returns:
        DataFrame with engineered features
    """
    df = ensure_datetime_index(df)
    
    # Add lag features
    df_feat = add_lag_features(df, base_cols=base_cols, lags=lags)
    
    # Add rolling features
    df_feat = add_rolling_features(df_feat, base_cols=base_cols, windows=roll_windows)
    
    # Build expected column list
    expected_cols = build_feature_column_names(base_cols, lags, roll_windows)
    
    # Select only expected columns
    X = df_feat[expected_cols]
    
    # Apply winsorization if requested
    if winsorize and winsor_bounds is not None:
        X = apply_winsorization(X, winsor_bounds)
    
    # Filter to low-variance-filtered columns if provided
    if low_var_cols is not None:
        X = X[[c for c in low_var_cols if c in X.columns]]
    
    # Drop NaNs introduced by shifts/rolling
    if dropna:
        X = X.dropna()
    
    # CRITICAL GUARDRAIL: no target lags allowed
    forbidden = [c for c in X.columns if any(
        c.endswith(suffix) for suffix in ('_lagy1', '_lagy3', '_lagy6', '_lagy12')
    )]
    if forbidden:
        raise ValueError(f"Target lags are forbidden in exogenous-only features: {forbidden}")
    
    return X


def validate_feature_columns(
    actual_cols: List[str],
    expected_path: Path
) -> Tuple[bool, List[str]]:
    """
    Validate that actual feature columns match the expected list from feature_columns.json.
    
    Returns:
        (is_valid, diff_messages)
    """
    if not expected_path.exists():
        return False, [f"Expected feature columns file not found: {expected_path}"]
    
    with open(expected_path, "r") as f:
        expected_cols = json.load(f)
    
    if actual_cols == expected_cols:
        return True, []
    
    diff_msgs = []
    
    # Check length
    if len(actual_cols) != len(expected_cols):
        diff_msgs.append(f"Column count mismatch: actual={len(actual_cols)}, expected={len(expected_cols)}")
    
    # Check set difference
    actual_set = set(actual_cols)
    expected_set = set(expected_cols)
    
    missing = expected_set - actual_set
    if missing:
        diff_msgs.append(f"Missing columns: {sorted(missing)[:10]}...")
    
    extra = actual_set - expected_set
    if extra:
        diff_msgs.append(f"Extra columns: {sorted(extra)[:10]}...")
    
    # Check order (if same set)
    if actual_set == expected_set and actual_cols != expected_cols:
        diff_msgs.append("Column order mismatch (same columns, different order)")
    
    return False, diff_msgs


def materialize_features_from_list(
    df: pd.DataFrame,
    feature_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Materialize engineered features from a feature column list.
    
    This is used for inference/export when you have a base parquet and need to
    recreate lag/rolling features on-the-fly.
    """
    df = ensure_datetime_index(df)
    df_out = df.copy()
    
    # Parse lag features
    lag_specs: Dict[str, List[int]] = {}
    for col in feature_cols:
        m = _LAG_RE.match(col)
        if m:
            base = m.group("base")
            lag = int(m.group("lag"))
            lag_specs.setdefault(base, []).append(lag)
    
    # Materialize lags
    for base, lags in lag_specs.items():
        if base not in df_out.columns:
            continue
        for lag in sorted(set(lags)):
            name = f"{base}_lag{lag}"
            if name not in df_out.columns:
                df_out[name] = df_out[base].shift(lag)
    
    # Parse rolling features
    roll_specs: Dict[str, Dict[int, List[str]]] = {}
    for col in feature_cols:
        m = _ROLL_RE.match(col)
        if m:
            base = m.group("base")
            w = int(m.group("w"))
            kind = m.group("kind")
            roll_specs.setdefault(base, {}).setdefault(w, []).append(kind)
    
    # Materialize rolling features
    for base, by_w in roll_specs.items():
        if base not in df_out.columns:
            continue
        s = df_out[base]
        for w, kinds in by_w.items():
            kinds_set = set(kinds)
            if "mean" in kinds_set:
                name = f"{base}_roll{w}_mean"
                if name not in df_out.columns:
                    df_out[name] = s.rolling(window=w, min_periods=w).mean()
            if "std" in kinds_set:
                name = f"{base}_roll{w}_std"
                if name not in df_out.columns:
                    df_out[name] = s.rolling(window=w, min_periods=w).std(ddof=0)
            if "delta" in kinds_set:
                name = f"{base}_roll{w}_delta"
                if name not in df_out.columns:
                    df_out[name] = s - s.shift(w)
    
    # Return only requested columns that exist
    existing = [c for c in feature_cols if c in df_out.columns]
    return df_out[existing]
