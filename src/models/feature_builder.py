from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>\d+)$")
_ROLL_RE = re.compile(r"^(?P<base>.+)_roll(?P<w>\d+)_(?P<kind>mean|std|delta)$")


@dataclass(frozen=True)
class SupervisedDataset:
    X: pd.DataFrame
    y: pd.DataFrame
    feature_cols: List[str]
    target_cols: List[str]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
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


def add_lag_features(df: pd.DataFrame, base_cols: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    df_out = df.copy()
    for col in base_cols:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found for lagging")
        for l in lags:
            df_out[f"{col}_lag{int(l)}"] = df_out[col].shift(int(l))
    return df_out


def build_feature_columns(
    base_feature_cols: Sequence[str],
    exog_lags: Sequence[int],
    target_cols: Sequence[str],
    target_lags: Sequence[int],
    include_target_lags: bool,
    rolling_windows: Sequence[int],
) -> List[str]:
    feat_cols: List[str] = []

    feat_cols.extend(list(base_feature_cols))
    for c in base_feature_cols:
        for l in exog_lags:
            feat_cols.append(f"{c}_lag{int(l)}")

    for c in base_feature_cols:
        for w in rolling_windows:
            w_i = int(w)
            feat_cols.append(f"{c}_roll{w_i}_mean")
            feat_cols.append(f"{c}_roll{w_i}_std")
            feat_cols.append(f"{c}_roll{w_i}_delta")

    if include_target_lags:
        for c in target_cols:
            for l in target_lags:
                feat_cols.append(f"{c}_lag{int(l)}")

    # De-duplicate but preserve order
    return list(dict.fromkeys(feat_cols))


def build_supervised_dataset(
    df: pd.DataFrame,
    *,
    base_feature_cols: Sequence[str],
    target_cols: Sequence[str],
    exog_lags: Sequence[int] = (1, 3, 6, 12),
    include_target_lags: bool = False,
    target_lags: Sequence[int] = (1, 3),
    rolling_windows: Sequence[int] = (3, 6, 12),
    k_ahead: int = 1,
    dropna: bool = True,
) -> SupervisedDataset:
    """
    Build X(t) and y(t+k) from a time-indexed dataframe.

    Official feature set:
    - base_feature_cols
    - lagged base_feature_cols for exog_lags
    - optional lagged target_cols for target_lags (autoregressive teacher)

    Labels:
    - y = target_cols shifted by -k_ahead (i.e., future)
    """
    if k_ahead < 0:
        raise ValueError("k_ahead must be >= 0")

    df = ensure_datetime_index(df)

    # Engineer features
    feature_cols = build_feature_columns(
        base_feature_cols=base_feature_cols,
        exog_lags=exog_lags,
        target_cols=target_cols,
        target_lags=target_lags,
        include_target_lags=include_target_lags,
        rolling_windows=rolling_windows,
    )

    df_feat = add_lag_features(df, base_cols=base_feature_cols, lags=exog_lags)
    if include_target_lags:
        df_feat = add_lag_features(df_feat, base_cols=target_cols, lags=target_lags)

    # Past-looking rolling windows over base (exogenous) features
    for c in base_feature_cols:
        if c not in df_feat.columns:
            raise KeyError(f"Column '{c}' not found for rolling features")
        s = df_feat[c]
        for w in rolling_windows:
            w_i = int(w)
            df_feat[f"{c}_roll{w_i}_mean"] = s.rolling(window=w_i, min_periods=w_i).mean()
            df_feat[f"{c}_roll{w_i}_std"] = s.rolling(window=w_i, min_periods=w_i).std(ddof=0)
            df_feat[f"{c}_roll{w_i}_delta"] = s - s.shift(w_i)

    # Labels: future targets
    y_future = df_feat[list(target_cols)].shift(-int(k_ahead))

    X = df_feat[feature_cols]
    y = y_future[list(target_cols)]

    if dropna:
        idx = X.index.intersection(y.index)
        df_xy = pd.concat([X.loc[idx], y.loc[idx]], axis=1).dropna()
        X = df_xy[feature_cols]
        y = df_xy[list(target_cols)]

    return SupervisedDataset(X=X, y=y, feature_cols=list(feature_cols), target_cols=list(target_cols))


def materialize_features_from_feature_list(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Given a base dataframe (typically the preprocessed parquet) and an *engineered*
    feature column list (like artifacts/feature_columns.json), materialize those
    features (currently supports *_lagN patterns).

    This is designed for export-time sanity checks and inference pipelines.
    """
    df = ensure_datetime_index(df)
    df_out = df.copy()

    requested = list(feature_cols)

    # Identify which lag features are needed
    lag_specs: Dict[str, List[int]] = {}
    for c in requested:
        m = _LAG_RE.match(c)
        if not m:
            continue
        base = m.group("base")
        lag = int(m.group("lag"))
        lag_specs.setdefault(base, []).append(lag)

    # Materialize requested lag features
    for base, lags in lag_specs.items():
        if base not in df_out.columns:
            # Base column absent; can't materialize these lags.
            continue
        for l in sorted(set(lags)):
            name = f"{base}_lag{l}"
            if name not in df_out.columns:
                df_out[name] = df_out[base].shift(l)

    # Identify which rolling features are needed
    roll_specs: Dict[str, Dict[int, List[str]]] = {}
    for c in requested:
        m = _ROLL_RE.match(c)
        if not m:
            continue
        base = m.group("base")
        w = int(m.group("w"))
        kind = m.group("kind")
        roll_specs.setdefault(base, {}).setdefault(w, []).append(kind)

    for base, by_w in roll_specs.items():
        if base not in df_out.columns:
            continue
        s = df_out[base]
        for w, kinds in by_w.items():
            kinds_u = set(kinds)
            if "mean" in kinds_u:
                name = f"{base}_roll{w}_mean"
                if name not in df_out.columns:
                    df_out[name] = s.rolling(window=w, min_periods=w).mean()
            if "std" in kinds_u:
                name = f"{base}_roll{w}_std"
                if name not in df_out.columns:
                    df_out[name] = s.rolling(window=w, min_periods=w).std(ddof=0)
            if "delta" in kinds_u:
                name = f"{base}_roll{w}_delta"
                if name not in df_out.columns:
                    df_out[name] = s - s.shift(w)

    # Return only requested columns that exist
    existing = [c for c in requested if c in df_out.columns]
    return df_out[existing]


def compute_winsor_bounds(X_train: pd.DataFrame, q_low: float = 0.01, q_high: float = 0.99) -> Dict[str, Dict[str, float]]:
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


def apply_winsorization(X: pd.DataFrame, bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    Xc = X.copy()
    for col, b in bounds.items():
        if col not in Xc.columns:
            continue
        Xc[col] = Xc[col].clip(lower=b["low"], upper=b["high"])
    return Xc


def drop_near_constant_features(
    X_train: pd.DataFrame,
    *,
    threshold: float = 1e-8,
) -> List[str]:
    variances = X_train.var(axis=0, ddof=0)
    keep = variances[variances >= threshold].index.tolist()
    return keep
