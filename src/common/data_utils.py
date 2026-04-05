"""
Data utilities: splitting, validation, resampling.

Shared by both RF and PINN models.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def time_split_indices(
    n: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices by time (contiguous blocks).
    
    Args:
        n: Total number of samples
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing (optional, computed from remainder)
    
    Returns:
        (train_idx, val_idx, test_idx) as numpy arrays
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"
    
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    
    idx_all = np.arange(n)
    return idx_all[:train_end], idx_all[train_end:val_end], idx_all[val_end:]


def validate_cadence(df: pd.DataFrame, expected_seconds: float = 10.0, tolerance: float = 1.0):
    """
    Validate that the DataFrame has uniform time cadence.
    
    Args:
        df: DataFrame with DatetimeIndex
        expected_seconds: Expected cadence in seconds
        tolerance: Tolerance in seconds
    """
    dt = df.index.to_series().diff().dt.total_seconds().dropna()
    off = dt[(dt < expected_seconds - tolerance) | (dt > expected_seconds + tolerance)]
    if len(off) > 0:
        print(f"[WARN] Found {len(off)} intervals off expected {expected_seconds}s cadence.")
        print(off.value_counts().head())
    else:
        print(f"[OK] Cadence looks uniform at {expected_seconds}s.")


def resample_with_interpolation(
    df: pd.DataFrame,
    rule: str = "10s",
    strict: bool = False,
    max_interp_seconds: float = 30.0
) -> pd.DataFrame:
    """
    Resample DataFrame to uniform cadence with optional interpolation.
    
    Args:
        df: DataFrame with DatetimeIndex
        rule: Resampling rule (e.g., '10s')
        strict: If True, no interpolation (forward fill only)
        max_interp_seconds: Maximum gap to interpolate (in seconds)
    
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")
    
    df_res = df.resample(rule).mean()
    
    if not strict:
        try:
            df_res = df_res.interpolate(method="time", limit=int(max_interp_seconds / 10), limit_direction="forward")
        except Exception:
            df_res = df_res.interpolate(limit=int(max_interp_seconds / 10), limit_direction="forward")
    
    df_res = df_res.dropna(how="all")
    return df_res
