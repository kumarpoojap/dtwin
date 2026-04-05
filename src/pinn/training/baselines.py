"""
Baseline models for comparison: k-ahead persistence.
"""

from typing import Dict

import numpy as np
import pandas as pd


def persistence_k_ahead_baseline(
    y_all: np.ndarray,
    k: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute k-ahead persistence baseline: y_hat(t+k) = y(t).
    
    Args:
        y_all: All target values (before k-ahead shift), shape (n_samples, n_targets)
        k: Number of steps ahead
        train_idx, val_idx, test_idx: Split indices
    
    Returns:
        Dict with baseline predictions and ground truth for each split
    """
    # For k-ahead persistence, we predict y(t+k) ≈ y(t)
    # Ground truth is y(t+k), prediction is y(t)
    
    # Shift ground truth by -k to get y(t+k)
    y_future = np.roll(y_all, -k, axis=0)
    
    # Valid indices (exclude last k samples where future is not available)
    valid_mask = np.ones(len(y_all), dtype=bool)
    valid_mask[-k:] = False
    
    results = {}
    
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        # Filter to split and valid samples
        split_mask = np.zeros(len(y_all), dtype=bool)
        split_mask[idx] = True
        final_mask = split_mask & valid_mask
        
        # Ground truth: y(t+k)
        y_true = y_future[final_mask]
        
        # Prediction: y(t)
        y_pred = y_all[final_mask]
        
        results[f"{split_name}_true"] = y_true
        results[f"{split_name}_pred"] = y_pred
    
    return results


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list
) -> pd.DataFrame:
    """
    Compute baseline metrics (MAE, RMSE) per target and overall.
    
    Args:
        y_true: Ground truth, shape (n_samples, n_targets)
        y_pred: Predictions, shape (n_samples, n_targets)
        target_cols: Target column names
    
    Returns:
        DataFrame with metrics
    """
    from .metrics import compute_mae, compute_rmse
    
    metrics = []
    
    # Per-target
    for i, col in enumerate(target_cols):
        mae = compute_mae(y_true[:, i], y_pred[:, i])
        rmse = compute_rmse(y_true[:, i], y_pred[:, i])
        metrics.append({"target": col, "MAE": mae, "RMSE": rmse})
    
    # Overall
    mae_overall = compute_mae(y_true.ravel(), y_pred.ravel())
    rmse_overall = compute_rmse(y_true.ravel(), y_pred.ravel())
    metrics.append({"target": "__overall__", "MAE": mae_overall, "RMSE": rmse_overall})
    
    return pd.DataFrame(metrics)
