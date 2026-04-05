"""
Evaluation metrics: MAE, RMSE, R², skill scores.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_skill_score(
    rmse_model: float,
    rmse_baseline: float
) -> float:
    """
    Skill score: 1 - RMSE_model / RMSE_baseline.
    
    Positive values indicate improvement over baseline.
    """
    if rmse_baseline < 1e-10:
        return 0.0
    return 1.0 - rmse_model / rmse_baseline


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    baseline_rmse: Optional[np.ndarray] = None,
    baseline_mae: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute per-target and overall metrics.
    
    Args:
        y_true: Ground truth, shape (n_samples, n_targets)
        y_pred: Predictions, shape (n_samples, n_targets)
        target_cols: Target column names
        baseline_rmse: Baseline RMSE per target (for skill score)
        baseline_mae: Baseline MAE per target (for skill score)
    
    Returns:
        DataFrame with metrics per target + overall
    """
    n_targets = y_true.shape[1]
    
    metrics = []
    
    # Per-target metrics
    for i, col in enumerate(target_cols):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        mae = compute_mae(y_t, y_p)
        rmse = compute_rmse(y_t, y_p)
        r2 = compute_r2(y_t, y_p)
        
        row = {
            "target": col,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
        
        # Skill scores if baseline provided
        if baseline_rmse is not None:
            skill_rmse = compute_skill_score(rmse, baseline_rmse[i])
            row["skill_RMSE"] = skill_rmse
        
        if baseline_mae is not None:
            skill_mae = compute_skill_score(mae, baseline_mae[i])
            row["skill_MAE"] = skill_mae
        
        metrics.append(row)
    
    # Overall metrics (flattened)
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    mae_overall = compute_mae(y_true_flat, y_pred_flat)
    rmse_overall = compute_rmse(y_true_flat, y_pred_flat)
    r2_overall = compute_r2(y_true_flat, y_pred_flat)
    
    overall_row = {
        "target": "__overall__",
        "MAE": mae_overall,
        "RMSE": rmse_overall,
        "R2": r2_overall
    }
    
    if baseline_rmse is not None:
        baseline_rmse_overall = np.sqrt(np.mean(baseline_rmse ** 2))
        overall_row["skill_RMSE"] = compute_skill_score(rmse_overall, baseline_rmse_overall)
    
    if baseline_mae is not None:
        baseline_mae_overall = np.mean(baseline_mae)
        overall_row["skill_MAE"] = compute_skill_score(mae_overall, baseline_mae_overall)
    
    metrics.append(overall_row)
    
    return pd.DataFrame(metrics)


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataset,
    device: str = "cpu",
    scaler = None,
    batch_size: int = 256
) -> Dict[str, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataset: KAheadDataset
        device: Device to run on
        scaler: TargetScaler (if targets are normalized)
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with 'y_true', 'y_pred' (both in original scale)
    """
    model.eval()
    
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            
            X_batch = []
            y_batch = []
            t_idx_batch = []
            
            for j in range(i, batch_end):
                X, y, t_idx = dataset[j]
                X_batch.append(X)
                y_batch.append(y)
                t_idx_batch.append(t_idx)
            
            X_batch = torch.stack(X_batch).to(device)
            y_batch = torch.stack(y_batch)
            t_idx_batch = torch.tensor(t_idx_batch, dtype=torch.float32).to(device)
            
            # Forward pass
            out = model(X_batch, t_idx_batch, return_physics_params=False)
            y_pred_batch = out["delta_y"]
            
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(y_pred_batch.cpu().numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Inverse transform if normalized
    if scaler is not None:
        y_true = scaler.inverse_transform_array(y_true)
        y_pred = scaler.inverse_transform_array(y_pred)
    
    return {
        "y_true": y_true,
        "y_pred": y_pred
    }
