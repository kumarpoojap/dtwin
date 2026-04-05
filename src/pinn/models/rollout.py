"""
Multi-step rollout utilities for surrogate model evaluation.

Implements autoregressive rollout for testing long-horizon prediction stability.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pinn.data.scalers import TargetScaler


def rollout_pinn_model(
    model: nn.Module,
    initial_features: torch.Tensor,
    initial_temp: torch.Tensor,
    n_steps: int,
    feature_cols: List[str],
    target_cols: List[str],
    scaler: Optional[TargetScaler] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform autoregressive rollout with PINN model.
    
    Args:
        model: Trained PINN model
        initial_features: Initial feature vector, shape (batch, n_features)
        initial_temp: Initial temperature, shape (batch, n_targets)
        n_steps: Number of steps to roll out
        feature_cols: List of feature column names
        target_cols: List of target column names
        scaler: Target scaler for denormalization (optional)
        device: Device to run on
    
    Returns:
        predictions: Predicted temperatures over time, shape (batch, n_steps, n_targets)
        predictions_denorm: Denormalized predictions (if scaler provided)
    """
    model.eval()
    batch_size = initial_features.shape[0]
    n_targets = len(target_cols)
    
    # Storage for predictions
    predictions = torch.zeros(batch_size, n_steps, n_targets, device=device)
    
    # Current state
    current_temp = initial_temp.clone()
    current_features = initial_features.clone()
    
    with torch.no_grad():
        for step in range(n_steps):
            # Create time index (could be actual time or just step number)
            t_idx = torch.full((batch_size,), step, dtype=torch.float32, device=device)
            
            # Predict next temperature delta
            out = model(current_features, t_idx, return_physics_params=False)
            delta_temp = out["delta_y"]  # (batch, n_targets)
            
            # Update temperature (assuming delta is in normalized space)
            current_temp = current_temp + delta_temp
            
            # Store prediction
            predictions[:, step, :] = current_temp
            
            # Update features for next step (simplified - assumes features don't change)
            # In reality, you'd need to update lag features, rolling windows, etc.
            # For now, we keep features constant (exogenous inputs)
    
    # Denormalize if scaler provided
    predictions_denorm = None
    if scaler is not None:
        predictions_denorm = torch.zeros_like(predictions)
        for i, target_col in enumerate(target_cols):
            mean = scaler.stats[target_col]["mean"]
            std = scaler.stats[target_col]["std"]
            predictions_denorm[:, :, i] = predictions[:, :, i] * std + mean
    
    return predictions, predictions_denorm


def rollout_rf_teacher(
    teacher_model,
    initial_features: np.ndarray,
    n_steps: int,
    feature_cols: List[str],
    target_cols: List[str]
) -> np.ndarray:
    """
    Perform autoregressive rollout with RF teacher model.
    
    Args:
        teacher_model: Trained RF teacher (TeacherRF object)
        initial_features: Initial feature vector, shape (batch, n_features)
        n_steps: Number of steps to roll out
        feature_cols: List of feature column names
        target_cols: List of target column names
    
    Returns:
        predictions: Predicted temperatures over time, shape (batch, n_steps, n_targets)
    """
    batch_size = initial_features.shape[0]
    n_targets = len(target_cols)
    
    # Storage for predictions
    predictions = np.zeros((batch_size, n_steps, n_targets))
    
    # Current features
    current_features = initial_features.copy()
    
    for step in range(n_steps):
        # Predict next temperature
        # Convert to DataFrame for teacher model
        X_df = pd.DataFrame(current_features, columns=feature_cols)
        y_pred = teacher_model.predict(X_df, return_tensor=False)  # (batch, n_targets)
        
        # Store prediction
        predictions[:, step, :] = y_pred
        
        # Update features for next step (simplified - keep constant)
        # In reality, you'd update lag features based on predictions
    
    return predictions


def rollout_rc_model(
    initial_temp: np.ndarray,
    power: np.ndarray,
    fan_speed: np.ndarray,
    ambient_temp: np.ndarray,
    n_steps: int,
    dt: float = 1.0,
    C: float = 100.0,
    h: float = 0.05,
    beta: float = -0.03,
    gamma: float = 0.001
) -> np.ndarray:
    """
    Perform rollout with RC (lumped-parameter) thermal model.
    
    Physics: T(t+1) = T(t) + dt * [gamma*P - beta*Fan - h*(T - T_amb)] / C
    
    Args:
        initial_temp: Initial temperature, shape (batch, n_targets)
        power: Power values over time, shape (batch, n_steps)
        fan_speed: Fan speed values over time, shape (batch, n_steps)
        ambient_temp: Ambient temperature over time, shape (batch, n_steps)
        n_steps: Number of steps to roll out
        dt: Time step in seconds
        C: Thermal capacity
        h: Heat transfer coefficient
        beta: Cooling effectiveness (negative for cooling)
        gamma: Power-to-heat conversion
    
    Returns:
        predictions: Predicted temperatures over time, shape (batch, n_steps, n_targets)
    """
    batch_size = initial_temp.shape[0]
    n_targets = initial_temp.shape[1]
    
    # Storage for predictions
    predictions = np.zeros((batch_size, n_steps, n_targets))
    
    # Current temperature
    current_temp = initial_temp.copy()
    
    for step in range(n_steps):
        # Get current inputs
        P = power[:, step]  # (batch,)
        Fan = fan_speed[:, step]  # (batch,)
        T_amb = ambient_temp[:, step]  # (batch,)
        
        # Expand to match targets
        P_exp = P[:, np.newaxis]  # (batch, 1)
        Fan_exp = Fan[:, np.newaxis]
        T_amb_exp = T_amb[:, np.newaxis]
        
        # Compute temperature change
        # dT/dt = [gamma*P - beta*Fan - h*(T - T_amb)] / C
        heat_gen = gamma * P_exp
        cooling = beta * Fan_exp
        heat_transfer = -h * (current_temp - T_amb_exp)
        
        dT_dt = (heat_gen + cooling + heat_transfer) / C
        delta_T = dT_dt * dt
        
        # Update temperature
        current_temp = current_temp + delta_T
        
        # Store prediction
        predictions[:, step, :] = current_temp
    
    return predictions


def compute_rollout_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    horizons: List[int] = [10, 30, 60, 90]
) -> Dict[str, np.ndarray]:
    """
    Compute rollout stability metrics at different horizons.
    
    Args:
        predictions: Model predictions, shape (batch, n_steps, n_targets)
        ground_truth: Ground truth values, shape (batch, n_steps, n_targets)
        horizons: List of horizons (in steps) to evaluate
    
    Returns:
        metrics: Dictionary with MAE, RMSE, and drift at each horizon
    """
    metrics = {
        "horizons": np.array(horizons),
        "mae": [],
        "rmse": [],
        "drift": []
    }
    
    for h in horizons:
        if h > predictions.shape[1]:
            continue
        
        # Get predictions and ground truth up to horizon
        pred_h = predictions[:, :h, :]  # (batch, h, n_targets)
        gt_h = ground_truth[:, :h, :]
        
        # Compute MAE
        mae = np.mean(np.abs(pred_h - gt_h))
        metrics["mae"].append(mae)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((pred_h - gt_h) ** 2))
        metrics["rmse"].append(rmse)
        
        # Compute drift (mean absolute deviation from initial)
        initial = ground_truth[:, 0:1, :]  # (batch, 1, n_targets)
        drift = np.mean(np.abs(pred_h - initial))
        metrics["drift"].append(drift)
    
    # Convert to arrays
    metrics["mae"] = np.array(metrics["mae"])
    metrics["rmse"] = np.array(metrics["rmse"])
    metrics["drift"] = np.array(metrics["drift"])
    
    return metrics


def evaluate_rollout_stability(
    model: nn.Module,
    dataloader: DataLoader,
    n_steps: int,
    feature_cols: List[str],
    target_cols: List[str],
    scaler: Optional[TargetScaler] = None,
    device: str = "cpu",
    max_batches: int = 10
) -> Dict[str, np.ndarray]:
    """
    Evaluate rollout stability over multiple batches.
    
    Args:
        model: Trained PINN model
        dataloader: DataLoader with test data
        n_steps: Number of steps to roll out
        feature_cols: List of feature column names
        target_cols: List of target column names
        scaler: Target scaler for denormalization
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
    
    Returns:
        metrics: Rollout stability metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, (X, y, t_idx) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            X = X.to(device)
            y = y.to(device)
            
            # Get initial temperature (assume y is the k-ahead target)
            # For simplicity, use zeros as initial (or extract from features if available)
            initial_temp = torch.zeros_like(y)
            
            # Perform rollout
            predictions, predictions_denorm = rollout_pinn_model(
                model, X, initial_temp, n_steps,
                feature_cols, target_cols, scaler, device
            )
            
            # Use denormalized predictions if available
            if predictions_denorm is not None:
                pred_np = predictions_denorm.cpu().numpy()
            else:
                pred_np = predictions.cpu().numpy()
            
            # Ground truth (simplified - would need actual sequence data)
            # For now, repeat the target as a placeholder
            gt_np = y.cpu().numpy()[:, np.newaxis, :].repeat(n_steps, axis=1)
            
            all_predictions.append(pred_np)
            all_ground_truth.append(gt_np)
    
    # Concatenate all batches
    predictions_all = np.concatenate(all_predictions, axis=0)
    ground_truth_all = np.concatenate(all_ground_truth, axis=0)
    
    # Compute metrics
    horizons = [10, 30, 60, 90] if n_steps >= 90 else [10, 20, 30]
    metrics = compute_rollout_metrics(predictions_all, ground_truth_all, horizons)
    
    return metrics


def compare_surrogate_rollouts(
    pinn_model: nn.Module,
    rf_teacher,
    initial_features: np.ndarray,
    initial_temp: np.ndarray,
    power: np.ndarray,
    fan_speed: np.ndarray,
    ambient_temp: np.ndarray,
    n_steps: int,
    feature_cols: List[str],
    target_cols: List[str],
    scaler: Optional[TargetScaler] = None,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Compare rollout predictions from PINN, RF, and RC models.
    
    Args:
        pinn_model: Trained PINN model
        rf_teacher: Trained RF teacher model
        initial_features: Initial features, shape (batch, n_features)
        initial_temp: Initial temperature, shape (batch, n_targets)
        power: Power values over time, shape (batch, n_steps)
        fan_speed: Fan speed values over time, shape (batch, n_steps)
        ambient_temp: Ambient temperature over time, shape (batch, n_steps)
        n_steps: Number of steps to roll out
        feature_cols: List of feature column names
        target_cols: List of target column names
        scaler: Target scaler for denormalization
        device: Device to run on
    
    Returns:
        results: Dictionary with predictions from each model
    """
    # PINN rollout
    initial_features_torch = torch.from_numpy(initial_features).float().to(device)
    initial_temp_torch = torch.from_numpy(initial_temp).float().to(device)
    
    pinn_pred, pinn_pred_denorm = rollout_pinn_model(
        pinn_model, initial_features_torch, initial_temp_torch,
        n_steps, feature_cols, target_cols, scaler, device
    )
    
    pinn_pred_np = pinn_pred_denorm.cpu().numpy() if pinn_pred_denorm is not None else pinn_pred.cpu().numpy()
    
    # RF rollout
    rf_pred = rollout_rf_teacher(
        rf_teacher, initial_features, n_steps, feature_cols, target_cols
    )
    
    # RC rollout
    rc_pred = rollout_rc_model(
        initial_temp, power, fan_speed, ambient_temp, n_steps
    )
    
    return {
        "pinn": pinn_pred_np,
        "rf": rf_pred,
        "rc": rc_pred
    }
