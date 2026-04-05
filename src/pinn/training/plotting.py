"""
Plotting utilities for training visualization.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8")


def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Training Loss Curves"
):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
        title: Plot title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    # Total loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train', alpha=0.8)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Data loss
    if 'train_loss_data' in history:
        axes[1].plot(history['train_loss_data'], label='Train', alpha=0.8)
    if 'val_loss_data' in history:
        axes[1].plot(history['val_loss_data'], label='Val', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Data Loss')
    axes[1].set_title('Data Loss (MSE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Physics loss
    if 'train_loss_physics' in history:
        axes[2].plot(history['train_loss_physics'], label='Train', alpha=0.8)
    if 'val_loss_physics' in history:
        axes[2].plot(history['val_loss_physics'], label='Val', alpha=0.8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Physics Loss')
    axes[2].set_title('Physics Residual')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Validation skill score
    if 'val_skill_rmse' in history:
        axes[3].plot(history['val_skill_rmse'], label='Skill (RMSE)', alpha=0.8, color='green')
        axes[3].axhline(0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Skill Score')
    axes[3].set_title('Validation Skill vs Persistence')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved loss curves: {save_path}")


def plot_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_name: str,
    save_path: Path,
    max_samples: int = 500
):
    """
    Plot actual vs predicted time series for a single target.
    
    Args:
        y_true: Ground truth, shape (n_samples,)
        y_pred: Predictions, shape (n_samples,)
        timestamps: DatetimeIndex
        target_name: Name of target
        save_path: Path to save figure
        max_samples: Maximum number of samples to plot
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Subsample if too many points
    if len(y_true) > max_samples:
        idx = np.linspace(0, len(y_true) - 1, max_samples, dtype=int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        timestamps = timestamps[idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series
    ax1.plot(timestamps, y_true, label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(timestamps, y_pred, label='Predicted', linewidth=1.2, alpha=0.8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Actual vs Predicted: {target_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    ax2.plot(timestamps, residuals, linewidth=1, alpha=0.7, color='red')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual (°C)')
    ax2.set_title('Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved actual vs pred plot: {save_path}")


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    save_path: Path
):
    """
    Plot residuals distribution for all targets.
    
    Args:
        y_true: Ground truth, shape (n_samples, n_targets)
        y_pred: Predictions, shape (n_samples, n_targets)
        target_cols: Target column names
        save_path: Path to save figure
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_targets = len(target_cols)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.ravel() if n_targets > 1 else [axes]
    
    for i, col in enumerate(target_cols):
        residuals = y_true[:, i] - y_pred[:, i]
        
        axes[i].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[i].set_xlabel('Residual (°C)')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{col}\nMean: {np.mean(residuals):.3f}, Std: {np.std(residuals):.3f}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Residuals Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved residuals distribution: {save_path}")


def plot_skill_comparison(
    metrics_df: pd.DataFrame,
    save_path: Path
):
    """
    Plot skill scores comparison across targets.
    
    Args:
        metrics_df: DataFrame with 'target', 'skill_RMSE', 'skill_MAE' columns
        save_path: Path to save figure
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter out overall row for per-target plot
    df = metrics_df[metrics_df['target'] != '__overall__'].copy()
    
    if 'skill_RMSE' not in df.columns:
        print("[WARN] No skill scores found in metrics, skipping skill comparison plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE skill
    ax1.barh(df['target'], df['skill_RMSE'], color='steelblue', alpha=0.8)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xlabel('Skill Score (RMSE)')
    ax1.set_ylabel('Target')
    ax1.set_title('RMSE Skill vs Persistence\n(positive = better than baseline)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # MAE skill
    if 'skill_MAE' in df.columns:
        ax2.barh(df['target'], df['skill_MAE'], color='darkorange', alpha=0.8)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_xlabel('Skill Score (MAE)')
        ax2.set_ylabel('Target')
        ax2.set_title('MAE Skill vs Persistence\n(positive = better than baseline)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved skill comparison: {save_path}")
