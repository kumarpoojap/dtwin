"""
Target normalization (TRAIN-only statistics).

Shared by both RF and PINN models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


class TargetScaler:
    """Z-score normalization for targets, fitted on TRAIN split only."""
    
    def __init__(self):
        self.stats: Optional[Dict[str, Dict[str, float]]] = None
        self.target_cols: Optional[list] = None
    
    def fit(self, y_train: pd.DataFrame) -> TargetScaler:
        """Compute mean and std from training targets."""
        self.target_cols = list(y_train.columns)
        self.stats = {}
        for col in self.target_cols:
            mu = float(y_train[col].mean())
            sd = float(y_train[col].std(ddof=0))
            if sd < 1e-8:
                sd = 1.0
            self.stats[col] = {"mean": mu, "std": sd}
        return self
    
    def transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization."""
        if self.stats is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        y_norm = {}
        for col in y.columns:
            if col not in self.stats:
                raise KeyError(f"Column '{col}' not found in fitted stats")
            mu = self.stats[col]["mean"]
            sd = self.stats[col]["std"]
            y_norm[col] = (y[col] - mu) / sd
        
        return pd.DataFrame(y_norm, index=y.index)
    
    def inverse_transform(self, y_norm: pd.DataFrame) -> pd.DataFrame:
        """Invert z-score normalization."""
        if self.stats is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        y_raw = {}
        for col in y_norm.columns:
            if col not in self.stats:
                raise KeyError(f"Column '{col}' not found in fitted stats")
            mu = self.stats[col]["mean"]
            sd = self.stats[col]["std"]
            y_raw[col] = y_norm[col] * sd + mu
        
        return pd.DataFrame(y_raw, index=y_norm.index)
    
    def inverse_transform_array(self, y_norm: np.ndarray) -> np.ndarray:
        """Invert z-score for numpy array (assumes column order matches target_cols)."""
        if self.stats is None or self.target_cols is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        y_raw = np.zeros_like(y_norm)
        for i, col in enumerate(self.target_cols):
            mu = self.stats[col]["mean"]
            sd = self.stats[col]["std"]
            y_raw[:, i] = y_norm[:, i] * sd + mu
        
        return y_raw
    
    def save(self, path: Path):
        """Save scaler stats to JSON."""
        if self.stats is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "target_cols": self.target_cols,
                "stats": self.stats
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> TargetScaler:
        """Load scaler stats from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        
        scaler = cls()
        scaler.target_cols = data["target_cols"]
        scaler.stats = data["stats"]
        return scaler


# Legacy function-based API for backward compatibility
def compute_train_target_scaler(y_train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Legacy function: compute target normalization stats from training data."""
    scaler = TargetScaler()
    scaler.fit(y_train_df)
    return scaler.stats


def apply_target_normalization(y_df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Legacy function: apply target normalization using pre-computed stats."""
    scaler = TargetScaler()
    scaler.stats = stats
    scaler.target_cols = list(y_df.columns)
    return scaler.transform(y_df)


def invert_target_normalization(y_pred_z: np.ndarray, y_stats: Dict[str, Dict[str, float]], cols: list) -> pd.DataFrame:
    """Legacy function: invert target normalization."""
    scaler = TargetScaler()
    scaler.stats = y_stats
    scaler.target_cols = cols
    y_raw = scaler.inverse_transform_array(y_pred_z)
    return pd.DataFrame(y_raw, columns=cols)
