"""
RandomForest teacher utilities for distillation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch


def sha256_str(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fingerprint_list(items: List[str]) -> str:
    """Deterministic fingerprint over an ordered list."""
    return sha256_str("|".join(items))


class TeacherRF:
    """Wrapper for RandomForest teacher model with caching support."""
    
    def __init__(
        self,
        model_path: Path,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Load RandomForest teacher bundle.
        
        Args:
            model_path: Path to joblib bundle (dict with 'model', 'feature_columns', etc.)
            cache_dir: Directory to cache teacher predictions
            use_cache: Whether to use cached predictions
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        if cache_dir and use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load bundle
        print(f"[INFO] Loading teacher bundle: {model_path}")
        bundle = joblib.load(model_path)
        
        # Validate bundle structure
        required_keys = ["model", "feature_columns", "target_columns"]
        for key in required_keys:
            if key not in bundle:
                raise ValueError(f"Teacher bundle missing required key: {key}")
        
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_columns"]
        self.target_cols = bundle["target_columns"]
        self.k_ahead = bundle.get("k_ahead", 12)
        self.cadence_seconds = bundle.get("cadence_seconds", 10.0)
        
        # Metadata
        self.feature_fp = bundle.get("feature_fingerprint", fingerprint_list(self.feature_cols))
        self.target_fp = bundle.get("target_fingerprint", fingerprint_list(self.target_cols))
        
        print(f"[INFO] Teacher loaded: {len(self.feature_cols)} features → {len(self.target_cols)} targets")
        print(f"[INFO] k_ahead={self.k_ahead}, cadence={self.cadence_seconds}s")
    
    def predict(
        self,
        X: pd.DataFrame,
        return_tensor: bool = True
    ) -> np.ndarray | torch.Tensor:
        """
        Predict using teacher model.
        
        Args:
            X: Features DataFrame (must have all required feature columns)
            return_tensor: Whether to return torch.Tensor (else numpy array)
        
        Returns:
            Predictions, shape (n_samples, n_targets)
        """
        # Validate columns
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns for teacher: {missing[:10]}...")
        
        # Predict
        X_ordered = X[self.feature_cols]
        y_pred = self.model.predict(X_ordered.values)
        
        if return_tensor:
            return torch.tensor(y_pred, dtype=torch.float32)
        return y_pred
    
    def get_or_compute_predictions(
        self,
        X: pd.DataFrame,
        split_name: str,
        return_tensor: bool = True
    ) -> np.ndarray | torch.Tensor:
        """
        Get cached predictions or compute and cache them.
        
        Args:
            X: Features DataFrame
            split_name: Name of split (e.g., 'train', 'val', 'test')
            return_tensor: Whether to return torch.Tensor
        
        Returns:
            Predictions
        """
        if not self.use_cache or self.cache_dir is None:
            return self.predict(X, return_tensor=return_tensor)
        
        # Compute cache key
        cache_key = f"{split_name}_{self.feature_fp[:12]}_{len(X)}.npy"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            print(f"[INFO] Loading cached teacher predictions: {cache_path}")
            y_pred = np.load(cache_path)
        else:
            print(f"[INFO] Computing teacher predictions for {split_name}...")
            y_pred = self.predict(X, return_tensor=False)
            np.save(cache_path, y_pred)
            print(f"[INFO] Cached predictions to: {cache_path}")
        
        if return_tensor:
            return torch.tensor(y_pred, dtype=torch.float32)
        return y_pred
    
    def validate_compatibility(
        self,
        feature_cols: List[str],
        target_cols: List[str]
    ) -> bool:
        """
        Validate that feature/target columns match teacher expectations.
        
        Returns:
            True if compatible, raises ValueError otherwise
        """
        if feature_cols != self.feature_cols:
            raise ValueError(
                f"Feature column mismatch!\n"
                f"Expected: {len(self.feature_cols)} cols (fp={self.feature_fp[:12]})\n"
                f"Got: {len(feature_cols)} cols"
            )
        
        if target_cols != self.target_cols:
            raise ValueError(
                f"Target column mismatch!\n"
                f"Expected: {self.target_cols}\n"
                f"Got: {target_cols}"
            )
        
        return True


def load_teacher(
    model_path: Path,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    allow_missing: bool = False
) -> Optional[TeacherRF]:
    """
    Load teacher model with graceful fallback.
    
    Args:
        model_path: Path to teacher bundle
        cache_dir: Cache directory
        use_cache: Whether to use cache
        allow_missing: If True, return None if model not found (else raise)
    
    Returns:
        TeacherRF instance or None
    """
    if not model_path.exists():
        if allow_missing:
            print(f"[WARN] Teacher model not found: {model_path}")
            print("[INFO] Training will proceed without teacher distillation.")
            return None
        else:
            raise FileNotFoundError(f"Teacher model not found: {model_path}")
    
    return TeacherRF(model_path, cache_dir=cache_dir, use_cache=use_cache)
