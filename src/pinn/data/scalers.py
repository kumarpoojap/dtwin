"""
PINN-specific scaler utilities (re-exports from common).

This module re-exports common target normalization utilities for backward compatibility.
The actual implementation is in src.common.scalers.
"""

from ...common.scalers import (
    TargetScaler,
    compute_train_target_scaler,
    apply_target_normalization,
    invert_target_normalization,
)

__all__ = [
    "TargetScaler",
    "compute_train_target_scaler",
    "apply_target_normalization",
    "invert_target_normalization",
]
