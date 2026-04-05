"""
PINN-specific feature utilities (re-exports from common).

This module re-exports common feature engineering utilities for backward compatibility.
The actual implementation is in src.common.features.
"""

from ...common.features import (
    ensure_datetime_index,
    add_lag_features,
    add_rolling_features,
    compute_winsor_bounds,
    apply_winsorization,
    drop_low_variance_features,
    build_feature_column_names,
    build_official_features,
    validate_feature_columns,
    materialize_features_from_list,
)

__all__ = [
    "ensure_datetime_index",
    "add_lag_features",
    "add_rolling_features",
    "compute_winsor_bounds",
    "apply_winsorization",
    "drop_low_variance_features",
    "build_feature_column_names",
    "build_official_features",
    "validate_feature_columns",
    "materialize_features_from_list",
]
