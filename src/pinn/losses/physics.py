"""
Physics-informed loss: thermal ODE residual.

Lumped thermal model (discrete-time):
    T(t+Δt) - T(t) = Δt * [ -h*(T(t) - T_supply(t)) + beta*u_cool(t) + gamma*Q_load(t) ] / C

Where:
    - T: rack temperature
    - T_supply: supply air temperature (from features)
    - u_cool: cooling actuator (airflow or fan speed)
    - Q_load: IT load proxy (from features, e.g., cool_demand_kwh)
    - C, h, beta, gamma: learnable physics parameters (per target)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class PhysicsODELoss(nn.Module):
    """
    Thermal ODE residual loss.
    
    Computes the mismatch between predicted temperature change and physics-based prediction.
    """
    
    def __init__(
        self,
        window_size: int = 12,
        dt: float = 10.0,
        use_load_proxy: bool = True
    ):
        """
        Args:
            window_size: Number of past steps to average supply/airflow/load
            dt: Time step in seconds (cadence)
            use_load_proxy: Whether to include load term in ODE
        """
        super().__init__()
        self.window_size = window_size
        self.dt = dt
        self.use_load_proxy = use_load_proxy
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_current: torch.Tensor,
        physics_params: Dict[str, torch.Tensor],
        supply_air: torch.Tensor,
        cooling_actuator: torch.Tensor,
        load_proxy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics residual.
        
        Args:
            y_pred: Predicted future temperature, shape (batch, n_targets)
            y_current: Current temperature, shape (batch, n_targets)
            physics_params: Dict with C, h, beta, gamma (each shape (n_targets,))
            supply_air: Supply air temperature (past-windowed mean), shape (batch,)
            cooling_actuator: Cooling actuator value (past-windowed mean), shape (batch,)
            load_proxy: Load proxy (past-windowed mean), shape (batch,) or None
        
        Returns:
            Scalar loss (mean squared residual)
        """
        C = physics_params["C"]  # (n_targets,)
        h = physics_params["h"]
        beta = physics_params["beta"]
        gamma = physics_params["gamma"]
        
        batch_size, n_targets = y_pred.shape
        
        # Expand supply_air and actuator to match targets
        supply_air_exp = supply_air.unsqueeze(-1).expand(batch_size, n_targets)  # (batch, n_targets)
        cooling_exp = cooling_actuator.unsqueeze(-1).expand(batch_size, n_targets)
        
        # Physics-based temperature change
        # dT/dt = [-h*(T - T_supply) + beta*u_cool + gamma*Q_load] / C
        heat_transfer = -h.unsqueeze(0) * (y_current - supply_air_exp)  # (batch, n_targets)
        cooling_effect = beta.unsqueeze(0) * cooling_exp
        
        if self.use_load_proxy and load_proxy is not None:
            load_exp = load_proxy.unsqueeze(-1).expand(batch_size, n_targets)
            load_effect = gamma.unsqueeze(0) * load_exp
        else:
            load_effect = 0.0
        
        dT_dt_physics = (heat_transfer + cooling_effect + load_effect) / C.unsqueeze(0)
        
        # Predicted change over k*dt seconds
        delta_T_physics = dT_dt_physics * self.dt * self.window_size  # Approximate over k steps
        
        # Actual predicted change
        delta_T_pred = y_pred - y_current
        
        # Residual: difference between predicted and physics-based change
        residual = delta_T_pred - delta_T_physics
        
        # Mean squared residual
        loss = torch.mean(residual ** 2)
        
        return loss


def extract_physics_drivers(
    X: torch.Tensor,
    feature_cols: list,
    supply_col: str = "supply_air_c",
    actuator_col: str = "evap_fan_speed_pct",
    load_col: Optional[str] = "cool_demand_kwh",
    window_size: int = 12
) -> Dict[str, torch.Tensor]:
    """
    Extract physics driver features from input tensor.
    
    Uses rolling mean features if available, else falls back to base columns.
    
    Args:
        X: Input features, shape (batch, n_features)
        feature_cols: List of feature column names
        supply_col: Base name for supply air temperature
        actuator_col: Base name for cooling actuator
        load_col: Base name for load proxy (or None)
        window_size: Window size for rolling mean
    
    Returns:
        Dict with 'supply_air', 'cooling_actuator', 'load_proxy' (each shape (batch,))
    """
    # Try to find rolling mean features first (past-looking)
    supply_roll_name = f"{supply_col}_roll{window_size}_mean"
    actuator_roll_name = f"{actuator_col}_roll{window_size}_mean"
    load_roll_name = f"{load_col}_roll{window_size}_mean" if load_col else None
    
    # Extract supply air
    if supply_roll_name in feature_cols:
        supply_idx = feature_cols.index(supply_roll_name)
        supply_air = X[:, supply_idx]
    elif supply_col in feature_cols:
        supply_idx = feature_cols.index(supply_col)
        supply_air = X[:, supply_idx]
    else:
        raise ValueError(f"Supply air column not found: {supply_col}")
    
    # Extract cooling actuator
    if actuator_roll_name in feature_cols:
        actuator_idx = feature_cols.index(actuator_roll_name)
        cooling_actuator = X[:, actuator_idx]
    elif actuator_col in feature_cols:
        actuator_idx = feature_cols.index(actuator_col)
        cooling_actuator = X[:, actuator_idx]
    else:
        raise ValueError(f"Cooling actuator column not found: {actuator_col}")
    
    # Extract load proxy
    load_proxy = None
    if load_col:
        if load_roll_name in feature_cols:
            load_idx = feature_cols.index(load_roll_name)
            load_proxy = X[:, load_idx]
        elif load_col in feature_cols:
            load_idx = feature_cols.index(load_col)
            load_proxy = X[:, load_idx]
    
    return {
        "supply_air": supply_air,
        "cooling_actuator": cooling_actuator,
        "load_proxy": load_proxy
    }
