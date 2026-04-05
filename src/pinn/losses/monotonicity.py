"""
Monotonic cooling constraint loss.

Ensures that increasing cooling effort (e.g., fan speed, airflow) does not increase predicted temperature.
Uses finite-difference perturbation to estimate dT/du_cool and penalizes positive gradients.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn


class MonotonicCoolingLoss(nn.Module):
    """
    Monotonic cooling constraint: dT/du_cool <= 0.
    
    Penalizes cases where increasing cooling actuator leads to higher predicted temperature.
    """
    
    def __init__(
        self,
        actuator_idx: int,
        epsilon: float = 0.01,
        penalty_type: Literal["relu", "quadratic"] = "relu"
    ):
        """
        Args:
            actuator_idx: Index of cooling actuator in feature vector
            epsilon: Perturbation magnitude for finite difference
            penalty_type: Type of penalty ('relu' or 'quadratic')
        """
        super().__init__()
        self.actuator_idx = actuator_idx
        self.epsilon = epsilon
        self.penalty_type = penalty_type
    
    def forward(
        self,
        model: nn.Module,
        X: torch.Tensor,
        t_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute monotonicity penalty.
        
        Args:
            model: PINN model
            X: Input features, shape (batch, n_features)
            t_idx: Time indices, shape (batch,)
        
        Returns:
            Scalar loss (penalty for violations)
        """
        # Baseline prediction
        with torch.no_grad():
            out_base = model(X, t_idx, return_physics_params=False)
            y_base = out_base["delta_y"]
        
        # Perturbed input (increase cooling actuator)
        X_perturbed = X.clone()
        X_perturbed[:, self.actuator_idx] += self.epsilon
        
        # Prediction with increased cooling
        out_perturbed = model(X_perturbed, t_idx, return_physics_params=False)
        y_perturbed = out_perturbed["delta_y"]
        
        # Gradient estimate: dT/du_cool ≈ (T_perturbed - T_base) / epsilon
        gradient = (y_perturbed - y_base) / self.epsilon
        
        # Penalize positive gradients (temperature should decrease or stay same with more cooling)
        if self.penalty_type == "relu":
            # ReLU penalty: only penalize violations
            penalty = torch.relu(gradient)
        elif self.penalty_type == "quadratic":
            # Quadratic penalty: penalize violations quadratically
            penalty = torch.relu(gradient) ** 2
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")
        
        # Mean penalty across batch and targets
        loss = torch.mean(penalty)
        
        return loss


def create_monotonic_loss(
    feature_cols: list,
    actuator_col: str = "evap_fan_speed_pct",
    epsilon: float = 0.01,
    penalty_type: Literal["relu", "quadratic"] = "relu"
) -> MonotonicCoolingLoss:
    """
    Factory function to create monotonic cooling loss.
    
    Args:
        feature_cols: List of feature column names
        actuator_col: Name of cooling actuator column
        epsilon: Perturbation magnitude
        penalty_type: Penalty type
    
    Returns:
        MonotonicCoolingLoss instance
    """
    if actuator_col not in feature_cols:
        raise ValueError(f"Actuator column '{actuator_col}' not found in features")
    
    actuator_idx = feature_cols.index(actuator_col)
    
    return MonotonicCoolingLoss(
        actuator_idx=actuator_idx,
        epsilon=epsilon,
        penalty_type=penalty_type
    )
