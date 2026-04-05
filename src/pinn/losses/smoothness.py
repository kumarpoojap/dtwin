"""
Temporal smoothness loss.

Penalizes abrupt changes in predictions across adjacent time steps.
"""

from typing import Literal

import torch
import torch.nn as nn


class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal smoothness penalty for predictions.
    
    Encourages smooth trajectories by penalizing large differences between
    predictions at consecutive time steps.
    """
    
    def __init__(
        self,
        order: int = 1,
        reduction: Literal["mean", "sum"] = "mean"
    ):
        """
        Args:
            order: Order of differences (1 = first-order, 2 = second-order)
            reduction: How to reduce the loss ('mean' or 'sum')
        """
        super().__init__()
        self.order = order
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss.
        
        Args:
            y_pred: Predictions, shape (batch, n_targets)
                    Assumes batch is ordered by time
        
        Returns:
            Scalar loss
        """
        if y_pred.shape[0] < 2:
            # Need at least 2 samples for first-order difference
            return torch.tensor(0.0, device=y_pred.device)
        
        if self.order == 1:
            # First-order differences: y[t+1] - y[t]
            diff = y_pred[1:] - y_pred[:-1]
        elif self.order == 2:
            # Second-order differences: (y[t+2] - y[t+1]) - (y[t+1] - y[t])
            if y_pred.shape[0] < 3:
                return torch.tensor(0.0, device=y_pred.device)
            first_diff = y_pred[1:] - y_pred[:-1]
            diff = first_diff[1:] - first_diff[:-1]
        else:
            raise ValueError(f"Unsupported order: {self.order}")
        
        # Squared differences
        loss = torch.sum(diff ** 2)
        
        if self.reduction == "mean":
            loss = loss / diff.numel()
        
        return loss
