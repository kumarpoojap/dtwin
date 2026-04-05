"""
Hybrid PINN model: MLP backbone + time embedding + optional physics parameters head.
"""

from typing import List, Literal, Optional

import torch
import torch.nn as nn

from .time_embedding import create_time_embedding


class PhysicsParamsHead(nn.Module):
    """Learnable physics parameters per target (C, h, beta, gamma)."""
    
    def __init__(self, n_targets: int):
        super().__init__()
        self.n_targets = n_targets
        
        # Initialize parameters (will be passed through softplus to ensure > 0)
        # C: thermal capacitance, h: heat transfer coeff, beta: cooling efficiency, gamma: load coupling
        self.C_raw = nn.Parameter(torch.randn(n_targets) * 0.1)
        self.h_raw = nn.Parameter(torch.randn(n_targets) * 0.1)
        self.beta_raw = nn.Parameter(torch.randn(n_targets) * 0.1)
        self.gamma_raw = nn.Parameter(torch.randn(n_targets) * 0.1)
    
    def forward(self) -> dict:
        """Return physics parameters (all positive via softplus)."""
        return {
            "C": torch.nn.functional.softplus(self.C_raw),
            "h": torch.nn.functional.softplus(self.h_raw),
            "beta": torch.nn.functional.softplus(self.beta_raw),
            "gamma": torch.nn.functional.softplus(self.gamma_raw)
        }


class HybridPINN(nn.Module):
    """
    Hybrid Physics-Informed Neural Network for k-ahead temperature prediction.
    
    Architecture:
        - Input: engineered features (exogenous-only) + time embedding
        - Backbone: MLP with SiLU activation
        - Output: Δy (change over k steps), reconstruct y_hat = y(t) + Δy
        - Optional: physics parameters head for ODE residual computation
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128, 128],
        activation: Literal["relu", "silu", "gelu"] = "silu",
        dropout: float = 0.1,
        time_embedding_enabled: bool = True,
        time_embedding_method: Literal["fourier", "sinusoidal"] = "fourier",
        time_embedding_n_freqs: int = 16,
        physics_head_enabled: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_embedding_enabled = time_embedding_enabled
        self.physics_head_enabled = physics_head_enabled
        
        # Time embedding
        if time_embedding_enabled:
            self.time_emb = create_time_embedding(
                method=time_embedding_method,
                n_freqs=time_embedding_n_freqs
            )
            time_emb_dim = self.time_emb.output_dim if hasattr(self.time_emb, 'output_dim') else 2 * time_embedding_n_freqs
        else:
            self.time_emb = None
            time_emb_dim = 0
        
        # MLP backbone
        backbone_input_dim = input_dim + time_emb_dim
        
        layers = []
        prev_dim = backbone_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (predicts Δy)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.backbone = nn.Sequential(*layers)
        
        # Physics parameters head (optional)
        if physics_head_enabled:
            self.physics_head = PhysicsParamsHead(n_targets=output_dim)
        else:
            self.physics_head = None
    
    def forward(
        self,
        x: torch.Tensor,
        t_idx: Optional[torch.Tensor] = None,
        return_physics_params: bool = False
    ) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (batch_size, input_dim)
            t_idx: Time indices, shape (batch_size,) or None
            return_physics_params: Whether to return physics parameters
        
        Returns:
            dict with:
                - delta_y: Predicted change, shape (batch_size, output_dim)
                - physics_params: dict of physics params (if requested and enabled)
        """
        batch_size = x.shape[0]
        
        # Time embedding
        if self.time_embedding_enabled:
            if t_idx is None:
                # Default: use batch indices as time
                t_idx = torch.arange(batch_size, device=x.device, dtype=torch.float32)
            
            t_emb = self.time_emb(t_idx)
            x_in = torch.cat([x, t_emb], dim=-1)
        else:
            x_in = x
        
        # MLP backbone
        delta_y = self.backbone(x_in)
        
        result = {"delta_y": delta_y}
        
        # Physics parameters
        if return_physics_params and self.physics_head is not None:
            result["physics_params"] = self.physics_head()
        
        return result
    
    def predict_absolute(
        self,
        x: torch.Tensor,
        y_current: torch.Tensor,
        t_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict absolute y(t+k) = y(t) + Δy.
        
        Args:
            x: Input features
            y_current: Current target values y(t), shape (batch_size, output_dim)
            t_idx: Time indices
        
        Returns:
            y_pred: Predicted y(t+k), shape (batch_size, output_dim)
        """
        out = self.forward(x, t_idx, return_physics_params=False)
        delta_y = out["delta_y"]
        y_pred = y_current + delta_y
        return y_pred
