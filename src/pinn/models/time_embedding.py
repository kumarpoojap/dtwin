"""
Time embedding modules for temporal awareness in PINN.
"""

import math
from typing import Literal

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """Fourier-based time embedding with learnable frequencies."""
    
    def __init__(self, n_freqs: int = 16, learnable: bool = False):
        super().__init__()
        self.n_freqs = n_freqs
        self.output_dim = 2 * n_freqs  # cos + sin for each frequency
        
        # Initialize frequencies (log-spaced by default)
        freqs = torch.logspace(-3, 3, n_freqs)
        
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time indices, shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Time embeddings, shape (batch_size, 2*n_freqs)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        
        # Compute angular frequencies
        angles = 2 * math.pi * t * self.freqs.unsqueeze(0)  # (B, n_freqs)
        
        # Concatenate cos and sin
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, 2*n_freqs)
        return emb


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding (Transformer-style)."""
    
    def __init__(self, dim: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time indices, shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Time embeddings, shape (batch_size, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        
        args = t * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        return emb


def create_time_embedding(
    method: Literal["fourier", "sinusoidal"] = "fourier",
    n_freqs: int = 16,
    dim: int = 32
) -> nn.Module:
    """Factory function to create time embedding module."""
    if method == "fourier":
        return FourierTimeEmbedding(n_freqs=n_freqs)
    elif method == "sinusoidal":
        return SinusoidalTimeEmbedding(dim=dim)
    else:
        raise ValueError(f"Unknown time embedding method: {method}")
