"""
Gym-like environment wrapper for Hybrid PINN.

Provides a minimal RL interface for future integration with RL algorithms.
The environment is deterministic and uses the PINN to predict next states.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.pinn.data.features import build_official_features
from src.pinn.data.scalers import TargetScaler
from src.pinn.models.hybrid_pinn import HybridPINN


class PINNEnv:
    """
    Gym-like environment using Hybrid PINN for state transitions.
    
    State: Compact feature vector (exogenous sensors/actuators)
    Action: Control inputs (e.g., fan speed, airflow)
    Reward: Negative temperature deviation from target (to be customized)
    
    This is a deterministic environment for now (no stochasticity).
    """
    
    def __init__(
        self,
        model_checkpoint: Path,
        config_path: Path,
        scaler_path: Optional[Path] = None,
        target_temp: float = 25.0,
        device: str = "cpu"
    ):
        """
        Initialize PINN environment.
        
        Args:
            model_checkpoint: Path to trained PINN checkpoint
            config_path: Path to training config YAML
            scaler_path: Path to target scaler JSON (if targets were normalized)
            target_temp: Target temperature for reward computation (°C)
            device: Device to run model on
        """
        self.device = device
        self.target_temp = target_temp
        
        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path and scaler_path.exists():
            self.scaler = TargetScaler.load(scaler_path)
        
        # Feature and target info
        self.feature_cols = None  # Will be set from config
        self.target_cols = None
        self.base_cols = self.cfg["data"]["features"]["base_cols"]
        self.lags = self.cfg["data"]["features"]["lags"]
        self.roll_windows = self.cfg["data"]["features"]["roll_windows"]
        
        # Load model
        self.model = self._load_model(model_checkpoint)
        self.model.eval()
        
        # State tracking
        self.current_state = None
        self.current_targets = None
        self.timestep = 0
        self.max_timesteps = 1000  # Episode length limit
        
        # Action space info (to be customized based on actuators)
        self.action_dim = len([c for c in self.base_cols if "fan" in c or "airflow" in c])
        
        # Observation space info
        self.obs_dim = len(self.base_cols)
    
    def _load_model(self, checkpoint_path: Path) -> HybridPINN:
        """Load trained PINN model."""
        # Infer dimensions from config
        # This is a simplification; ideally load from saved metadata
        n_features = len(self.base_cols) * (1 + len(self.lags) + 3 * len(self.roll_windows))
        
        # Load target cols from spec
        import json
        spec_path = Path(self.cfg["data"]["spec_path"])
        with open(spec_path, "r") as f:
            spec = json.load(f)
        self.target_cols = spec["target_cols_raw"]
        n_targets = len(self.target_cols)
        
        model = HybridPINN(
            input_dim=n_features,
            output_dim=n_targets,
            hidden_dims=self.cfg["model"]["hidden_dims"],
            activation=self.cfg["model"]["activation"],
            dropout=0.0,  # No dropout at inference
            time_embedding_enabled=self.cfg["model"]["time_embedding"]["enabled"],
            time_embedding_method=self.cfg["model"]["time_embedding"]["method"],
            time_embedding_n_freqs=self.cfg["model"]["time_embedding"]["n_freqs"],
            physics_head_enabled=self.cfg["model"]["physics_head"]["enabled"]
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            initial_state: Optional initial state (base features). If None, use defaults.
        
        Returns:
            Initial observation (state)
        """
        self.timestep = 0
        
        if initial_state is not None:
            self.current_state = initial_state.copy()
        else:
            # Default initial state (example values)
            self.current_state = np.array([
                10.0,   # airflow_lps
                5.0,    # cool_output_kwh
                4.0,    # cool_demand_kwh
                50.0,   # evap_fan_speed_pct
                22.0,   # return_air_c
                18.0,   # suction_c
                16.0,   # supply_air_c
                20.0    # min_rack_inlet_c
            ])
        
        # Initialize targets to target_temp
        self.current_targets = np.full(len(self.target_cols), self.target_temp)
        
        return self.current_state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Control action (e.g., fan speed adjustment)
        
        Returns:
            (next_state, reward, done, info)
        """
        # Apply action to state (modify actuators)
        # This is a simplified example; customize based on your action space
        next_state = self.current_state.copy()
        
        # Example: action modifies fan speed (index 3)
        if self.action_dim > 0:
            next_state[3] = np.clip(action[0], 0.0, 100.0)  # evap_fan_speed_pct
        
        # Build features from current state
        # (This is simplified; in practice, you'd maintain a history buffer for lags/rolling)
        df_state = pd.DataFrame([next_state], columns=self.base_cols)
        
        # For simplicity, assume no lags/rolling (just use current state)
        # In a real implementation, you'd maintain a sliding window buffer
        X_features = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict next temperatures using PINN
        with torch.no_grad():
            t_idx = torch.tensor([self.timestep], dtype=torch.float32).to(self.device)
            out = self.model(X_features, t_idx, return_physics_params=False)
            delta_y = out["delta_y"].cpu().numpy()[0]
        
        # Update targets (absolute prediction)
        next_targets = self.current_targets + delta_y
        
        # Inverse transform if normalized
        if self.scaler is not None:
            next_targets = self.scaler.inverse_transform_array(next_targets.reshape(1, -1))[0]
        
        # Compute reward (negative deviation from target)
        temp_deviation = np.abs(next_targets - self.target_temp)
        reward = -np.mean(temp_deviation)  # Negative mean deviation
        
        # Update state
        self.current_state = next_state
        self.current_targets = next_targets
        self.timestep += 1
        
        # Check if done
        done = self.timestep >= self.max_timesteps
        
        # Info dict
        info = {
            "timestep": self.timestep,
            "temperatures": next_targets.tolist(),
            "mean_temp": float(np.mean(next_targets)),
            "max_temp": float(np.max(next_targets)),
            "min_temp": float(np.min(next_targets))
        }
        
        return self.current_state.copy(), reward, done, info
    
    def render(self, mode: str = "human"):
        """Render environment state (optional)."""
        if mode == "human":
            print(f"Timestep: {self.timestep}")
            print(f"State: {self.current_state}")
            print(f"Temperatures: {self.current_targets}")
            print(f"Mean Temp: {np.mean(self.current_targets):.2f} °C")
    
    def close(self):
        """Cleanup (optional)."""
        pass


def create_pinn_env(
    model_checkpoint: Path,
    config_path: Path,
    scaler_path: Optional[Path] = None,
    target_temp: float = 25.0,
    device: str = "cpu"
) -> PINNEnv:
    """
    Factory function to create PINN environment.
    
    Args:
        model_checkpoint: Path to trained model
        config_path: Path to config YAML
        scaler_path: Path to scaler JSON
        target_temp: Target temperature
        device: Device
    
    Returns:
        PINNEnv instance
    """
    return PINNEnv(
        model_checkpoint=model_checkpoint,
        config_path=config_path,
        scaler_path=scaler_path,
        target_temp=target_temp,
        device=device
    )


# Example usage
if __name__ == "__main__":
    # Example: create and test environment
    env = create_pinn_env(
        model_checkpoint=Path("artifacts/best_model.pt"),
        config_path=Path("configs/train_hybrid_pinn.yaml"),
        scaler_path=Path("artifacts/scalers.json"),
        target_temp=25.0,
        device="cpu"
    )
    
    # Reset
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    # Take a few steps
    for i in range(5):
        action = np.array([50.0 + i * 5])  # Increase fan speed
        obs, reward, done, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Mean Temp: {info['mean_temp']:.2f} °C")
        
        if done:
            break
    
    env.close()
