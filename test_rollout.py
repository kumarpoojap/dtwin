"""
Quick test script for multi-step rollout functionality.
"""

import numpy as np
import torch
from pathlib import Path

from src.pinn.models.rollout import rollout_rc_model, compute_rollout_metrics


def test_rc_rollout():
    """Test RC model rollout."""
    print("="*60)
    print("Testing RC Model Rollout")
    print("="*60)
    
    # Simulate a simple scenario
    batch_size = 5
    n_steps = 90  # 90 seconds
    n_targets = 1
    
    # Initial conditions
    initial_temp = np.ones((batch_size, n_targets)) * 40.0  # Start at 40°C
    
    # Inputs over time
    power = np.ones((batch_size, n_steps)) * 150.0  # Constant 150W
    fan_speed = np.ones((batch_size, n_steps)) * 50.0  # 50% fan speed
    ambient_temp = np.ones((batch_size, n_steps)) * 25.0  # 25°C ambient
    
    # Add some variation to one sample
    power[0, 30:60] = 250.0  # Power spike
    fan_speed[1, 40:70] = 80.0  # Fan increase
    
    # Run rollout
    predictions = rollout_rc_model(
        initial_temp, power, fan_speed, ambient_temp,
        n_steps, dt=1.0,
        C=100.0, h=0.05, beta=-0.03, gamma=0.01  # Increased from 0.001
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Initial temp: {initial_temp[0, 0]:.2f}°C")
    print(f"Final temp (sample 0): {predictions[0, -1, 0]:.2f}°C")
    print(f"Final temp (sample 1): {predictions[1, -1, 0]:.2f}°C")
    print(f"Temperature range: [{predictions.min():.2f}, {predictions.max():.2f}]°C")
    
    # Check for stability (no NaN or extreme values)
    assert not np.isnan(predictions).any(), "NaN values detected!"
    assert predictions.min() > 0, "Negative temperatures detected!"
    assert predictions.max() < 150, "Unrealistic high temperatures!"
    
    print("\n[SUCCESS] RC rollout test passed!")
    return predictions


def test_rollout_metrics():
    """Test rollout metrics computation."""
    print("\n" + "="*60)
    print("Testing Rollout Metrics")
    print("="*60)
    
    # Create dummy predictions and ground truth
    batch_size = 10
    n_steps = 90
    n_targets = 1
    
    predictions = np.random.randn(batch_size, n_steps, n_targets) * 5 + 50
    ground_truth = np.random.randn(batch_size, n_steps, n_targets) * 5 + 50
    
    # Compute metrics
    horizons = [10, 30, 60, 90]
    metrics = compute_rollout_metrics(predictions, ground_truth, horizons)
    
    print(f"Horizons: {metrics['horizons']}")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"Drift: {metrics['drift']}")
    
    # Check metrics are reasonable
    assert len(metrics['mae']) == len(horizons), "Metric length mismatch!"
    assert all(metrics['mae'] >= 0), "Negative MAE detected!"
    assert all(metrics['rmse'] >= 0), "Negative RMSE detected!"
    
    print("\n[SUCCESS] Metrics test passed!")
    return metrics


def test_physics_consistency():
    """Test that RC model follows expected physics."""
    print("\n" + "="*60)
    print("Testing Physics Consistency")
    print("="*60)
    
    batch_size = 1
    n_steps = 60
    n_targets = 1
    
    # Test 1: No power, no fan → temperature should approach ambient
    print("\nTest 1: Cooling to ambient")
    initial_temp = np.ones((batch_size, n_targets)) * 60.0
    power = np.zeros((batch_size, n_steps))
    fan_speed = np.zeros((batch_size, n_steps))
    ambient_temp = np.ones((batch_size, n_steps)) * 25.0
    
    pred1 = rollout_rc_model(
        initial_temp, power, fan_speed, ambient_temp, n_steps,
        dt=1.0, C=100.0, h=0.05, beta=-0.03, gamma=0.01
    )
    
    print(f"  Initial: {initial_temp[0, 0]:.2f}°C")
    print(f"  Final: {pred1[0, -1, 0]:.2f}°C")
    print(f"  Ambient: {ambient_temp[0, 0]:.2f}°C")
    assert pred1[0, -1, 0] < initial_temp[0, 0], "Temperature should decrease!"
    assert pred1[0, -1, 0] > ambient_temp[0, 0], "Should not go below ambient!"
    
    # Test 2: High power → temperature should increase
    print("\nTest 2: Heating with power")
    initial_temp = np.ones((batch_size, n_targets)) * 40.0
    power = np.ones((batch_size, n_steps)) * 300.0  # High power
    fan_speed = np.ones((batch_size, n_steps)) * 30.0  # Low fan
    ambient_temp = np.ones((batch_size, n_steps)) * 25.0
    
    pred2 = rollout_rc_model(
        initial_temp, power, fan_speed, ambient_temp, n_steps,
        dt=1.0, C=100.0, h=0.05, beta=-0.03, gamma=0.01
    )
    
    print(f"  Initial: {initial_temp[0, 0]:.2f}°C")
    print(f"  Final: {pred2[0, -1, 0]:.2f}°C")
    assert pred2[0, -1, 0] > initial_temp[0, 0], "Temperature should increase with power!"
    
    # Test 3: High fan → temperature should decrease more
    print("\nTest 3: Cooling with fan")
    initial_temp = np.ones((batch_size, n_targets)) * 60.0
    power = np.ones((batch_size, n_steps)) * 100.0
    fan_speed_low = np.ones((batch_size, n_steps)) * 30.0
    fan_speed_high = np.ones((batch_size, n_steps)) * 90.0
    ambient_temp = np.ones((batch_size, n_steps)) * 25.0
    
    pred3_low = rollout_rc_model(
        initial_temp, power, fan_speed_low, ambient_temp, n_steps,
        dt=1.0, C=100.0, h=0.05, beta=-0.03, gamma=0.01
    )
    pred3_high = rollout_rc_model(
        initial_temp, power, fan_speed_high, ambient_temp, n_steps,
        dt=1.0, C=100.0, h=0.05, beta=-0.03, gamma=0.01
    )
    
    print(f"  Final (low fan): {pred3_low[0, -1, 0]:.2f}°C")
    print(f"  Final (high fan): {pred3_high[0, -1, 0]:.2f}°C")
    assert pred3_high[0, -1, 0] < pred3_low[0, -1, 0], "High fan should cool more!"
    
    print("\n[SUCCESS] Physics consistency tests passed!")


if __name__ == "__main__":
    print("Multi-Step Rollout Test Suite\n")
    
    # Run tests
    predictions = test_rc_rollout()
    metrics = test_rollout_metrics()
    test_physics_consistency()
    
    print("\n" + "="*60)
    print("All Tests Passed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python eval/evaluate_surrogate.py --config configs/train_gpu_pinn.yaml")
    print("2. Check results in: results/surrogate_eval/")
