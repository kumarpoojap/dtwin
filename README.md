# Hybrid PINN + Teacher (Datacenter Rack Temperature)

Production-grade **Hybrid Physics-Informed Neural Network** for **k-step ahead rack temperature forecasting** with teacher distillation, physics constraints, and RL integration.

## Overview

This project implements a complete training and evaluation pipeline for datacenter thermal management:

- **k-ahead prediction**: Forecast rack temperatures 120s ahead (k=12 @ 10s cadence)
- **Exogenous-only features**: No target lags (RL-compatible)
- **Teacher distillation**: Optional knowledge transfer from RandomForest
- **Physics-informed**: Lumped thermal ODE residual
- **Monotonic cooling**: Ensures increasing cooling effort reduces temperature
- **Curriculum learning**: 3-phase progressive training
- **Comprehensive evaluation**: Skill scores vs persistence baseline

## Architecture

```
Data Flow:
  Parquet (10s) → Feature Builder → k-ahead Labels → Hybrid PINN
                                                          ↓
  Teacher (RF) ────────────────────────────────→ Distillation Loss

Model Components:
  - MLP Backbone (3 layers, SiLU activation)
  - Fourier Time Embedding (temporal awareness)
  - Physics Parameters Head (learnable C, h, β, γ)
  - Multi-output prediction (12 temperature probes)

Loss Functions:
  L_total = λ_data·MSE + λ_teacher·MSE + λ_physics·ODE + λ_mono·Constraint + λ_smooth·Temporal
```

## Quick Start

See **[QUICKSTART.md](QUICKSTART.md)** for detailed walkthrough.

### Minimal Example

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Dev run (smoke test)
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml --dev-run

# 3. Evaluate
python -m eval.evaluate_model --config configs/train_hybrid_pinn.yaml
```

## Project Structure

```
Digital-twin1/
├── configs/
│   └── train_hybrid_pinn.yaml      # All hyperparameters
├── src/
│   └── pinn/
│       ├── data/
│       │   ├── features.py         # Official feature builder ⭐
│       │   ├── scalers.py          # Target normalization
│       │   └── dataset_k_ahead.py  # Data loader + splits
│       ├── models/
│       │   ├── hybrid_pinn.py      # Main model
│       │   ├── time_embedding.py   # Fourier/sinusoidal
│       │   └── teacher_rf.py       # Teacher wrapper
│       ├── losses/
│       │   ├── physics.py          # Thermal ODE
│       │   ├── monotonicity.py     # Cooling constraint
│       │   └── smoothness.py       # Temporal smoothness
│       └── training/
│           ├── metrics.py          # MAE/RMSE/R²/skill
│           ├── baselines.py        # Persistence baseline
│           └── plotting.py         # Visualization
├── training/
│   └── train_pinn_hybrid.py        # Main trainer
├── eval/
│   ├── evaluate_model.py           # Test evaluation
│   └── export_teacher_preds.py     # Cache teacher
├── rl/
│   └── env_pinn.py                 # Gym-like wrapper
├── artifacts/                       # Generated outputs
├── Makefile                         # Convenience commands
└── README.md                        # This file
```

## Key Features

### 1. Official Feature Builder

**Single source of truth** for feature engineering at `src/pinn/data/features.py`:

```python
from src.pinn.data.features import build_official_features

X = build_official_features(
    df,
    base_cols=["airflow_lps", "cool_output_kwh", ...],
    lags=[1, 3, 6, 12],
    roll_windows=[3, 6, 12],
    winsorize=True,
    winsor_bounds=bounds,
    low_var_cols=keep_cols
)
```

**Strict validation**: Training fails if `X.columns` ≠ `feature_columns.json`

**Exogenous-only guarantee**: Forbidden patterns (`_lagy1`, `_lagy3`, etc.) trigger errors

### 2. Curriculum Training

Three progressive phases with adaptive loss weights:

| Phase | Focus | Epochs | Data | Teacher | Physics | Monotonic | Smoothness |
|-------|-------|--------|------|---------|---------|-----------|------------|
| 1. Stabilize | Learn patterns | 30 | 1.0 | 0.2 | 0.0 | 0.0 | 0.05 |
| 2. Physics On | Add constraints | 40 | 1.0 | 0.1 | 0.5 | 0.1 | 0.05 |
| 3. Control Ready | RL preparation | 30 | 1.0 | 0.05 | 0.5 | 0.2 | 0.05 |

### 3. Comprehensive Evaluation

- **Per-target metrics**: MAE, RMSE, R² for each temperature probe
- **Skill scores**: `1 - RMSE_model / RMSE_baseline`
- **Persistence baseline**: Fair comparison for exogenous-only models
- **Visualization**: Loss curves, actual vs predicted, residual distributions

### 4. RL Integration

Gym-like environment for future RL algorithms:

```python
from rl.env_pinn import create_pinn_env

env = create_pinn_env(
    model_checkpoint="artifacts/best_model.pt",
    config_path="configs/train_hybrid_pinn.yaml",
    target_temp=25.0
)

obs = env.reset()
action = np.array([60.0])  # Fan speed
obs, reward, done, info = env.step(action)
```

## Configuration

All settings in `configs/train_hybrid_pinn.yaml`:

**Data**:
- Paths, k-ahead, cadence, splits
- Feature engineering (lags, rolling windows, winsorization)

**Model**:
- Architecture (hidden dims, activation, dropout)
- Time embedding (Fourier/sinusoidal)
- Physics head (learnable parameters)

**Training**:
- Optimizer, learning rate, scheduler
- Batch size, epochs, early stopping
- Curriculum phases and loss weights

**Losses**:
- Physics ODE config (window size, load proxy)
- Monotonic constraint (actuator, epsilon, penalty type)
- Smoothness order

## Outputs

After training and evaluation:

```
artifacts/
├── best_model.pt                           # Trained model
├── scalers.json                            # Target normalization
├── feature_winsor_bounds.json              # Outlier clipping
├── metrics/
│   ├── test_metrics.csv                    # Model performance
│   ├── persistence_baseline_test.csv       # Baseline comparison
│   └── skill_vs_baseline.csv               # Skill scores
├── plots/
│   ├── loss_curves.png                     # Training progress
│   ├── actual_vs_pred_*.png                # Predictions
│   ├── residuals_distribution.png          # Error analysis
│   └── skill_comparison.png                # Per-target skill
└── logs/
    └── training_history.json               # Full training log
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- pandas, numpy, scikit-learn
- matplotlib, PyYAML, tqdm, rich

See `requirements.txt` for full list.

## Commands

```powershell
# Setup
pip install -r requirements.txt

# Train (full)
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml

# Train (dev mode)
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml --dev-run

# Train (no teacher)
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml --no-teacher

# Evaluate
python -m eval.evaluate_model --config configs/train_hybrid_pinn.yaml

# Cache teacher predictions
python -m eval.export_teacher_preds --config configs/train_hybrid_pinn.yaml
```

## Performance Expectations

**Typical results** (120s ahead, exogenous-only):

- **MAE**: 0.05-0.15 °C
- **RMSE**: 0.08-0.20 °C
- **Skill vs Persistence**: -0.2 to +0.3 (negative is expected for smooth temps)

**What matters**:
- Absolute error should be small (< 0.5°C)
- Model responds correctly to control inputs
- Physics residual is low (if enabled)

## Troubleshooting

See **[QUICKSTART.md](QUICKSTART.md)** for common issues and solutions.

## Citation

If you use this code, please cite:

```bibtex
@software{hybrid_pinn_datacenter,
  title={Hybrid PINN for Datacenter Thermal Management},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Digital-twin1}
}
```

## License

MIT License - see LICENSE file for details.
