# Hybrid PINN Quick Start Guide

This guide walks you through training and evaluating the Hybrid Physics-Informed Neural Network for datacenter rack temperature forecasting.

## Prerequisites

1. **Preprocessed data**: `artifacts/previousDesign_merged.parquet` (10s cadence, already cleaned)
2. **Feature spec**: `artifacts/feature_target_spec.json`
3. **Feature columns**: `artifacts/feature_columns.json` (from RF training)
4. **Teacher model** (optional): `artifacts/model_random_forest.pkl`

## Installation

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Verify Configuration

Edit `configs/train_hybrid_pinn.yaml` to match your paths:

```yaml
data:
  parquet_path: "artifacts/previousDesign_merged.parquet"
  spec_path: "artifacts/feature_target_spec.json"
  feature_columns_path: "artifacts/feature_columns.json"

teacher:
  enabled: true  # Set false if no teacher model
  model_path: "artifacts/model_random_forest.pkl"
```

## Step 2: Quick Smoke Test (Dev Run)

Test the pipeline end-to-end on a small subset:

```powershell
# Train (dev mode: 1000 samples, 2 epochs)
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml --dev-run

# Evaluate
python -m eval.evaluate_model --config configs/train_hybrid_pinn.yaml
```

**Expected outputs** in `artifacts/`:
- `best_model.pt` - Trained model checkpoint
- `scalers.json` - Target normalization stats
- `metrics/test_metrics.csv` - Test performance
- `metrics/persistence_baseline_test.csv` - Baseline comparison
- `metrics/skill_vs_baseline.csv` - Skill scores
- `plots/loss_curves.png` - Training curves
- `plots/actual_vs_pred_*.png` - Predictions vs ground truth
- `plots/residuals_distribution.png` - Error distribution
- `plots/skill_comparison.png` - Skill by target

## Step 3: Full Training

Once dev run succeeds, train on full dataset:

```powershell
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml
```

**Training phases** (curriculum learning):
1. **Phase 1 (Stabilize)**: Data + teacher + smoothness (30 epochs)
2. **Phase 2 (Physics On)**: Add physics residual + monotonicity (40 epochs)
3. **Phase 3 (Control Ready)**: Fine-tune for RL (30 epochs)

**Training time**: ~15-30 minutes on CPU, ~5-10 minutes on GPU (depending on dataset size)

## Step 4: Evaluation

```powershell
python -m eval.evaluate_model --config configs/train_hybrid_pinn.yaml
```

**Outputs**:
- Console prints overall test metrics and skill scores
- CSV files in `artifacts/metrics/`
- Plots in `artifacts/plots/`

**Interpreting results**:
- **Skill > 0**: Model beats persistence baseline
- **Skill < 0**: Persistence is better (expected for smooth temps at 120s horizon)
- **MAE/RMSE**: Absolute error in °C

## Step 5: Optional - Cache Teacher Predictions

Pre-compute teacher predictions to speed up training:

```powershell
python -m eval.export_teacher_preds --config configs/train_hybrid_pinn.yaml
```

Cached predictions will be saved to `artifacts/teacher_cache/`.

## Step 6: RL Integration (Future)

The trained model can be used in an RL environment:

```python
from rl.env_pinn import create_pinn_env

env = create_pinn_env(
    model_checkpoint="artifacts/best_model.pt",
    config_path="configs/train_hybrid_pinn.yaml",
    scaler_path="artifacts/scalers.json",
    target_temp=25.0
)

obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Your RL policy here
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Troubleshooting

### Feature column mismatch error

```
ValueError: Feature columns do not match artifacts/feature_columns.json
```

**Solution**: Regenerate the RF model or ensure `feature_columns.json` matches the official feature builder output.

### Teacher model not found

```
FileNotFoundError: Teacher model not found: artifacts/model_random_forest.pkl
```

**Solution**: Either train the RF teacher first, or disable teacher in config:

```yaml
teacher:
  enabled: false
```

Or run with `--no-teacher` flag:

```powershell
python -m training.train_pinn_hybrid --config configs/train_hybrid_pinn.yaml --no-teacher
```

### CUDA out of memory

**Solution**: Reduce batch size in config:

```yaml
training:
  batch_size: 128  # Default is 256
```

### Poor skill scores (negative)

This is **expected** for exogenous-only models at 120s horizon when temperatures are smooth. The persistence baseline is very strong for short horizons.

**What matters**:
- Absolute MAE/RMSE should be small (< 0.5°C is good)
- Model should respond sensibly to control inputs (check monotonicity)
- Physics residual should be low (if physics loss is enabled)

## Advanced: Tuning Hyperparameters

Edit `configs/train_hybrid_pinn.yaml`:

**Model architecture**:
```yaml
model:
  hidden_dims: [128, 128, 128]  # Increase for more capacity
  dropout: 0.1  # Increase to reduce overfitting
```

**Loss weights** (per phase):
```yaml
training:
  curriculum:
    phase2:
      loss_weights:
        physics: 0.5  # Increase for stronger physics constraint
        monotonic: 0.1  # Increase for stricter cooling constraint
```

**Learning rate**:
```yaml
training:
  lr_initial: 2.0e-3  # Increase for faster convergence
  lr_final: 5.0e-4
```

## Next Steps

1. **Analyze results**: Check `artifacts/plots/` and `artifacts/metrics/`
2. **Tune hyperparameters**: Adjust loss weights, architecture, learning rate
3. **Integrate with RL**: Use `rl/env_pinn.py` as starting point
4. **Deploy**: Export model for production inference

## Support

For issues or questions, check:
- `README.md` - Full documentation
- `configs/train_hybrid_pinn.yaml` - All configuration options
- Training logs in `artifacts/logs/training_history.json`
