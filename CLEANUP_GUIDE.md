# Cleanup Guide - Datacenter to GPU Dataset Migration

This guide helps you identify and clean up files specific to the old datacenter cooling dataset now that you've migrated to the GPU thermal dataset.

## Files You Can Safely Remove

### ❌ Old Datacenter Data Files (Keep for Reference or Delete)

**Raw data** (in `data/` directory):
```
data/previousDesign/
├── inrow_cooler_sensor_door_opened.txt
└── netbotz_sensor_values_door_opened_temperature.txt

data/retrofittedDesign/
├── inrow_cooler_sensor_door_closed.txt
└── netbotz_sensor_values_door_closed_temperature.txt

data/Plots/
├── previousDesign/
│   ├── data.csv
│   └── exhaust_temp.csv
└── retrofittedDesign/
    ├── data.csv
    └── exhaust_temp.csv
```

**Decision**: 
- ✅ **Keep if**: You might switch back to datacenter dataset
- ❌ **Delete if**: You're fully committed to GPU dataset only

### ❌ Old Data Processing Scripts

```
src/data_processing/prepare_data_previous_design.py
```

**Decision**: 
- ❌ **Delete**: Specific to datacenter data format
- Or rename to `prepare_data_previous_design.py.old` for reference

### ❌ Old Visualization Scripts

```
data/Plots/c2.py
data/Plots/Temp1.png
src/visualization/plot_vertical_profile.py
```

**Decision**: 
- ❌ **Delete**: Specific to datacenter rack vertical temperature profiles
- Not applicable to single GPU temperature

## Files to Keep (Still Useful)

### ✅ Core Pipeline (Works with Any Dataset)

**All of these are dataset-agnostic:**
```
src/common/              # Shared utilities - KEEP
├── features.py          # Feature engineering
├── scalers.py           # Target normalization
└── data_utils.py        # Time splits, validation

src/pinn/                # PINN architecture - KEEP
├── models/
├── losses/
├── training/
└── data/

src/models/              # RF training - KEEP
├── feature_builder.py
└── ml_surrogate_rack_temp.py

training/                # Training scripts - KEEP
└── train_pinn_hybrid.py

eval/                    # Evaluation - KEEP
├── evaluate_model.py
└── export_teacher_preds.py

rl/                      # RL wrapper - KEEP
└── env_pinn.py
```

### ✅ Configuration Files

**Keep both configs** (you might want to switch back):
```
configs/
├── train_hybrid_pinn.yaml         # Datacenter config - KEEP
├── train_gpu_pinn.yaml            # GPU config - KEEP
├── feature_target_spec.json       # Datacenter spec - KEEP
└── gpu_feature_target_spec.json   # GPU spec - KEEP
```

### ✅ Documentation

**Keep all documentation:**
```
README.md                      # General overview
QUICKSTART.md                  # Still relevant
IMPLEMENTATION_SUMMARY.md      # Architecture docs
CONSOLIDATION_SUMMARY.md       # Refactoring history
GPU_MIGRATION_GUIDE.md         # Migration guide
```

## Temporary/Generated Files (Safe to Delete Anytime)

### 🗑️ Artifacts Directory

**These are regenerated during training:**
```
artifacts/
├── *.parquet                          # Old datacenter parquet
├── *.pkl                              # Old models
├── *.json                             # Old configs
├── *.csv                              # Old metrics
├── *.png                              # Old plots
├── checkpoints/                       # Old checkpoints
├── plots/                             # Old plots
├── logs/                              # Old logs
└── teacher_cache/                     # Old teacher predictions

artifacts_rf/
└── model_random_forest.pkl            # Old RF teacher bundle
```

**Decision**: 
- ❌ **Delete all**: Will be regenerated for GPU dataset
- Or move to `artifacts_old/` for comparison

### 🗑️ Analysis Scripts (Temporary)

```
analyze_synthetic.py              # Temporary analysis script
prepare_synthetic_data.py         # One-time conversion script
```

**Decision**: 
- ✅ **Keep**: Useful for re-running if needed
- Or move to `scripts/` directory

## Recommended Cleanup Actions

### Option 1: Archive Old Datacenter Files

```powershell
# Create archive directory
New-Item -ItemType Directory -Path "archive_datacenter" -Force

# Move old data
Move-Item data/previousDesign archive_datacenter/
Move-Item data/retrofittedDesign archive_datacenter/
Move-Item data/Plots archive_datacenter/

# Move old artifacts (if they exist)
if (Test-Path artifacts) {
    Move-Item artifacts archive_datacenter/artifacts_old
}
if (Test-Path artifacts_rf) {
    Move-Item artifacts_rf archive_datacenter/artifacts_rf_old
}

# Move old data processing
Move-Item src/data_processing/prepare_data_previous_design.py archive_datacenter/

# Move old visualization
Move-Item src/visualization/plot_vertical_profile.py archive_datacenter/
```

### Option 2: Delete Old Datacenter Files (Aggressive)

```powershell
# Delete old data
Remove-Item -Recurse data/previousDesign
Remove-Item -Recurse data/retrofittedDesign
Remove-Item -Recurse data/Plots

# Delete old artifacts
Remove-Item -Recurse artifacts -ErrorAction SilentlyContinue
Remove-Item -Recurse artifacts_rf -ErrorAction SilentlyContinue

# Delete old scripts
Remove-Item src/data_processing/prepare_data_previous_design.py
Remove-Item src/visualization/plot_vertical_profile.py

# Keep configs for reference (optional)
```

### Option 3: Minimal Cleanup (Recommended)

Just clean up generated artifacts, keep source data:

```powershell
# Delete old generated artifacts only
Remove-Item -Recurse artifacts -ErrorAction SilentlyContinue
Remove-Item -Recurse artifacts_rf -ErrorAction SilentlyContinue

# Keep everything else for reference
```

## What Gets Regenerated for GPU Dataset

After running `train_gpu_rf.ps1` and PINN training, you'll have:

```
artifacts/
├── synthetic_gpu_thermal.parquet      # NEW: GPU data
├── gpu_rf/                            # NEW: RF teacher artifacts
│   ├── model_random_forest.pkl
│   ├── feature_columns.json
│   ├── targets_used.json
│   ├── target_normalization_stats.json
│   └── metrics_summary.csv
├── gpu_rf_teacher.pkl                 # NEW: Teacher bundle
├── gpu_feature_columns.json           # NEW: Feature list
├── checkpoints_gpu/                   # NEW: PINN checkpoints
├── plots_gpu/                         # NEW: Training plots
├── logs_gpu/                          # NEW: Training logs
└── teacher_cache_gpu/                 # NEW: Cached predictions
```

## Files That Don't Need Changes

**These work with both datasets:**
- All Python source code in `src/`
- Training scripts in `training/`
- Evaluation scripts in `eval/`
- RL wrapper in `rl/`
- Export script `export_rf_teacher.py`
- Requirements `requirements.txt`

## Summary

### Must Keep ✅
- `src/` directory (all code)
- `training/`, `eval/`, `rl/` directories
- `configs/` directory (both old and new configs)
- Documentation files
- `export_rf_teacher.py`
- `requirements.txt`

### Can Delete ❌
- `data/previousDesign/` and `data/retrofittedDesign/`
- `data/Plots/`
- `artifacts/` and `artifacts_rf/` (will regenerate)
- `src/data_processing/prepare_data_previous_design.py`
- `src/visualization/plot_vertical_profile.py`

### Optional Cleanup 🗑️
- `analyze_synthetic.py` (temporary script)
- `prepare_synthetic_data.py` (one-time conversion)
- Old `.png` plot files

## Recommendation

**Start with Option 3 (Minimal Cleanup)**:
1. Delete `artifacts/` and `artifacts_rf/` directories
2. Keep all source code and data for reference
3. After confirming GPU pipeline works, do Option 1 (Archive) if you want to clean up further

This way you can always revert to the datacenter dataset if needed!
