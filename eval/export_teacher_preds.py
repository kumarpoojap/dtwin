"""
Export teacher predictions to cache for faster training.

Pre-computes and caches teacher predictions for train/val/test splits.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pinn.data.dataset_k_ahead import prepare_k_ahead_data
from src.pinn.models.teacher_rf import load_teacher


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Export teacher predictions to cache")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = load_config(config_path)
    
    if not cfg["teacher"]["enabled"]:
        print("[INFO] Teacher is disabled in config. Nothing to export.")
        return
    
    # Prepare dataset
    print("[INFO] Preparing dataset...")
    data_result = prepare_k_ahead_data(
        parquet_path=Path(cfg["data"]["parquet_path"]),
        spec_path=Path(cfg["data"]["spec_path"]),
        feature_columns_path=Path(cfg["data"]["feature_columns_path"]),
        base_cols=cfg["data"]["features"]["base_cols"],
        lags=cfg["data"]["features"]["lags"],
        roll_windows=cfg["data"]["features"]["roll_windows"],
        k_ahead=cfg["data"]["k_ahead"],
        train_frac=cfg["data"]["train_frac"],
        val_frac=cfg["data"]["val_frac"],
        normalize_targets=cfg["data"]["normalize_targets"],
        winsorize=cfg["data"]["features"]["winsorize"],
        winsor_quantiles=tuple(cfg["data"]["features"]["winsor_quantiles"]),
        low_var_threshold=cfg["data"]["features"]["low_var_threshold"],
        cadence_seconds=cfg["data"]["cadence_seconds"],
        dev_run=False
    )
    
    train_dataset = data_result["train_dataset"]
    val_dataset = data_result["val_dataset"]
    test_dataset = data_result["test_dataset"]
    feature_cols = data_result["feature_cols"]
    target_cols = data_result["target_cols"]
    
    # Load teacher
    print("[INFO] Loading teacher model...")
    teacher = load_teacher(
        model_path=Path(cfg["teacher"]["model_path"]),
        cache_dir=Path(cfg["teacher"]["cache_dir"]) if cfg["teacher"].get("cache_dir") else None,
        use_cache=True,  # Force cache creation
        allow_missing=False
    )
    
    # Validate compatibility
    teacher.validate_compatibility(feature_cols, target_cols)
    
    # Export predictions for each split
    for split_name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        print(f"\n[INFO] Exporting {split_name} predictions...")
        
        # Build DataFrame
        X_df = pd.DataFrame(
            dataset.X.numpy(),
            columns=feature_cols,
            index=dataset.timestamps
        )
        
        # Get predictions (will be cached)
        y_pred = teacher.get_or_compute_predictions(X_df, split_name, return_tensor=False)
        
        print(f"[OK] {split_name} predictions cached: shape {y_pred.shape}")
    
    print(f"\n[INFO] Teacher predictions exported successfully!")
    print(f"[INFO] Cache directory: {cfg['teacher']['cache_dir']}")


if __name__ == "__main__":
    main()
