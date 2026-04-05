"""
Evaluate trained Hybrid PINN on test set.

Computes metrics, skill scores vs persistence baseline, and generates plots.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pinn.data.dataset_k_ahead import prepare_k_ahead_data
from src.pinn.data.scalers import TargetScaler
from src.pinn.models.hybrid_pinn import HybridPINN
from src.pinn.training.baselines import compute_baseline_metrics, persistence_k_ahead_baseline
from src.pinn.training.metrics import evaluate_model_on_dataset, evaluate_predictions
from src.pinn.training.plotting import (
    plot_actual_vs_pred,
    plot_residuals_distribution,
    plot_skill_comparison,
)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, model_config: dict, n_features: int, n_targets: int, device: str):
    """Load trained model from checkpoint."""
    model = HybridPINN(
        input_dim=n_features,
        output_dim=n_targets,
        hidden_dims=model_config["hidden_dims"],
        activation=model_config["activation"],
        dropout=model_config["dropout"],
        time_embedding_enabled=model_config["time_embedding"]["enabled"],
        time_embedding_method=model_config["time_embedding"]["method"],
        time_embedding_n_freqs=model_config["time_embedding"]["n_freqs"],
        physics_head_enabled=model_config["physics_head"]["enabled"]
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid PINN on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (default: best_model.pt)")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = load_config(config_path)
    
    # Device
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    # Paths
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    plots_dir = Path(cfg["output"]["plots_dir"])
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else artifacts_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    
    # Prepare dataset (same as training)
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
    
    test_dataset = data_result["test_dataset"]
    scaler = data_result["scaler"]
    feature_cols = data_result["feature_cols"]
    target_cols = data_result["target_cols"]
    
    # Load model
    print("[INFO] Loading model...")
    model = load_model(
        checkpoint_path,
        cfg["model"],
        len(feature_cols),
        len(target_cols),
        device
    )
    
    # Evaluate on test set
    print("[INFO] Evaluating on test set...")
    test_results = evaluate_model_on_dataset(
        model, test_dataset, device, scaler, batch_size=cfg["training"]["batch_size"]
    )
    
    y_test_true = test_results["y_true"]
    y_test_pred = test_results["y_pred"]
    
    # Compute persistence baseline
    print("[INFO] Computing persistence baseline...")
    
    # Load full data for baseline computation
    import pandas as pd
    df_full = pd.read_parquet(cfg["data"]["parquet_path"])
    from src.pinn.data.features import ensure_datetime_index
    df_full = ensure_datetime_index(df_full)
    
    # Get raw targets (before k-ahead shift)
    y_all_raw = df_full[target_cols].values
    
    # Compute split indices
    from src.pinn.data.dataset_k_ahead import time_split_indices
    n_total = len(y_all_raw)
    train_idx, val_idx, test_idx = time_split_indices(
        n_total,
        train_frac=cfg["data"]["train_frac"],
        val_frac=cfg["data"]["val_frac"]
    )
    
    # Persistence baseline
    baseline_results = persistence_k_ahead_baseline(
        y_all_raw,
        k=cfg["data"]["k_ahead"],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )
    
    baseline_test_true = baseline_results["test_true"]
    baseline_test_pred = baseline_results["test_pred"]
    
    # Compute baseline metrics
    baseline_metrics = compute_baseline_metrics(
        baseline_test_true,
        baseline_test_pred,
        target_cols
    )
    
    # Save baseline metrics
    baseline_metrics["split"] = "test_persistence"
    baseline_path = metrics_dir / "persistence_baseline_test.csv"
    baseline_metrics.to_csv(baseline_path, index=False)
    print(f"[INFO] Saved baseline metrics: {baseline_path}")
    
    # Extract baseline RMSE and MAE for skill computation
    baseline_rmse = baseline_metrics[baseline_metrics["target"] != "__overall__"]["RMSE"].values
    baseline_mae = baseline_metrics[baseline_metrics["target"] != "__overall__"]["MAE"].values
    
    # Compute model metrics with skill scores
    print("[INFO] Computing model metrics...")
    
    # Align test predictions with baseline (may have different lengths due to dropna)
    min_len = min(len(y_test_true), len(baseline_test_true))
    y_test_true_aligned = y_test_true[:min_len]
    y_test_pred_aligned = y_test_pred[:min_len]
    
    test_metrics = evaluate_predictions(
        y_test_true_aligned,
        y_test_pred_aligned,
        target_cols,
        baseline_rmse=baseline_rmse,
        baseline_mae=baseline_mae
    )
    
    test_metrics["split"] = "test"
    test_path = metrics_dir / "test_metrics.csv"
    test_metrics.to_csv(test_path, index=False)
    print(f"[INFO] Saved test metrics: {test_path}")
    
    # Skill vs baseline summary
    skill_df = test_metrics[["target", "skill_RMSE", "skill_MAE"]].copy()
    skill_path = metrics_dir / "skill_vs_baseline.csv"
    skill_df.to_csv(skill_path, index=False)
    print(f"[INFO] Saved skill scores: {skill_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SET EVALUATION SUMMARY")
    print("="*60)
    
    overall_model = test_metrics[test_metrics["target"] == "__overall__"].iloc[0]
    overall_baseline = baseline_metrics[baseline_metrics["target"] == "__overall__"].iloc[0]
    
    print("\nOverall Metrics:")
    print(f"  Model     - MAE: {overall_model['MAE']:.4f} °C, RMSE: {overall_model['RMSE']:.4f} °C")
    print(f"  Baseline  - MAE: {overall_baseline['MAE']:.4f} °C, RMSE: {overall_baseline['RMSE']:.4f} °C")
    
    if "skill_RMSE" in overall_model:
        print(f"\nSkill Scores (positive = better than baseline):")
        print(f"  RMSE Skill: {overall_model['skill_RMSE']:.4f}")
        print(f"  MAE Skill:  {overall_model['skill_MAE']:.4f}")
    
    print("\n" + "="*60)
    
    # Generate plots
    print("\n[INFO] Generating plots...")
    
    # Actual vs predicted for first target
    timestamps_test = test_dataset.timestamps[:min_len]
    plot_actual_vs_pred(
        y_test_true_aligned[:, 0],
        y_test_pred_aligned[:, 0],
        timestamps_test,
        target_cols[0],
        plots_dir / f"actual_vs_pred_{target_cols[0]}.png"
    )
    
    # Residuals distribution
    plot_residuals_distribution(
        y_test_true_aligned,
        y_test_pred_aligned,
        target_cols,
        plots_dir / "residuals_distribution.png"
    )
    
    # Skill comparison
    plot_skill_comparison(
        test_metrics,
        plots_dir / "skill_comparison.png"
    )
    
    print(f"\n[INFO] Evaluation complete!")
    print(f"[INFO] Metrics saved to: {metrics_dir}")
    print(f"[INFO] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
