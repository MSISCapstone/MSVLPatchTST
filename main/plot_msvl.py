#!/usr/bin/env python3
"""
Plotting helper for the MSVL (Multi-Scale Variable-Length) PatchTST experiments.

- Creates a 2x3 figure with the following layout:
  Row 1: 'p (mbar)', 'T (degC)', 'wv (m/s)'
  Row 2: 'max. wv (m/s)', 'rain (mm)', 'raining (s)'

- Reads predictions from a results folder (expects `pred.npy` saved by experiments)
- Reconstructs ground-truth test values by loading the dataset
- Computes per-feature metrics (MAE, MSE, RMSE, RSE)
- Generates test source data statistics per feature
- Saves all outputs to the specified output directory

Usage:
  python main/plot_msvl.py \
    --model_id_name weather --seq_len 336 --pred_len 96 \
    [--results_src PATH] [--output_dir PATH] [--data_file weather_with_hour.csv]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

from MSVLPatchTST.data_provider.data_factory import data_provider
from PatchTST_supervised.utils.metrics import MAE, MSE, RMSE, RSE

FEATURES_TO_PLOT = [
    'p (mbar)',
    'T (degC)',
    'wv (m/s)',
    'max. wv (m/s)',
    'rain (mm)',
    'raining (s)'
]

PLOT_ORDER = FEATURES_TO_PLOT


def find_pred_file(results_src, git_root, model_id_name, seq_len, pred_len):
    """Find the prediction file in the output folder or fallback locations."""
    model_id = f"{model_id_name}_{seq_len}_{pred_len}"
    filenames_to_check = ['pred.npy', f'{model_id}_pred.npy']
    
    # Helper to search a directory for pred files
    def search_dir(base_path):
        p = Path(base_path)
        if not p.exists():
            return None, None
        # Direct files in directory
        for filename in filenames_to_check:
            pred_file = p / filename
            if pred_file.exists():
                return pred_file, p
        # Check in model_id subdirectory
        subdir = p / model_id
        if subdir.exists():
            for filename in filenames_to_check:
                pred_file = subdir / filename
                if pred_file.exists():
                    return pred_file, subdir
        return None, None
    
    # Primary location: output/MSVLPatchTST/test_results/model_id/
    primary_loc = Path(git_root) / 'output' / 'MSVLPatchTST' / 'test_results' / model_id
    pred_file, found_dir = search_dir(primary_loc)
    if pred_file:
        return pred_file, found_dir
    
    # If results_src is provided, search there
    if results_src:
        pred_file, found_dir = search_dir(results_src)
        if pred_file:
            return pred_file, found_dir
    
    # Build error message with expected location
    raise FileNotFoundError(
        f"No pred.npy found.\n"
        f"Expected location: {primary_loc}\n"
        f"Run inference first: ./main/scripts/weather_msvl_test.sh"
    )


def build_test_truths(root_path_name, data_path_name, seq_len, label_len, pred_len, batch_size=128, num_workers=0):
    """Build ground truth values from test dataset."""
    args = SimpleNamespace()
    args.data = 'custom'
    args.root_path = root_path_name
    args.data_path = data_path_name
    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len
    args.features = 'M'
    args.target = 'OT'
    args.embed = 'timeF'
    args.freq = 'h'
    args.batch_size = batch_size
    args.num_workers = num_workers

    dataset, loader = data_provider(args, flag='test')

    trues = []
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        f_dim = -1 if args.features == 'MS' else 0
        batch_y = batch_y[:, -pred_len:, f_dim:].numpy()
        trues.append(batch_y)
    
    if not trues:
        raise RuntimeError('No test batches found')
    
    trues = np.concatenate(trues, axis=0)
    return trues, dataset


def compute_feature_columns(root_path_name, data_path_name, target='OT'):
    """Get feature column names from data file."""
    path = Path(root_path_name) / data_path_name
    df = pd.read_csv(path)
    cols = list(df.columns)
    if 'date' in cols:
        cols.remove('date')
    if target in cols:
        cols.remove(target)
    return cols + [target]


def compute_test_data_statistics(root_path_name, data_path_name, data_columns, out_dir):
    """Compute and save test data statistics for TARGET features only."""
    path = Path(root_path_name) / data_path_name
    df = pd.read_csv(path)
    
    # Get test split (last 20% as in Dataset_Custom)
    n = len(df)
    border1 = int(n * 0.7)  # end of validation
    border2 = n
    test_df = df.iloc[border1:border2]
    
    stats = []
    for col in FEATURES_TO_PLOT:  # Only target features
        if col in test_df.columns:
            series = test_df[col]
            stats.append({
                'feature': col,
                'count': series.count(),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                '25%': series.quantile(0.25),
                '50%': series.quantile(0.50),
                '75%': series.quantile(0.75),
                'max': series.max()
            })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(Path(out_dir) / 'test_data_statistics.csv', index=False)
    print(f"Saved test data statistics (target features) to {out_dir}/test_data_statistics.csv")
    return stats_df


def save_stats_and_plot(preds, trues, data_columns, out_dir, seq_len, pred_len):
    """Save per-feature metrics and create prediction plots."""
    os.makedirs(out_dir, exist_ok=True)

    num_samples = preds.shape[0]
    D = preds.shape[-1]
    actual_pred_len = preds.shape[1]  # Actual prediction length from data
    true_len = trues.shape[1]  # Ground truth length
    
    print(f"Predictions shape: {preds.shape} (num_samples={num_samples}, pred_len={actual_pred_len})")
    print(f"Ground truth shape: {trues.shape} (true_len={true_len})")
    
    # For metrics, use only the overlapping portion (where we have ground truth)
    min_len = min(actual_pred_len, true_len)
    preds_for_metrics = preds[:, :min_len, :]
    trues_for_metrics = trues[:, :min_len, :]

    # Compute per-feature metrics ONLY for target features (FEATURES_TO_PLOT)
    metrics = []
    for feat_name in FEATURES_TO_PLOT:
        if feat_name not in data_columns:
            print(f"Warning: Target feature '{feat_name}' not found in data columns")
            continue
        idx = data_columns.index(feat_name)
        if idx >= D:
            continue
        p = preds_for_metrics[..., idx]
        t = trues_for_metrics[..., idx]
        mae = MAE(p, t)
        mse = MSE(p, t)
        rmse = RMSE(p, t)
        rse = RSE(p, t)
        metrics.append({
            'feature': feat_name,
            'index': idx,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'rse': float(rse)
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Path(out_dir) / 'per_feature_metrics.csv', index=False)
    print(f"Saved per-feature metrics to {out_dir}/per_feature_metrics.csv")

    # Create 2x3 plot grid for selected features - CONTINUOUS TIME SERIES
    feature_indices = []
    for f in PLOT_ORDER:
        if f in data_columns:
            feature_indices.append(data_columns.index(f))
        else:
            feature_indices.append(None)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    # Total timesteps = num_samples * pred_len (e.g., 4 * 96 = 384)
    total_timesteps = num_samples * actual_pred_len

    for ax, feat_name, feat_idx in zip(axes, PLOT_ORDER, feature_indices):
        if feat_idx is None or feat_idx >= preds.shape[-1]:
            ax.text(0.5, 0.5, f'Feature not found: {feat_name}', ha='center', va='center')
            ax.set_title(feat_name)
            continue
        
        p = preds[..., feat_idx]  # [num_samples, pred_len]
        t = trues[..., feat_idx]  # [num_samples, pred_len]
        
        # Flatten to continuous time series: [num_samples * pred_len]
        # Each sample's 96 predictions are consecutive in time
        continuous_pred = p.flatten()  # All 384 timesteps
        continuous_true = t.flatten()  # All 384 timesteps
        
        # Time axis
        x_axis = np.arange(total_timesteps)
        
        # Compute metrics
        mae = MAE(p, t)
        mse = MSE(p, t)

        # Plot continuous time series
        ax.plot(x_axis, continuous_true, label='Ground Truth', linewidth=1.5, color='blue', alpha=0.8)
        ax.plot(x_axis, continuous_pred, label='Prediction', linewidth=1.5, color='orange', alpha=0.8)
        
        # Add vertical lines at sample boundaries
        for i in range(1, num_samples):
            ax.axvline(x=i * actual_pred_len, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_title(f"{feat_name}\nMAE={mae:.4f}, MSE={mse:.4f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value (normalized)')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'MSVLPatchTST Weather Predictions - {total_timesteps} Timesteps\n({num_samples} samples x {actual_pred_len} steps, seq_len={seq_len})', fontsize=14)
    fig_file = Path(out_dir) / f'prediction_grid_sl{seq_len}_pl{actual_pred_len}.png'
    fig.savefig(fig_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved prediction plot to {fig_file}")

    # Save summary text file
    with open(Path(out_dir) / 'summary.txt', 'w') as fh:
        fh.write('MSVLPatchTST Test Results Summary\n')
        fh.write('=' * 60 + '\n\n')
        fh.write(f'Sequence Length (lookback): {seq_len}\n')
        fh.write(f'Prediction Length per sample: {actual_pred_len}\n')
        fh.write(f'Number of samples: {num_samples}\n')
        fh.write(f'Total predicted timesteps: {total_timesteps}\n')
        fh.write(f'Number of target features: {len(FEATURES_TO_PLOT)}\n\n')
        fh.write(f'Target Features: {FEATURES_TO_PLOT}\n\n')
        fh.write(f'Per-feature Metrics (target features only):\n')
        fh.write('-' * 60 + '\n')
        fh.write(metrics_df.to_string(index=False))
        fh.write('\n\n')
        
        # Overall metrics for TARGET FEATURES ONLY
        target_indices = [data_columns.index(f) for f in FEATURES_TO_PLOT if f in data_columns and data_columns.index(f) < D]
        if target_indices:
            preds_target = preds_for_metrics[..., target_indices]
            trues_target = trues_for_metrics[..., target_indices]
            overall_mae = MAE(preds_target, trues_target)
            overall_mse = MSE(preds_target, trues_target)
            overall_rmse = RMSE(preds_target, trues_target)
        else:
            overall_mae = overall_mse = overall_rmse = float('nan')
        fh.write(f'Overall Metrics (target features only):\n')
        fh.write('-' * 60 + '\n')
        fh.write(f'MAE: {overall_mae:.6f}\n')
        fh.write(f'MSE: {overall_mse:.6f}\n')
        fh.write(f'RMSE: {overall_rmse:.6f}\n')

    print(f"Saved summary to {out_dir}/summary.txt")


def main():
    parser = argparse.ArgumentParser(description='MSVL PatchTST Plotting and Metrics')
    parser.add_argument('--model_id_name', default='weather')
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--results_src', default=None, help='Path to folder containing pred.npy')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--data_root', default=None, help='Path to datasets root')
    parser.add_argument('--data_file', default='weather_with_hour.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    # Determine git root
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    if not git_root:
        git_root = os.getcwd()

    root_path_name = args.data_root if args.data_root else os.path.join(git_root, 'datasets', 'weather')
    
    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(git_root) / 'output' / 'MSVLPatchTST' / 'test_results' / f"{args.model_id_name}_{args.seq_len}_{args.pred_len}"
    
    os.makedirs(out_dir, exist_ok=True)

    # Find and load predictions
    pred_file, found_dir = find_pred_file(args.results_src, git_root, args.model_id_name, args.seq_len, args.pred_len)
    print(f"Loading predictions from: {pred_file}")
    preds = np.load(pred_file)
    print(f"Predictions shape: {preds.shape}")

    # Load ground truth from saved file (saved alongside predictions)
    true_file = found_dir / 'true.npy'
    if true_file.exists():
        print(f"Loading ground truth from: {true_file}")
        trues = np.load(true_file)
    else:
        # Fallback: rebuild from data loader
        print("Warning: true.npy not found, rebuilding from data loader...")
        trues, dataset = build_test_truths(
            root_path_name, args.data_file,
            args.seq_len, args.label_len, args.pred_len,
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    print(f"Ground truth shape: {trues.shape}")
    
    # Verify shapes match
    if preds.shape != trues.shape:
        print(f"Warning: Shape mismatch - preds {preds.shape} vs trues {trues.shape}")
        # Use minimum number of features
        n_features = min(preds.shape[-1], trues.shape[-1])
        preds = preds[..., :n_features]
        trues = trues[..., :n_features]
        print(f"Truncated to {n_features} features")

    # Get feature column names
    data_columns = compute_feature_columns(root_path_name, args.data_file)
    # Adjust columns to match actual prediction dimensions
    if len(data_columns) > preds.shape[-1]:
        data_columns = data_columns[:preds.shape[-1]]
    print(f"Data columns ({len(data_columns)}): {data_columns[:5]}...")

    # Save test data statistics
    compute_test_data_statistics(root_path_name, args.data_file, data_columns, out_dir)

    # Save metrics and plots
    save_stats_and_plot(preds, trues, data_columns, out_dir, args.seq_len, args.pred_len)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
