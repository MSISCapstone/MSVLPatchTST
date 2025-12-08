"""
Evaluation utilities for Physics-Integrated PatchTST
"""

import torch
import numpy as np
from typing import Tuple


def metric(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        pred: Predictions [samples, time_steps, channels]
        true: Ground truth [samples, time_steps, channels]
        
    Returns:
        mae, mse, rmse, mape, mspe, rse, corr
    """
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100
    mspe = np.mean(np.square((pred - true) / (true + 1e-8))) * 100
    
    # RSE (Root Relative Squared Error)
    rse = np.sqrt(np.sum((pred - true) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
    
    # Correlation
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    corr = np.corrcoef(pred_flat, true_flat)[0, 1]
    
    return mae, mse, rmse, mape, mspe, rse, corr


def evaluate_model(model, test_loader, device, args):
    """
    Evaluate Physics-Integrated PatchTST on test set.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to run on
        args: Configuration arguments
        
    Returns:
        Dictionary with predictions, ground truth, and inputs
    """
    model.eval()
    
    preds = []
    trues = []
    inputs = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Store predictions and ground truth
            pred = outputs[:, -args.pred_len:, :].cpu().numpy()
            true = batch_y[:, -args.pred_len:, :args.c_out].cpu().numpy()
            inp = batch_x[:, :, :args.c_out].cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            inputs.append(inp)
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    
    print(f"Evaluation complete:")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Ground truth shape: {trues.shape}")
    print(f"  Inputs shape: {inputs.shape}")
    
    # Calculate overall metrics
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    
    print(f"\nOverall Test Metrics:")
    print(f"  MSE: {mse:.7f}")
    print(f"  MAE: {mae:.7f}")
    print(f"  RMSE: {rmse:.7f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Correlation: {corr:.4f}")
    
    return {
        'preds': preds,
        'trues': trues,
        'inputs': inputs,
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'rse': rse,
            'corr': corr
        }
    }


def evaluate_per_channel(preds, trues, target_indices, target_names):
    """
    Calculate per-channel metrics for target variables.
    
    Args:
        preds: Predictions [samples, time_steps, channels]
        trues: Ground truth [samples, time_steps, channels]
        target_indices: List of channel indices
        target_names: List of channel names
        
    Returns:
        Dictionary with per-channel metrics
    """
    per_channel_metrics = {}
    
    for ch_idx, ch_name in zip(target_indices, target_names):
        pred_ch = preds[:, :, ch_idx]
        true_ch = trues[:, :, ch_idx]
        
        mae = np.mean(np.abs(pred_ch - true_ch))
        mse = np.mean((pred_ch - true_ch) ** 2)
        rmse = np.sqrt(mse)
        
        per_channel_metrics[ch_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    
    return per_channel_metrics
