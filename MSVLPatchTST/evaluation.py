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
    Evaluate MSVLPatchTST on test set.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to run on
        args: Configuration arguments
        
    Returns:
        Dictionary with predictions, ground truth, and inputs
    """
    # Get target input indices from model (p=0, T=1, wv=11, max.wv=12, rain=14, raining=15)
    target_input_indices = []
    for group_name in args.channel_groups.keys():
        target_input_indices.extend(model.group_info[group_name].get('target_indices', []))
    
    model.eval()
    
    preds = []
    trues = []
    inputs = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Forward pass - outputs [bs, pred_len, 6] for 6 target features
            outputs = model(batch_x)
            
            # Store predictions and ground truth for 6 target features only
            pred = outputs[:, -model.pred_len:, :].cpu().numpy()
            # Extract ground truth for target features from batch_y
            true = batch_y[:, -model.pred_len:, target_input_indices].cpu().numpy()
            inp = batch_x[:, :, :].cpu().numpy()
            
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
    
    print(f"\nOverall Test Metrics (6 target features):")
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


def evaluate_model_sliding_window(model, dataset, device, args, num_iterations=1, max_samples=None, window_stride=96):
    """
    Evaluate MSVLPatchTST with sliding window prediction.
    
    Args:
        model: The model to evaluate
        dataset: Test dataset (not loader, to access raw data for sliding windows)
        device: Device to run on
        args: Configuration arguments
        num_iterations: Number of prediction iterations per sample (default 1)
        max_samples: Maximum number of samples to use (default: all)
        window_stride: Stride between sliding windows (default: 96 for non-overlapping)
    
    Example with window_stride=96, seq_len=336, pred_len=96, max_samples=4:
        Sample 0: Input [0:336]   → Predict [336:432]
        Sample 1: Input [96:432]  → Predict [432:528]
        Sample 2: Input [192:528] → Predict [528:624]
        Sample 3: Input [288:624] → Predict [624:720]
        
    Returns:
        Dictionary with predictions, ground truth, and metrics
    """
    # Get target input indices from model (p=0, T=1, wv=11, max.wv=12, rain=14, raining=15)
    target_input_indices = []
    for group_name in args.channel_groups.keys():
        target_input_indices.extend(model.group_info[group_name].get('target_indices', []))
    
    model.eval()
    pred_len = model.pred_len
    seq_len = args.seq_len
    c_out = model.c_out  # 6 target features
    
    print(f"\nSliding window prediction with stride={window_stride}")
    print(f"Each sample: {seq_len}-step lookback → {pred_len}-step prediction")
    print(f"Predicting {c_out} target features: indices {target_input_indices}")
    
    # Access raw data from dataset
    data_x = dataset.data_x  # Full normalized data [total_len, channels]
    data_y = dataset.data_y  # Full normalized data for targets
    
    total_len = len(data_x)
    
    # Calculate how many samples we can create with the given stride
    # We need: seq_len for input + pred_len for prediction
    required_len = seq_len + pred_len
    max_possible_samples = (total_len - required_len) // window_stride + 1
    
    if max_possible_samples <= 0:
        raise ValueError(f"Dataset too short. Need {required_len} timesteps, have {total_len}")
    
    # Apply max_samples limit if specified
    num_samples = max_possible_samples
    if max_samples is not None and max_samples > 0:
        num_samples = min(num_samples, max_samples)
    
    print(f"Total data length: {total_len}")
    print(f"Creating {num_samples} samples (stride={window_stride}, max_possible={max_possible_samples})")
    print(f"Total predicted timesteps: {num_samples * pred_len}")
    
    all_preds = []  # [num_samples, pred_len, c_out]
    all_trues = []  # [num_samples, pred_len, c_out]
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Calculate the start position for this sample's input window
            input_start = sample_idx * window_stride
            input_end = input_start + seq_len
            
            # Calculate the target position (what we're predicting)
            target_start = input_end
            target_end = target_start + pred_len
            
            if target_end > total_len:
                break  # Not enough data
            
            # Get input sequence
            seq_x = data_x[input_start:input_end]  # [seq_len, channels]
            seq_x = torch.FloatTensor(seq_x).unsqueeze(0).to(device)  # [1, seq_len, channels]
            
            # Get ground truth for 6 target features only
            true_y = data_y[target_start:target_end, target_input_indices]  # [pred_len, 6]
            
            # Forward pass - outputs [1, pred_len, 6]
            outputs = model(seq_x)
            pred = outputs[:, -pred_len:, :].cpu().numpy()[0]  # [pred_len, 6]
            
            all_preds.append(pred)
            all_trues.append(true_y)
            
            if sample_idx % 100 == 0:
                print(f"  Processed {sample_idx}/{num_samples} samples...")
    
    all_preds = np.array(all_preds)  # [N, pred_len, c_out]
    all_trues = np.array(all_trues)  # [N, pred_len, c_out]
    
    total_pred_steps = num_samples * pred_len
    
    print(f"\nSliding Window Evaluation complete:")
    print(f"  Predictions shape: {all_preds.shape}")
    print(f"  Ground truth shape: {all_trues.shape}")
    print(f"  Total predicted timesteps: {total_pred_steps}")
    
    # Calculate metrics
    mae, mse, rmse, mape, mspe, rse, corr = metric(all_preds, all_trues)
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {mse:.7f}")
    print(f"  MAE: {mae:.7f}")
    print(f"  RMSE: {rmse:.7f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Correlation: {corr:.4f}")
    
    return {
        'preds': all_preds,  # [N, pred_len, c_out]
        'trues': all_trues,  # [N, pred_len, c_out]
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'rse': rse,
            'corr': corr
        },
        'num_samples': num_samples,
        'pred_len': pred_len,
        'total_pred_steps': total_pred_steps
    }


def evaluate_per_channel(preds, trues, target_indices, target_names):
    """
    Calculate per-channel metrics for target variables.
    
    Args:
        preds: Predictions [samples, time_steps, num_targets] - already filtered to target channels
        trues: Ground truth [samples, time_steps, num_targets] - already filtered to target channels
        target_indices: List of original channel indices (not used for indexing preds/trues)
        target_names: List of channel names
        
    Returns:
        Dictionary with per-channel metrics
    """
    per_channel_metrics = {}
    
    # preds and trues are already filtered to contain only target channels in order
    # so we iterate through channels sequentially (0, 1, 2, ...)
    for i, ch_name in enumerate(target_names):
        pred_ch = preds[:, :, i]
        true_ch = trues[:, :, i]
        
        mae = np.mean(np.abs(pred_ch - true_ch))
        mse = np.mean((pred_ch - true_ch) ** 2)
        rmse = np.sqrt(mse)
        
        per_channel_metrics[ch_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    
    return per_channel_metrics
