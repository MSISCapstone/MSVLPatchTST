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
    from .training_utils import get_target_indices
    target_indices, _ = get_target_indices(args.channel_groups)
    
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
            
            # Store predictions and ground truth for weather features only (c_out channels)
            pred = outputs[:, -model.pred_len:, :].cpu().numpy()
            true = batch_y[:, -model.pred_len:, :model.c_out].cpu().numpy()  # Only weather channels
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


def evaluate_model_sliding_window(model, dataset, device, args, num_iterations=4):
    """
    Evaluate Physics-Integrated PatchTST with sliding window multi-step prediction.
    
    Each iteration uses REAL historical data (not previous predictions) to predict the next 96 steps.
    The prediction windows are consecutive, covering num_iterations * pred_len total steps.
    
    Example with num_iterations=4, seq_len=336, pred_len=96:
        Step 1: Input [0:336]    → Predict [336:432]   (ground truth available)
        Step 2: Input [96:432]   → Predict [432:528]   (ground truth available)
        Step 3: Input [192:528]  → Predict [528:624]   (ground truth available)
        Step 4: Input [288:624]  → Predict [624:720]   (ground truth available)
    
    Args:
        model: The model to evaluate
        dataset: Test dataset (not loader, to access raw data for sliding windows)
        device: Device to run on
        args: Configuration arguments
        num_iterations: Number of prediction iterations (default 4 for 384 total steps)
        
    Returns:
        Dictionary with predictions, ground truth, and metrics
    """
    from .training_utils import get_target_indices
    target_indices, _ = get_target_indices(args.channel_groups)
    
    model.eval()
    pred_len = model.pred_len
    seq_len = args.seq_len
    total_pred_len = num_iterations * pred_len
    c_out = model.c_out
    
    print(f"\nSliding window prediction: {num_iterations} iterations x {pred_len} steps = {total_pred_len} total steps")
    print(f"Each prediction uses real {seq_len}-step lookback window")
    
    # Access raw data from dataset
    # Dataset returns (seq_x, seq_y, seq_x_mark, seq_y_mark) for each index
    # We need to create custom sliding windows
    
    # Get the full data array from dataset
    data_x = dataset.data_x  # Full normalized data [total_len, channels]
    data_y = dataset.data_y  # Full normalized data for targets
    
    total_len = len(data_x)
    label_len = args.label_len
    
    # Calculate how many complete sliding window sequences we can create
    # We need: seq_len for first input + (num_iterations-1)*pred_len for shifts + pred_len for last prediction ground truth
    required_len = seq_len + num_iterations * pred_len
    num_samples = total_len - required_len + 1
    
    if num_samples <= 0:
        raise ValueError(f"Dataset too short. Need {required_len} timesteps, have {total_len}")
    
    print(f"Total data length: {total_len}, Creating {num_samples} sliding window samples")
    
    all_preds = []  # [num_samples, total_pred_len, c_out]
    all_trues = []  # [num_samples, total_pred_len, c_out]
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            iteration_preds = []
            iteration_trues = []
            
            for iteration in range(num_iterations):
                # Calculate the start position for this iteration's input window
                input_start = sample_idx + iteration * pred_len
                input_end = input_start + seq_len
                
                # Calculate the target position (what we're predicting)
                target_start = input_end
                target_end = target_start + pred_len
                
                if target_end > total_len:
                    break  # Not enough data for this iteration
                
                # Get input sequence
                seq_x = data_x[input_start:input_end]  # [seq_len, channels]
                seq_x = torch.FloatTensor(seq_x).unsqueeze(0).to(device)  # [1, seq_len, channels]
                
                # Get ground truth
                true_y = data_y[target_start:target_end, :c_out]  # [pred_len, c_out]
                
                # Forward pass
                outputs = model(seq_x)  # [1, pred_len, c_out]
                pred = outputs[:, -pred_len:, :].cpu().numpy()[0]  # [pred_len, c_out]
                
                iteration_preds.append(pred)
                iteration_trues.append(true_y)
            
            if len(iteration_preds) == num_iterations:
                # Concatenate all iterations: [total_pred_len, c_out]
                full_pred = np.concatenate(iteration_preds, axis=0)
                full_true = np.concatenate(iteration_trues, axis=0)
                
                all_preds.append(full_pred)
                all_trues.append(full_true)
            
            if sample_idx % 1000 == 0:
                print(f"  Processed {sample_idx}/{num_samples} samples...")
    
    all_preds = np.array(all_preds)  # [N, total_pred_len, c_out]
    all_trues = np.array(all_trues)  # [N, total_pred_len, c_out]
    
    print(f"\nSliding Window Evaluation complete:")
    print(f"  Predictions shape: {all_preds.shape}")
    print(f"  Ground truth shape: {all_trues.shape}")
    
    # Calculate metrics on full predictions (all 384 steps have ground truth now!)
    mae, mse, rmse, mape, mspe, rse, corr = metric(all_preds, all_trues)
    
    print(f"\nTest Metrics (all {total_pred_len} steps):")
    print(f"  MSE: {mse:.7f}")
    print(f"  MAE: {mae:.7f}")
    print(f"  RMSE: {rmse:.7f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Correlation: {corr:.4f}")
    
    # Also compute per-iteration metrics
    print(f"\nPer-iteration Metrics:")
    for i in range(num_iterations):
        start_idx = i * pred_len
        end_idx = (i + 1) * pred_len
        iter_preds = all_preds[:, start_idx:end_idx, :]
        iter_trues = all_trues[:, start_idx:end_idx, :]
        iter_mae, iter_mse, _, _, _, _, _ = metric(iter_preds, iter_trues)
        print(f"  Iteration {i+1} (steps {start_idx}-{end_idx}): MSE={iter_mse:.7f}, MAE={iter_mae:.7f}")
    
    return {
        'preds': all_preds,  # [N, 384, c_out]
        'trues': all_trues,  # [N, 384, c_out] - full ground truth!
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'rse': rse,
            'corr': corr
        },
        'num_iterations': num_iterations,
        'pred_len_per_iteration': pred_len,
        'total_pred_len': total_pred_len
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
