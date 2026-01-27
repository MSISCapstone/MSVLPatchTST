"""
Training module for Physics-Integrated PatchTST
Supports:
  A) Multichannel forecast output: [B, pred_len, C_out]
  B) Fusion head output:          [B, out_dim] OR [B, pred_len] OR [B, pred_len, 1]
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from .training_utils import validate, adjust_learning_rate


def _safe_torch_load(path, device):
    """
    Compatibility loader for different torch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _select_target_from_batch_y(batch_y, args, target_indices):
    """
    Returns target tensor from batch_y based on args.pred_len and target_indices.

    batch_y: [B, label_len + pred_len, C]  (typical PatchTST style)
    Returns: [B, pred_len, K] where K = len(target_indices)
    """
    # Ensure indices are list-like
    if isinstance(target_indices, (int, np.integer)):
        target_indices = [int(target_indices)]
    else:
        target_indices = list(target_indices)

    y = batch_y[:, -args.pred_len:, target_indices]  # [B, pred_len, K]
    return y


def _align_outputs_to_targets(outputs, y_true, args):
    """
    Align model outputs to y_true for loss computation.

    y_true: [B, pred_len, K]
    outputs can be:
      - [B, pred_len, C]   (old full forecast)
      - [B, out_dim]       (fusion head)
      - [B, pred_len]      (fusion head multi-horizon)
      - [B, pred_len, 1]   (fusion head with last dim)
      - [B, 1]             (fusion head single-step)

    Returns:
      y_pred_aligned, y_true_aligned with matching shapes.
    """

    # Case 1: old model output [B, pred_len, C] (we will select targets outside before calling this)
    if outputs.ndim == 3:
        # Expected already selected outside OR full channels.
        # If it is full channels, caller should slice it. Here we just pass through.
        return outputs, y_true

    # Case 2: fusion head outputs [B, out_dim]
    if outputs.ndim == 2:
        B, out_dim = outputs.shape
        K = y_true.shape[-1]

        # If K==1:
        # - If out_dim == pred_len: compare with y_true squeezed to [B, pred_len]
        # - If out_dim == 1: compare with last step y_true [B,1]
        if K == 1:
            y_true_1 = y_true.squeeze(-1)  # [B, pred_len]
            if out_dim == args.pred_len:
                return outputs, y_true_1
            elif out_dim == 1:
                return outputs, y_true_1[:, -1:].contiguous()
            else:
                raise ValueError(
                    f"Fusion output out_dim={out_dim} does not match pred_len={args.pred_len} "
                    f"and is not 1. Choose out_dim = 1 or out_dim = pred_len."
                )

        # If K>1 and fusion returns vector, you must decide mapping (not supported silently)
        raise ValueError(
            f"y_true has K={K} target channels, but fusion output is 2D {outputs.shape}. "
            f"Fusion head typically supports single target (K=1)."
        )

    # Case 3: outputs [B, pred_len, 1]
    if outputs.ndim == 3 and outputs.shape[-1] == 1:
        # Align to y_true [B,pred_len,1] (if K==1)
        if y_true.shape[-1] != 1:
            raise ValueError("outputs last dim is 1 but y_true has multiple targets.")
        return outputs, y_true

    raise ValueError(f"Unsupported outputs shape: {outputs.shape}")


def train_model(model, train_loader, val_loader, test_loader, optimizer, scheduler,
                criterion, args, device, target_indices, checkpoint_path):
    """
    Training loop for Physics-Integrated PatchTST.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        args: Configuration arguments
        device: Device to train on
        target_indices: Indices of target channels to compute loss
        checkpoint_path: Path to save checkpoints

    Returns:
        Dictionary with training history
    """
    from .training_utils import EarlyStopping

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training history
    train_losses = []
    val_losses = []
    test_losses = []

    # Per-group losses (only meaningful for old 3D outputs)
    group_train_losses = {name: [] for name in args.channel_groups.keys()}

    # Target variable losses (only meaningful for old 3D outputs)
    target_variable_losses = {'long': [], 'short': []}

    print(f"\nStarting Physics-Integrated PatchTST Training...")
    print(f"Checkpoint path: {checkpoint_path}")
    print("=" * 70)
    print(f"Input channels: {args.enc_in} (21 weather + 2 hour)")
    print(f"Output channels: {args.c_out} (21 weather only)")
    print("─" * 70)
    print(f"Physics Groups (with integrated hour features):")
    for name, cfg in args.patch_configs.items():
        info = model.group_info[name]
        hour_str = f" + hour" if len(info['hour_indices']) > 0 else ""
        print(f"  {name}: {info['n_output']} weather{hour_str} → patch={cfg['patch_len']} , stride={cfg['stride']}")
    print("=" * 70)

    for epoch in range(args.train_epochs):
        model.train()
        epoch_time = time.time()
        train_loss = []

        batch_group_losses = {name: [] for name in args.channel_groups.keys()}
        batch_target_losses = {'long': [], 'short': []}

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # ---------- Forward ----------
            # Old: outputs = model(batch_x)  -> [B, pred_len, C]
            # New fusion: outputs = model(...) -> [B, out_dim] etc.
            outputs = model(batch_x)

            # ---------- Targets ----------
            y_true = _select_target_from_batch_y(batch_y, args, target_indices)  # [B, pred_len, K]

            # ---------- Align for loss ----------
            # If old output is 3D full channels, we select the same targets first.
            if outputs.ndim == 3:
                # outputs is expected [B, pred_len, C_out] or [B, pred_len, C]
                # select only target_indices (same as y_true)
                if isinstance(target_indices, (int, np.integer)):
                    ti = [int(target_indices)]
                else:
                    ti = list(target_indices)
                # NOTE: your original code uses outputs[:, -pred_len:, target_indices]
                # we enforce pred_len alignment here:
                outputs_selected = outputs[:, -args.pred_len:, ti]
                y_pred_aligned, y_true_aligned = outputs_selected, y_true
            else:
                # fusion head outputs
                y_pred_aligned, y_true_aligned = _align_outputs_to_targets(outputs, y_true, args)

            # ---------- Loss ----------
            loss = criterion(y_pred_aligned, y_true_aligned)
            train_loss.append(loss.item())

            # ---------- Optional tracking (only if old-style 3D outputs) ----------
            if outputs.ndim == 3:
                with torch.no_grad():
                    # Per-group loss tracking
                    for group_name in args.channel_groups.keys():
                        info = model.group_info[group_name]
                        weather_indices = info['weather_indices']
                        if len(weather_indices) > 0:
                            # outputs is [B, pred_len, C]; align true to last pred_len
                            group_out = outputs[:, -args.pred_len:, weather_indices]
                            group_true = batch_y[:, -args.pred_len:, weather_indices]
                            group_loss = criterion(group_out, group_true)
                            batch_group_losses[group_name].append(group_loss.item())

                    # Target variable losses (your original indices)
                    long_indices = args.channel_groups['long_channel']['output_indices']   # e.g. [10,1,2,7]
                    short_indices = args.channel_groups['short_channel']['output_indices'] # e.g. [11,15,8]

                    long_out = outputs[:, -args.pred_len:, long_indices]
                    short_out = outputs[:, -args.pred_len:, short_indices]
                    long_true = batch_y[:, -args.pred_len:, long_indices]
                    short_true = batch_y[:, -args.pred_len:, short_indices]

                    batch_target_losses['long'].append(criterion(long_out, long_true).item())
                    batch_target_losses['short'].append(criterion(short_out, short_true).item())

            # ---------- Backward ----------
            loss.backward()
            optimizer.step()

        # ---------- Epoch statistics ----------
        train_loss_avg = float(np.mean(train_loss))

        # validate() currently expects old signature: validate(model, loader, criterion, device, target_indices)
        # It will still work if your validate() also handles fusion outputs.
        val_loss = validate(model, val_loader, criterion, device, target_indices)
        test_loss = validate(model, test_loader, criterion, device, target_indices)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        # Store per-group losses
        for group_name in args.channel_groups.keys():
            if len(batch_group_losses[group_name]) > 0:
                group_train_losses[group_name].append(float(np.mean(batch_group_losses[group_name])))

        # Store target variable losses
        for target_name in ['long', 'short']:
            if len(batch_target_losses[target_name]) > 0:
                target_variable_losses[target_name].append(float(np.mean(batch_target_losses[target_name])))

        epoch_duration = time.time() - epoch_time

        print(f"\nEpoch {epoch+1}/{args.train_epochs} | Time: {epoch_duration:.2f}s")
        print(f"  Train Loss: {train_loss_avg:.7f} | Val Loss: {val_loss:.7f} | Test Loss: {test_loss:.7f}")
        print(f"  Epoch duration: {epoch_duration:.2f} seconds")

        # Print target variable losses only if tracked (old-style outputs)
        if len(batch_target_losses['long']) > 0 or len(batch_target_losses['short']) > 0:
            print(f"  Target Variable Losses:")
            for target_name in ['long', 'short']:
                if len(batch_target_losses[target_name]) > 0:
                    print(f"    {target_name.capitalize()}: {float(np.mean(batch_target_losses[target_name])):.7f}")

        # ---------- Early stopping ----------
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break

        # ---------- Adjust learning rate ----------
        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)

    print("=" * 70)
    print("Training completed!")

    # ---------- Load best model ----------
    best_model_path = os.path.join(checkpoint_path, 'checkpoint.pth')
    state = _safe_torch_load(best_model_path, device)
    model.load_state_dict(state)
    print(f"Best model loaded from: {best_model_path}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'group_train_losses': group_train_losses,
        'target_variable_losses': target_variable_losses
    }
