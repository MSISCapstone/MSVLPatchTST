"""
Training module for Physics-Integrated PatchTST
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from .training_utils import validate, adjust_learning_rate


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
    
    # Per-group losses
    group_train_losses = {name: [] for name in args.channel_groups.keys()}
    
    # Target variable losses
    target_variable_losses = {
        'long': [],
        'short': []
    }
    
    print(f"\nStarting MSVLPatchTST Training...")
    print(f"Checkpoint path: {checkpoint_path}")
    print("=" * 70)
    print(f"Input channels: {args.enc_in} (20 weather + 2 hour)")
    print(f"Output channels: {args.c_out} (6 target features)")
    print("─" * 70)
    print(f"Physics-Based Channel Groups:")
    for name, cfg in args.patch_configs.items():
        info = model.group_info[name]
        target_names = args.channel_groups[name].get('target_names', [])
        print(f"  {name}: {target_names}")
        print(f"    → patch={cfg['patch_len']}, stride={cfg['stride']}, weight={cfg['weight']}")
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
            
            # Forward pass - outputs [bs, pred_len, 6] for 6 target features
            outputs = model(batch_x)
            
            # Get target input indices from model (p=0, T=1, wv=11, max.wv=12, rain=14, raining=15)
            target_input_indices = []
            for group_name in args.channel_groups.keys():
                target_input_indices.extend(model.group_info[group_name].get('target_indices', []))
            
            # Extract ground truth for target features only
            # batch_y: [bs, label_len+pred_len, enc_in] where enc_in=22
            # Select only the 6 target feature columns
            batch_y_targets = batch_y[:, -args.pred_len:, target_input_indices]  # [bs, pred_len, 6]
            
            # Loss - MSE only on 6 target features
            outputs_selected = outputs[:, -args.pred_len:, :]  # [bs, pred_len, 6]
            
            loss = criterion(outputs_selected, batch_y_targets)
            train_loss.append(loss.item())
            
            # Per-group loss tracking
            with torch.no_grad():
                for group_name in args.channel_groups.keys():
                    info = model.group_info[group_name]
                    output_indices = info['output_indices']
                    if len(output_indices) > 0:
                        # Get outputs and ground truth for this group
                        group_out = outputs[:, :, output_indices]  # [bs, pred_len, 3]
                        group_true = batch_y_targets[:, :, output_indices]  # [bs, pred_len, 3]
                        group_loss = criterion(group_out, group_true)
                        batch_group_losses[group_name].append(group_loss.item())
                
                # Target variable losses per group
                long_indices = model.group_info['long_channel']['output_indices']
                short_indices = model.group_info['short_channel']['output_indices']
                batch_target_losses['long'].append(criterion(outputs[:, :, long_indices], batch_y_targets[:, :, long_indices]).item())
                batch_target_losses['short'].append(criterion(outputs[:, :, short_indices], batch_y_targets[:, :, short_indices]).item())
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Epoch statistics
        train_loss_avg = np.mean(train_loss)
        val_loss = validate(model, val_loader, criterion, device, target_indices)
        test_loss = validate(model, test_loader, criterion, device, target_indices)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        
        # Store per-group losses
        for group_name in args.channel_groups.keys():
            if len(batch_group_losses[group_name]) > 0:
                group_train_losses[group_name].append(np.mean(batch_group_losses[group_name]))
        
        # Store target variable losses
        for target_name in ['long', 'short']:
            if len(batch_target_losses[target_name]) > 0:
                target_variable_losses[target_name].append(np.mean(batch_target_losses[target_name]))
        
        epoch_duration = time.time() - epoch_time
        
        print(f"\nEpoch {epoch+1}/{args.train_epochs} | Time: {epoch_duration:.2f}s")
        print(f"  Train Loss: {train_loss_avg:.7f} | Val Loss: {val_loss:.7f} | Test Loss: {test_loss:.7f}")
        print(f"  Epoch duration: {epoch_duration:.2f} seconds")
        print(f"  Target Variable Losses:")
        for target_name in ['long', 'short']:
            if len(batch_target_losses[target_name]) > 0:
                print(f"    {target_name.capitalize()}: {np.mean(batch_target_losses[target_name]):.7f}")
        
        # Early stopping check
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break
        
        # Adjust learning rate
        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)
    
    print("=" * 70)
    print("Training completed!")
    
    # Load best model
    best_model_path = os.path.join(checkpoint_path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    print(f"Best model loaded from: {best_model_path}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'group_train_losses': group_train_losses,
        'target_variable_losses': target_variable_losses
    }
