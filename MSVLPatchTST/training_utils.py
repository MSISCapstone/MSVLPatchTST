"""
Utility functions for training and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, ConcatDataset


def create_full_train_loader(train_loader, val_loader, test_loader, batch_size, num_workers=0):
    """
    Create a combined data loader containing all data for training.
    
    Args:
        train_loader: Original train data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        batch_size: Batch size for the combined loader
        num_workers: Number of workers for data loading
        
    Returns:
        Combined DataLoader with all data
    """
    # Get the datasets from the loaders
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
    # Combine all datasets
    combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    # Create new loader with all data
    full_train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    print(f"Train loader - 2 created: {len(combined_dataset)} total samples")
    print(f"  Original train: {len(train_dataset)} samples")
    print(f"  Original val: {len(val_dataset)} samples")
    print(f"  Original test: {len(test_dataset)} samples")
    
    return full_train_loader


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_target_indices(channel_groups: dict) -> Tuple[list, list]:
    """
    Get target indices from all channel groups.
    
    Args:
        channel_groups: Dictionary of channel groups
        
    Returns:
        target_indices: List of channel indices
        target_names: List of channel names
    """
    target_indices = []
    target_names = []
    
    for group in channel_groups.values():
        for idx in group['output_indices']:
            target_indices.append(idx)
            name_idx = group['indices'].index(idx)
            target_names.append(group['names'][name_idx])
    
    return target_indices, target_names


def validate(model, val_loader, criterion, device, target_indices):
    """
    Validation function for MSVLPatchTST.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        target_indices: Indices of target channels (not used, kept for compatibility)
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = []
    
    # Get target input indices from model (p=0, T=1, wv=11, max.wv=12, rain=14, raining=15)
    target_input_indices = []
    for group_name in model.channel_groups.keys():
        target_input_indices.extend(model.group_info[group_name].get('target_indices', []))
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Forward pass - outputs [bs, pred_len, 6] for 6 target features
            outputs = model(batch_x)
            outputs_selected = outputs[:, -model.pred_len:, :]  # [bs, pred_len, 6]
            
            # Extract ground truth for target features only
            batch_y_targets = batch_y[:, -model.pred_len:, target_input_indices]  # [bs, pred_len, 6]
            
            loss = criterion(outputs_selected.cpu(), batch_y_targets.cpu())
            total_loss.append(loss.item())
    
    model.train()
    return np.mean(total_loss)


def get_scheduler(optimizer, num_warmup_steps=1000, num_training_steps=10000):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        
    Returns:
        Scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.1, 0.5 * (1 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience=10, verbose=True, delta=0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: If True, prints a message for each validation loss improvement
            delta: Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        import os
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, scheduler, epoch, args):
    """Adjust learning rate based on scheduler"""
    if args.lradj == 'type3':
        scheduler.step()
    else:
        # Standard learning rate adjustment
        if epoch < args.train_epochs // 2:
            lr = args.learning_rate
        else:
            lr = args.learning_rate * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
