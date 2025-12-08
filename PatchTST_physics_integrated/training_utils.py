"""
Utility functions for training and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


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
    Get target indices from rain, temperature, and wind predictors.
    
    Args:
        channel_groups: Dictionary of channel groups
        
    Returns:
        target_indices: List of channel indices
        target_names: List of channel names
    """
    target_indices = []
    target_names = []
    
    for group_key in ['rain_predictors', 'temperature_predictors', 'wind_predictors']:
        group = channel_groups[group_key]
        for idx in group['output_indices']:
            target_indices.append(idx)
            name_idx = group['indices'].index(idx)
            target_names.append(group['names'][name_idx])
    
    return target_indices, target_names


def validate(model, val_loader, criterion, device, target_indices):
    """
    Validation function for Physics-Integrated PatchTST.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        target_indices: Indices of target channels
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Forward pass - outputs only weather channels (21)
            outputs = model(batch_x)  # [bs, pred_len, 21]
            outputs_selected = outputs[:, -model.pred_len:, :][:, :, target_indices]
            batch_y_selected = batch_y[:, -model.pred_len:, :][:, :, target_indices]
            
            loss = criterion(outputs_selected.cpu(), batch_y_selected.cpu())
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
        self.val_loss_min = np.Inf
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
