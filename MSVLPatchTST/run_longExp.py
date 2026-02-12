"""
Main training script for MSVLPatchTST (Multi-Scale Variable-Length PatchTST)
Compatible with the original PatchTST command-line interface
"""

import argparse
import torch
import numpy as np
import random
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from MSVLPatchTST.config import MSVLConfig
from MSVLPatchTST.models import MSVLPatchTST
from MSVLPatchTST.trainer import train_model
from MSVLPatchTST.evaluation import evaluate_model, evaluate_model_sliding_window
from MSVLPatchTST.training_utils import get_target_indices, set_seed

# Import data loader from local copy
from MSVLPatchTST.data_provider.data_factory import data_provider


def parse_args():
    """Parse command line arguments to match original PatchTST interface"""
    parser = argparse.ArgumentParser(description='MSVLPatchTST for Long-term Time Series Forecasting')
    
    # Basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='MSVLPatchTST',
                        help='model name')
    
    # Data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/weather/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='direct path to checkpoint file (overrides checkpoints)')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # not used in PatchTST
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # Model define
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=6, help='output size (6 target features)')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    
    # Patching
    parser.add_argument('--patch_len', type=int, default=16, help='patch length (used when patch_len_short/long not specified)')
    parser.add_argument('--stride', type=int, default=8, help='stride (used when stride_short/long not specified)')
    # Multi-scale patching for short and long channels
    parser.add_argument('--patch_len_short', type=int, default=None, help='patch length for short channel (fast dynamics)')
    parser.add_argument('--stride_short', type=int, default=None, help='stride for short channel')
    parser.add_argument('--patch_len_long', type=int, default=None, help='patch length for long channel (slow dynamics)')
    parser.add_argument('--stride_long', type=int, default=None, help='stride for long channel')
    # Loss weights per channel group (higher weight = more focus during training)
    parser.add_argument('--weight_short', type=float, default=1.5, help='loss weight for short channel (sparse features)')
    parser.add_argument('--weight_long', type=float, default=0.5, help='loss weight for long channel')
    # Cross-group attention configuration
    parser.add_argument('--cross_group_ffn_ratio', type=int, default=2, help='FFN expansion ratio for cross-group attention (default: 2)')
    parser.add_argument('--padding_patch', type=str, default='end', help='padding patch, options:[None, end]')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='huber', help='loss function (mse, mae, huber)')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    
    # Other
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    
    # Iterative prediction
    parser.add_argument('--num_iterations', type=int, default=1, 
                        help='Number of prediction iterations per sample (default: 1)')
    parser.add_argument('--iterative', action='store_true', default=False,
                        help='Use iterative multi-step prediction (default: single-step)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use for sliding window evaluation (default: all)')
    parser.add_argument('--window_stride', type=int, default=96,
                        help='Stride between sliding windows (default: 96 for non-overlapping predictions)')
    
    args = parser.parse_args()
    return args


def main():
    """Main training loop"""
    # Parse arguments
    args = parse_args()
    
    # Create MSVLPatchTST config and update with command line arguments
    config = MSVLConfig()
    
    # Update config with command line arguments
    config.random_seed = args.random_seed
    config.root_path = args.root_path
    config.data_path = args.data_path
    config.features = args.features
    config.target = args.target
    config.freq = args.freq
    config.seq_len = args.seq_len
    config.label_len = args.label_len
    config.pred_len = args.pred_len
    config.enc_in = args.enc_in
    config.dec_in = args.dec_in
    config.c_out = args.c_out
    config.d_model = args.d_model
    config.n_heads = args.n_heads
    config.e_layers = args.e_layers
    config.d_layers = args.d_layers
    config.d_ff = args.d_ff
    config.dropout = args.dropout
    config.fc_dropout = args.fc_dropout
    config.head_dropout = args.head_dropout
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.train_epochs = args.train_epochs
    config.patience = args.patience
    config.num_workers = args.num_workers
    config.use_gpu = args.use_gpu
    config.gpu = args.gpu
    config.use_multi_gpu = args.use_multi_gpu
    config.devices = args.devices
    config.checkpoints = args.checkpoints
    config.des = args.des
    config.loss = args.loss
    config.lradj = args.lradj
    config.use_amp = args.use_amp
    config.pct_start = args.pct_start
    
    # Update patch_configs with command line arguments for multi-scale patching
    # Use channel-specific args if provided, otherwise fall back to default patch_len/stride
    patch_len_short = args.patch_len_short if args.patch_len_short is not None else args.patch_len
    stride_short = args.stride_short if args.stride_short is not None else args.stride
    patch_len_long = args.patch_len_long if args.patch_len_long is not None else args.patch_len
    stride_long = args.stride_long if args.stride_long is not None else args.stride
    
    config.patch_configs = {
        'short_channel': {'patch_len': patch_len_short, 'stride': stride_short, 'weight': args.weight_short},
        'long_channel': {'patch_len': patch_len_long, 'stride': stride_long, 'weight': args.weight_long}
    }
    
    # Cross-group attention configuration
    config.cross_group_ffn_ratio = args.cross_group_ffn_ratio
    
    # Set seed
    set_seed(config.random_seed)
    
    # Setup device
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.gpu}')
        print(f'Using GPU: {device}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    # Print configuration
    print('\n' + '='*80)
    print('MSVLPatchTST Training Configuration')
    print('='*80)
    print(f'Model: {config.model}')
    print(f'Dataset: {config.data_path}')
    print(f'Sequence Length: {config.seq_len}')
    print(f'Prediction Length: {config.pred_len}')
    print(f'Batch Size: {config.batch_size}')
    print(f'Learning Rate: {config.learning_rate}')
    print(f'Training Epochs: {config.train_epochs}')
    print(f'Encoder Input Channels: {config.enc_in}')
    print(f'Model Dimension: {config.d_model}')
    print(f'Number of Heads: {config.n_heads}')
    print(f'Number of Layers: {config.e_layers}')
    print('='*80 + '\n')
    
    # Create data loaders using original PatchTST data provider
    print('Loading data...')
    
    # Create a dummy args object for data_provider
    class DataArgs:
        pass
    
    data_args = DataArgs()
    data_args.data = config.data
    data_args.root_path = config.root_path
    data_args.data_path = config.data_path
    data_args.features = config.features
    data_args.target = config.target
    data_args.freq = config.freq
    data_args.seq_len = config.seq_len
    data_args.label_len = config.label_len
    data_args.pred_len = config.pred_len
    data_args.batch_size = config.batch_size
    data_args.num_workers = config.num_workers
    data_args.embed = 'timeF'
    
    train_dataset, train_loader = data_provider(data_args, 'train')
    val_dataset, val_loader = data_provider(data_args, 'val')
    test_dataset, test_loader = data_provider(data_args, 'test')
    
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')
    
    # Create model
    print('\nInitializing MSVLPatchTST model...')
    model = MSVLPatchTST(config).float().to(device)
    
    # Print model architecture
    print('\n' + '='*80)
    print('Model Architecture')
    print('='*80)
    print(model)
    print('='*80 + '\n')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Get target indices for loss computation
    target_indices, target_names = get_target_indices(config.channel_groups)
    print(f'Target indices: {target_indices}')
    print(f'Target names: {target_names}')
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Create learning rate scheduler
    num_training_steps = len(train_loader) * config.train_epochs
    num_warmup_steps = int(config.pct_start * num_training_steps)
    
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=num_training_steps,
        pct_start=config.pct_start,
        anneal_strategy='cos'
    )
    
    # Loss function
    if config.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.loss == 'mae':
        criterion = torch.nn.L1Loss()
    elif config.loss == 'huber':
        criterion = torch.nn.HuberLoss(delta=1.0)
    else:
        criterion = torch.nn.HuberLoss(delta=1.0)
    
    # Setup checkpoint directory
    if args.checkpoint_path:
        # Direct checkpoint path provided - extract directory
        checkpoint_path = os.path.dirname(args.checkpoint_path)
        if not checkpoint_path:
            checkpoint_path = '.'
        checkpoint_dir = checkpoint_path
    else:
        # Construct checkpoint directory
        checkpoint_dir = os.path.join(config.checkpoints, f'{args.model_id}')
        checkpoint_path = checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f'\nCheckpoint directory: {checkpoint_dir}')
    
    if args.is_training:
        print(f'Starting training...\n')
        
        # Train model
        history = train_model(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, criterion, config, device,
            target_indices, checkpoint_path
        )
        
        print('\n' + '='*80)
        print('Loading best model for final evaluation...')
        print('='*80)
        
        best_model_file = os.path.join(checkpoint_path, 'checkpoint.pth')
        if os.path.exists(best_model_file):
            model.load_state_dict(torch.load(best_model_file))
            print(f'Loaded best model from: {best_model_file}')
        else:
            print('No checkpoint found, using final model')
    else:
        print('Inference mode: Loading existing checkpoint...')
        
        best_model_file = os.path.join(checkpoint_path, 'checkpoint.pth')
        if os.path.exists(best_model_file):
            model.load_state_dict(torch.load(best_model_file))
            print(f'Loaded model from: {best_model_file}')
        else:
            print(f'Error: No checkpoint found at {best_model_file}')
            print('Please run training first or check the checkpoint path.')
            return None
    
    # Final evaluation on test set
    print('\nEvaluating on test set...')
    if args.iterative:
        print(f'Using sliding window prediction with stride={args.window_stride}')
        print(f'Each sample: {config.seq_len}-step lookback → {config.pred_len}-step prediction')
        if args.max_samples:
            print(f'Limiting to {args.max_samples} samples')
        results = evaluate_model_sliding_window(
            model, test_dataset, device, config, 
            num_iterations=args.num_iterations, 
            max_samples=args.max_samples,
            window_stride=args.window_stride
        )
    else:
        results = evaluate_model(model, test_loader, device, config)
    
    # Print results
    print('\n' + '='*80)
    print('MSVLPatchTST Test Results')
    if args.iterative:
        total_steps = args.max_samples * config.pred_len if args.max_samples else 'all'
        print(f'(Sliding window: stride={args.window_stride}, {args.max_samples} samples x {config.pred_len} = {total_steps} steps)')
    print('='*80)
    print(f'MSE: {results["metrics"]["mse"]:.6f}')
    print(f'MAE: {results["metrics"]["mae"]:.6f}')
    print(f'RMSE: {results["metrics"]["rmse"]:.6f}')
    print(f'Correlation: {results["metrics"]["corr"]:.4f}')
    print('='*80 + '\n')
    
    # Save results to corresponding output folder
    # Get git root or use current directory
    git_root = os.popen('git rev-parse --show-toplevel 2>/dev/null').read().strip()
    if not git_root:
        git_root = os.getcwd()
    
    # Create output directory structure: output/MSVLPatchTST/test_results/weather_336_96/
    results_dir = os.path.join(git_root, 'output', 'MSVLPatchTST', 'test_results', args.model_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Build file suffix from patch/stride config
    patch_len_short = config.patch_configs['short_channel']['patch_len']
    stride_short = config.patch_configs['short_channel']['stride']
    patch_len_long = config.patch_configs['long_channel']['patch_len']
    stride_long = config.patch_configs['long_channel']['stride']
    file_suffix = f'_sl{config.seq_len}_pl{config.pred_len}_sp{patch_len_short}_ss{stride_short}_lp{patch_len_long}_ls{stride_long}'
    
    results_file = os.path.join(results_dir, f'results{file_suffix}.txt')
    
    # Save pred.npy for plotting script
    try:
        pred_file = os.path.join(results_dir, f'pred{file_suffix}.npy')
        np.save(pred_file, results['preds'])
        print(f'Saved predictions to: {pred_file}')
        print(f'Predictions shape: {results["preds"].shape}')
        
        # Also save ground truth for plotting
        true_file = os.path.join(results_dir, f'true{file_suffix}.npy')
        np.save(true_file, results['trues'])
        print(f'Saved ground truth to: {true_file}')
        
        # Save combined CSV with predictions and ground truth
        import pandas as pd
        preds = results['preds']  # [num_samples, pred_len, num_features]
        trues = results['trues']  # [num_samples, pred_len, num_features]
        num_samples, pred_len_actual, num_features = preds.shape
        
        # Build CSV rows: sample_idx, step, feature_idx, pred, true
        rows = []
        for sample_idx in range(num_samples):
            for step in range(pred_len_actual):
                for feat_idx in range(num_features):
                    rows.append({
                        'sample_idx': sample_idx,
                        'step': step,
                        'feature_idx': feat_idx,
                        'prediction': preds[sample_idx, step, feat_idx],
                        'ground_truth': trues[sample_idx, step, feat_idx]
                    })
        
        csv_df = pd.DataFrame(rows)
        csv_file = os.path.join(results_dir, f'predictions{file_suffix}.csv')
        csv_df.to_csv(csv_file, index=False)
        print(f'Saved combined predictions CSV to: {csv_file}')
        print(f'CSV shape: {len(csv_df)} rows ({num_samples} samples x {pred_len_actual} steps x {num_features} features)')
    except Exception as e:
        print(f'Error saving predictions: {e}')
    
    try:
        with open(results_file, 'w') as f:
            f.write('MSVLPatchTST Test Results\n')
            f.write('='*80 + '\n')
            f.write(f'Model ID: {args.model_id}\n')
            f.write(f'MSE: {results["metrics"]["mse"]:.6f}\n')
            f.write(f'MAE: {results["metrics"]["mae"]:.6f}\n')
            f.write(f'RMSE: {results["metrics"]["rmse"]:.6f}\n')
            f.write(f'Correlation: {results["metrics"]["corr"]:.4f}\n')
            f.write('='*80 + '\n')
        
        print(f'Results saved to: {results_file}')
    except Exception as e:
        print(f'Error saving results: {e}')
    
    return results


if __name__ == '__main__':
    main()
