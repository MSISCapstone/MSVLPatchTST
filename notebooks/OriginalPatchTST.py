import os
import sys


sys.path.insert(0, '/content/PatchTST/PatchTST_supervised')
sys.path.insert(0, '/content/PatchTST')

# Setup paths
os.chdir('/content/PatchTST/PatchTST_supervised')
dataset_path = '/content/PatchTST/datasets/weather'
model_checkpoints = '/content/model/checkpoints_weather'

# Configuration
seq_len = 336
model_name = 'PatchTST'

import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# Create argument namespace
args = argparse.Namespace()

# Basic config
args.is_training = 1
args.model_id = f'weather_{seq_len}_{seq_len}'
args.model = model_name

# Data loader
args.data = 'custom'
args.root_path = dataset_path  # Use the dataset_path variable defined earlier
args.data_path = 'weather.csv'
args.features = 'M'
args.target = 'OT'
args.freq = 'h'
args.checkpoints = model_checkpoints

# Forecasting task
args.seq_len = seq_len
args.label_len = 48
args.pred_len = 336

# PatchTST parameters
args.fc_dropout = 0.05
args.head_dropout = 0.0
args.patch_len = 16
args.stride = 8
args.padding_patch = 'end'
args.revin = 1
args.affine = 0
args.subtract_last = 0
args.decomposition = 0
args.kernel_size = 25
args.individual = 0
args.channel_independent = 1

# Multi-scale
args.multi_scale = 0
args.patch_lengths = '16'
args.patch_strides = '8'
args.patch_weights = '1.0'

# Model architecture
args.embed_type = 0
args.enc_in = 21
args.dec_in = 21
args.c_out = 21
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.d_ff = 2048
args.moving_avg = 25
args.factor = 1
args.distil = True
args.dropout = 0.05
args.embed = 'timeF'
args.activation = 'gelu'
args.output_attention = False
args.do_predict = False

# Optimization
args.num_workers = 0
args.itr = 1
args.train_epochs = 100
args.batch_size = 16
args.patience = 10
args.learning_rate = 0.0001
args.des = 'Exp'
args.loss = 'mse'
args.lradj = 'type3'
args.pct_start = 0.3
args.use_amp = False

# GPU
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'
args.test_flop = False

# Random seed
args.random_seed = 2021
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('Args in experiment:')
print(args)

# Verify test data for comparison
exp = Exp_Main(args)
print("\n" + "="*60)
print("TEST DATA VERIFICATION (DLinear Baseline)")
print("="*60)
# Get test data
test_data, test_loader = exp._get_data(flag='test')

# Get a sample batch to inspect
sample_batch = next(iter(test_loader))
batch_x, batch_y, batch_x_mark, batch_y_mark = sample_batch

print(f"Test loader info:")
print(f"  Total batches: {len(test_loader)}")
print(f"  Batch size: {batch_x.shape[0]}")
print(f"  Input shape (batch_x): {batch_x.shape}")
print(f"  Output shape (batch_y): {batch_y.shape}")
print(f"  Features in test_data: {test_data.data_x.shape[1] if hasattr(test_data, 'data_x') else 'N/A'}")

# Print first sample statistics for verification (FIRST 20 WEATHER FEATURES ONLY)
if hasattr(test_data, 'data_x') and len(test_data.data_x) > 0:
    # Load the CSV to get column names AFTER data loader reordering
    import pandas as pd
    df_raw = pd.read_csv(os.path.join(dataset_path, 'weather.csv'))
    
    # Simulate the data loader column reordering logic
    cols = list(df_raw.columns)
    cols.remove('date')
    cols.remove('OT')  # target
    reordered_cols = cols + ['OT']  # Data loader puts target at the end
    
    print(f"\nColumn order AFTER data loader reordering (excluding 'date'):")
    print(f"  Total features: {len(reordered_cols)}")
    print(f"  First 20: {reordered_cols[:20]}")
    print(f"  Last feature: {reordered_cols[-1]}")
    
    # Get first 20 weather feature names (should exclude OT)
    weather_cols = reordered_cols[:20]
    
    # Statistics for first 20 columns
    first_sample_20 = test_data.data_x[0, :20]
    print(f"\nFirst test sample statistics (First 20 features):")
    print(f"  Mean: {first_sample_20.mean():.6f}")
    print(f"  Std: {first_sample_20.std():.6f}")
    print(f"  Min: {first_sample_20.min():.6f}")
    print(f"  Max: {first_sample_20.max():.6f}")
    print(f"  Sum: {first_sample_20.sum():.6f}")

print("="*60)