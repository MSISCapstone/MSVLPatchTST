import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set repository root path and change to it
repo_root_path = '/content/PatchTST'
os.chdir(repo_root_path)
print(f"Working directory: {os.getcwd()}")

# Build clean sys.path with supervised ahead of physics to avoid utils shadowing
supervised_path = os.path.join(repo_root_path, 'PatchTST_supervised')
physics_path = os.path.join(repo_root_path, 'PatchTST_physics_integrated')
new_paths = [p for p in [supervised_path, physics_path, repo_root_path] if p not in sys.path]
sys.path = new_paths + sys.path  # prepend in desired order

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Python path head: {sys.path[:5]}")

# Numpy fix
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    np.NaN = np.nan
    np.NAN = np.nan
    np.NINF = np.NINF if hasattr(np, 'NINF') else -np.inf
    print("NumPy compatibility patch applied for np.Inf -> np.inf")
else:
    print("NumPy already has np.Inf attribute")


from MSVLPatchTST.config import PhysicsIntegratedConfig
from MSVLPatchTST.models import PhysicsIntegratedPatchTST
from MSVLPatchTST.training_utils import set_seed, get_target_indices, get_scheduler
from MSVLPatchTST.trainer import train_model
from MSVLPatchTST.evaluation import evaluate_model, evaluate_per_channel
from MSVLPatchTST.data_preprocessing import add_hour_of_day_features

print("✓ All modules imported successfully")


# Create configuration
args = PhysicsIntegratedConfig()
args.random_seed = 2021
set_seed(args.random_seed)

# Override key parameters for comparison with DLinear baseline
args.seq_len = 336       # Look back window (same as DLinear baseline)
args.pred_len = 336      # Prediction horizon (same as DLinear baseline)
args.patience = 3
args.train_epochs = 15
args.batch_size = 16     # Same batch size as DLinear baseline
args.use_cross_channel_encoder = False
args.use_cross_group_attention = True

# CRITICAL: Make model predict ALL 24 features (20 weather + OT + hour_sin + hour_cos)
# Update channel groups to output all indices
for group_name in args.channel_groups:
    # Set output_indices to all indices (0-23 for 24 features)
    args.channel_groups[group_name]['output_indices'] = list(range(24))

print(f"Configuration:")
print(f"  seq_len (look back): {args.seq_len}")
print(f"  pred_len (forecast): {args.pred_len}")
print(f"  batch_size: {args.batch_size}")
print(f"  train_epochs: {args.train_epochs}")
print(f"  enc_in (features): {args.enc_in}")
print(f"  c_out (output features): {args.c_out}")
print(f"\nChannel groups output all 24 features")


# Add hour-of-day features to dataset
# ALWAYS regenerate from original source - no caching
original_path = os.path.join(args.root_path, 'weather.csv')
enhanced_path = os.path.join(args.root_path, args.data_path)

# Set random seed for reproducibility
np.random.seed(2021)

# ALWAYS delete cached file to force regeneration from original source
if os.path.exists(enhanced_path):
    os.remove(enhanced_path)
    print(f"Deleted cached file: {enhanced_path}")

# Regenerate with NO max pooling
df_enhanced = add_hour_of_day_features(
    original_path, 
    enhanced_path,
    apply_pooling=False  # NO max pooling
)

print(f"\n✓ Data regenerated from original source")
print(f"  Original: {original_path}")
print(f"  Enhanced: {enhanced_path}")
print(f"  Rows: {len(df_enhanced)}, Columns: {len(df_enhanced.columns)}")
print(f"  Max pooling: DISABLED")


# Change to PatchTST_supervised directory for data_provider imports
import importlib

os.chdir(os.path.join(repo_root_path, 'PatchTST_supervised'))
print(f"Changed to: {os.getcwd()}")

# Clear cached modules to avoid stale 'utils' shadowing
for m in [
    'utils', 'utils.timefeatures',
    'data_provider', 'data_provider.data_loader', 'data_provider.data_factory'
]:
    if m in sys.modules:
        sys.modules.pop(m, None)

from data_provider.data_factory import data_provider

os.chdir(repo_root_path)

# Set seed before data loading to ensure reproducible splits
import random
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

# Create data loaders
train_data, train_loader = data_provider(args, 'train')
val_data, val_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')

print(f"\nData loaded:")
print(f"  Train samples: {len(train_data)}")
print(f"  Val samples: {len(val_data)}")
print(f"  Test samples: {len(test_data)}")

# Verify data normalization parameters match
if hasattr(train_data, 'scaler'):
    print(f"\nData normalization:")
    print(f"  Mean shape: {train_data.scaler.mean_.shape if hasattr(train_data.scaler, 'mean_') else 'N/A'}")
    print(f"  Std shape: {train_data.scaler.scale_.shape if hasattr(train_data.scaler, 'scale_') else 'N/A'}")

# Verify test data for comparison
print("\n" + "="*60)
print("TEST DATA VERIFICATION (Physics-Integrated)")
print("="*60)

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
    df_raw = pd.read_csv(enhanced_path)
    
    # Simulate the data loader column reordering logic
    cols = list(df_raw.columns)
    cols.remove('date')
    cols.remove('OT')  # target (args.target is T (degC) but let's use OT for consistency)
    reordered_cols = cols + ['OT']  # Data loader puts target at the end
    
    print(f"\nColumn order AFTER data loader reordering (excluding 'date'):")
    print(f"  Total features: {len(reordered_cols)}")
    print(f"  First 20: {reordered_cols[:20]}")
    print(f"  Features 20-22: {reordered_cols[20:23] if len(reordered_cols) >= 23 else reordered_cols[20:]}")
    print(f"  Last feature: {reordered_cols[-1]}")
    
    # Get first 20 weather feature names (should match baseline)
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