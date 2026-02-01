"""
Data preprocessing utilities for Physics-Integrated PatchTST
"""

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d


def apply_max_pooling_to_channels(df, channel_indices, kernel_size=4, stride=1):
    """
    Apply max pooling to specific channels in the dataframe while preserving dimensions.
    
    Args:
        df: DataFrame with time series data
        channel_indices: List of column indices to apply max pooling
        kernel_size: Size of the max pooling kernel
        stride: Stride for max pooling
        
    Returns:
        DataFrame with max pooling applied to specified channels
    """
    df_copy = df.copy()
    
    # Get column names
    all_cols = df.columns.tolist()
    
    # Skip 'date' column
    data_cols = [col for col in all_cols if col != 'date']
    
    for idx in channel_indices:
        if idx < len(data_cols):
            col_name = data_cols[idx]
            
            # Get the data as numpy array
            data = df_copy[col_name].values
            
            # Apply max pooling with edge preservation
            pooled = maximum_filter1d(data, size=kernel_size, mode='nearest')
            
            # Update the dataframe
            df_copy[col_name] = pooled
            
            print(f"  Applied max pooling (kernel={kernel_size}) to channel {idx}: {col_name}")
    
    return df_copy


def add_hour_of_day_features(input_path, output_path, apply_pooling=False, 
                             pool_channel_indices=None, pool_kernel=4, pool_stride=1):
    """
    Add hour_sin and hour_cos columns to the weather dataset.
    Optionally apply max pooling to specific channels before saving.
    
    Args:
        input_path: Path to original weather.csv
        output_path: Path to save weather_with_hour.csv
        apply_pooling: Whether to apply max pooling to specific channels
        pool_channel_indices: List of channel indices to apply pooling (e.g., [11, 12, 15] for wind speed, max wind, raining)
        pool_kernel: Kernel size for max pooling
        pool_stride: Stride for max pooling
    
    Returns:
        DataFrame with added hour features and optional pooling
    """
    print(f"Loading original dataset from: {input_path}")
    df = pd.read_csv(input_path)

    # Keep all columns including 'OT' for consistency with baseline
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Parse datetime from the 'date' column
    df['datetime'] = pd.to_datetime(df['date'])
    
    # Extract hour (0-23)
    df['hour'] = df['datetime'].dt.hour
    
    # Create cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Drop temporary columns
    df = df.drop(columns=['datetime', 'hour'])
    
    # Reorder columns: keep 'date' first, then original features, then hour features
    original_cols = [col for col in df.columns if col not in ['hour_sin', 'hour_cos']]
    df = df[original_cols + ['hour_sin', 'hour_cos']]
    
    # Apply max pooling if requested (before adding hour features to indices)
    if apply_pooling and pool_channel_indices is not None:
        print(f"\nApplying max pooling to long channel features:")
        df = apply_max_pooling_to_channels(df, pool_channel_indices, pool_kernel, pool_stride)
    
    print(f"\nNew shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")
    print(f"\nHour feature statistics:")
    print(df[['hour_sin', 'hour_cos']].describe())
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    print(f"\n✓ Enhanced dataset saved to: {output_path}")
    
    return df
