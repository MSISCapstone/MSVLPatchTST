"""
Data preprocessing utilities for Physics-Integrated PatchTST
"""

import numpy as np
import pandas as pd


def add_hour_of_day_features(input_path, output_path):
    """
    Add hour_sin and hour_cos columns to the weather dataset.
    
    Args:
        input_path: Path to original weather.csv
        output_path: Path to save weather_with_hour.csv
    
    Returns:
        DataFrame with added hour features
    """
    print(f"Loading original dataset from: {input_path}")
    df = pd.read_csv(input_path)

    # Remove unwanted columns (if present)
    # Some CSV exports include an extra index or OT column; drop 'OT' when present
    if 'OT' in df.columns:
        df = df.drop(columns=['OT'])

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
    
    print(f"\nNew shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")
    print(f"\nHour feature statistics:")
    print(df[['hour_sin', 'hour_cos']].describe())
    
    # Apply quantization to short_channel variables
    # Short channel indices (after dropping OT): raining (15), Tdew (3), max. wv (12)
    # Column names in order: date, p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, H2OC, rho, wv, max. wv, wd, rain, raining, SWDR, PAR, max. PAR, Tlog, hour_sin, hour_cos
    short_channel_cols = ['raining (s)', 'Tdew (degC)', 'max. wv (m/s)']
    
    print(f"\nApplying global quantization to short_channel variables...")
    for col in short_channel_cols:
        if col in df.columns:
            values = df[col].values
            
            # Compute global percentiles
            p10 = np.percentile(values, 10)
            p50 = np.percentile(values, 50)
            p90 = np.percentile(values, 90)
            
            # Create masks
            is_bottom_10 = values <= p10
            is_top_10 = values >= p90
            is_middle = ~(is_bottom_10 | is_top_10)
            
            # Quantize middle values
            quantized = values.copy()
            middle_above_50 = is_middle & (values > p50)
            middle_below_50 = is_middle & (values <= p50)
            quantized[middle_above_50] = p90
            quantized[middle_below_50] = p10
            
            df[col] = quantized
            
            print(f"  {col}:")
            print(f"    p10={p10:.4f}, p50={p50:.4f}, p90={p90:.4f}")
            print(f"    Quantized {np.sum(middle_below_50 | middle_above_50)} / {len(values)} values")
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    print(f"\n✓ Enhanced dataset with quantization saved to: {output_path}")
    
    return df
