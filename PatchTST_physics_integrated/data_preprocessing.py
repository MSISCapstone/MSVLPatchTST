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
    
    # Save enhanced dataset
    df.to_csv(output_path, index=False)
    print(f"\n✓ Enhanced dataset saved to: {output_path}")
    
    return df
