"""
Data preparation module for multi-output intensity forecasting
Handles loading, preprocessing, and splitting data for model training
Supports both single and multi-output targets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import config


def load_data(filepath):
    """
    Load training data from CSV file
    
    Expected columns:
    - event_id: Unique identifier for each event
    - forecast_hour: Hour of the forecast (0, 3, 6, etc.)
    - met_var_1, met_var_2, ...: Meteorological variables
    - intensity, precipitation_target: Target variables
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Data shape: {df.shape}")
    print(f"Number of unique events: {df[config.EVENT_ID_COLUMN].nunique()}")
    print(f"Forecast hours present: {sorted(df[config.FORECAST_HOUR_COLUMN].unique())}")
    print(f"Target variables: {config.TARGET_COLUMNS}")
    
    return df


def validate_data(df):
    """
    Validate data quality and completeness
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Boolean indicating if data is valid
    """
    required_columns = (
        [config.EVENT_ID_COLUMN, config.FORECAST_HOUR_COLUMN] +
        config.FEATURE_COLUMNS +
        config.TARGET_COLUMNS
    )
    
    # Check for missing columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False
    
    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        print("Warning: Missing values detected:")
        print(missing_values[missing_values > 0])
    
    # Check for duplicate entries
    duplicates = df.duplicated(subset=[config.EVENT_ID_COLUMN, config.FORECAST_HOUR_COLUMN])
    if duplicates.any():
        print(f"Warning: {duplicates.sum()} duplicate event-hour combinations found")
    
    return True


def prepare_data_for_hour(df, forecast_hour):
    """
    Prepare training data for a specific forecast hour
    
    Args:
        df: Full DataFrame with all events and hours
        forecast_hour: Specific hour to extract (e.g., 0, 3, 6)
        
    Returns:
        X: Feature matrix (meteorological variables)
        y: Target matrix (intensity and/or precipitation) - 2D array for multi-output
        event_ids: Event identifiers for tracking
    """
    # Filter data for specific forecast hour
    hour_data = df[df[config.FORECAST_HOUR_COLUMN] == forecast_hour].copy()
    
    if len(hour_data) == 0:
        raise ValueError(f"No data found for forecast hour {forecast_hour}")
    
    # Extract features and targets
    X = hour_data[config.FEATURE_COLUMNS].values
    
    # For multi-output: y should be a 2D array with shape (n_samples, n_targets)
    # For single-output: y should be a 1D array
    if len(config.TARGET_COLUMNS) > 1:
        y = hour_data[config.TARGET_COLUMNS].values  # Shape: (n_samples, n_targets)
    else:
        y = hour_data[config.TARGET_COLUMNS[0]].values  # Shape: (n_samples,)
    
    event_ids = hour_data[config.EVENT_ID_COLUMN].values
    
    print(f"Hour {forecast_hour:02d}: {len(hour_data)} samples, targets: {config.TARGET_COLUMNS}")
    
    return X, y, event_ids


def split_data_by_events(df, test_size=0.2, random_state=42):
    """
    Split data by events (not by individual rows) to prevent data leakage
    This ensures all hours from an event are in the same split
    
    Args:
        df: Full DataFrame
        test_size: Proportion of events for test set
        random_state: Random seed for reproducibility
        
    Returns:
        train_df: Training DataFrame
        test_df: Test DataFrame
    """
    # Get unique event IDs
    unique_events = df[config.EVENT_ID_COLUMN].unique()
    
    # Split events
    train_events, test_events = train_test_split(
        unique_events,
        test_size=test_size,
        random_state=random_state
    )
    
    # Create train and test DataFrames
    train_df = df[df[config.EVENT_ID_COLUMN].isin(train_events)].copy()
    test_df = df[df[config.EVENT_ID_COLUMN].isin(test_events)].copy()
    
    print(f"\nData split:")
    print(f"Training events: {len(train_events)} ({len(train_df)} samples)")
    print(f"Test events: {len(test_events)} ({len(test_df)} samples)")
    
    return train_df, test_df


def get_feature_scaler(X_train):
    """
    Fit a StandardScaler on training data
    
    Args:
        X_train: Training feature matrix
        
    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def print_data_summary(df):
    """
    Print summary statistics of the dataset
    
    Args:
        df: DataFrame to summarize
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Number of events: {df[config.EVENT_ID_COLUMN].nunique()}")
    print(f"Forecast hours: {sorted(df[config.FORECAST_HOUR_COLUMN].unique())}")
    
    print(f"\nTarget variables ({', '.join(config.TARGET_COLUMNS)}) statistics:")
    print(df[config.TARGET_COLUMNS].describe())
    
    print(f"\nFeature statistics:")
    print(df[config.FEATURE_COLUMNS].describe())
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    filepath = os.path.join(config.DATA_DIR, 'training_data.csv')
    
    # Load and validate data
    df = load_data(filepath)
    
    if validate_data(df):
        print_data_summary(df)
        
        # Split data
        train_df, test_df = split_data_by_events(df, test_size=config.TEST_SIZE)
        
        # Example: Prepare data for 00h forecast
        X_train, y_train, _ = prepare_data_for_hour(train_df, forecast_hour=0)
        X_test, y_test, _ = prepare_data_for_hour(test_df, forecast_hour=0)
        
        print(f"\nExample for 00h forecast:")
        print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")
