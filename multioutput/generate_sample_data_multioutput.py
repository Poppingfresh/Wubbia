"""
Generate sample data for testing the multi-output intensity forecasting pipeline

This script creates synthetic data with multiple target variables (intensity + precipitation)
"""

import pandas as pd
import numpy as np
import os
import config


def generate_sample_event(event_id, forecast_hours):
    """
    Generate data for a single event with multiple targets
    
    Args:
        event_id: Unique identifier for the event
        forecast_hours: List of forecast hours to generate
        
    Returns:
        DataFrame with event data
    """
    np.random.seed(hash(event_id) % 2**32)
    
    event_data = []
    
    # Generate base values that evolve over time
    base_intensity = np.random.uniform(20, 60)
    intensity_trend = np.random.uniform(-0.5, 1.5)
    
    # Precipitation should be correlated with intensity
    base_precip = np.random.uniform(0, 20)
    
    for hour in forecast_hours:
        # Evolve intensity over time
        intensity = base_intensity + intensity_trend * hour + np.random.normal(0, 3)
        intensity = max(15, min(100, intensity))
        
        # Precipitation correlated with intensity (stronger storms = more rain)
        # With some additional randomness
        precip = base_precip + intensity * 0.15 + np.random.normal(0, 2)
        precip = max(0, min(50, precip))  # Clip to reasonable range
        
        # Build row with all features and targets
        row = {
            config.EVENT_ID_COLUMN: event_id,
            config.FORECAST_HOUR_COLUMN: hour,
        }
        
        # Add meteorological features (correlated with both targets)
        row['temperature'] = round(25 + np.random.normal(0, 3) + intensity * 0.05, 2)
        row['pressure'] = round(1013 - intensity * 0.2 + np.random.normal(0, 2), 2)
        row['humidity'] = round(65 + intensity * 0.1 + precip * 0.2 + np.random.normal(0, 5), 2)
        row['wind_speed'] = round(10 + intensity * 0.3 + np.random.normal(0, 2), 2)
        row['wind_direction'] = round(np.random.uniform(0, 360), 2)
        row['precipitation'] = round(max(0, intensity * 0.05 + np.random.normal(0, 1)), 2)
        row['cloud_cover'] = round(min(100, max(0, 40 + intensity * 0.2 + precip * 0.3 + np.random.normal(0, 10))), 2)
        row['sea_surface_temp'] = round(27 + np.random.normal(0, 1) + intensity * 0.02, 2)
        row['vorticity'] = round(intensity * 0.01 + np.random.normal(0, 0.05), 4)
        row['shear'] = round(max(0, 15 - intensity * 0.05 + np.random.normal(0, 2)), 2)
        
        # Add target variables
        row['intensity'] = round(intensity, 2)
        row['precipitation_target'] = round(precip, 2)
        
        event_data.append(row)
    
    return pd.DataFrame(event_data)


def generate_sample_dataset(n_events=200, forecast_hours=None):
    """
    Generate a complete sample dataset with multi-output targets
    
    Args:
        n_events: Number of events to generate
        forecast_hours: List of forecast hours (default: config.FORECAST_HOURS)
        
    Returns:
        DataFrame with all events
    """
    if forecast_hours is None:
        forecast_hours = config.FORECAST_HOURS
    
    print(f"Generating {n_events} sample events with multi-output targets...")
    print(f"Targets: {config.TARGET_COLUMNS}")
    
    all_events = []
    
    for i in range(n_events):
        event_id = f"event_{i+1:03d}"
        event_df = generate_sample_event(event_id, forecast_hours)
        all_events.append(event_df)
    
    df = pd.concat(all_events, ignore_index=True)
    
    print(f"Generated {len(df)} total samples")
    print(f"Events: {df[config.EVENT_ID_COLUMN].nunique()}")
    print(f"Forecast hours: {sorted(df[config.FORECAST_HOUR_COLUMN].unique())}")
    
    # Print correlation between targets
    print(f"\nTarget correlation:")
    print(df[config.TARGET_COLUMNS].corr())
    
    return df


def generate_new_events(n_events=20, forecast_hours=None):
    """
    Generate new events for prediction (without target columns)
    
    Args:
        n_events: Number of new events to generate
        forecast_hours: List of forecast hours
        
    Returns:
        DataFrame with new events (no target columns)
    """
    if forecast_hours is None:
        forecast_hours = config.FORECAST_HOURS
    
    print(f"Generating {n_events} new events for prediction...")
    
    # Generate events with temporary targets
    df = generate_sample_dataset(n_events, forecast_hours)
    
    # Remove target columns for prediction dataset
    df_new = df.drop(columns=config.TARGET_COLUMNS)
    
    # Rename events
    event_mapping = {
        old: f"new_event_{i+1:03d}"
        for i, old in enumerate(df_new[config.EVENT_ID_COLUMN].unique())
    }
    df_new[config.EVENT_ID_COLUMN] = df_new[config.EVENT_ID_COLUMN].map(event_mapping)
    
    print(f"Generated {len(df_new)} samples for {df_new[config.EVENT_ID_COLUMN].nunique()} new events")
    
    return df_new


if __name__ == "__main__":
    # Create data directory
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Generate training data
    print("\n" + "="*60)
    print("GENERATING MULTI-OUTPUT SAMPLE TRAINING DATA")
    print("="*60)
    
    training_data = generate_sample_dataset(n_events=200)
    training_path = os.path.join(config.DATA_DIR, 'training_data.csv')
    training_data.to_csv(training_path, index=False)
    print(f"\nTraining data saved to: {training_path}")
    
    # Print sample
    print("\nSample of training data:")
    print(training_data[['event_id', 'forecast_hour'] + config.TARGET_COLUMNS].head(10))
    
    # Generate new events for prediction
    print("\n" + "="*60)
    print("GENERATING SAMPLE NEW EVENTS")
    print("="*60)
    
    new_events = generate_new_events(n_events=20)
    new_events_path = os.path.join(config.DATA_DIR, 'new_events.csv')
    new_events.to_csv(new_events_path, index=False)
    print(f"\nNew events data saved to: {new_events_path}")
    
    print("\n" + "="*60)
    print("Multi-output sample data generation complete!")
    print("You can now run:")
    print("  1. python train_models_multioutput.py - to train models")
    print("  2. python predict_multioutput.py - to make predictions")
    print("="*60)
