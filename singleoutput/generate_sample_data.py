"""
Generate sample data for testing the intensity forecasting pipeline

This script creates synthetic data that matches the expected format.
Use this to test your pipeline before using real data.
"""

import pandas as pd
import numpy as np
import os
import config


def generate_sample_event(event_id, forecast_hours):
    """
    Generate data for a single event
    
    Args:
        event_id: Unique identifier for the event
        forecast_hours: List of forecast hours to generate
        
    Returns:
        DataFrame with event data
    """
    np.random.seed(hash(event_id) % 2**32)
    
    event_data = []
    
    # Generate a base intensity that evolves over time
    base_intensity = np.random.uniform(20, 60)
    intensity_trend = np.random.uniform(-0.5, 1.5)  # Intensity change per hour
    
    for hour in forecast_hours:
        # Evolve intensity over time with some randomness
        intensity = base_intensity + intensity_trend * hour + np.random.normal(0, 3)
        intensity = max(15, min(100, intensity))  # Clip to reasonable range
        
        # Generate meteorological variables
        # These have some correlation with intensity
        row = {
            config.EVENT_ID_COLUMN: event_id,
            config.FORECAST_HOUR_COLUMN: hour,
            config.TARGET_COLUMN: round(intensity, 2)
        }
        
        # Add meteorological features with some relationship to intensity
        row['temperature'] = round(25 + np.random.normal(0, 3) + intensity * 0.05, 2)
        row['pressure'] = round(1013 - intensity * 0.2 + np.random.normal(0, 2), 2)
        row['humidity'] = round(65 + intensity * 0.1 + np.random.normal(0, 5), 2)
        row['wind_speed'] = round(10 + intensity * 0.3 + np.random.normal(0, 2), 2)
        row['wind_direction'] = round(np.random.uniform(0, 360), 2)
        row['precipitation'] = round(max(0, intensity * 0.05 + np.random.normal(0, 1)), 2)
        row['cloud_cover'] = round(min(100, max(0, 40 + intensity * 0.2 + np.random.normal(0, 10))), 2)
        row['sea_surface_temp'] = round(27 + np.random.normal(0, 1) + intensity * 0.02, 2)
        row['vorticity'] = round(intensity * 0.01 + np.random.normal(0, 0.05), 4)
        row['shear'] = round(max(0, 15 - intensity * 0.05 + np.random.normal(0, 2)), 2)
        
        event_data.append(row)
    
    return pd.DataFrame(event_data)


def generate_sample_dataset(n_events=200, forecast_hours=None):
    """
    Generate a complete sample dataset
    
    Args:
        n_events: Number of events to generate
        forecast_hours: List of forecast hours (default: config.FORECAST_HOURS)
        
    Returns:
        DataFrame with all events
    """
    if forecast_hours is None:
        forecast_hours = config.FORECAST_HOURS
    
    print(f"Generating {n_events} sample events...")
    
    all_events = []
    
    for i in range(n_events):
        event_id = f"event_{i+1:03d}"
        event_df = generate_sample_event(event_id, forecast_hours)
        all_events.append(event_df)
    
    df = pd.concat(all_events, ignore_index=True)
    
    print(f"Generated {len(df)} total samples")
    print(f"Events: {df[config.EVENT_ID_COLUMN].nunique()}")
    print(f"Forecast hours: {sorted(df[config.FORECAST_HOUR_COLUMN].unique())}")
    
    return df


def generate_new_events(n_events=20, forecast_hours=None):
    """
    Generate new events for prediction (without intensity column)
    
    Args:
        n_events: Number of new events to generate
        forecast_hours: List of forecast hours
        
    Returns:
        DataFrame with new events (no intensity column)
    """
    if forecast_hours is None:
        forecast_hours = config.FORECAST_HOURS
    
    print(f"Generating {n_events} new events for prediction...")
    
    # Generate events with temporary intensity
    df = generate_sample_dataset(n_events, forecast_hours)
    
    # Remove intensity column for prediction dataset
    df_new = df.drop(columns=[config.TARGET_COLUMN])
    
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
    print("GENERATING SAMPLE TRAINING DATA")
    print("="*60)
    
    training_data = generate_sample_dataset(n_events=200)
    training_path = os.path.join(config.DATA_DIR, 'training_data.csv')
    training_data.to_csv(training_path, index=False)
    print(f"\nTraining data saved to: {training_path}")
    
    # Generate new events for prediction
    print("\n" + "="*60)
    print("GENERATING SAMPLE NEW EVENTS")
    print("="*60)
    
    new_events = generate_new_events(n_events=20)
    new_events_path = os.path.join(config.DATA_DIR, 'new_events.csv')
    new_events.to_csv(new_events_path, index=False)
    print(f"\nNew events data saved to: {new_events_path}")
    
    print("\n" + "="*60)
    print("Sample data generation complete!")
    print("You can now run:")
    print("  1. python train_models.py - to train models")
    print("  2. python evaluate.py - to evaluate models")
    print("  3. python predict.py - to make predictions")
    print("="*60)
