"""
Make predictions on new events using trained models
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import os
import argparse
import config


def load_model(forecast_hour, model_dir=None):
    """
    Load a trained model for a specific forecast hour
    
    Args:
        forecast_hour: Hour to load model for
        model_dir: Directory where models are stored
        
    Returns:
        Loaded XGBoost model
    """
    if model_dir is None:
        model_dir = config.MODEL_DIR
    
    model_path = os.path.join(model_dir, f'model_{forecast_hour:02d}h.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    return model


def load_new_event_data(filepath):
    """
    Load data for new events to predict
    
    Expected format: Same as training data but without intensity column
    Columns: event_id, forecast_hour, met_var_1, met_var_2, ...
    
    Args:
        filepath: Path to CSV with new event data
        
    Returns:
        DataFrame with event data
    """
    print(f"Loading new event data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Loaded {df[config.EVENT_ID_COLUMN].nunique()} events")
    print(f"Forecast hours: {sorted(df[config.FORECAST_HOUR_COLUMN].unique())}")
    
    # Validate required columns
    required_cols = [config.EVENT_ID_COLUMN, config.FORECAST_HOUR_COLUMN] + config.FEATURE_COLUMNS
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def predict_for_event(event_df, models):
    """
    Make predictions for a single event across all forecast hours
    
    Args:
        event_df: DataFrame with data for one event (multiple hours)
        models: Dictionary of trained models by forecast hour
        
    Returns:
        DataFrame with predictions for each hour
    """
    predictions = []
    
    for hour in sorted(event_df[config.FORECAST_HOUR_COLUMN].unique()):
        # Get data for this hour
        hour_data = event_df[event_df[config.FORECAST_HOUR_COLUMN] == hour]
        
        if hour not in models:
            print(f"Warning: No model available for hour {hour}, skipping")
            continue
        
        # Extract features
        X = hour_data[config.FEATURE_COLUMNS].values
        
        # Make prediction
        model = models[hour]
        intensity_pred = model.predict(X)[0]
        
        predictions.append({
            'event_id': hour_data[config.EVENT_ID_COLUMN].iloc[0],
            'forecast_hour': hour,
            'predicted_intensity': intensity_pred
        })
    
    return pd.DataFrame(predictions)


def predict_all_events(data_path, model_dir=None):
    """
    Make predictions for all new events
    
    Args:
        data_path: Path to CSV with new event data
        model_dir: Directory where models are stored
        
    Returns:
        DataFrame with predictions for all events and hours
    """
    print("\n" + "="*60)
    print("MAKING PREDICTIONS FOR NEW EVENTS")
    print("="*60)
    
    # Load new event data
    df = load_new_event_data(data_path)
    
    # Load all trained models
    print(f"\nLoading models from {model_dir or config.MODEL_DIR}...")
    models = {}
    for hour in config.FORECAST_HOURS:
        try:
            models[hour] = load_model(hour, model_dir)
            print(f"  Loaded model for {hour:02d}h")
        except FileNotFoundError:
            print(f"  Warning: Model for {hour:02d}h not found")
    
    if not models:
        raise ValueError("No models loaded. Please train models first.")
    
    # Make predictions for each event
    all_predictions = []
    unique_events = df[config.EVENT_ID_COLUMN].unique()
    
    print(f"\nMaking predictions for {len(unique_events)} events...")
    
    for event_id in unique_events:
        event_df = df[df[config.EVENT_ID_COLUMN] == event_id]
        event_predictions = predict_for_event(event_df, models)
        all_predictions.append(event_predictions)
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\nCompleted {len(predictions_df)} predictions")
    
    return predictions_df


def format_predictions_wide(predictions_df):
    """
    Format predictions in wide format (one row per event)
    
    Args:
        predictions_df: DataFrame with predictions in long format
        
    Returns:
        DataFrame with predictions in wide format
    """
    wide_df = predictions_df.pivot(
        index='event_id',
        columns='forecast_hour',
        values='predicted_intensity'
    )
    
    # Rename columns to include 'h' suffix
    wide_df.columns = [f'{int(col):02d}h' for col in wide_df.columns]
    wide_df = wide_df.reset_index()
    
    return wide_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions on new events')
    parser.add_argument(
        '--input', 
        type=str, 
        default=os.path.join(config.DATA_DIR, 'new_events.csv'),
        help='Path to input CSV with new event data'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='predictions.csv',
        help='Path to save predictions CSV'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['long', 'wide'],
        default='long',
        help='Output format: long (one row per hour) or wide (one row per event)'
    )
    
    args = parser.parse_args()
    
    # Make predictions
    predictions_df = predict_all_events(args.input)
    
    # Format output
    if args.format == 'wide':
        predictions_df = format_predictions_wide(predictions_df)
    
    # Save predictions
    predictions_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    
    # Display sample
    print(f"\nSample predictions:")
    print(predictions_df.head(10))
