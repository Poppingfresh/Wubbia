"""
Train XGBoost models for multi-output forecasting
Supports both single and multi-output targets (e.g., intensity + precipitation)
"""

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
import os
import pickle
import config
from data_preparation_multioutput import (
    load_data, 
    validate_data, 
    split_data_by_events, 
    prepare_data_for_hour
)


def train_model_for_hour(X_train, y_train, X_test, y_test, forecast_hour):
    """
    Train an XGBoost model for a specific forecast hour
    Handles both single-output and multi-output cases
    
    Args:
        X_train: Training features
        y_train: Training targets (1D for single output, 2D for multi-output)
        X_test: Test features
        y_test: Test targets
        forecast_hour: Hour being modeled
        
    Returns:
        model: Trained XGBoost model
        metrics: Dictionary of performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model for {forecast_hour:02d}h forecast")
    print(f"{'='*60}")
    
    # Determine if multi-output
    is_multioutput = len(y_train.shape) > 1 and y_train.shape[1] > 1
    
    if is_multioutput:
        print(f"Multi-output mode: predicting {y_train.shape[1]} targets")
        print(f"Targets: {config.TARGET_COLUMNS}")
        
        # Use MultiOutputRegressor wrapper for multi-output
        base_model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        model = MultiOutputRegressor(base_model)
        
        # Train without eval_set (MultiOutputRegressor doesn't support it)
        model.fit(X_train, y_train)
        
    else:
        print("Single-output mode")
        # Standard single-output XGBoost
        model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        
        # Train with evaluation set for monitoring
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    if is_multioutput:
        metrics = calculate_multioutput_metrics(
            y_train, y_train_pred, y_test, y_test_pred, forecast_hour
        )
    else:
        metrics = calculate_singleoutput_metrics(
            y_train, y_train_pred, y_test, y_test_pred, forecast_hour
        )
    
    # Print results
    print(f"\nTraining Results:")
    print_metrics(metrics, is_multioutput)
    
    return model, metrics


def calculate_singleoutput_metrics(y_train, y_train_pred, y_test, y_test_pred, forecast_hour):
    """Calculate metrics for single-output models"""
    metrics = {
        'forecast_hour': forecast_hour,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test)
    }
    return metrics


def calculate_multioutput_metrics(y_train, y_train_pred, y_test, y_test_pred, forecast_hour):
    """Calculate metrics for multi-output models"""
    n_targets = y_train.shape[1]
    
    metrics = {
        'forecast_hour': forecast_hour,
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'n_targets': n_targets
    }
    
    # Calculate metrics for each output
    for i, target_name in enumerate(config.TARGET_COLUMNS):
        metrics[f'{target_name}_train_rmse'] = np.sqrt(
            mean_squared_error(y_train[:, i], y_train_pred[:, i])
        )
        metrics[f'{target_name}_test_rmse'] = np.sqrt(
            mean_squared_error(y_test[:, i], y_test_pred[:, i])
        )
        metrics[f'{target_name}_train_mae'] = mean_absolute_error(
            y_train[:, i], y_train_pred[:, i]
        )
        metrics[f'{target_name}_test_mae'] = mean_absolute_error(
            y_test[:, i], y_test_pred[:, i]
        )
        metrics[f'{target_name}_train_r2'] = r2_score(
            y_train[:, i], y_train_pred[:, i]
        )
        metrics[f'{target_name}_test_r2'] = r2_score(
            y_test[:, i], y_test_pred[:, i]
        )
    
    # Calculate overall metrics (average across outputs)
    metrics['overall_train_rmse'] = np.mean([
        metrics[f'{name}_train_rmse'] for name in config.TARGET_COLUMNS
    ])
    metrics['overall_test_rmse'] = np.mean([
        metrics[f'{name}_test_rmse'] for name in config.TARGET_COLUMNS
    ])
    metrics['overall_train_r2'] = np.mean([
        metrics[f'{name}_train_r2'] for name in config.TARGET_COLUMNS
    ])
    metrics['overall_test_r2'] = np.mean([
        metrics[f'{name}_test_r2'] for name in config.TARGET_COLUMNS
    ])
    
    return metrics


def print_metrics(metrics, is_multioutput):
    """Print metrics in a readable format"""
    if is_multioutput:
        for target_name in config.TARGET_COLUMNS:
            print(f"\n  {target_name.upper()}:")
            print(f"    Train RMSE: {metrics[f'{target_name}_train_rmse']:.4f}")
            print(f"    Test RMSE:  {metrics[f'{target_name}_test_rmse']:.4f}")
            print(f"    Train MAE:  {metrics[f'{target_name}_train_mae']:.4f}")
            print(f"    Test MAE:   {metrics[f'{target_name}_test_mae']:.4f}")
            print(f"    Train R²:   {metrics[f'{target_name}_train_r2']:.4f}")
            print(f"    Test R²:    {metrics[f'{target_name}_test_r2']:.4f}")
        
        print(f"\n  OVERALL (averaged):")
        print(f"    Train RMSE: {metrics['overall_train_rmse']:.4f}")
        print(f"    Test RMSE:  {metrics['overall_test_rmse']:.4f}")
        print(f"    Train R²:   {metrics['overall_train_r2']:.4f}")
        print(f"    Test R²:    {metrics['overall_test_r2']:.4f}")
    else:
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"  Test RMSE:  {metrics['test_rmse']:.4f}")
        print(f"  Train MAE:  {metrics['train_mae']:.4f}")
        print(f"  Test MAE:   {metrics['test_mae']:.4f}")
        print(f"  Train R²:   {metrics['train_r2']:.4f}")
        print(f"  Test R²:    {metrics['test_r2']:.4f}")


def save_model(model, forecast_hour, model_dir=None):
    """
    Save trained model to disk
    
    Args:
        model: Trained model (XGBoost or MultiOutputRegressor)
        forecast_hour: Hour the model predicts
        model_dir: Directory to save model (default: config.MODEL_DIR)
    """
    if model_dir is None:
        model_dir = config.MODEL_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # For MultiOutputRegressor, we need to use pickle
    # For single XGBoost, we can use native format
    if isinstance(model, MultiOutputRegressor):
        model_path = os.path.join(model_dir, f'model_{forecast_hour:02d}h.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_path = os.path.join(model_dir, f'model_{forecast_hour:02d}h.json')
        model.save_model(model_path)
    
    print(f"Model saved to {model_path}")


def get_feature_importance(model, forecast_hour):
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained model
        forecast_hour: Hour the model predicts
        
    Returns:
        DataFrame with feature names and importance scores
    """
    if isinstance(model, MultiOutputRegressor):
        # For multi-output, average importance across all estimators
        importances = []
        for estimator in model.estimators_:
            importances.append(estimator.feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': config.FEATURE_COLUMNS,
            'importance': avg_importance,
            'forecast_hour': forecast_hour
        }).sort_values('importance', ascending=False)
        
    else:
        importance_scores = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': config.FEATURE_COLUMNS,
            'importance': importance_scores,
            'forecast_hour': forecast_hour
        }).sort_values('importance', ascending=False)
    
    return importance_df


def train_all_models(data_path):
    """
    Train models for all forecast hours
    
    Args:
        data_path: Path to training data CSV
        
    Returns:
        models: Dictionary of trained models by forecast hour
        all_metrics: DataFrame with metrics for all models
        all_importance: DataFrame with feature importance for all models
    """
    print("\n" + "="*60)
    print("TRAINING MULTI-OUTPUT MODELS FOR ALL FORECAST HOURS")
    print("="*60)
    print(f"Predicting: {', '.join(config.TARGET_COLUMNS)}")
    
    # Load and split data
    df = load_data(data_path)
    
    if not validate_data(df):
        raise ValueError("Data validation failed")
    
    train_df, test_df = split_data_by_events(
        df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Train models for each forecast hour
    models = {}
    all_metrics = []
    all_importance = []
    
    for hour in config.FORECAST_HOURS:
        try:
            # Prepare data for this hour
            X_train, y_train, _ = prepare_data_for_hour(train_df, hour)
            X_test, y_test, _ = prepare_data_for_hour(test_df, hour)
            
            # Train model
            model, metrics = train_model_for_hour(
                X_train, y_train, X_test, y_test, hour
            )
            
            # Save model
            save_model(model, hour)
            
            # Store results
            models[hour] = model
            all_metrics.append(metrics)
            
            # Get feature importance
            importance_df = get_feature_importance(model, hour)
            all_importance.append(importance_df)
            
        except Exception as e:
            print(f"Error training model for hour {hour}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary DataFrames
    metrics_df = pd.DataFrame(all_metrics)
    importance_df = pd.concat(all_importance, ignore_index=True)
    
    # Print overall summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    
    return models, metrics_df, importance_df


if __name__ == "__main__":
    # Path to training data
    data_path = os.path.join(config.DATA_DIR, 'training_data.csv')
    
    # Train all models
    models, metrics_df, importance_df = train_all_models(data_path)
    
    # Save metrics
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(
        os.path.join(config.RESULTS_DIR, 'training_metrics.csv'), 
        index=False
    )
    importance_df.to_csv(
        os.path.join(config.RESULTS_DIR, 'feature_importance.csv'), 
        index=False
    )
    
    print(f"\nMetrics saved to {config.RESULTS_DIR}/training_metrics.csv")
    print(f"Feature importance saved to {config.RESULTS_DIR}/feature_importance.csv")
