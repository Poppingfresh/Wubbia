"""
Evaluate trained models and compare performance across forecast hours
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from data_preparation import load_data, split_data_by_events, prepare_data_for_hour
from predict import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a single model on test data
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }
    
    return metrics, y_pred


def evaluate_all_models(data_path):
    """
    Evaluate all trained models on test data
    
    Args:
        data_path: Path to training data
        
    Returns:
        DataFrame with metrics for all forecast hours
    """
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)
    
    # Load and split data
    df = load_data(data_path)
    train_df, test_df = split_data_by_events(
        df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Evaluate each model
    all_metrics = []
    
    for hour in config.FORECAST_HOURS:
        try:
            # Load model
            model = load_model(hour)
            
            # Prepare test data
            X_test, y_test, _ = prepare_data_for_hour(test_df, hour)
            
            # Evaluate
            metrics, y_pred = evaluate_model(model, X_test, y_test)
            
            metrics['forecast_hour'] = hour
            metrics['n_samples'] = len(y_test)
            
            all_metrics.append(metrics)
            
            print(f"\nHour {hour:02d}:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating hour {hour}: {e}")
            continue
    
    # Create summary DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values('forecast_hour')
    
    return metrics_df


def plot_metrics_by_hour(metrics_df, save_path=None):
    """
    Plot performance metrics across forecast hours
    
    Args:
        metrics_df: DataFrame with metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    hours = metrics_df['forecast_hour']
    
    # RMSE
    axes[0].plot(hours, metrics_df['rmse'], marker='o', linewidth=2)
    axes[0].set_xlabel('Forecast Hour')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Forecast Hour')
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(hours, metrics_df['mae'], marker='o', linewidth=2, color='orange')
    axes[1].set_xlabel('Forecast Hour')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by Forecast Hour')
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].plot(hours, metrics_df['r2'], marker='o', linewidth=2, color='green')
    axes[2].set_xlabel('Forecast Hour')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² by Forecast Hour')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance_by_hour(save_path=None):
    """
    Plot top feature importance for each forecast hour
    
    Args:
        save_path: Optional path to save figure
    """
    # Load feature importance data
    importance_path = os.path.join(config.RESULTS_DIR, 'feature_importance.csv')
    
    if not os.path.exists(importance_path):
        print(f"Feature importance file not found: {importance_path}")
        return
    
    importance_df = pd.read_csv(importance_path)
    
    # Get top 5 features for each hour
    fig, axes = plt.subplots(
        len(config.FORECAST_HOURS), 1, 
        figsize=(10, 3 * len(config.FORECAST_HOURS))
    )
    
    if len(config.FORECAST_HOURS) == 1:
        axes = [axes]
    
    for idx, hour in enumerate(config.FORECAST_HOURS):
        hour_importance = importance_df[
            importance_df['forecast_hour'] == hour
        ].nlargest(5, 'importance')
        
        axes[idx].barh(hour_importance['feature'], hour_importance['importance'])
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'Top 5 Features - {hour:02d}h Forecast')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def generate_prediction_scatter(data_path, forecast_hours=None, save_path=None):
    """
    Generate scatter plots of actual vs predicted values
    
    Args:
        data_path: Path to training data
        forecast_hours: List of hours to plot (default: first 4 from config)
        save_path: Optional path to save figure
    """
    if forecast_hours is None:
        forecast_hours = config.FORECAST_HOURS[:4]
    
    # Load and split data
    df = load_data(data_path)
    train_df, test_df = split_data_by_events(
        df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Create subplots
    n_plots = len(forecast_hours)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, hour in enumerate(forecast_hours):
        try:
            # Load model and data
            model = load_model(hour)
            X_test, y_test, _ = prepare_data_for_hour(test_df, hour)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Plot
            axes[idx].scatter(y_test, y_pred, alpha=0.5)
            
            # Add diagonal line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            axes[idx].set_xlabel('Actual Intensity')
            axes[idx].set_ylabel('Predicted Intensity')
            axes[idx].set_title(f'{hour:02d}h Forecast')
            axes[idx].grid(True, alpha=0.3)
            
            # Add R² to plot
            r2 = r2_score(y_test, y_pred)
            axes[idx].text(
                0.05, 0.95, f'R² = {r2:.3f}',
                transform=axes[idx].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
        except Exception as e:
            print(f"Error plotting hour {hour}: {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Paths
    data_path = os.path.join(config.DATA_DIR, 'training_data.csv')
    
    # Evaluate all models
    metrics_df = evaluate_all_models(data_path)
    
    # Save metrics
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(
        os.path.join(config.RESULTS_DIR, 'evaluation_metrics.csv'),
        index=False
    )
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_metrics_by_hour(
        metrics_df,
        save_path=os.path.join(config.RESULTS_DIR, 'metrics_by_hour.png')
    )
    
    plot_feature_importance_by_hour(
        save_path=os.path.join(config.RESULTS_DIR, 'feature_importance.png')
    )
    
    generate_prediction_scatter(
        data_path,
        save_path=os.path.join(config.RESULTS_DIR, 'prediction_scatter.png')
    )
