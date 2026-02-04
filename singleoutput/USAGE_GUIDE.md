# Usage Guide: Event Intensity Forecasting

## Overview

This guide walks you through using the XGBoost intensity forecasting system step-by-step.

## Prerequisites

Install required packages:

```bash
pip install -r requirements.txt
```

## Step 1: Prepare Your Data

### Option A: Use Sample Data (for testing)

Generate sample synthetic data to test the pipeline:

```bash
python generate_sample_data.py
```

This creates:
- `data/training_data.csv` - 200 sample events for training
- `data/new_events.csv` - 20 sample events for prediction

### Option B: Use Your Own Data

Format your data as a CSV with these columns:

**Training Data Format:**
```
event_id,forecast_hour,temperature,pressure,humidity,wind_speed,wind_direction,precipitation,cloud_cover,sea_surface_temp,vorticity,shear,intensity
event_001,0,25.3,1013.2,65.0,15.2,180.5,0.5,45.0,27.1,0.015,12.3,30.5
event_001,3,24.8,1012.8,68.0,16.1,185.2,0.8,48.0,27.2,0.018,11.8,32.1
event_001,6,24.2,1012.5,70.0,17.5,190.0,1.2,52.0,27.3,0.021,11.2,35.8
...
```

**Required Columns:**
- `event_id`: Unique identifier for each event
- `forecast_hour`: Time point (0, 3, 6, 9, 12, etc.)
- 10 meteorological feature columns (customize in `config.py`)
- `intensity`: Target variable to predict

**New Events Format (for prediction):**
Same as above but WITHOUT the `intensity` column.

### Update config.py

Edit `config.py` to match your data:

```python
# Update feature column names to match your data
FEATURE_COLUMNS = [
    'your_feature_1',
    'your_feature_2',
    # ... add all your features
]

# Update forecast hours you want to model
FORECAST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21, 24]

# Update column names if different
EVENT_ID_COLUMN = 'event_id'
FORECAST_HOUR_COLUMN = 'forecast_hour'
TARGET_COLUMN = 'intensity'
```

## Step 2: Train Models

Train separate XGBoost models for each forecast hour:

```bash
python train_models.py
```

**What this does:**
- Loads training data from `data/training_data.csv`
- Splits data by events (80/20 train/test split)
- Trains one model per forecast hour
- Saves models to `models/` directory
- Saves training metrics to `results/training_metrics.csv`
- Saves feature importance to `results/feature_importance.csv`

**Output files:**
```
models/
├── model_00h.json
├── model_03h.json
├── model_06h.json
...
```

**Example output:**
```
Training model for 00h forecast
  Train RMSE: 2.3451
  Test RMSE:  2.8923
  Train MAE:  1.7234
  Test MAE:   2.1456
  Train R²:   0.9234
  Test R²:    0.8876
```

## Step 3: Evaluate Models

Assess model performance and generate visualizations:

```bash
python evaluate.py
```

**What this does:**
- Loads all trained models
- Evaluates on test data
- Generates performance plots:
  - RMSE, MAE, R² across forecast hours
  - Feature importance by hour
  - Actual vs Predicted scatter plots
- Saves plots to `results/` directory

**Generated visualizations:**
- `results/metrics_by_hour.png` - Performance trends
- `results/feature_importance.png` - Top features per hour
- `results/prediction_scatter.png` - Prediction accuracy

## Step 4: Make Predictions

Predict intensity for new events:

```bash
python predict.py --input data/new_events.csv --output predictions.csv
```

**Options:**
- `--input`: Path to new events CSV (default: `data/new_events.csv`)
- `--output`: Where to save predictions (default: `predictions.csv`)
- `--format`: Output format `long` or `wide` (default: `long`)

**Output format (long):**
```
event_id,forecast_hour,predicted_intensity
new_event_001,0,28.45
new_event_001,3,30.12
new_event_001,6,33.89
...
```

**Output format (wide):**
```
event_id,00h,03h,06h,09h,12h,...
new_event_001,28.45,30.12,33.89,37.23,41.56,...
new_event_002,25.67,26.89,28.45,30.23,32.11,...
```

## Step 5: Analyze Results

### View Training Metrics

```python
import pandas as pd

metrics = pd.read_csv('results/training_metrics.csv')
print(metrics)
```

### View Feature Importance

```python
importance = pd.read_csv('results/feature_importance.csv')

# Top features for 24h forecast
hour_24 = importance[importance['forecast_hour'] == 24].nlargest(5, 'importance')
print(hour_24)
```

### Compare Predictions

```python
predictions = pd.read_csv('predictions.csv')

# Get predictions for a specific event
event_preds = predictions[predictions['event_id'] == 'new_event_001']
print(event_preds)
```

## Customization

### Tune Hyperparameters

Edit `config.py` to adjust XGBoost parameters:

```python
XGBOOST_PARAMS = {
    'n_estimators': 300,        # Increase for better performance
    'max_depth': 8,             # Increase for more complex patterns
    'learning_rate': 0.03,      # Decrease for more robust learning
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,          # L1 regularization
    'reg_lambda': 1.0,         # L2 regularization
}
```

### Train Models for Specific Hours Only

Edit `config.py`:

```python
FORECAST_HOURS = [0, 6, 12, 18, 24]  # Only these hours
```

### Use Feature Scaling

If your features have very different scales:

```python
# In config.py
SCALE_FEATURES = True
```

## Troubleshooting

### "No data found for forecast hour X"

Check that your data includes all forecast hours specified in `FORECAST_HOURS`.

### "Missing required columns"

Ensure your CSV has all columns listed in `FEATURE_COLUMNS` plus `event_id`, `forecast_hour`, and `intensity`.

### Poor model performance

Try:
1. Increasing `n_estimators` (more trees)
2. Adjusting `learning_rate` (lower = more robust)
3. Tuning `max_depth` (higher = more complex)
4. Adding more training data
5. Engineering new features

### Models not found during prediction

Make sure you've run `train_models.py` first to create the model files.

## Best Practices

1. **Event-based splitting**: Always split by events, not by individual rows, to avoid data leakage
2. **Feature engineering**: Consider derived features like temporal trends
3. **Cross-validation**: For production, implement k-fold CV per hour
4. **Model versioning**: Keep track of model versions and hyperparameters
5. **Regular retraining**: Update models as new event data becomes available

## Example Workflow

Complete example from start to finish:

```bash
# 1. Generate sample data (or prepare your own)
python generate_sample_data.py

# 2. Train models
python train_models.py

# 3. Evaluate performance
python evaluate.py

# 4. Make predictions
python predict.py --input data/new_events.csv --output predictions.csv

# 5. View results
cat predictions.csv
```

## Next Steps

- Implement cross-validation for more robust evaluation
- Add ensemble methods (e.g., stack predictions from multiple hours)
- Explore SHAP values for interpretability
- Add confidence intervals to predictions
- Deploy models as a REST API
