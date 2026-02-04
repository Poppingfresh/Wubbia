# Multi-Output Forecasting Guide

## Overview

This guide explains how to use the multi-output version of the intensity forecasting system, which can predict **multiple dependent targets simultaneously** (e.g., intensity AND precipitation).

## Why Multi-Output?

When your targets are **dependent on each other**, training multi-output models has several advantages:

1. **Captures dependencies**: The model learns the relationship between intensity and precipitation
2. **Shared learning**: Both outputs benefit from the same underlying patterns in the data
3. **Consistency**: Predictions are internally consistent (e.g., high intensity → high precipitation)
4. **Efficiency**: One model predicts both outputs instead of training separate models

## How It Works

### Architecture

Instead of training separate models for each target, we use **MultiOutputRegressor**:

```
Input Features (10 meteorological variables)
              ↓
     XGBoost Multi-Output Model
              ↓
   ┌──────────┴──────────┐
   ↓                     ↓
Intensity          Precipitation
```

For each forecast hour (00h, 03h, 06h, etc.):
- **One model** predicts **both** intensity and precipitation
- The model learns joint patterns between the two outputs
- Predictions maintain the statistical relationship between targets

### Technical Implementation

We use scikit-learn's `MultiOutputRegressor` wrapper:

```python
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# Base XGBoost model
base_model = xgb.XGBRegressor(**params)

# Wrap for multi-output
model = MultiOutputRegressor(base_model)

# Train on multiple targets
model.fit(X, y)  # y has shape (n_samples, n_targets)
```

## Setup Instructions

### 1. Update Configuration

Edit `config.py` to specify multiple targets:

```python
# Change from single output:
# TARGET_COLUMNS = ['intensity']

# To multi-output:
TARGET_COLUMNS = ['intensity', 'precipitation_target']
```

**Important**: 
- Use different column names for input features vs. target precipitation
- Example: `'precipitation'` as input feature, `'precipitation_target'` as output

### 2. Prepare Your Data

Your training CSV should have:

```csv
event_id,forecast_hour,temperature,pressure,...,intensity,precipitation_target
event_001,0,25.3,1013.2,...,30.5,12.3
event_001,3,24.8,1012.8,...,32.1,14.8
event_001,6,24.2,1012.5,...,35.8,18.2
```

**Required columns:**
- Event identifier: `event_id`
- Time: `forecast_hour`
- Input features: All meteorological variables (defined in `FEATURE_COLUMNS`)
- **Multiple targets**: `intensity`, `precipitation_target`, etc.

### 3. Generate Sample Data

Test the system with synthetic multi-output data:

```bash
python generate_sample_data_multioutput.py
```

This creates correlated intensity and precipitation data.

### 4. Train Multi-Output Models

```bash
python train_models_multioutput.py
```

**What this does:**
- Loads data with multiple target columns
- Trains one multi-output model per forecast hour
- Each model predicts ALL targets simultaneously
- Saves models as `.pkl` files (required for multi-output)

**Example output:**
```
Training model for 00h forecast
Multi-output mode: predicting 2 targets
Targets: ['intensity', 'precipitation_target']

  INTENSITY:
    Train RMSE: 2.3451
    Test RMSE:  2.8923
    Train R²:   0.9234
    Test R²:    0.8876

  PRECIPITATION_TARGET:
    Train RMSE: 1.5678
    Test RMSE:  1.9234
    Train R²:   0.8765
    Test R²:    0.8321

  OVERALL (averaged):
    Train RMSE: 1.9565
    Test RMSE:  2.4079
    Train R²:   0.9000
    Test R²:    0.8599
```

### 5. Make Predictions

```bash
python predict_multioutput.py --input data/new_events.csv --output predictions.csv
```

**Output format (long):**
```csv
event_id,forecast_hour,predicted_intensity,predicted_precipitation_target
new_event_001,0,28.45,11.23
new_event_001,3,30.12,13.67
new_event_001,6,33.89,16.89
```

**Output format (wide):**
```bash
python predict_multioutput.py --format wide
```

```csv
event_id,intensity_00h,intensity_03h,...,precipitation_target_00h,precipitation_target_03h,...
new_event_001,28.45,30.12,...,11.23,13.67,...
```

## Understanding the Results

### Metrics by Target

Each target gets separate metrics:

```python
import pandas as pd

metrics = pd.read_csv('results/training_metrics.csv')

# View intensity metrics
print(metrics[['forecast_hour', 'intensity_test_rmse', 'intensity_test_r2']])

# View precipitation metrics
print(metrics[['forecast_hour', 'precipitation_target_test_rmse', 'precipitation_target_test_r2']])
```

### Correlation Analysis

Check if the model maintains the dependency:

```python
import pandas as pd

# Load predictions
preds = pd.read_csv('predictions.csv')

# Check correlation between predicted outputs
corr = preds[['predicted_intensity', 'predicted_precipitation_target']].corr()
print(f"Correlation between predictions: {corr.iloc[0,1]:.3f}")
```

A high correlation (e.g., 0.7+) indicates the model learned the dependency.

## Comparison: Multi-Output vs. Independent Models

### Multi-Output (Recommended for Dependent Targets)

**Pros:**
- ✅ Learns dependencies between targets
- ✅ Predictions are consistent with each other
- ✅ More efficient (one model per hour instead of N models per hour)
- ✅ Better generalization when targets are correlated

**Cons:**
- ❌ Slightly more complex to implement
- ❌ Must use pickle for saving (not native XGBoost format)
- ❌ Can't tune hyperparameters separately per target

### Independent Models

**Pros:**
- ✅ Simpler implementation
- ✅ Can tune each target separately
- ✅ Each target optimized independently

**Cons:**
- ❌ Ignores dependencies
- ❌ Predictions may be inconsistent
- ❌ Less efficient (2x the models)

## Advanced Topics

### Different Hyperparameters per Target

If you want different parameters for different targets, you can subclass `MultiOutputRegressor`:

```python
class CustomMultiOutput(MultiOutputRegressor):
    def __init__(self, estimators):
        self.estimators = estimators
    
    def fit(self, X, y):
        for i, estimator in enumerate(self.estimators):
            estimator.fit(X, y[:, i])
        return self

# Use different params for each target
intensity_model = xgb.XGBRegressor(max_depth=6, learning_rate=0.05)
precip_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1)

model = CustomMultiOutput([intensity_model, precip_model])
```

### Adding More Targets

Simply add to `config.py`:

```python
TARGET_COLUMNS = ['intensity', 'precipitation_target', 'wind_speed_target']
```

The system automatically handles any number of targets.

### Weighted Loss

To prioritize certain targets:

```python
from sklearn.multioutput import MultiOutputRegressor

# Weight intensity 2x more than precipitation
sample_weights = np.column_stack([
    np.ones(len(y)) * 2,  # intensity weight
    np.ones(len(y)) * 1   # precipitation weight
])

model.fit(X, y, sample_weight=sample_weights)
```

## Troubleshooting

### "Cannot stack 1D arrays"

Make sure your targets are in a 2D array:
```python
# Wrong: y = df['intensity'].values  # 1D
# Right: y = df[['intensity', 'precipitation_target']].values  # 2D
```

### Predictions not correlated

- Check that your training data shows correlation between targets
- Increase model complexity (more trees, deeper trees)
- Add interaction features between meteorological variables

### Poor performance on one target

- Consider independent models for that target
- Adjust `reg_alpha` and `reg_lambda` per target
- Use weighted loss to prioritize problematic target

## Example Workflow

Complete multi-output workflow:

```bash
# 1. Update config
# Edit config.py: TARGET_COLUMNS = ['intensity', 'precipitation_target']

# 2. Generate sample data
python generate_sample_data_multioutput.py

# 3. Train models
python train_models_multioutput.py

# 4. Make predictions
python predict_multioutput.py

# 5. Analyze results
python
>>> import pandas as pd
>>> preds = pd.read_csv('predictions.csv')
>>> preds.head()
>>> preds[['predicted_intensity', 'predicted_precipitation_target']].corr()
```

## Files for Multi-Output

Use these files for multi-output forecasting:

- `config.py` - Set `TARGET_COLUMNS` to multiple targets
- `data_preparation_multioutput.py` - Handles multi-output data
- `train_models_multioutput.py` - Trains multi-output models
- `predict_multioutput.py` - Makes multi-output predictions
- `generate_sample_data_multioutput.py` - Creates test data

## Key Takeaways

1. **Dependencies matter**: Use multi-output when targets are related
2. **One model per hour**: Each hour gets one model that predicts all targets
3. **Consistent predictions**: Outputs maintain their statistical relationship
4. **Easy to extend**: Add more targets by updating `TARGET_COLUMNS`
5. **Better efficiency**: Fewer models to train and maintain
