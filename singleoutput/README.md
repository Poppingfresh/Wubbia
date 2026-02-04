# Event Intensity Forecasting with XGBoost

## Project Overview

This project uses XGBoost boosted decision trees to predict event intensity at various forecast hours. The approach trains separate models for each forecast hour (00h, 03h, 06h, 24h, etc.), where each model independently predicts intensity based on meteorological inputs at that specific time.

## Problem Structure

### Data Format
- **Events**: 100s of historical events for training
- **Forecast Hours**: Multiple time points per event (00h, 03h, 06h, ..., 24h, etc.)
- **Features**: ~10 meteorological variables at each forecast hour
- **Target**: Intensity value at each forecast hour

### Model Architecture
- **Approach**: Independent models per forecast hour
- **Model Type**: XGBoost Regressor (regression task)
- **Input**: Meteorological features at hour X
- **Output**: Intensity prediction at hour X

### Key Assumption
Meteorological forecasts are available for future hours when making predictions on new events.

## Installation

```bash
pip install xgboost pandas numpy scikit-learn matplotlib
```

## Project Structure

```
project/
├── README.md
├── data_preparation.py      # Data loading and preprocessing
├── train_models.py          # Model training for all forecast hours
├── predict.py               # Make predictions on new events
├── evaluate.py              # Model evaluation and comparison
├── config.py                # Configuration and parameters
├── models/                  # Saved trained models
│   ├── model_00h.json
│   ├── model_03h.json
│   └── ...
└── data/
    ├── training_data.csv
    └── new_events.csv
```

## Quick Start

### 1. Prepare Your Data

Your training data should be in CSV format with columns:
```
event_id, forecast_hour, met_var_1, met_var_2, ..., met_var_10, intensity
```

Example:
```
event_id,forecast_hour,temperature,pressure,humidity,...,intensity
event_001,00,25.3,1013.2,65.0,...,30.5
event_001,03,24.8,1012.8,68.0,...,32.1
event_001,06,24.2,1012.5,70.0,...,35.8
event_002,00,26.1,1014.0,62.0,...,28.3
...
```

### 2. Train Models

```bash
python train_models.py
```

This will train separate XGBoost models for each forecast hour and save them to the `models/` directory.

### 3. Make Predictions

```bash
python predict.py --input data/new_events.csv --output predictions.csv
```

### 4. Evaluate Performance

```bash
python evaluate.py
```

## Detailed Usage

See the individual Python files for complete implementation details:
- `data_preparation.py` - How to structure and preprocess your data
- `train_models.py` - Training pipeline with cross-validation
- `predict.py` - Inference on new events
- `evaluate.py` - Performance metrics across forecast hours

## Key XGBoost Parameters

Since this is a regression problem predicting intensity values, important parameters include:

```python
{
    'n_estimators': 200,           # Number of trees
    'max_depth': 6,                # Maximum tree depth
    'learning_rate': 0.05,         # Step size (lower = more robust)
    'subsample': 0.8,              # Fraction of samples per tree
    'colsample_bytree': 0.8,       # Fraction of features per tree
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 1.0,             # L2 regularization
    'objective': 'reg:squarederror', # Regression objective
    'eval_metric': 'rmse'          # Root mean squared error
}
```

## Expected Workflow

1. **Data Collection**: Gather historical event data with meteorological variables and intensity measurements at multiple forecast hours
2. **Data Preparation**: Structure data so each row represents one event at one forecast hour
3. **Model Training**: Train one XGBoost model per forecast hour using historical data
4. **Model Evaluation**: Assess performance using RMSE, MAE, and R² for each forecast hour
5. **Prediction**: For new events, use meteorological forecasts at each hour to predict intensity
6. **Analysis**: Compare model performance across forecast hours and identify important features

## Performance Monitoring

Track these metrics for each forecast hour:
- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute deviation
- **R²** (Coefficient of Determination): Proportion of variance explained
- **Feature Importance**: Which meteorological variables matter most

## Tips for This Use Case

1. **Forecast Hour Comparison**: Some forecast hours may be easier to predict than others (e.g., 00h vs 24h)
2. **Feature Engineering**: Consider derived features like trends or differences between variables
3. **Data Quality**: Ensure meteorological data quality is consistent across all forecast hours
4. **Model Validation**: Use event-based splitting (not random) to avoid data leakage
5. **Ensemble Consideration**: You could ensemble predictions across similar forecast hours if needed

## Next Steps

- Implement the code files based on your specific data format
- Tune hyperparameters for each forecast hour independently
- Analyze feature importance to understand what drives intensity predictions
- Consider adding temporal features if they improve performance
- Evaluate prediction uncertainty/confidence intervals

## Resources

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
