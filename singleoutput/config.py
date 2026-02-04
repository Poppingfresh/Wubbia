"""
Configuration file for XGBoost intensity forecasting models
"""

# Forecast hours to train models for
FORECAST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21, 24]

# Feature column names (meteorological variables)
# Replace these with your actual column names
FEATURE_COLUMNS = [
    'temperature',
    'pressure',
    'humidity',
    'wind_speed',
    'wind_direction',
    'precipitation',
    'cloud_cover',
    'sea_surface_temp',
    'vorticity',
    'shear'
]

# Target column
TARGET_COLUMN = 'intensity'

# Data columns
EVENT_ID_COLUMN = 'event_id'
FORECAST_HOUR_COLUMN = 'forecast_hour'

# Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 3,
    'gamma': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1
}

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature scaling (set to True if you want to scale features)
SCALE_FEATURES = False
