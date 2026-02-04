# XGBoost Hyperparameter Tuning Guide (Native API)

## Overview

This guide explains every XGBoost parameter using **native XGBoost parameter names** (not scikit-learn wrapper names), what each does, and how to tune it for your intensity forecasting problem.

**Your situation**: Training MAE is much lower than test MAE (3x difference) → **Classic overfitting**

The model is learning the training data too well and not generalizing to new events.

---

## Parameter Name Differences: Native vs Scikit-Learn

XGBoost has two APIs with different parameter names:

| Scikit-Learn Wrapper | Native XGBoost | Description |
|---------------------|----------------|-------------|
| `n_estimators` | `num_boost_round` | Number of trees |
| `learning_rate` | `eta` | Learning rate |
| `max_depth` | `max_depth` | Same in both |
| `n_jobs` | `nthread` | Number of threads |
| `random_state` | `seed` | Random seed |

**This guide uses NATIVE parameter names** for use with `xgb.train()` or when passing params directly.

---

## Understanding Your Problem: Overfitting

### Signs of Overfitting:
- ✅ Training MAE/RMSE much lower than test MAE/RMSE (you have this!)
- Training R² is high (>0.95) but test R² is lower
- Model performs well on training events but poorly on new events

### Root Causes:
1. **Model too complex** for the amount of data
2. **Not enough regularization**
3. **Too many trees** without early stopping
4. **Noisy features** being overfit

### Solutions:
Focus on parameters that **increase bias** and **reduce variance**:
- Reduce tree depth
- Increase regularization
- Use early stopping
- Subsample data
- Increase minimum samples per leaf

---

## Core Parameters (Start Here!)

### 1. `num_boost_round` (Number of Trees)

**Native XGBoost name**: `num_boost_round` (passed to `xgb.train()`, not in params dict)  
**Scikit-learn equivalent**: `n_estimators`

**What it does**: Total number of boosting rounds (trees to build)

**Default**: 10 (native), 100 (scikit-learn)

**Impact**:
- More trees → Better training performance
- Too many trees → Overfitting (especially without early stopping)
- Too few trees → Underfitting

**Tuning strategy**:
```python
# Start conservative
num_boost_round = 100

# If underfitting, increase
num_boost_round = 200-500

# If overfitting, decrease
num_boost_round = 50-100
```

**For your overfitting problem**:
- Try **reducing** to 100-150
- Use early stopping (see below)

**Best practice**: Set high (500+) and use early stopping

**Usage**:
```python
# Native API
model = xgb.train(params, dtrain, num_boost_round=100)

# Scikit-learn API
model = xgb.XGBRegressor(n_estimators=100)
```

---

### 2. `max_depth` (Tree Depth)

**Native XGBoost name**: `max_depth`  
**Scikit-learn equivalent**: `max_depth` (same)

**What it does**: Maximum depth of each tree (how many splits)

**Default**: 6

**Impact**:
- Deep trees → Capture complex interactions, but overfit easily
- Shallow trees → More robust, better generalization
- Most important parameter for controlling overfitting!

**Tuning strategy**:
```python
# For small datasets (<1000 samples)
params = {'max_depth': 3}  # or 4

# For medium datasets (1000-10000 samples)
params = {'max_depth': 4}  # to 6

# For large datasets (>10000 samples)
params = {'max_depth': 6}  # to 10

# When overfitting
params = {'max_depth': 3}  # Start here!
```

**For your overfitting problem**:
- **Reduce to 3-4** (this is your primary lever!)
- Deeper trees memorize training data patterns

**Example impact**:
```
max_depth=3:  Train MAE=2.5, Test MAE=2.8  ✓ Good generalization
max_depth=6:  Train MAE=1.2, Test MAE=3.6  ✗ Overfitting
max_depth=10: Train MAE=0.5, Test MAE=5.2  ✗ Severe overfitting
```

---

### 3. `eta` (Learning Rate)

**Native XGBoost name**: `eta`  
**Scikit-learn equivalent**: `learning_rate`

**What it does**: Step size for each tree's contribution (shrinkage factor)

**Default**: 0.3

**Impact**:
- Lower rate → More robust, better generalization, needs more trees
- Higher rate → Faster training, more aggressive updates, can overfit

**Tuning strategy**:
```python
# Conservative (recommended)
params = {'eta': 0.01}  # to 0.05

# Moderate
params = {'eta': 0.05}  # to 0.1

# Aggressive (faster but risky)
params = {'eta': 0.1}  # to 0.3
```

**For your overfitting problem**:
- **Reduce to 0.01-0.03**
- Lower eta = smoother learning = better generalization
- Increase `num_boost_round` to compensate

**Rule of thumb**: `eta × num_boost_round ≈ constant`
```python
# These are roughly equivalent:
Option 1: eta=0.1,  num_boost_round=100
Option 2: eta=0.05, num_boost_round=200
Option 3: eta=0.01, num_boost_round=1000  # Best for generalization
```

---

### 4. `subsample` (Row Sampling)

**Native XGBoost name**: `subsample`  
**Scikit-learn equivalent**: `subsample` (same)

**What it does**: Fraction of training samples to use for each tree

**Default**: 1.0 (use all samples)

**Impact**:
- <1.0 → Random sampling, reduces overfitting, adds randomness
- More robust to outliers
- Too low → Underfitting

**Tuning strategy**:
```python
# No sampling (default)
params = {'subsample': 1.0}

# Moderate sampling (recommended for overfitting)
params = {'subsample': 0.7}  # to 0.8

# Aggressive sampling
params = {'subsample': 0.5}  # to 0.7
```

**For your overfitting problem**:
- **Set to 0.7-0.8**
- Each tree sees only 70-80% of data → better generalization

---

### 5. `colsample_bytree` (Feature Sampling per Tree)

**Native XGBoost name**: `colsample_bytree`  
**Scikit-learn equivalent**: `colsample_bytree` (same)

**What it does**: Fraction of features to use for each tree

**Default**: 1.0 (use all features)

**Impact**:
- <1.0 → Random feature selection, reduces overfitting
- Helps when some features are noisy
- Makes trees more diverse

**Tuning strategy**:
```python
# No sampling (default)
params = {'colsample_bytree': 1.0}

# Moderate sampling (recommended for overfitting)
params = {'colsample_bytree': 0.7}  # to 0.8

# Aggressive sampling
params = {'colsample_bytree': 0.5}  # to 0.7
```

**For your overfitting problem**:
- **Set to 0.7-0.8**
- Especially helpful if you have 10+ features

---

## Regularization Parameters (Critical for Overfitting!)

### 6. `alpha` (L1 Regularization)

**Native XGBoost name**: `alpha`  
**Scikit-learn equivalent**: `reg_alpha`

**What it does**: L1 penalty on leaf weights (encourages sparsity)

**Default**: 0

**Impact**:
- Higher values → Simpler model, feature selection
- Pushes some leaf weights to zero
- Good when many features are irrelevant

**Tuning strategy**:
```python
# No regularization
params = {'alpha': 0}

# Light regularization
params = {'alpha': 0.1}  # to 1.0

# Strong regularization (for overfitting)
params = {'alpha': 1.0}  # to 10.0
```

**For your overfitting problem**:
- **Start with 0.5-1.0**
- Increase if still overfitting

---

### 7. `lambda` (L2 Regularization)

**Native XGBoost name**: `lambda`  
**Scikit-learn equivalent**: `reg_lambda`

**What it does**: L2 penalty on leaf weights (smooth weights)

**Default**: 1.0

**Impact**:
- Higher values → Smoother predictions, less extreme weights
- More conservative than L1
- Default=1.0 provides some regularization

**Tuning strategy**:
```python
# Light regularization
params = {'lambda': 1.0}  # Default, often sufficient

# Moderate regularization (for overfitting)
params = {'lambda': 2.0}  # to 5.0

# Strong regularization
params = {'lambda': 5.0}  # to 10.0
```

**For your overfitting problem**:
- **Increase to 2.0-5.0**
- Works well combined with `alpha`

---

### 8. `min_child_weight` (Minimum Sum of Instance Weight)

**Native XGBoost name**: `min_child_weight`  
**Scikit-learn equivalent**: `min_child_weight` (same)

**What it does**: Minimum sum of instance weight needed in a child node

**Default**: 1

**Impact**:
- Higher values → More conservative splits, prevents overfitting
- Stops splitting when samples are too few
- Similar to "min_samples_leaf" in other algorithms

**Tuning strategy**:
```python
# Permissive (default)
params = {'min_child_weight': 1}

# Conservative (recommended for overfitting)
params = {'min_child_weight': 3}  # to 5

# Very conservative
params = {'min_child_weight': 5}  # to 10
```

**For your overfitting problem**:
- **Increase to 3-5**
- Prevents fitting on small sample groups

---

### 9. `gamma` (Minimum Loss Reduction)

**Native XGBoost name**: `gamma`  
**Scikit-learn equivalent**: `gamma` (same)

**What it does**: Minimum loss reduction required to make a split

**Default**: 0

**Impact**:
- Higher values → Fewer splits, simpler trees
- Acts as pruning mechanism
- Makes model more conservative

**Tuning strategy**:
```python
# No constraint (default)
params = {'gamma': 0}

# Light constraint
params = {'gamma': 0.1}  # to 1.0

# Strong constraint (for overfitting)
params = {'gamma': 1.0}  # to 5.0
```

**For your overfitting problem**:
- **Start with 0.1-0.5**
- Increase if still overfitting

---

## Advanced Parameters

### 10. `colsample_bylevel` (Per-Level Feature Sampling)

**Native XGBoost name**: `colsample_bylevel`  
**Scikit-learn equivalent**: `colsample_bylevel` (same)

**What it does**: Fraction of features to sample at each tree level

**Default**: 1.0

**Impact**:
- Similar to `colsample_bytree` but per level
- Additional randomization

**When to use**:
- When `colsample_bytree` alone isn't enough
- For very deep trees

```python
params = {'colsample_bylevel': 0.7}  # to 0.9
```

---

### 11. `colsample_bynode` (Per-Node Feature Sampling)

**Native XGBoost name**: `colsample_bynode`  
**Scikit-learn equivalent**: `colsample_bynode` (same)

**What it does**: Fraction of features to sample at each node

**Default**: 1.0

**Impact**:
- Most aggressive feature sampling
- Maximum randomization

**When to use**:
- Extreme overfitting cases
- Many noisy features

```python
params = {'colsample_bynode': 0.7}  # to 0.9
```

---

### 12. `max_delta_step` (Maximum Delta Step)

**Native XGBoost name**: `max_delta_step`  
**Scikit-learn equivalent**: `max_delta_step` (same)

**What it does**: Maximum delta step for each tree's output

**Default**: 0 (no constraint)

**Impact**:
- Constrains leaf values
- Useful for imbalanced data or extreme values

**When to use**:
- When predictions are unstable
- For classification with imbalanced classes

```python
params = {'max_delta_step': 1}  # to 5
```

---

### 13. `tree_method` (Tree Construction Algorithm)

**Native XGBoost name**: `tree_method`  
**Scikit-learn equivalent**: `tree_method` (same)

**Options**:
- `'auto'`: Default, lets XGBoost decide
- `'exact'`: Exact greedy algorithm (slower, more accurate)
- `'approx'`: Approximate algorithm (faster)
- `'hist'`: Histogram-based (fastest for large datasets)

**Recommendation**: 
```python
params = {'tree_method': 'hist'}  # For >10k samples
```

---

### 14. `seed` (Random Seed)

**Native XGBoost name**: `seed`  
**Scikit-learn equivalent**: `random_state`

**What it does**: Random seed for reproducibility

```python
params = {'seed': 42}
```

---

### 15. `nthread` (Number of Threads)

**Native XGBoost name**: `nthread`  
**Scikit-learn equivalent**: `n_jobs`

**What it does**: Number of parallel threads

```python
params = {'nthread': -1}  # Use all available cores
# or
params = {'nthread': 4}   # Use 4 cores
```

---

## Parameter Tuning Strategy for Your Overfitting Problem

### Phase 1: Immediate Changes (Try This First!)

**Current config.py (using scikit-learn API)**:
```python
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
}
```

**Recommended anti-overfitting config (scikit-learn API)**:
```python
XGBOOST_PARAMS = {
    'n_estimators': 500,           # Increase (we'll use early stopping)
    'max_depth': 3,                # REDUCE from 6 to 3 ← KEY CHANGE
    'learning_rate': 0.01,         # REDUCE from 0.05 to 0.01 ← KEY CHANGE
    'subsample': 0.7,              # REDUCE from 0.8 to 0.7
    'colsample_bytree': 0.7,       # REDUCE from 0.8 to 0.7
    'reg_alpha': 1.0,              # INCREASE from 0.1 to 1.0 ← KEY CHANGE
    'reg_lambda': 3.0,             # INCREASE from 1.0 to 3.0 ← KEY CHANGE
    'min_child_weight': 5,         # INCREASE from 3 to 5
    'gamma': 0.5,                  # INCREASE from 0.1 to 0.5
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1
}
```

**If using native XGBoost API (xgb.train)**:
```python
# Parameter dictionary for xgb.train()
params = {
    'max_depth': 3,                # REDUCE from 6 to 3 ← KEY CHANGE
    'eta': 0.01,                   # REDUCE from 0.05 to 0.01 ← KEY CHANGE (note: eta not learning_rate)
    'subsample': 0.7,              # REDUCE from 0.8 to 0.7
    'colsample_bytree': 0.7,       # REDUCE from 0.8 to 0.7
    'alpha': 1.0,                  # INCREASE from 0.1 to 1.0 ← KEY CHANGE (note: alpha not reg_alpha)
    'lambda': 3.0,                 # INCREASE from 1.0 to 3.0 ← KEY CHANGE (note: lambda not reg_lambda)
    'min_child_weight': 5,         # INCREASE from 3 to 5
    'gamma': 0.5,                  # INCREASE from 0.1 to 0.5
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42,
    'nthread': -1
}

# Train with native API
num_boost_round = 500
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
```

### Phase 2: Add Early Stopping

Early stopping prevents overfitting by monitoring validation performance:

**Using scikit-learn API**:
```python
# In train_models_multioutput.py, modify the fit call:

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,      # Stop if no improvement for 50 rounds
    verbose=False
)
```

**Using native API**:
```python
# Native XGBoost with early stopping
model = xgb.train(
    params, 
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=False
)

# Get actual number of trees used
print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score}")
```

**Note**: For MultiOutputRegressor, early stopping isn't directly supported. You need to:
1. Use single-output models with early stopping, OR
2. Implement custom early stopping wrapper, OR
3. Just reduce `n_estimators` manually

### Phase 3: Systematic Grid Search

After Phase 1 improvements, fine-tune with grid search:

**Using scikit-learn API**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.02, 0.03],
    'reg_alpha': [0.5, 1.0, 2.0],
    'reg_lambda': [2.0, 3.0, 5.0],
    'min_child_weight': [3, 5, 7]
}

# For each forecast hour
base_model = xgb.XGBRegressor(
    n_estimators=500,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")
```

**Using native API with manual grid search**:
```python
import itertools

# Define parameter grid (using native names)
param_grid = {
    'max_depth': [3, 4, 5],
    'eta': [0.01, 0.02, 0.03],
    'alpha': [0.5, 1.0, 2.0],
    'lambda': [2.0, 3.0, 5.0],
    'min_child_weight': [3, 5, 7]
}

# Base parameters
base_params = {
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'seed': 42
}

# Grid search
best_score = float('inf')
best_params = None

# Generate all combinations
keys = param_grid.keys()
for values in itertools.product(*param_grid.values()):
    params = base_params.copy()
    params.update(dict(zip(keys, values)))
    
    # Cross-validation
    cv_scores = []
    for fold in range(5):
        # ... split data into folds ...
        dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval_fold = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        model = xgb.train(params, dtrain_fold, num_boost_round=500)
        preds = model.predict(dval_fold)
        mae = mean_absolute_error(y_fold_val, preds)
        cv_scores.append(mae)
    
    avg_score = np.mean(cv_scores)
    if avg_score < best_score:
        best_score = avg_score
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best CV MAE: {best_score:.4f}")
```

---

## Debugging Overfitting: Step-by-Step

### Step 1: Measure the Gap

```python
# In your evaluation
train_mae = 2.5
test_mae = 7.5
overfit_ratio = test_mae / train_mae  # 3.0 in your case

if overfit_ratio > 1.5:
    print("WARNING: Significant overfitting detected!")
```

### Step 2: Try Progressive Changes

Test each change individually to see impact:

```python
# Baseline (your current config)
params_baseline = {'max_depth': 6, 'learning_rate': 0.05, ...}

# Test 1: Reduce max_depth
params_test1 = {**params_baseline, 'max_depth': 3}

# Test 2: Add regularization
params_test2 = {**params_baseline, 'reg_alpha': 1.0, 'reg_lambda': 3.0}

# Test 3: Reduce learning rate
params_test3 = {**params_baseline, 'learning_rate': 0.01, 'n_estimators': 500}

# Test 4: Increase sampling
params_test4 = {**params_baseline, 'subsample': 0.7, 'colsample_bytree': 0.7}

# Test all and compare
```

### Step 3: Learning Curves

Plot train vs. test performance over number of trees:

```python
import matplotlib.pyplot as plt

results = model.evals_result()

plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['rmse'], label='Train')
plt.plot(results['validation_1']['rmse'], label='Test')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.legend()
plt.title('Learning Curve: Train vs Test')
plt.show()

# Look for:
# - Train keeps decreasing, test plateaus → overfitting
# - Both decrease together → good!
```

---

## Parameter Combinations for Different Scenarios

### Scenario 1: Small Dataset (<500 events)

**Challenge**: Easy to overfit with limited data

**Scikit-learn API**:
```python
XGBOOST_PARAMS = {
    'n_estimators': 100,           # Fewer trees
    'max_depth': 3,                # Shallow trees
    'learning_rate': 0.05,         # Moderate rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 2.0,              # Strong L1
    'reg_lambda': 5.0,             # Strong L2
    'min_child_weight': 5,         # Conservative splits
    'gamma': 1.0,                  # High pruning
}
```

**Native API**:
```python
params = {
    'max_depth': 3,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 2.0,
    'lambda': 5.0,
    'min_child_weight': 5,
    'gamma': 1.0,
    'objective': 'reg:squarederror'
}
num_boost_round = 100
```

### Scenario 2: Large Dataset (>5000 events)

**Challenge**: Can afford more complexity

**Scikit-learn API**:
```python
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,                # Deeper trees OK
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 3,
    'gamma': 0.1,
}
```

**Native API**:
```python
params = {
    'max_depth': 6,
    'eta': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.1,
    'lambda': 1.0,
    'min_child_weight': 3,
    'gamma': 0.1,
    'objective': 'reg:squarederror'
}
num_boost_round = 500
```

### Scenario 3: Noisy Features

**Challenge**: Some features don't help prediction

**Scikit-learn API**:
```python
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.6,       # Aggressive feature sampling
    'colsample_bylevel': 0.7,      # Additional sampling
    'reg_alpha': 2.0,              # Feature selection via L1
    'reg_lambda': 2.0,
    'min_child_weight': 5,
    'gamma': 0.5,
}
```

**Native API**:
```python
params = {
    'max_depth': 4,
    'eta': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 0.7,
    'alpha': 2.0,
    'lambda': 2.0,
    'min_child_weight': 5,
    'gamma': 0.5,
    'objective': 'reg:squarederror'
}
num_boost_round = 200
```

### Scenario 4: Extreme Overfitting (Your Case!)

**Challenge**: Train MAE 3x better than test MAE

**Scikit-learn API**:
```python
XGBOOST_PARAMS = {
    'n_estimators': 500,           # High, with early stopping
    'max_depth': 2,                # VERY shallow ← Start here
    'learning_rate': 0.01,         # Very low
    'subsample': 0.6,              # Aggressive subsampling
    'colsample_bytree': 0.6,       # Aggressive feature sampling
    'reg_alpha': 2.0,              # Strong L1
    'reg_lambda': 5.0,             # Strong L2
    'min_child_weight': 7,         # Very conservative
    'gamma': 1.0,                  # Strong pruning
}
```

**Native API**:
```python
params = {
    'max_depth': 2,                # VERY shallow ← Start here
    'eta': 0.01,                   # Very low
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'alpha': 2.0,                  # Strong L1
    'lambda': 5.0,                 # Strong L2
    'min_child_weight': 7,
    'gamma': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae'
}
num_boost_round = 500
```

### Scenario 5: Need Fast Training

**Challenge**: Quick experiments, acceptable accuracy loss

**Scikit-learn API**:
```python
XGBOOST_PARAMS = {
    'n_estimators': 100,           # Fewer trees
    'max_depth': 4,
    'learning_rate': 0.1,          # Higher rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'tree_method': 'hist',         # Faster algorithm
    'n_jobs': -1,                  # Use all CPUs
}
```

**Native API**:
```python
params = {
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.5,
    'lambda': 1.0,
    'tree_method': 'hist',
    'nthread': -1,
    'objective': 'reg:squarederror'
}
num_boost_round = 100
```

---

## Other Important Parameters

### `tree_method`

**Options**:
- `'auto'`: Default, lets XGBoost decide
- `'exact'`: Exact greedy algorithm (slower, more accurate)
- `'approx'`: Approximate algorithm (faster)
- `'hist'`: Histogram-based (fastest for large datasets)

**Recommendation**: Use `'hist'` for >10k samples

### `scale_pos_weight`

For imbalanced data (if some intensity ranges are rare):

```python
# Calculate imbalance
n_negative = sum(y < threshold)
n_positive = sum(y >= threshold)
scale_pos_weight = n_negative / n_positive
```

### `base_score`

Initial prediction score:

```python
'base_score': y_train.mean()  # Initialize at mean target value
```

---

## Monitoring and Validation

### Track Multiple Metrics

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric=['rmse', 'mae'],  # Track both
    verbose=True
)
```

### Cross-Validation

For robust parameter selection:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5,  # 5-fold
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

print(f"CV MAE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Quick Reference: Native vs Scikit-Learn Parameter Names

| Native XGBoost | Scikit-Learn | Description |
|----------------|--------------|-------------|
| `eta` | `learning_rate` | Learning rate / shrinkage |
| `alpha` | `reg_alpha` | L1 regularization |
| `lambda` | `reg_lambda` | L2 regularization |
| `seed` | `random_state` | Random seed |
| `nthread` | `n_jobs` | Number of threads |
| (passed separately) | `n_estimators` | Number of trees (use `num_boost_round` in `xgb.train()`) |
| `max_depth` | `max_depth` | Same in both |
| `subsample` | `subsample` | Same in both |
| `colsample_bytree` | `colsample_bytree` | Same in both |
| `gamma` | `gamma` | Same in both |
| `min_child_weight` | `min_child_weight` | Same in both |

---

## Summary: Your Action Plan

### Immediate Actions (Try Today!)

**For Scikit-Learn API (XGBRegressor)**:
1. **Reduce `max_depth` to 3** ← Most important!
2. **Reduce `learning_rate` to 0.01**
3. **Increase regularization**: `reg_alpha=1.0`, `reg_lambda=3.0`
4. **Reduce sampling**: `subsample=0.7`, `colsample_bytree=0.7`

**For Native API (xgb.train)**:
1. **Reduce `max_depth` to 3** ← Most important!
2. **Reduce `eta` to 0.01** (note: `eta` not `learning_rate`)
3. **Increase regularization**: `alpha=1.0`, `lambda=3.0` (note: no `reg_` prefix)
4. **Reduce sampling**: `subsample=0.7`, `colsample_bytree=0.7`

### Expected Results

Before:
```
Train MAE: 2.5
Test MAE:  7.5  (3x worse)
```

After (conservative params):
```
Train MAE: 3.5  (higher - that's OK!)
Test MAE:  4.5  (much better generalization!)
```

### If Still Overfitting

1. Reduce `max_depth` to 2
2. Increase `min_child_weight` to 7-10
3. Increase `gamma` to 1.0-2.0
4. Reduce `subsample` to 0.5-0.6

### Remember

- **Lower training error ≠ better model**
- **Goal**: Minimize test error, even if train error increases
- **Iterate**: Test one change at a time
- **Validate**: Use cross-validation for final model selection
- **Use correct names**: `eta` not `learning_rate`, `alpha` not `reg_alpha` when using native API

Good luck with the tuning!
