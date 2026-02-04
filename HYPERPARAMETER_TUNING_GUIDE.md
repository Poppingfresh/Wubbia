# XGBoost Hyperparameter Tuning Guide

## Overview

This guide explains every XGBoost parameter, what it does, and how to tune it for your intensity forecasting problem. 

**Your situation**: Training MAE is much lower than test MAE (3x difference) → **Classic overfitting**

The model is learning the training data too well and not generalizing to new events.

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

### 1. `n_estimators` (Number of Trees)

**What it does**: Total number of boosting rounds (trees to build)

**Default**: 100

**Impact**:
- More trees → Better training performance
- Too many trees → Overfitting (especially without early stopping)
- Too few trees → Underfitting

**Tuning strategy**:
```python
# Start conservative
'n_estimators': 100

# If underfitting, increase
'n_estimators': 200-500

# If overfitting, decrease
'n_estimators': 50-100
```

**For your overfitting problem**:
- Try **reducing** to 100-150
- Use early stopping (see below)

**Best practice**: Set high (500+) and use early stopping

---

### 2. `max_depth` (Tree Depth)

**What it does**: Maximum depth of each tree (how many splits)

**Default**: 6

**Impact**:
- Deep trees → Capture complex interactions, but overfit easily
- Shallow trees → More robust, better generalization
- Most important parameter for controlling overfitting!

**Tuning strategy**:
```python
# For small datasets (<1000 samples)
'max_depth': 3-4

# For medium datasets (1000-10000 samples)
'max_depth': 4-6

# For large datasets (>10000 samples)
'max_depth': 6-10

# When overfitting
'max_depth': 3  # Start here!
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

### 3. `learning_rate` (or `eta`)

**What it does**: Step size for each tree's contribution

**Default**: 0.3

**Impact**:
- Lower rate → More robust, better generalization, needs more trees
- Higher rate → Faster training, more aggressive updates, can overfit

**Tuning strategy**:
```python
# Conservative (recommended)
'learning_rate': 0.01-0.05

# Moderate
'learning_rate': 0.05-0.1

# Aggressive (faster but risky)
'learning_rate': 0.1-0.3
```

**For your overfitting problem**:
- **Reduce to 0.01-0.03**
- Lower learning rate = smoother learning = better generalization
- Increase `n_estimators` to compensate

**Rule of thumb**: `learning_rate × n_estimators ≈ constant`
```python
# These are roughly equivalent:
Option 1: learning_rate=0.1,  n_estimators=100
Option 2: learning_rate=0.05, n_estimators=200
Option 3: learning_rate=0.01, n_estimators=1000  # Best for generalization
```

---

### 4. `subsample` (Row Sampling)

**What it does**: Fraction of training samples to use for each tree

**Default**: 1.0 (use all samples)

**Impact**:
- <1.0 → Random sampling, reduces overfitting, adds randomness
- More robust to outliers
- Too low → Underfitting

**Tuning strategy**:
```python
# No sampling (default)
'subsample': 1.0

# Moderate sampling (recommended for overfitting)
'subsample': 0.7-0.8

# Aggressive sampling
'subsample': 0.5-0.7
```

**For your overfitting problem**:
- **Set to 0.7-0.8**
- Each tree sees only 70-80% of data → better generalization

---

### 5. `colsample_bytree` (Feature Sampling)

**What it does**: Fraction of features to use for each tree

**Default**: 1.0 (use all features)

**Impact**:
- <1.0 → Random feature selection, reduces overfitting
- Helps when some features are noisy
- Makes trees more diverse

**Tuning strategy**:
```python
# No sampling (default)
'colsample_bytree': 1.0

# Moderate sampling (recommended for overfitting)
'colsample_bytree': 0.7-0.8

# Aggressive sampling
'colsample_bytree': 0.5-0.7
```

**For your overfitting problem**:
- **Set to 0.7-0.8**
- Especially helpful if you have 10+ features

---

## Regularization Parameters (Critical for Overfitting!)

### 6. `reg_alpha` (L1 Regularization)

**What it does**: L1 penalty on leaf weights (encourages sparsity)

**Default**: 0

**Impact**:
- Higher values → Simpler model, feature selection
- Pushes some leaf weights to zero
- Good when many features are irrelevant

**Tuning strategy**:
```python
# No regularization
'reg_alpha': 0

# Light regularization
'reg_alpha': 0.1-1.0

# Strong regularization (for overfitting)
'reg_alpha': 1.0-10.0
```

**For your overfitting problem**:
- **Start with 0.5-1.0**
- Increase if still overfitting

---

### 7. `reg_lambda` (L2 Regularization)

**What it does**: L2 penalty on leaf weights (smooth weights)

**Default**: 1.0

**Impact**:
- Higher values → Smoother predictions, less extreme weights
- More conservative than L1
- Default=1.0 provides some regularization

**Tuning strategy**:
```python
# Light regularization
'reg_lambda': 1.0  # Default, often sufficient

# Moderate regularization (for overfitting)
'reg_lambda': 2.0-5.0

# Strong regularization
'reg_lambda': 5.0-10.0
```

**For your overfitting problem**:
- **Increase to 2.0-5.0**
- Works well combined with `reg_alpha`

---

### 8. `min_child_weight` (Minimum Sum of Instance Weight)

**What it does**: Minimum sum of instance weight needed in a child node

**Default**: 1

**Impact**:
- Higher values → More conservative splits, prevents overfitting
- Stops splitting when samples are too few
- Similar to "min_samples_leaf" in other algorithms

**Tuning strategy**:
```python
# Permissive (default)
'min_child_weight': 1

# Conservative (recommended for overfitting)
'min_child_weight': 3-5

# Very conservative
'min_child_weight': 5-10
```

**For your overfitting problem**:
- **Increase to 3-5**
- Prevents fitting on small sample groups

---

### 9. `gamma` (Minimum Loss Reduction)

**What it does**: Minimum loss reduction required to make a split

**Default**: 0

**Impact**:
- Higher values → Fewer splits, simpler trees
- Acts as pruning mechanism
- Makes model more conservative

**Tuning strategy**:
```python
# No constraint (default)
'gamma': 0

# Light constraint
'gamma': 0.1-1.0

# Strong constraint (for overfitting)
'gamma': 1.0-5.0
```

**For your overfitting problem**:
- **Start with 0.1-0.5**
- Increase if still overfitting

---

## Advanced Parameters

### 10. `colsample_bylevel` (Per-Level Feature Sampling)

**What it does**: Fraction of features to sample at each tree level

**Default**: 1.0

**Impact**:
- Similar to `colsample_bytree` but per level
- Additional randomization

**When to use**:
- When `colsample_bytree` alone isn't enough
- For very deep trees

```python
'colsample_bylevel': 0.7-0.9
```

---

### 11. `colsample_bynode` (Per-Node Feature Sampling)

**What it does**: Fraction of features to sample at each node

**Default**: 1.0

**Impact**:
- Most aggressive feature sampling
- Maximum randomization

**When to use**:
- Extreme overfitting cases
- Many noisy features

```python
'colsample_bynode': 0.7-0.9
```

---

### 12. `max_delta_step` (Maximum Delta Step)

**What it does**: Maximum delta step for each tree's output

**Default**: 0 (no constraint)

**Impact**:
- Constrains leaf values
- Useful for imbalanced data or extreme values

**When to use**:
- When predictions are unstable
- For classification with imbalanced classes

```python
'max_delta_step': 1-5
```

---

## Parameter Tuning Strategy for Your Overfitting Problem

### Phase 1: Immediate Changes (Try This First!)

**Current config.py**:
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

**Recommended anti-overfitting config**:
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

### Phase 2: Add Early Stopping

Early stopping prevents overfitting by monitoring validation performance:

```python
# In train_models_multioutput.py, modify the fit call:

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,      # Stop if no improvement for 50 rounds
    verbose=False
)
```

**Note**: For MultiOutputRegressor, early stopping isn't directly supported. You need to:
1. Use single-output models with early stopping, OR
2. Implement custom early stopping wrapper, OR
3. Just reduce `n_estimators` manually

### Phase 3: Systematic Grid Search

After Phase 1 improvements, fine-tune with grid search:

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

### Scenario 2: Large Dataset (>5000 events)

**Challenge**: Can afford more complexity

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

### Scenario 3: Noisy Features

**Challenge**: Some features don't help prediction

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

### Scenario 4: Extreme Overfitting (Your Case!)

**Challenge**: Train MAE 3x better than test MAE

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

### Scenario 5: Need Fast Training

**Challenge**: Quick experiments, acceptable accuracy loss

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

## Summary: Your Action Plan

### Immediate Actions (Try Today!)

1. **Reduce `max_depth` to 3** ← Most important!
2. **Reduce `learning_rate` to 0.01**
3. **Increase regularization**: `reg_alpha=1.0`, `reg_lambda=3.0`
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

Good luck with the tuning!
