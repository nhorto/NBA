# PCA vs Raw Features: Strategic Model Training

## The Problem with Current Approach

Currently, ALL your models are trained on PCA-transformed data (27 components). This hurts some models while helping others:

### Models That BENEFIT from PCA:
- **Logistic Regression**: Reduces multicollinearity, creates linear separability
- **Naive Bayes**: Orthogonal features better match independence assumption
- **Simple models**: Less prone to overfitting with fewer features

### Models That are HURT by PCA:
- **XGBoost**: Loses feature interactions (e.g., "missing star player AND back-to-back game")
- **Random Forest**: Can't leverage raw feature relationships
- **Tree-based models**: Work better with interpretable, raw features

## Recommended Strategy: Hybrid Approach

### Option 1: Train Base Models on Different Feature Sets (BEST)

```python
# Train some models on PCA data
logreg_pca = LogisticRegression().fit(X_train_pca, y_train)
nb_pca = GaussianNB().fit(X_train_pca, y_train)

# Train other models on RAW scaled data
xgb_raw = XGBClassifier().fit(X_train_scaled, y_train)
rf_raw = RandomForestClassifier().fit(X_train_scaled, y_train)

# Stack them together
base_models = [
    ('logreg_pca', logreg_pca),
    ('nb_pca', nb_pca),
    ('xgb_raw', xgb_raw),
    ('rf_raw', rf_raw)
]
```

**Why this works:**
- Maximum diversity: Different models + different feature representations
- Each model sees data in its optimal format
- LogReg captures linear PCA patterns, XGBoost captures raw feature interactions
- Meta-learner learns to combine complementary signals

**Implementation considerations:**
- Need to maintain two feature pipelines (PCA and scaled)
- At prediction time, transform input through both pipelines
- More complex but potentially much better performance

### Option 2: All Raw Features

Train everything on raw scaled features (119 features):

**Pros:**
- XGBoost/Random Forest will perform MUCH better
- Can interpret feature importance (which raw stats matter most)
- Tree models can find interactions like "missing_impact_home > 2 AND PACE_rolling_away < 95"

**Cons:**
- Logistic Regression may suffer from multicollinearity
  - Many features are highly correlated (FGA and POSS, for example)
  - Can mitigate with stronger L1/L2 regularization
- Training is slower (119 features vs 27)
- Higher risk of overfitting for simple models

**When to use:**
- If tree-based models are your primary interest
- If you want interpretable feature importance
- If you have enough data (1000+ samples is borderline okay)

### Option 3: All PCA Features (CURRENT)

**Pros:**
- Dimensionality reduction (27 vs 119 features)
- Removes multicollinearity
- Faster training

**Cons:**
- Destroys feature interactions for tree models
- Loses interpretability (what does PC7 mean?)
- Hurts XGBoost/Random Forest performance

**When to use:**
- If Logistic Regression is your best performer and you want to optimize it
- If you need speed (fewer features = faster)
- If interpretability at the feature level isn't important

## Experiment Results Prediction

Based on your current results, here's what I predict will happen:

### Current (All PCA):
- LogReg: 66.0% (best base model)
- XGBoost: 63.8% (worst base model)

### If you switch to All Raw:
- XGBoost: ~68-70% (predicted best base model)
- Random Forest: ~66-68%
- LogReg: ~64-65% (will drop due to multicollinearity)

### If you use Hybrid (PCA + Raw):
- Best stacked model: ~70-72% (predicted)
- LogReg_PCA captures linear strength differential
- XGBoost_Raw captures "missing star + back-to-back" type interactions
- Meta-learner combines both signals

## Recommended Next Steps

1. **Test XGBoost on raw features:**
   ```python
   xgb_raw = XGBClassifier(eval_metric='logloss', random_state=42)
   xgb_raw.fit(X_train_scaled, y_train)  # Not X_train_pca!
   y_pred_xgb_raw = xgb_raw.predict(X_test_scaled)
   ```
   - I bet it will jump from 63.8% to ~68%+
   - Check feature importance: `xgb_raw.feature_importances_`

2. **Add Random Forest on raw features:**
   ```python
   rf_raw = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
   rf_raw.fit(X_train_scaled, y_train)
   ```
   - Should perform ~66-68%
   - More robust than XGBoost (less overfitting)

3. **Create hybrid stack:**
   - Train LogReg, GaussianNB on PCA features
   - Train XGBoost, RandomForest, LightGBM on raw features
   - Stack all 5 together with Logistic meta-learner
   - This should be your best model


## Technical Note: Why XGBoost Needs Raw Features

**Example interaction XGBoost can learn from raw features:**
```
if missing_star_home == 1:
    if PACE_rolling_home > 100:
        predict_loss  # Fast pace without star = disorganized
    else:
        check_ELO_differential  # Slow pace = can still win with system
else:
    predict based on offensive_rating
```

**After PCA, this becomes:**
```
PC2 = 0.05*missing_star + 0.02*PACE + 0.15*offensive_rating + ...
```

The discrete interaction is LOST—it's blended into a continuous component. XGBoost can't separate it back out.

## Conclusion

**Answer to your question:** Yes, you should absolutely experiment with non-PCA data, especially for tree-based models. The optimal strategy is likely a **hybrid approach** where different models use different feature representations, then stack them together for maximum diversity.
