# Model Diversity Analysis: Are You Covering All Bases?

## Current Model Lineup

| Model               | Type          | Learning Style        | Strengths                    | Weakness in Your Stack          |
|---------------------|---------------|-----------------------|------------------------------|---------------------------------|
| Logistic Regression | Linear        | Gradient-based        | Linear boundaries, probs     | Similar to SVM                  |
| SVM (RBF)           | Kernel        | Margin-based          | Non-linear boundaries        | Similar to LogReg on PCA data   |
| XGBoost             | Tree ensemble | Boosting              | Feature interactions         | Hurt by PCA transformation      |
| Naive Bayes         | Probabilistic | Bayesian              | Fast, probabilistic          | Independence assumption flawed  |

## Diversity Assessment: ⚠️ NEEDS IMPROVEMENT

### Problem 1: LogReg and SVM Are Too Similar

On PCA-transformed data, LogReg and SVM are capturing almost identical patterns:
- Both find (approximately) linear decision boundaries
- Correlation between their predictions is probably >0.9
- This is redundant—not true diversity


**Evidence:**
- LogReg accuracy: 66.0%
- SVM accuracy: 65.8%
- Nearly identical performance = they're learning the same thing

### Problem 2: Only One Tree-Based Model

XGBoost is your only tree-based model, and it's being handicapped by PCA.
- Trees are fundamentally different from linear models
- But you need more variety in tree approaches

### Problem 3: Naive Bayes Is Weak

BernoulliNB on continuous PCA components is a poor fit:
- Wrong distribution (should use GaussianNB)
- Weak performance as base model (64.2%)
- Weak contribution to ensemble

## Diversity Scores (Qualitative)

### High Diversity Pairs (Good):
- **XGBoost + Logistic Regression**: Tree vs Linear (⭐⭐⭐⭐⭐)
- **Naive Bayes + SVM**: Probabilistic vs Geometric (⭐⭐⭐⭐)
- **XGBoost + Naive Bayes**: Tree vs Probabilistic (⭐⭐⭐⭐)

### Low Diversity Pairs (Redundant):
- **Logistic Regression + SVM**: Both linear on PCA data (⭐⭐)
  - On raw data, SVM would be more different
  - On PCA data, they're too similar

## Recommended Models to Add

### 1. Random Forest (HIGH PRIORITY)

**Why add it:**
- Tree-based (complements LogReg/SVM)
- Different from XGBoost:
  - XGBoost = boosting (sequential, corrects errors)
  - Random Forest = bagging (parallel, reduces variance)
- More robust than XGBoost (less prone to overfitting)
- Works great on raw features

**Expected performance:**
- On PCA: ~64-66%
- On raw features: ~66-69%

**Configuration:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

### 2. K-Nearest Neighbors (MEDIUM PRIORITY)

**Why add it:**
- Completely different paradigm: Distance-based, not model-based
- Non-parametric: Makes no assumptions about data distribution
- Captures local patterns: "This game looks like these 5 past games, which home team won 4/5"

**Expected performance:**
- On PCA: ~62-65%
- On raw features (scaled): ~64-67%

**Configuration:**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=15,  # Try 10, 15, 20
    weights='distance',  # Closer neighbors have more influence
    metric='euclidean'
)
```

**Diversity contribution:** ⭐⭐⭐⭐⭐ (Very different from all current models)

### 3. LightGBM (MEDIUM PRIORITY)

**Why add it:**
- Another gradient boosting library, but different algorithm than XGBoost
- Faster training
- Often finds different patterns than XGBoost
  - Uses histogram-based splitting
  - Grows trees leaf-wise vs level-wise

**Expected performance:**
- Similar to XGBoost (~63-68% depending on features)
- But predictions will be different enough to add value

**Configuration:**
```python
import lightgbm as lgb

lgbm = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
```

### 4. Neural Network (LOW PRIORITY)

**Why add it:**
- Completely different learning paradigm
- Can learn complex non-linear interactions
- Good for diverse ensemble

**Why LOW priority:**
- Needs more data than you have (~1000 samples is borderline)
- Harder to tune (many hyperparameters)
- Likely to overfit without careful regularization

**Only add if:**
- You get more data (5,000+ games)
- You use strong regularization (dropout, L2)

### 5. GaussianNB (HIGH PRIORITY - REPLACEMENT)

**Replace BernoulliNB with GaussianNB:**
- Bernoulli assumes binary features (0/1)
- Gaussian assumes continuous features (which PCA components are)
- Should improve performance by ~2-3%

**Configuration:**
```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()  # No hyperparameters needed
```

## Recommended Final Model Lineup

### Option A: Conservative (5 models)
```python
# On PCA features:
logreg_pca = LogisticRegression(penalty='l2', C=1.0)
gnb_pca = GaussianNB()

# On raw features:
xgb_raw = XGBClassifier(max_depth=5, learning_rate=0.05)
rf_raw = RandomForestClassifier(n_estimators=200, max_depth=10)
knn_raw = KNeighborsClassifier(n_neighbors=15, weights='distance')
```

**Diversity score:** ⭐⭐⭐⭐⭐
- 2 linear models (PCA space)
- 1 probabilistic model (PCA space)
- 2 tree models (raw space)
- 1 distance-based model (raw space)

### Option B: Aggressive (7 models)
Add to Option A:
```python
svm_pca = SVC(kernel='rbf', probability=True)
lgbm_raw = LGBMClassifier(max_depth=5, learning_rate=0.05)
```

**Diversity score:** ⭐⭐⭐⭐
- More models, but some redundancy (SVM+LogReg, LGBM+XGBoost)
- Might improve ensemble performance by 1-2%
- Risk of overfitting the meta-learner increases

## How to Measure Diversity

After training your models, check prediction correlation:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions from all models
predictions = pd.DataFrame({
    'LogReg': logreg.predict_proba(X_test)[:, 1],
    'SVM': svm.predict_proba(X_test)[:, 1],
    'XGBoost': xgb_model.predict_proba(X_test)[:, 1],
    'RandomForest': rf.predict_proba(X_test)[:, 1],
    'KNN': knn.predict_proba(X_test)[:, 1]
})

# Calculate correlation matrix
corr_matrix = predictions.corr()

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True)
plt.title('Model Prediction Correlation (Diversity Check)')
plt.show()

print(corr_matrix)
```

**What to look for:**
- **Good diversity:** Most correlations between 0.5-0.8
  - Models agree on general direction but differ on specifics
- **Too similar:** Correlations > 0.9
  - Models are redundant, remove one
- **Too different:** Correlations < 0.3
  - Models might be learning noise, check for errors

## Expected Correlation Matrix (My Predictions)

|            | LogReg | SVM  | XGB  | RF   | KNN  |
|------------|--------|------|------|------|------|
| LogReg     | 1.00   | 0.92 | 0.65 | 0.68 | 0.58 |
| SVM        | 0.92   | 1.00 | 0.63 | 0.66 | 0.56 |
| XGBoost    | 0.65   | 0.63 | 1.00 | 0.85 | 0.62 |
| RF         | 0.68   | 0.66 | 0.85 | 1.00 | 0.64 |
| KNN        | 0.58   | 0.56 | 0.62 | 0.64 | 1.00 |

**Interpretation:**
- LogReg ↔ SVM: 0.92 (too high, consider removing SVM)
- XGBoost ↔ RF: 0.85 (acceptable, both trees but different algorithms)
- KNN ↔ Everyone: 0.56-0.64 (good diversity)

## Diversity vs Performance Trade-off

More diversity ≠ always better. You want:
1. **Diversity:** Models capture different patterns
2. **Performance:** Each model is reasonably accurate

A terrible model that's very different doesn't help—it just adds noise.

**Sweet spot:** 3-5 models with correlation between 0.6-0.8 and individual accuracy >60%

## Summary: Your Diversity Grade

**Current lineup:** C+ (⭐⭐⭐)
- Good: Have tree-based (XGB) + linear (LogReg) + probabilistic (NB)
- Bad: LogReg and SVM are redundant on PCA data
- Bad: Only one tree model, handicapped by PCA

**With additions (LogReg_PCA, GaussianNB_PCA, XGB_raw, RF_raw, KNN_raw):** A (⭐⭐⭐⭐⭐)
- Excellent diversity across model types
- Excellent diversity across feature representations (PCA vs raw)
- Each model in optimal environment
- Expected ensemble performance: ~70-72%
