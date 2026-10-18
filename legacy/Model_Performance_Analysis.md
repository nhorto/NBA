# Model Performance Analysis: Understanding Base Models vs Stacked Ensembles

## Executive Summary
After testing various machine learning models on NBA game predictions, the key findings are:
- **Best base model:** Logistic Regression (66% accuracy, 0.719 ROC AUC)
- **Best stacked model:** SVM meta-learner with XGBoost + Naive Bayes (68.7% accuracy, 0.745 ROC AUC)
- **Improvement:** Stacking provides ~2-7% accuracy boost and ~2-5% ROC AUC improvement over base models
- **Key insight:** Different models capture different patterns, and combining them extracts complementary information

---

## Baseline Performance
Before analyzing the models, here are the baselines:
- **Always predict home team wins:** 57.2% accuracy
- **Always predict higher ELO team wins:** 62.6% accuracy

The ELO baseline is surprisingly strong, showing that raw team strength (captured by ELO) is a powerful predictor. Any model needs to beat 62.6% to be valuable.

---


## Base Model Performance Analysis

### Results Summary
| Model               | Accuracy | ROC AUC | Key Strength                          |
|---------------------|----------|---------|---------------------------------------|
| Logistic Regression | 0.660    | 0.719   | Linear patterns, probabilities        |
| SVM (RBF kernel)    | 0.658    | 0.696   | Non-linear decision boundaries        |
| XGBoost             | 0.638    | 0.692   | Feature interactions, non-linearity   |
| Naive Bayes         | 0.642    | 0.700   | Probabilistic, feature independence   |

### Why Logistic Regression Performs Best

**Logistic Regression outperforms all other base models.** Here's why:

1. **PCA Creates Linear Separability**
   - After PCA transformation, the principal components are orthogonal (uncorrelated)
   - PC2 captures away team strength, PC3 captures home team strength
   - The decision boundary between win/loss becomes approximately linear in this transformed space
   - LogReg excels at finding linear decision boundaries

2. **Probabilistic Outputs**
   - LogReg naturally produces well-calibrated probabilities
   - Your target (home team win) is inherently probabilistic—games aren't deterministic
   - ROC AUC of 0.719 shows it's ranking games correctly by win probability

3. **No Overfitting**
   - With 27 PCA components and ~980 training samples, LogReg has enough data to avoid overfitting
   - L2 regularization (penalty='l2') prevents coefficient explosion
   - Tree-based models (XGBoost) can overfit more easily on this dataset size

4. **Data Matches Assumptions**
   - PCA-transformed features are continuous and roughly normally distributed
   - No strong non-linear interactions that LogReg would miss
   - The relationship between team strength differential (PC2 vs PC3) and win probability is naturally logistic

### Why SVM Underperforms Logistic Regression

**SVM achieves 65.8% accuracy (0.696 ROC AUC)—slightly worse than LogReg.** Reasons:

1. **PCA Already Linearizes the Space**
   - The RBF (Radial Basis Function) kernel is designed to find non-linear boundaries
   - But after PCA, the data is already linearly separable
   - SVM's complexity doesn't add value; it just adds noise

2. **Probability Calibration Issues**
   - SVM with `probability=True` uses Platt scaling to estimate probabilities
   - This is a post-hoc calibration that's less accurate than LogReg's native probability estimates
   - Hurts ROC AUC performance

3. **Sensitivity to Scaling**
   - While you scaled the data, SVM can still be sensitive to the scale of PCA components
   - PCA components have different variances (PC1 explains 30%, PC27 explains <1%)
   - SVM treats all dimensions equally, which may not be optimal

**When would SVM be better?**
- If you used raw features (not PCA) with complex non-linear patterns
- If you had more data (thousands of samples per feature)
- If you had clear non-linear decision boundaries that LogReg couldn't capture

### Why XGBoost Underperforms

**XGBoost achieves 63.8% accuracy (0.692 ROC AUC)—surprisingly the worst performer.** This seems counterintuitive since XGBoost often dominates competitions. Here's why it struggles:

1. **PCA Removes Feature Interactions**
   - XGBoost's strength is discovering interactions between raw features
   - For example: "if team A is missing a star AND they're on a back-to-back, then..."
   - PCA components are linear combinations that blend features together
   - This destroys the discrete feature interactions that trees exploit

2. **Smooth Decision Space**
   - After PCA, the decision boundary is smooth and continuous
   - Tree-based models create rectangular decision boundaries (if X < threshold, then...)
   - Trees excel at capturing discrete regions, not smooth gradients
   - LogReg's smooth sigmoid function fits better

3. **Limited Data for Complex Model**
   - XGBoost has many hyperparameters (tree depth, learning rate, number of trees)
   - With only ~980 training samples, it's easy to overfit
   - Even with your tuned parameters (max_depth=5, learning_rate=0.05), it may be finding spurious patterns

4. **PCA Components Ranked by Variance, Not Predictive Power**
   - PC1 (season progression) explains 30% of variance but may not predict wins well
   - Later PCs might be more predictive but have less weight
   - XGBoost can't easily re-weight these; LogReg learns optimal weights via coefficients

**When would XGBoost be better?**
- If you used raw features (not PCA) where interactions matter
- If you had more training data (5,000+ games)
- If you had categorical features or discrete decision rules

### Why Naive Bayes is Middle-of-the-Pack

**Naive Bayes achieves 64.2% accuracy (0.700 ROC AUC)—better than XGBoost but worse than LogReg/SVM.**

1. **Feature Independence Assumption**
   - Naive Bayes assumes all features are independent given the class
   - PCA components ARE orthogonal (independent), which helps
   - This is why NB performs reasonably well

2. **But: Conditional Independence is Violated**
   - PC2 (away team strength) and PC3 (home team strength) are not independent given the outcome
   - If away team is strong (low PC2), home team strength (PC3) matters more
   - Naive Bayes can't capture this interaction

3. **Bernoulli Distribution Mismatch**
   - You used BernoulliNB, which expects binary features
   - PCA components are continuous, not binary
   - This is a poor distributional fit (GaussianNB would be better)

---

## Stacking Ensemble Performance

### Top Performing Combinations

| Stack Configuration            | Accuracy | ROC AUC | Why It Works                              |
|--------------------------------|----------|---------|-------------------------------------------|
| SVMmeta-xgb+nb                 | 0.687    | 0.745   | Diverse base models + SVM finds boundary  |
| LRmeta-logreg+xgb              | 0.675    | 0.737   | Linear + non-linear combo                 |
| LRmeta-logreg+xgb+nb           | 0.654    | 0.733   | Three diverse perspectives                |
| SVMmeta-logreg+xgb+nb          | 0.663    | 0.733   | SVM meta-learner, 3 base models           |
| LRmeta-xgb+nb                  | 0.679    | 0.731   | Complements XGBoost's trees with NB probs |

### Why Stacking Outperforms Base Models

**Core Principle:** Different models make different types of errors. Stacking learns to correct these errors.

#### 1. **Complementary Strengths**
Each base model captures different patterns:
- **Logistic Regression:** Linear separation in PCA space (PC2 vs PC3)
- **XGBoost:** Non-linear interactions and threshold-based rules
- **Naive Bayes:** Probabilistic reasoning under independence assumptions
- **SVM:** Maximum margin decision boundary

When one model is uncertain, another might be confident. The meta-learner learns which model to trust in which situations.

#### 2. **Error Correction**
Example scenario:
- LogReg predicts: 55% chance home team wins (uncertain)
- XGBoost predicts: 70% chance home team wins (confident)
- Naive Bayes predicts: 45% chance home team wins (thinks away team wins)

The meta-learner learns: "When LogReg is uncertain but XGBoost is confident, trust XGBoost 80% of the time."

#### 3. **Ensemble Diversity = Lower Variance**
Statistical principle: Averaging predictions from diverse models reduces variance.
- If all models agreed perfectly, stacking wouldn't help
- Because base models disagree (they capture different signals), averaging reduces random errors
- The meta-learner does "smart averaging" weighted by each model's reliability

#### 4. **Second Layer Captures Meta-Patterns**
The meta-learner learns higher-level patterns like:
- "When SVM and LogReg agree, they're usually right"
- "When XGBoost disagrees with everyone, it's often seeing something important"
- "When all models predict ~50%, just predict the ELO favorite"

This is information the base models can't learn individually.

### Why Certain Combinations Work Best

#### **SVMmeta-xgb+nb: The Champion (68.7% acc, 0.745 ROC AUC)**

**Why this works:**
1. **Maximum Diversity:** XGBoost (tree-based) + Naive Bayes (probabilistic) use completely different algorithms
2. **No LogReg/SVM Redundancy:** LogReg and SVM both find linear boundaries; excluding LogReg avoids redundancy
3. **SVM Meta-Learner:** The meta-layer uses SVM to find a non-linear combination of XGB and NB predictions
   - This is appropriate because XGB and NB predictions might have non-linear interactions
   - SVM's RBF kernel can capture: "If XGB says >70% AND NB says >60%, confidence is very high"

**Intuition:** XGBoost finds discrete patterns, Naive Bayes provides probabilistic calibration, and SVM learns the optimal non-linear blend.

#### **LRmeta-logreg+xgb: Strong Runner-Up (67.5% acc, 0.737 ROC AUC)**

**Why this works:**
1. **Linear + Non-linear Combo:** LogReg captures smooth decision boundary, XGBoost captures discrete rules
2. **Logistic Meta-Learner:** Simple weighted average of the two predictions
3. **Interpretability:** The meta-weights tell you how much to trust each model
   - If LogReg coefficient > XGBoost coefficient, linear patterns dominate
   - This combination balances simplicity (LogReg) with complexity (XGBoost)

#### **Why Adding More Models Isn't Always Better**

Notice that 4-model stacks often perform WORSE than 2-3 model stacks:
- **LRmeta-logreg+svm+xgb+nb:** 0.667 accuracy (worse than many 2-model combos)
- **Reason:** Redundancy and overfitting
  - LogReg and SVM are highly correlated (both linear)
  - The meta-learner has 4 inputs instead of 2-3, increasing overfitting risk
  - More models = more noise for the meta-learner to filter

**Optimal diversity:** 2-3 highly diverse models > 4 somewhat redundant models

### Why Meta-Learner Choice Matters

#### **Logistic Regression Meta-Learner: Most Reliable**
- Averages well across all combinations
- Produces calibrated probabilities
- Doesn't overfit the meta-layer
- **Best for:** Most use cases

#### **SVM Meta-Learner: Best Ceiling, Higher Variance**
- Can capture non-linear interactions between base model predictions
- Achieves the highest ROC AUC (0.745) with right combination
- But also produces some poor combinations (SVMmeta-svm+nb: 0.666 ROC AUC)
- **Best for:** When base models have non-linear relationships

#### **XGBoost Meta-Learner: Mediocre**
- Best result: XGBmeta-logreg+svm+xgb at 0.725 ROC AUC
- Tends to overfit the meta-layer (only 243 test samples to learn from)
- Tree-based meta-learner is overkill for 2-4 input features
- **Best for:** When you have thousands of base models (not applicable here)

#### **Naive Bayes Meta-Learner: Complete Failure**
- **ROC AUC of exactly 0.500** across ALL combinations = random guessing!
- **What happened:** NB as meta-learner just predicts the majority class (home team wins) for every game
  - Accuracy = 57.2% (% of home wins in test set)
  - But no discrimination between games (ROC AUC = 0.5)
- **Why it failed:**
  - BernoulliNB expects binary inputs, but base model probabilities are continuous
  - NB's independence assumption is severely violated (base model predictions are correlated)
  - Poor probability calibration leads to predicting same class always
- **Never use Naive Bayes as a meta-learner** for stacking

---

## Key Takeaways for Stakeholders

### 1. **Why Model Selection Matters**
- The same data with different models produces 63.8%-66.0% accuracy for base models
- Choice of algorithm matters, but only within ~2-3 percentage points
- Understanding your data structure (linear vs non-linear) guides model choice

### 2. **Why Stacking Improves Performance**
- Combining models provides ~2-5% accuracy improvement
- Different models capture different signals in the data
- The meta-learner learns which model to trust in different situations
- Real-world analogy: "Getting a second opinion from a doctor with a different specialty"

### 3. **Diminishing Returns of Complexity**
- 2-3 diverse models > 4 redundant models
- SVM meta-learner has highest ceiling but more variance
- Simple Logistic meta-learner is most consistent
- More complexity doesn't always = better performance

### 4. **Model Performance in Context**
- Best model: 68.7% accuracy, 0.745 ROC AUC
- Baseline (ELO): 62.6% accuracy
- **Improvement:** 6.1 percentage points over ELO alone
- This means: In ~6 extra games out of 100, the model correctly predicts the outcome vs. just using ELO
- ROC AUC improvement (0.745 vs estimated ~0.65 for ELO) shows better probability calibration

---

## Technical Summary

### Why PCA Changes Model Performance

**Before PCA (raw features):**
- XGBoost would likely dominate (capturing feature interactions)
- SVM would find complex non-linear boundaries
- LogReg would struggle with multicollinearity

**After PCA (27 orthogonal components):**
- LogReg dominates (linear separation in transformed space)
- XGBoost loses its advantage (no discrete feature interactions)
- SVM's non-linearity isn't needed (space is already linearized)

**Lesson:** Feature engineering (PCA) can completely change which algorithm works best.

### Stacking Architecture
```
Raw Features (119 features)
    ↓
StandardScaler
    ↓
PCA (27 components)
    ↓
Base Models (LogReg, SVM, XGB, NB)
    ↓ (produce predictions)
Meta-Learner (combines predictions)
    ↓
Final Prediction
```

### Why Stacking Works Mathematically
- **Base models:** Learn f₁(X), f₂(X), f₃(X) from features X
- **Meta-learner:** Learns g(f₁(X), f₂(X), f₃(X)) from base predictions
- **Key insight:** g can learn to weight models differently for different regions of feature space
- **Result:** Lower generalization error than any single base model

---

## Recommendations

### For Model Selection:
1. **Production model:** Use SVMmeta-xgb+nb (best ROC AUC: 0.745)
   - Highest discrimination between wins/losses
   - Good probability calibration for betting/decision-making

2. **Interpretable alternative:** Use LRmeta-logreg+xgb (ROC AUC: 0.737)
   - Easier to explain to stakeholders
   - Meta-weights show relative importance of linear vs non-linear patterns

### For Future Improvements:
1. **Try GaussianNB instead of BernoulliNB**
   - Better fit for continuous PCA components
   - May improve base NB performance

2. **Experiment with fewer PCA components**
   - Try 10-15 components instead of 27 (95% variance)
   - Reducing dimensionality might help XGBoost performance
   - May reduce overfitting

3. **Consider using raw features with XGBoost**
   - Test XGBoost on original features (not PCA)
   - It might outperform LogReg if it can leverage feature interactions
   - Then stack raw-XGB with PCA-LogReg for best of both worlds

4. **Collect more data**
   - Current dataset: ~1,230 games
   - With 5,000+ games, complex models (XGBoost, deep stacking) would improve
   - More data makes ensemble methods even more powerful
