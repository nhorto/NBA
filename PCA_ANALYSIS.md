# PCA Analysis: PC1, PC2, and PC3 Loadings Interpretation

## Overview

This document analyzes the principal component loadings from PCA performed on NBA game prediction features. The analysis focuses on understanding what patterns PC1, PC2, and PC3 capture and how these insights can inform model building decisions.

---

## PC1 Analysis: "Season Progression / Game Volume"

### Key Pattern
PC1 is almost entirely dominated by **cumulative statistics** (all negative loadings ~-0.16 to -0.17)

### Top Contributors (all negative)
- FGA, POSS, REB, PTS, FGM (cumulative for both home/away)
- All counting stats accumulated over the season
- Player availability metrics (missing_impact, missing_players)

### What This Means
- PC1 is essentially a **time-in-season axis**
- Games early in season → low cumulative stats → positive PC1 values
- Games late in season → high cumulative stats → negative PC1 values
- The rolling stats show slight positive loadings (PACE, POSS, FTA) as a counter-balance

### Impact on Your Model
This explains **29.3% of variance** but **doesn't tell you much about who wins**. It's mostly capturing when in the season a game occurs, not team quality.

---

## PC2 Analysis: "Away Team Quality" (THIS IS YOUR GOLDMINE)

### Key Pattern
PC2 captures a **clear quality gradient for the away team**

### Strong Negative Loadings (= Strong Away Team)
- `NET_RATING_rolling_away` (-0.262)
- `PLUS_MINUS_rolling_away` (-0.262)
- `E_NET_RATING_rolling_away` (-0.262)
- `E_OFF_RATING_rolling_away` (-0.259)
- `PIE_rolling_away` (-0.259)
- `TS_PCT_rolling_away`, `EFG_PCT_rolling_away` (shooting efficiency)
- `AWAY_ELO_PRE` (-0.221)

### Strong Positive Loadings (= Weak Away Team / Strong Home Advantage)
- `HOME_WIN_PROB` (+0.168)
- `missing_players_away` (+0.139)
- `missing_impact_away` (+0.131)
- `TM_TOV_PCT_rolling_away` (+0.126) - away team turnovers
- `DEF_RATING_rolling_away` (+0.099) - worse defense

### What This Means
- **When PC2 is negative** → Strong away team (good ratings, efficiency, Elo)
- **When PC2 is positive** → Weak/depleted away team or strong home advantage
- This explains **10.1% of variance** and is **directly predictive** of game outcomes

---

## PC3 Analysis: "Home Team Quality" (Perfect Mirror of PC2)

### Key Pattern
PC3 captures a **clear quality gradient for the home team** - this is the exact mirror image of what PC2 did for the away team!

### Strong Negative Loadings (= Strong Home Team)
- `E_NET_RATING_rolling_home` (-0.266)
- `PLUS_MINUS_rolling_home` (-0.266)
- `NET_RATING_rolling_home` (-0.266)
- `E_OFF_RATING_rolling_home` (-0.265)
- `OFF_RATING_rolling_home` (-0.263)
- `PIE_rolling_home` (-0.262)
- `TS_PCT_rolling_home`, `EFG_PCT_rolling_home` (shooting efficiency)
- `HOME_ELO_PRE` (-0.210)
- `HOME_WIN_PROB` (-0.149)

### Strong Positive Loadings (= Weak Home Team)
- `missing_players_home` (+0.120)
- `missing_impact_home` (+0.112)
- `TM_TOV_PCT_rolling_home` (+0.112) - home team turnovers
- `DEF_RATING_rolling_home` (+0.094) - worse defense
- `E_DEF_RATING_rolling_home` (+0.096)
- `missing_star_home` (+0.087)

### What This Means
- **When PC3 is negative** → Strong home team (good ratings, efficiency, Elo)
- **When PC3 is positive** → Weak/depleted home team
- This explains **9.4% of variance** and is **directly predictive** of game outcomes

### Critical Insight: PC2 + PC3 = Complete Team Quality Picture
**Together, PC2 and PC3 explain ~19.5% of variance** and capture the complete matchup:
- **PC2 (10.1%)** = Away team quality
- **PC3 (9.4%)** = Home team quality
- The fact that PC2 explains slightly more variance than PC3 suggests **away team quality varies more** than home team quality, which makes sense given home court advantage provides a baseline boost to all home teams

---

## Key Insights for Model Building

### 1. Feature Redundancy is Real
Your loadings show that:
- Cumulative stats are basically all measuring the same thing (season progression)
- Advanced metrics (NET_RATING, PIE, OFF_RATING) are highly correlated
- PCA has successfully collapsed ~120 features down to 27 components while keeping 95% variance

### 2. Perfect Home/Away Symmetry in PC2 and PC3
PC2 and PC3 form a perfect pair:
- **PC2** = Away team quality (10.1% variance)
- **PC3** = Home team quality (9.4% variance)
- Together they explain nearly 20% of variance with almost identical feature patterns
- The slight advantage of PC2 suggests away team quality is more variable/important
- This symmetry validates your feature engineering approach

### 3. Rolling Stats > Cumulative Stats
For rate/efficiency metrics:
- Rolling windows capture **current form**
- Cumulative versions just measure **games played**
- You were right to filter out cumulative versions of rate stats

### 4. Player Availability Matters
All three components show missing player metrics:
- **PC1**: General injury impact across season progression
- **PC2**: Away team injuries specifically boost home win probability
- **PC3**: Home team injuries hurt home win probability
- These features are clearly pulling their weight in all major components

---

## Actionable Recommendations

### For Your Current Model

1. **Don't drop cumulative stats entirely** - They help the model understand "cold start" early season
2. **Consider interaction features** between home/away quality metrics
3. **Try models that handle correlated features well** (Ridge, XGBoost, Random Forest)

### For Feature Engineering

#### 1. Create explicit home vs. away differentials:
```python
df['NET_RATING_diff'] = df['NET_RATING_rolling_home'] - df['NET_RATING_rolling_away']
df['ELO_diff'] = df['HOME_ELO_PRE'] - df['AWAY_ELO_PRE']
df['OFF_RATING_diff'] = df['OFF_RATING_rolling_home'] - df['OFF_RATING_rolling_away']
df['DEF_RATING_diff'] = df['DEF_RATING_rolling_home'] - df['DEF_RATING_rolling_away']
df['TS_PCT_diff'] = df['TS_PCT_rolling_home'] - df['TS_PCT_rolling_away']
```

#### 2. Emphasize recent form over cumulative:
- Your 5-game rolling window is good
- Maybe try 3-game for "hot streaks" vs 10-game for "true talent"
- Consider exponentially weighted moving averages for smoother trends

#### 3. Home/away splits matter:
- Consider team performance specifically in home/away contexts
- Some teams travel poorly
- Back-to-back games, time zone changes, etc.

### For Model Selection

Your current results show:
- **Logistic Regression**: 66% accuracy, 0.72 AUC ✅
- **Beating ELO baseline** (62.6%) and home team baseline (57.2%)
- **PCA is helping** by reducing noise and multicollinearity

#### Potential Improvements:

1. **Try using just PC2-PC10** (drop PC1 which is mostly time)
   - PC1 might be adding noise rather than signal for outcome prediction
   - PC2 and PC3 are your most predictive components - they capture team quality directly

2. **Use PCA components as features in tree-based models**
   - Random Forest
   - XGBoost with PCA features
   - These can capture non-linear relationships in the principal component space

3. **Or skip PCA entirely** and use:
   - **Regularization** (Ridge/Lasso) with differential features
   - **Feature selection** based on PC2 loadings (focus on what matters)
   - Tree-based models naturally handle multicollinearity

---

## Bottom Line

Your PCA has revealed a beautiful structure:
- **PC1 (29.3%)**: Season progression / game volume (not predictive of wins)
- **PC2 (10.1%)**: Away team quality (highly predictive)
- **PC3 (9.4%)**: Home team quality (highly predictive)

Together, **PC2 and PC3 explain ~19.5% of variance** and capture the complete team quality matchup that determines game outcomes.

**Key Takeaway:** Build features that capture home vs. away quality differentials and focus on rolling windows over cumulative stats. The symmetry between PC2 and PC3 validates that your features are well-balanced between home and away perspectives.

### Next Steps:
1. Create differential features (home - away) for key metrics
2. Experiment with dropping PC1 from your model
3. Test tree-based models on PCA-transformed data
4. Consider rolling window variations (3-game, 10-game)
5. Add interaction terms between player availability and team quality metrics

---

## Current Model Performance Summary

| Model | Accuracy | ROC AUC | Notes |
|-------|----------|---------|-------|
| Always Home Wins | 57.2% | - | Baseline |
| Higher ELO Wins | 62.6% | - | Baseline |
| Logistic Regression (PCA) | 66.0% | 0.72 | Best so far |
| SVM (PCA) | 65.8% | 0.70 | Similar to LogReg |
| XGBoost (PCA) | 63.8% | 0.69 | Underperforming |
| Naive Bayes (PCA) | 64.2% | 0.70 | Decent |

The models are all performing reasonably well and beating the baselines, which validates that:
- Your feature engineering is working
- PCA is reducing noise effectively
- There's room for improvement with better feature engineering
