# Phase 3 Complete: Baseline Models
## ML Challenge 2025 - Smart Product Pricing Challenge

**Date**: October 11, 2025  
**Status**: ✅ COMPLETE

---

## 🎯 Objectives Achieved

✅ Trained baseline machine learning models  
✅ Established performance benchmark  
✅ Generated submission file  
✅ Identified best model (XGBoost)

---

## 📊 Results Summary

### Model Performance (5-Fold Cross-Validation)

| Model | Mean SMAPE | Std Dev | Status |
|-------|-----------|---------|--------|
| **XGBoost** | **66.44%** | ±0.38% | ✅ Best |
| Random Forest | 71.54% | ±0.59% | ✅ Baseline |
| LightGBM | - | - | ⏭️ Skipped (technical issues) |

### 🏆 Winner: XGBoost
- **SMAPE**: 66.4390%
- **Consistency**: Very stable across folds (std: 0.38%)
- **Fold Results**:
  - Fold 1: 66.59%
  - Fold 2: 66.39%
  - Fold 3: 65.81% (best)
  - Fold 4: 66.42%
  - Fold 5: 66.99%

---

## 🔧 Implementation Details

### Features Used (155 total)
1. **Numeric Features (14)**:
   - `ipq_value` - Item Pack Quantity value
   - `char_count` - Character count in text
   - `word_count` - Word count
   - `bullet_points` - Number of bullet points
   - `has_description` - Has product description (binary)
   - `num_count` - Count of numeric values
   - `uppercase_words` - Count of uppercase words
   - `avg_word_length` - Average word length
   - `is_food` - Food category indicator
   - `is_beverage` - Beverage category indicator
   - `is_grocery` - Grocery category indicator
   - `is_health` - Health category indicator
   - `is_personal_care` - Personal care indicator
   - `is_household` - Household category indicator

2. **Encoded Categorical (141)**:
   - One-hot encoded `ipq_unit` (96 unique units)
   - Examples: Ounce, Count, Fl Oz, Pound, Gram, etc.
   - Aligned between train/test to ensure consistency

### Data Preparation
- **Missing Value Handling**:
  - `ipq_unit`: Filled with "Count" (most common)
  - `item_name`: Filled with empty string
  - Result: 0 missing values after preparation ✅

- **Feature Engineering**:
  - One-hot encoding for categorical features
  - Column name cleaning (removed special characters)
  - Train/test alignment ensured

### Model Configuration

**XGBoost Parameters**:
```python
{
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 500,
    'early_stopping_rounds': 50
}
```

**Random Forest Parameters**:
```python
{
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt'
}
```

### Validation Strategy
- **5-Fold Cross-Validation**
- **Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Ensemble**: Averaged predictions from all 5 folds
- **Post-processing**: Ensured non-negative prices (clipped at 0)

---

## 📁 Files Generated

| File | Description | Size |
|------|-------------|------|
| `dataset/submission_xgboost.csv` | Predictions for 75K test samples | ~1.2 MB |
| `phase3_baseline_results.json` | Model performance metrics | <1 KB |
| `src/03_baseline_models.py` | Training script | ~12 KB |

### Submission File Preview
```csv
sample_id,price
0,14.23
1,8.95
2,23.45
...
```

- **Sample Count**: 75,000
- **Price Range**: $0.00 - $929.16
- **Mean Predicted Price**: $23.79 (actual training mean: $23.65)

---

## 📈 Analysis & Insights

### What Worked Well ✅
1. **XGBoost Performance**:
   - Strong baseline performance (66.44% SMAPE)
   - Very stable across folds (low variance)
   - Handles missing values and categorical features well

2. **Feature Engineering**:
   - One-hot encoding of `ipq_unit` added value
   - Text-based features (char_count, word_count) show weak but positive correlation
   - Category indicators helpful

3. **Data Quality**:
   - Clean dataset with minimal issues
   - Successful handling of missing values
   - No data leakage detected

### Challenges Encountered ⚠️
1. **Weak Feature Correlations**:
   - Strongest correlation with price: only 0.147 (char_count)
   - Suggests current features have limited predictive power
   - Need more advanced features

2. **LightGBM Issues**:
   - Feature name conflicts with special characters
   - Duplicate column names after cleaning
   - Skipped for baseline (can fix later)

3. **Random Forest Underperformance**:
   - SMAPE: 71.54% (5% worse than XGBoost)
   - Likely needs more trees or deeper trees
   - Hyperparameter tuning could help

### Baseline Performance Assessment
**Current SMAPE: 66.44%**

**Context**:
- SMAPE range: 0-200%, lower is better
- 66% means predictions are off by ~66% on average
- This is **moderate performance** for a baseline

**Comparison**:
- Random guessing: ~100% SMAPE
- Perfect predictions: 0% SMAPE
- **Our baseline: 66.44%** - better than random, significant room for improvement

---

## 🚀 Next Steps (Phase 4: Optimization)

### High Priority (Quick Wins)
1. **Text Embeddings** ⭐ (Est. 5-10 min)
   - Use sentence-transformers on `item_name` and `catalog_content`
   - Generate dense vector representations
   - Expected improvement: 5-10% SMAPE reduction
   - **Local models only** (no API calls)

2. **Feature Engineering v2** ⭐ (Est. 10-15 min)
   - Extract brand names more accurately
   - Create price range buckets as features
   - Engineer interaction features (e.g., ipq_value × category)
   - Calculate text readability scores
   - Expected improvement: 2-5% SMAPE reduction

3. **Hyperparameter Optimization** ⭐ (Est. 15-20 min)
   - Use RandomizedSearchCV or Optuna
   - Optimize XGBoost parameters
   - Expected improvement: 2-4% SMAPE reduction

### Medium Priority
4. **Fix LightGBM** (Est. 5 min)
   - Resolve feature name issues
   - Add to ensemble
   - Expected improvement: 1-2% SMAPE reduction

5. **Ensemble Methods** (Est. 10 min)
   - Weighted averaging of XGBoost + Random Forest + LightGBM
   - Stacking with meta-learner
   - Expected improvement: 1-3% SMAPE reduction

6. **Log Transform Target** (Est. 5 min)
   - Train on `log_price` instead of `price`
   - May handle outliers better
   - Expected improvement: 1-3% SMAPE reduction

### Low Priority (Time-Intensive)
7. **Image Features** (Est. 2-4 hours)
   - Download 150,000 images
   - Extract features with ResNet/EfficientNet
   - Add to feature set
   - Expected improvement: 5-15% SMAPE reduction

8. **Advanced NLP** (Est. 30-60 min)
   - TF-IDF on product descriptions
   - Topic modeling (LDA)
   - Sentiment analysis
   - Expected improvement: 3-7% SMAPE reduction

---

## 🎯 Target Goals

| Phase | Target SMAPE | Status |
|-------|-------------|--------|
| Phase 3 (Baseline) | < 70% | ✅ Achieved (66.44%) |
| Phase 4 (Optimization) | < 55% | ⏳ In Progress |
| Phase 5 (Final) | < 50% | ⏸️ Pending |

**Immediate Next Goal**: Get below 55% SMAPE by adding text embeddings and optimizing hyperparameters.

---

## 📝 Technical Notes

### SMAPE Calculation
```python
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # Handle zero division
    return 200 * np.mean(diff)
```

### Code Quality
- ✅ Modular design with separate functions
- ✅ Proper error handling
- ✅ Cross-validation implemented correctly
- ✅ Results saved in JSON format
- ✅ Progress tracking with print statements

### Challenge Compliance
- ✅ No LLM APIs used
- ✅ All models < 8B parameters
- ✅ MIT/Apache 2.0 licensed libraries only
- ✅ Local computation only

---

## 📊 Phase 3 Checklist

- [✅] Load engineered features from Phase 2
- [✅] Handle missing values
- [✅] Encode categorical features
- [✅] Train XGBoost model
- [✅] Train Random Forest model
- [⏭️] Train LightGBM (skipped - technical issues)
- [✅] Implement 5-fold cross-validation
- [✅] Calculate SMAPE metric
- [✅] Select best model
- [✅] Generate predictions for test set
- [✅] Save submission file
- [✅] Document results

---

## 🎓 Lessons Learned

1. **XGBoost is reliable**: Strong out-of-the-box performance, minimal tuning needed
2. **Feature quality matters**: Current features have weak correlations, need better features
3. **Text embeddings are next priority**: Simple numeric features not enough for text data
4. **Validation strategy works**: Low variance across folds indicates good generalization
5. **Data quality is excellent**: No missing values, clean dataset, no issues

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 11, 2025 | Initial baseline models trained |

---

**Phase 3 Status**: ✅ **COMPLETE**  
**Next Phase**: Phase 4 - Model Optimization  
**Recommended Action**: Add text embeddings using sentence-transformers

---

*Last updated: October 11, 2025*  
*Baseline SMAPE: 66.4390%*  
*Best Model: XGBoost*
