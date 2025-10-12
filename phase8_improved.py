"""
Phase 8: Strategic Improvement on Phase 5
==========================================

Goal: Beat 58.16% leaderboard score
Strategy:
1. Start from Phase 5 (58.38% CV = 58.16% LB) ✅ Working baseline
2. Add careful improvements:
   - Better hyperparameters (deeper search)
   - Light ensemble (XGBoost + LightGBM only)
   - More TF-IDF dimensions (100 → 150)
   - Better regularization

Target: 55-57% CV → 55-57% LB (improvement of 1-3%)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 8: STRATEGIC IMPROVEMENT ON PHASE 5")
print("=" * 80)
print()
print("Building on proven Phase 5 approach (58.38% CV = 58.16% LB)")
print("Target: 55-57% SMAPE")
print()

# ============================================================================
# SMAPE METRIC (CORRECT FORMULA!)
# ============================================================================

def smape(y_true, y_pred):
    """Calculate SMAPE - CORRECT FORMULA with 200*"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)  # CORRECT!

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("-" * 80)
print("STEP 1: Load Data")
print("-" * 80)
print()

train_df = pd.read_csv('dataset/train_features.csv')
test_df = pd.read_csv('dataset/test_features.csv')
train_orig = pd.read_csv('dataset/train.csv')
test_orig = pd.read_csv('dataset/test.csv')

print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}")
print()

# ============================================================================
# STEP 2: GENERATE IMPROVED TF-IDF FEATURES
# ============================================================================

print("-" * 80)
print("STEP 2: Generate TF-IDF Features (Improved)")
print("-" * 80)
print()

# Increased dimensions: 100 → 150 for better text representation
print("Creating TF-IDF features (150 dimensions)...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    strip_accents='unicode',
    lowercase=True
)

train_texts = train_orig['catalog_content'].fillna('')
test_texts = test_orig['catalog_content'].fillna('')

train_tfidf = tfidf.fit_transform(train_texts)
test_tfidf = tfidf.transform(test_texts)

print("Reducing with SVD to 150 dimensions...")
svd = TruncatedSVD(n_components=150, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

train_tfidf_df = pd.DataFrame(train_svd, columns=[f'tfidf_{i}' for i in range(150)])
test_tfidf_df = pd.DataFrame(test_svd, columns=[f'tfidf_{i}' for i in range(150)])

print(f"TF-IDF features: {train_tfidf_df.shape[1]}")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
print()

# ============================================================================
# STEP 3: PREPARE FEATURES (SAME AS PHASE 5)
# ============================================================================

print("-" * 80)
print("STEP 3: Prepare Features")
print("-" * 80)
print()

# Handle missing values
train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')
train_df['item_name'] = train_df['item_name'].fillna('')
test_df['item_name'] = test_df['item_name'].fillna('')

# Numeric features from Phase 5
numeric_features = [
    'ipq_value', 'char_count', 'word_count', 'bullet_points',
    'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
    'is_food', 'is_beverage', 'is_grocery', 'is_health',
    'is_personal_care', 'is_household'
]

# One-hot encode ipq_unit
train_encoded = pd.get_dummies(train_df[['ipq_unit']], prefix='unit', drop_first=True)
test_encoded = pd.get_dummies(test_df[['ipq_unit']], prefix='unit', drop_first=True)

# Align columns
all_columns = list(set(train_encoded.columns) | set(test_encoded.columns))
for col in all_columns:
    if col not in train_encoded.columns:
        train_encoded[col] = 0
    if col not in test_encoded.columns:
        test_encoded[col] = 0

train_encoded = train_encoded[sorted(all_columns)]
test_encoded = test_encoded[sorted(all_columns)]

# Combine all features
X_train = pd.concat([
    train_df[numeric_features].reset_index(drop=True),
    train_encoded.reset_index(drop=True),
    train_tfidf_df.reset_index(drop=True)
], axis=1)

X_test = pd.concat([
    test_df[numeric_features].reset_index(drop=True),
    test_encoded.reset_index(drop=True),
    test_tfidf_df.reset_index(drop=True)
], axis=1)

# Clean column names
for char in ['[', ']', '<', '>', '{', '}', '"', ':', ',']:
    X_train.columns = X_train.columns.str.replace(char, '_', regex=False)
    X_test.columns = X_test.columns.str.replace(char, '_', regex=False)

# Remove duplicate columns (for LightGBM compatibility)
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

y_train = train_df['price'].values

print(f"Total features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print()

# ============================================================================
# STEP 4: TRAIN XGBOOST (IMPROVED PARAMETERS)
# ============================================================================

print("-" * 80)
print("STEP 4: Train XGBoost (Improved Parameters)")
print("-" * 80)
print()

# Better hyperparameters based on Phase 5 + optimization
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.03,  # Slower = more careful
    'max_depth': 7,  # Balanced depth
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.1,
    'reg_alpha': 0.3,  # L1 regularization
    'reg_lambda': 1.5,  # L2 regularization
    'n_estimators': 800,  # More trees with slow learning
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros((len(X_test), 5))
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold}/5 (XGBoost)...")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_pred = model.predict(X_val)
    xgb_oof[val_idx] = val_pred
    xgb_test_preds[:, fold-1] = model.predict(X_test)
    
    fold_smape = smape(y_val, val_pred)
    xgb_scores.append(fold_smape)
    print(f"   SMAPE: {fold_smape:.4f}%")

xgb_cv = smape(y_train, xgb_oof)
print()
print(f"XGBoost CV SMAPE: {xgb_cv:.4f}% (±{np.std(xgb_scores):.4f}%)")
print()

# ============================================================================
# STEP 5: TRAIN LIGHTGBM (ENSEMBLE)
# ============================================================================

print("-" * 80)
print("STEP 5: Train LightGBM (Light Ensemble)")
print("-" * 80)
print()

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 60,  # Balanced
    'max_depth': 7,
    'min_child_samples': 25,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'n_estimators': 800,
    'random_state': 42,
    'verbose': -1
}

lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros((len(X_test), 5))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold}/5 (LightGBM)...")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    val_pred = model.predict(X_val)
    lgb_oof[val_idx] = val_pred
    lgb_test_preds[:, fold-1] = model.predict(X_test)
    
    fold_smape = smape(y_val, val_pred)
    lgb_scores.append(fold_smape)
    print(f"   SMAPE: {fold_smape:.4f}%")

lgb_cv = smape(y_train, lgb_oof)
print()
print(f"LightGBM CV SMAPE: {lgb_cv:.4f}% (±{np.std(lgb_scores):.4f}%)")
print()

# ============================================================================
# STEP 6: CREATE ENSEMBLE
# ============================================================================

print("-" * 80)
print("STEP 6: Create Weighted Ensemble")
print("-" * 80)
print()

# Weight by inverse SMAPE (better model gets more weight)
xgb_weight = 1 / xgb_cv
lgb_weight = 1 / lgb_cv
total_weight = xgb_weight + lgb_weight

xgb_w = xgb_weight / total_weight
lgb_w = lgb_weight / total_weight

print(f"XGBoost weight: {xgb_w:.3f}")
print(f"LightGBM weight: {lgb_w:.3f}")
print()

# Ensemble predictions
ensemble_oof = xgb_w * xgb_oof + lgb_w * lgb_oof
ensemble_test = xgb_w * xgb_test_preds.mean(axis=1) + lgb_w * lgb_test_preds.mean(axis=1)

ensemble_cv = smape(y_train, ensemble_oof)
print(f"Ensemble CV SMAPE: {ensemble_cv:.4f}%")
print()

# ============================================================================
# STEP 7: GENERATE SUBMISSION
# ============================================================================

print("-" * 80)
print("STEP 7: Generate Submission")
print("-" * 80)
print()

# Use best single model or ensemble
if ensemble_cv < min(xgb_cv, lgb_cv):
    print("Using ENSEMBLE predictions (best CV)")
    final_predictions = ensemble_test
    final_cv = ensemble_cv
    method = "Ensemble"
else:
    print("Using best SINGLE MODEL predictions")
    if xgb_cv < lgb_cv:
        final_predictions = xgb_test_preds.mean(axis=1)
        final_cv = xgb_cv
        method = "XGBoost"
    else:
        final_predictions = lgb_test_preds.mean(axis=1)
        final_cv = lgb_cv
        method = "LightGBM"

print(f"Selected method: {method}")
print(f"CV Score: {final_cv:.4f}%")
print()

# Create submission
submission_df = pd.DataFrame({
    'sample_id': test_orig['sample_id'],
    'price': final_predictions
})

# Save submission
output_csv = Path('dataset/test_out.csv')
submission_df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

# Also save as phase8 backup
backup_csv = Path('dataset/submission_phase8.csv')
submission_df.to_csv(backup_csv, index=False)
print(f"✅ Saved backup: {backup_csv}")
print()

# Prediction stats
print(f"Predictions: {len(final_predictions):,}")
print(f"Price range: ${final_predictions.min():.2f} - ${final_predictions.max():.2f}")
print(f"Mean: ${final_predictions.mean():.2f}")
print(f"Median: ${np.median(final_predictions):.2f}")
print()

# Save results
results = {
    'phase': 'Phase 8 - Strategic Improvement',
    'xgboost_cv': xgb_cv,
    'xgboost_std': np.std(xgb_scores),
    'lightgbm_cv': lgb_cv,
    'lightgbm_std': np.std(lgb_scores),
    'ensemble_cv': ensemble_cv,
    'final_cv': final_cv,
    'final_method': method,
    'features': {
        'baseline': len(numeric_features),
        'encoded': len(all_columns),
        'tfidf': 150,
        'total': X_train.shape[1]
    },
    'predictions': {
        'count': len(final_predictions),
        'min': float(final_predictions.min()),
        'max': float(final_predictions.max()),
        'mean': float(final_predictions.mean()),
        'median': float(np.median(final_predictions))
    }
}

results_json = Path('phase8_results.json')
with open(results_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {results_json}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 8 COMPLETE!")
print("=" * 80)
print()
print(f"XGBoost CV:  {xgb_cv:.4f}% (±{np.std(xgb_scores):.4f}%)")
print(f"LightGBM CV: {lgb_cv:.4f}% (±{np.std(lgb_scores):.4f}%)")
print(f"Ensemble CV: {ensemble_cv:.4f}%")
print()
print(f"Final Method: {method}")
print(f"Final CV: {final_cv:.4f}%")
print()

# Comparison with Phase 5
phase5_cv = 58.38
improvement = phase5_cv - final_cv
print(f"Phase 5 baseline: {phase5_cv:.2f}%")
print(f"Phase 8 result:   {final_cv:.2f}%")
if improvement > 0:
    print(f"Improvement: {improvement:.2f}% ✅")
    print(f"Expected leaderboard: {final_cv:.2f}% - {final_cv+2:.2f}%")
else:
    print(f"Change: {improvement:.2f}%")
    print("⚠️ No improvement - consider using Phase 5 submission")
print()
print(f"Submit: {output_csv}")
print()
print("=" * 80)
