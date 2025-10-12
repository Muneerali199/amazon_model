"""
Phase 9: Quick Strategic Improvements
======================================

Goal: Beat 57.900% leaderboard score
Strategy: Test 3 focused improvements on Phase 5 baseline

Improvements to test:
1. More TF-IDF dimensions (100 → 150) for better text representation
2. Slower learning rate (0.05 → 0.03) for more careful learning
3. Light ensemble (XGBoost + LightGBM) for diversity

Target: 56.5-57.5% SMAPE
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
print("PHASE 9: QUICK STRATEGIC IMPROVEMENTS")
print("=" * 80)
print()
print("Current best: 57.900% LB (Phase 5)")
print("Target: 56.5-57.5% SMAPE")
print()

def smape(y_true, y_pred):
    """SMAPE - CORRECT FORMULA"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)

# ============================================================================
# LOAD DATA
# ============================================================================

print("-" * 80)
print("STEP 1: Load Data")
print("-" * 80)

train_df = pd.read_csv('dataset/train_features.csv')
test_df = pd.read_csv('dataset/test_features.csv')
train_orig = pd.read_csv('dataset/train.csv')
test_orig = pd.read_csv('dataset/test.csv')

print(f"Training: {train_df.shape}")
print(f"Test: {test_df.shape}")
print()

# ============================================================================
# IMPROVEMENT 1: MORE TF-IDF DIMENSIONS (100 → 150)
# ============================================================================

print("-" * 80)
print("STEP 2: Generate TF-IDF Features (IMPROVED: 150 dims)")
print("-" * 80)

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

# More dimensions for better text representation
svd = TruncatedSVD(n_components=150, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

train_tfidf_df = pd.DataFrame(train_svd, columns=[f'tfidf_{i}' for i in range(150)])
test_tfidf_df = pd.DataFrame(test_svd, columns=[f'tfidf_{i}' for i in range(150)])

print(f"TF-IDF features: {train_tfidf_df.shape[1]} (was 100)")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
print()

# ============================================================================
# PREPARE FEATURES
# ============================================================================

print("-" * 80)
print("STEP 3: Prepare Features")
print("-" * 80)

train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')
train_df['item_name'] = train_df['item_name'].fillna('')
test_df['item_name'] = test_df['item_name'].fillna('')

numeric_features = [
    'ipq_value', 'char_count', 'word_count', 'bullet_points',
    'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
    'is_food', 'is_beverage', 'is_grocery', 'is_health',
    'is_personal_care', 'is_household'
]

train_encoded = pd.get_dummies(train_df[['ipq_unit']], prefix='unit', drop_first=True)
test_encoded = pd.get_dummies(test_df[['ipq_unit']], prefix='unit', drop_first=True)

all_columns = list(set(train_encoded.columns) | set(test_encoded.columns))
for col in all_columns:
    if col not in train_encoded.columns:
        train_encoded[col] = 0
    if col not in test_encoded.columns:
        test_encoded[col] = 0

train_encoded = train_encoded[sorted(all_columns)]
test_encoded = test_encoded[sorted(all_columns)]

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

for char in ['[', ']', '<', '>', '{', '}', '"', ':', ',']:
    X_train.columns = X_train.columns.str.replace(char, '_', regex=False)
    X_test.columns = X_test.columns.str.replace(char, '_', regex=False)

# Remove duplicates for LightGBM
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

y_train = train_df['price'].values

print(f"Total features: {X_train.shape[1]} (was 255)")
print(f"Training samples: {X_train.shape[0]:,}")
print()

# ============================================================================
# IMPROVEMENT 2: OPTIMIZED XGBOOST (Slower learning)
# ============================================================================

print("-" * 80)
print("STEP 4: Train XGBoost (IMPROVED: slower learning)")
print("-" * 80)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.03,  # Slower (was 0.05)
    'max_depth': 7,
    'min_child_weight': 2,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.05,
    'reg_alpha': 0.1,
    'reg_lambda': 1.2,
    'n_estimators': 700,  # More trees to compensate for slower learning
    'random_state': 42,
    'tree_method': 'hist',
    'verbosity': 0
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros((len(X_test), 5))
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"  Fold {fold}/5...", end=' ', flush=True)
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    val_pred = model.predict(X_val)
    xgb_oof[val_idx] = val_pred
    xgb_test_preds[:, fold-1] = model.predict(X_test)
    
    fold_smape = smape(y_val, val_pred)
    xgb_scores.append(fold_smape)
    print(f"{fold_smape:.4f}%")

xgb_cv = smape(y_train, xgb_oof)
print(f"\nXGBoost CV: {xgb_cv:.4f}% (±{np.std(xgb_scores):.4f}%)")
print()

# ============================================================================
# IMPROVEMENT 3: ADD LIGHTGBM FOR ENSEMBLE
# ============================================================================

print("-" * 80)
print("STEP 5: Train LightGBM (IMPROVED: ensemble diversity)")
print("-" * 80)

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 50,
    'max_depth': 7,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.1,
    'reg_lambda': 1.2,
    'n_estimators': 700,
    'random_state': 42,
    'verbose': -1
}

lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros((len(X_test), 5))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"  Fold {fold}/5...", end=' ', flush=True)
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    
    val_pred = model.predict(X_val)
    lgb_oof[val_idx] = val_pred
    lgb_test_preds[:, fold-1] = model.predict(X_test)
    
    fold_smape = smape(y_val, val_pred)
    lgb_scores.append(fold_smape)
    print(f"{fold_smape:.4f}%")

lgb_cv = smape(y_train, lgb_oof)
print(f"\nLightGBM CV: {lgb_cv:.4f}% (±{np.std(lgb_scores):.4f}%)")
print()

# ============================================================================
# CREATE WEIGHTED ENSEMBLE
# ============================================================================

print("-" * 80)
print("STEP 6: Create Ensemble")
print("-" * 80)

# Weight by inverse SMAPE
xgb_weight = 1 / xgb_cv
lgb_weight = 1 / lgb_cv
total_weight = xgb_weight + lgb_weight

xgb_w = xgb_weight / total_weight
lgb_w = lgb_weight / total_weight

ensemble_oof = xgb_w * xgb_oof + lgb_w * lgb_oof
ensemble_test = xgb_w * xgb_test_preds.mean(axis=1) + lgb_w * lgb_test_preds.mean(axis=1)
ensemble_cv = smape(y_train, ensemble_oof)

print(f"XGBoost:  {xgb_cv:.4f}% (weight: {xgb_w:.3f})")
print(f"LightGBM: {lgb_cv:.4f}% (weight: {lgb_w:.3f})")
print(f"Ensemble: {ensemble_cv:.4f}%")
print()

# Choose best
best_cv = min(xgb_cv, lgb_cv, ensemble_cv)
if ensemble_cv == best_cv:
    final_preds = ensemble_test
    method = "Ensemble"
elif xgb_cv == best_cv:
    final_preds = xgb_test_preds.mean(axis=1)
    method = "XGBoost"
else:
    final_preds = lgb_test_preds.mean(axis=1)
    method = "LightGBM"

print(f"Best method: {method} ({best_cv:.4f}%)")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("-" * 80)
print("STEP 7: Save Predictions")
print("-" * 80)

submission_df = pd.DataFrame({
    'sample_id': test_orig['sample_id'],
    'price': final_preds
})

output_csv = Path('dataset/test_out.csv')
submission_df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

backup_csv = Path('dataset/submission_phase9.csv')
submission_df.to_csv(backup_csv, index=False)
print(f"✅ Saved backup: {backup_csv}")
print()

print(f"Predictions: {len(final_preds):,}")
print(f"Price range: ${final_preds.min():.2f} - ${final_preds.max():.2f}")
print(f"Mean: ${final_preds.mean():.2f}")
print(f"Median: ${np.median(final_preds):.2f}")
print()

# Save results
results = {
    'phase': 'Phase 9 - Quick Improvements',
    'improvements': [
        'TF-IDF: 100 → 150 dimensions',
        'Learning rate: 0.05 → 0.03',
        'Added LightGBM ensemble'
    ],
    'xgboost_cv': xgb_cv,
    'lightgbm_cv': lgb_cv,
    'ensemble_cv': ensemble_cv,
    'best_cv': best_cv,
    'best_method': method,
    'features': X_train.shape[1],
    'predictions': {
        'count': len(final_preds),
        'min': float(final_preds.min()),
        'max': float(final_preds.max()),
        'mean': float(final_preds.mean()),
        'median': float(np.median(final_preds))
    }
}

results_json = Path('phase9_results.json')
with open(results_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {results_json}")
print()

# ============================================================================
# COMPARISON
# ============================================================================

print("=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)
print()

phase5_cv = 58.38
phase5_lb = 57.900

print(f"Phase 5:")
print(f"  CV: {phase5_cv:.2f}%")
print(f"  LB: {phase5_lb:.3f}% ✅ Current best")
print()

print(f"Phase 9:")
print(f"  CV: {best_cv:.2f}%")
print(f"  Expected LB: {best_cv:.2f}% - {best_cv+1:.2f}%")
print()

if best_cv < phase5_cv:
    improvement = phase5_cv - best_cv
    print(f"✅ IMPROVEMENT: {improvement:.2f} points better CV!")
    print(f"Expected LB: {best_cv:.2f}% - {best_cv+1:.2f}%")
    print(f"If LB follows CV: ~{phase5_lb - improvement:.3f}%")
    print()
    print("🎯 RECOMMEND: Submit this result!")
else:
    print(f"⚠️ No CV improvement")
    print(f"Phase 5 remains best (57.900% LB)")
    print()
    print("Note: Sometimes LB can still be better than CV suggests.")
    print("Your choice: Submit Phase 9 or keep Phase 5.")

print()
print("=" * 80)
