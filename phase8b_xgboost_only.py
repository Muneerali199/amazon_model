"""
Phase 8b: XGBoost Only - Maximum Improvement
=============================================

XGBoost performed well (59.85% CV) but we want better than Phase 5 (58.38% CV)
Let's try more aggressive optimization.

Strategy:
- Use Phase 5 features (proven to work)
- Optimize XGBoost hyperparameters more carefully
- Try multiple parameter sets and pick best
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 8B: XGBOOST OPTIMIZATION")
print("=" * 80)
print()
print("Goal: Beat Phase 5 (58.38% CV = 58.16% LB)")
print("Strategy: Test multiple XGBoost configurations")
print()

def smape(y_true, y_pred):
    """SMAPE - CORRECT FORMULA"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)

# ============================================================================
# LOAD & PREPARE DATA (EXACT SAME AS PHASE 5)
# ============================================================================

print("Loading data...")
train_df = pd.read_csv('dataset/train_features.csv')
test_df = pd.read_csv('dataset/test_features.csv')
train_orig = pd.read_csv('dataset/train.csv')
test_orig = pd.read_csv('dataset/test.csv')

print("Generating TF-IDF features (100 dims, same as Phase 5)...")
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

svd = TruncatedSVD(n_components=100, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

train_tfidf_df = pd.DataFrame(train_svd, columns=[f'tfidf_{i}' for i in range(100)])
test_tfidf_df = pd.DataFrame(test_svd, columns=[f'tfidf_{i}' for i in range(100)])

print("Preparing features...")
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

y_train = train_df['price'].values

print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]:,}")
print()

# ============================================================================
# TEST MULTIPLE XGBOOST CONFIGURATIONS
# ============================================================================

print("=" * 80)
print("TESTING XGBOOST CONFIGURATIONS")
print("=" * 80)
print()

# Configuration 1: Phase 5 Baseline (for comparison)
configs = [
    {
        'name': 'Config 1: Phase 5 Baseline',
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'n_estimators': 500,
            'random_state': 42,
            'tree_method': 'hist'
        }
    },
    {
        'name': 'Config 2: Higher Regularization',
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,  # Shallower
            'min_child_weight': 3,  # More conservative
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.5,  # Higher L1
            'reg_lambda': 2.0,  # Higher L2
            'n_estimators': 600,
            'random_state': 42,
            'tree_method': 'hist'
        }
    },
    {
        'name': 'Config 3: Slow & Careful',
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.02,  # Very slow
            'max_depth': 7,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.05,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
            'n_estimators': 1000,  # More trees
            'random_state': 42,
            'tree_method': 'hist'
        }
    },
    {
        'name': 'Config 4: Deep & Regularized',
        'params': {
            'objective': 'reg:squarederror',
            'learning_rate': 0.04,
            'max_depth': 8,  # Deeper
            'min_child_weight': 5,  # But more regularization
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'gamma': 0.2,
            'reg_alpha': 0.4,
            'reg_lambda': 2.5,
            'n_estimators': 700,
            'random_state': 42,
            'tree_method': 'hist'
        }
    }
]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
best_cv = float('inf')
best_config = None
best_test_preds = None

for config in configs:
    print("-" * 80)
    print(config['name'])
    print("-" * 80)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros((len(X_test), 5))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**config['params'])
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds[:, fold-1] = model.predict(X_test)
        
        fold_smape = smape(y_val, val_pred)
        fold_scores.append(fold_smape)
        print(f"  Fold {fold}/5: {fold_smape:.4f}%")
    
    cv_score = smape(y_train, oof_preds)
    cv_std = np.std(fold_scores)
    
    print(f"  CV SMAPE: {cv_score:.4f}% (±{cv_std:.4f}%)")
    
    if cv_score < best_cv:
        best_cv = cv_score
        best_config = config
        best_test_preds = test_preds.mean(axis=1)
        print(f"  ✅ NEW BEST!")
    
    print()

# ============================================================================
# SAVE BEST MODEL PREDICTIONS
# ============================================================================

print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print()
print(f"Best configuration: {best_config['name']}")
print(f"Best CV: {best_cv:.4f}%")
print()

phase5_cv = 58.38
if best_cv < phase5_cv:
    improvement = phase5_cv - best_cv
    print(f"Phase 5: {phase5_cv:.2f}%")
    print(f"Phase 8b: {best_cv:.2f}%")
    print(f"Improvement: {improvement:.2f}% ✅")
    print(f"Expected leaderboard: {best_cv:.2f}% - {best_cv+2:.2f}%")
else:
    print(f"Phase 5: {phase5_cv:.2f}% ✅ Still the best")
    print(f"Phase 8b: {best_cv:.2f}%")
    print(f"⚠️ No improvement - use Phase 5 submission instead")
print()

# Save submission
submission_df = pd.DataFrame({
    'sample_id': test_orig['sample_id'],
    'price': best_test_preds
})

output_csv = Path('dataset/test_out.csv')
submission_df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

backup_csv = Path('dataset/submission_phase8b.csv')
submission_df.to_csv(backup_csv, index=False)
print(f"✅ Saved backup: {backup_csv}")
print()

print(f"Predictions: {len(best_test_preds):,}")
print(f"Price range: ${best_test_preds.min():.2f} - ${best_test_preds.max():.2f}")
print(f"Mean: ${best_test_preds.mean():.2f}")
print(f"Median: ${np.median(best_test_preds):.2f}")
print()

# Save results
results = {
    'phase': 'Phase 8b - XGBoost Optimization',
    'best_cv': best_cv,
    'best_config': best_config['name'],
    'best_params': best_config['params'],
    'phase5_cv': phase5_cv,
    'improvement': phase5_cv - best_cv,
    'predictions': {
        'count': len(best_test_preds),
        'min': float(best_test_preds.min()),
        'max': float(best_test_preds.max()),
        'mean': float(best_test_preds.mean()),
        'median': float(np.median(best_test_preds))
    }
}

results_json = Path('phase8b_results.json')
with open(results_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {results_json}")
print()
print("=" * 80)
