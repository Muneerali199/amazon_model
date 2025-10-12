"""
FINAL PUSH - 3 Hour Deadline Strategy
======================================

Current: 57.900% (stuck after 2 submissions)
Time: 3 hours
Goal: Beat 57.900% with aggressive but smart changes

Strategy:
1. Feature selection (remove noisy features)
2. Different random seed (sometimes makes big difference)
3. Optimized hyperparameters (tested combinations)
4. Quick ensemble (best 2-3 models)

Target: 55-57% SMAPE
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚨 FINAL PUSH - 3 HOUR DEADLINE")
print("=" * 80)
print()
print("Current: 57.900% SMAPE")
print("Target: 55-57% SMAPE")
print("Time: ~30 minutes training")
print()

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)

# ============================================================================
# LOAD DATA
# ============================================================================

print("-" * 80)
print("Loading data...")
print("-" * 80)

train_df = pd.read_csv('dataset/train_features.csv')
test_df = pd.read_csv('dataset/test_features.csv')
train_orig = pd.read_csv('dataset/train.csv')
test_orig = pd.read_csv('dataset/test.csv')

# ============================================================================
# STRATEGY 1: OPTIMAL TF-IDF (120 dims - middle ground)
# ============================================================================

print("\n" + "-" * 80)
print("Creating TF-IDF features (120 dims)...")
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

svd = TruncatedSVD(n_components=120, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

train_tfidf_df = pd.DataFrame(train_svd, columns=[f'tfidf_{i}' for i in range(120)])
test_tfidf_df = pd.DataFrame(test_svd, columns=[f'tfidf_{i}' for i in range(120)])

print(f"TF-IDF: 120 dimensions (between 100 and 150)")

# ============================================================================
# PREPARE BASE FEATURES
# ============================================================================

print("\n" + "-" * 80)
print("Preparing features...")
print("-" * 80)

train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')

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

X_train_full = pd.concat([
    train_df[numeric_features].reset_index(drop=True),
    train_encoded.reset_index(drop=True),
    train_tfidf_df.reset_index(drop=True)
], axis=1)

X_test_full = pd.concat([
    test_df[numeric_features].reset_index(drop=True),
    test_encoded.reset_index(drop=True),
    test_tfidf_df.reset_index(drop=True)
], axis=1)

y_train = train_df['price'].values

# ============================================================================
# STRATEGY 2: FEATURE SELECTION (Remove noisy features)
# ============================================================================

print("\n" + "-" * 80)
print("Feature selection (keep best 200 features)...")
print("-" * 80)

selector = SelectKBest(f_regression, k=200)
X_train = selector.fit_transform(X_train_full, y_train)
X_test = selector.transform(X_test_full)

selected_features = X_train_full.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} best features from {X_train_full.shape[1]}")

# ============================================================================
# STRATEGY 3: TEST MULTIPLE CONFIGS WITH DIFFERENT SEEDS
# ============================================================================

print("\n" + "=" * 80)
print("Testing 4 optimized configurations...")
print("=" * 80)

configs = [
    {
        'name': 'Config 1: Optimal Depth',
        'params': {
            'learning_rate': 0.04,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.1,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'n_estimators': 600,
            'random_state': 42
        }
    },
    {
        'name': 'Config 2: Different Seed',
        'params': {
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 500,
            'random_state': 123  # Different seed!
        }
    },
    {
        'name': 'Config 3: Conservative',
        'params': {
            'learning_rate': 0.03,
            'max_depth': 5,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'n_estimators': 800,
            'random_state': 42
        }
    },
    {
        'name': 'Config 4: Aggressive',
        'params': {
            'learning_rate': 0.06,
            'max_depth': 8,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0.0,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'n_estimators': 400,
            'random_state': 42
        }
    }
]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

for config in configs:
    print("\n" + "-" * 80)
    print(config['name'])
    print("-" * 80)
    
    params = config['params'].copy()
    params['objective'] = 'reg:squarederror'
    params['tree_method'] = 'hist'
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros((len(X_test), 5))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds[:, fold-1] = model.predict(X_test)
        
        fold_smape = smape(y_val, val_pred)
        fold_scores.append(fold_smape)
        print(f"  Fold {fold}: {fold_smape:.4f}%")
    
    cv_score = smape(y_train, oof_preds)
    print(f"  CV: {cv_score:.4f}% (±{np.std(fold_scores):.4f}%)")
    
    all_results.append({
        'name': config['name'],
        'cv': cv_score,
        'oof': oof_preds,
        'test': test_preds.mean(axis=1),
        'params': params
    })

# ============================================================================
# STRATEGY 4: ENSEMBLE TOP MODELS
# ============================================================================

print("\n" + "=" * 80)
print("Creating ensemble of best models...")
print("=" * 80)

# Sort by CV score
all_results.sort(key=lambda x: x['cv'])

# Take top 3 models
top_3 = all_results[:3]

print("\nTop 3 models:")
for i, result in enumerate(top_3, 1):
    print(f"  {i}. {result['name']}: {result['cv']:.4f}%")

# Weighted ensemble
weights = [1/r['cv'] for r in top_3]
total_weight = sum(weights)
weights = [w/total_weight for w in weights]

ensemble_oof = sum(w * r['oof'] for w, r in zip(weights, top_3))
ensemble_test = sum(w * r['test'] for w, r in zip(weights, top_3))
ensemble_cv = smape(y_train, ensemble_oof)

print(f"\nEnsemble CV: {ensemble_cv:.4f}%")

# Choose best: single model or ensemble
best_single = all_results[0]
if ensemble_cv < best_single['cv']:
    print(f"✅ Using ENSEMBLE (better than best single)")
    final_preds = ensemble_test
    final_cv = ensemble_cv
    method = "Ensemble (Top 3)"
else:
    print(f"✅ Using SINGLE MODEL: {best_single['name']}")
    final_preds = best_single['test']
    final_cv = best_single['cv']
    method = best_single['name']

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(f"\nBest Method: {method}")
print(f"CV Score: {final_cv:.4f}%")
print()

phase5_cv = 58.38
phase5_lb = 57.900

print(f"Phase 5: {phase5_cv:.2f}% CV → {phase5_lb:.3f}% LB")
print(f"Final Push: {final_cv:.2f}% CV → Expected {final_cv-0.5:.2f}%-{final_cv+0.5:.2f}% LB")
print()

if final_cv < phase5_cv:
    improvement = phase5_cv - final_cv
    print(f"✅ IMPROVEMENT: {improvement:.2f} points better!")
    print(f"🎯 Expected LB: ~{phase5_lb - improvement:.3f}%")
    print("\n🚀 SUBMIT THIS!")
else:
    print(f"⚠️ CV not better, but might still improve on LB")
    print("Your choice to submit or not")

print()

# Save submission
submission_df = pd.DataFrame({
    'sample_id': test_orig['sample_id'],
    'price': final_preds
})

output_csv = Path('dataset/test_out.csv')
submission_df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

backup_csv = Path('dataset/submission_final_push.csv')
submission_df.to_csv(backup_csv, index=False)
print(f"✅ Saved backup: {backup_csv}")

# Save results
results = {
    'method': method,
    'cv_score': final_cv,
    'all_configs': [{'name': r['name'], 'cv': r['cv']} for r in all_results],
    'ensemble_cv': ensemble_cv,
    'predictions': {
        'count': len(final_preds),
        'min': float(final_preds.min()),
        'max': float(final_preds.max()),
        'mean': float(final_preds.mean()),
        'median': float(np.median(final_preds))
    }
}

with open('final_push_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Ready to submit!")
print("=" * 80)
