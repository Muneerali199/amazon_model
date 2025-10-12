"""
Phase 11: FINAL AGGRESSIVE ATTEMPT
===================================

NEW STRATEGIES (not tried before):
1. Interaction features (price-related patterns)
2. Target encoding for high-cardinality categoricals
3. Multiple models with different random states -> AVERAGE
4. Slightly adjusted hyperparameters based on all learnings

Time: ~20-25 minutes
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 PHASE 11: FINAL AGGRESSIVE ATTEMPT")
print("="*80)
print("\nNEW Strategies:")
print("  1. Interaction features (char_count * word_count, etc.)")
print("  2. Target encoding for categorical features")
print("  3. Train 3 models with different random seeds")
print("  4. Average predictions for stability")
print("  5. Optimized hyperparameters (lr=0.04, depth=6)")
print()
print("Phase 5: 58.38% CV → 57.900% LB")
print("Target: < 58.0% CV (beat Phase 5!)")
print()

def smape(y_true, y_pred):
    """Correct SMAPE formula"""
    diff = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return 200 * np.mean(diff)

# Load data
print("-"*80)
print("Step 1: Loading data...")
print("-"*80)
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
train_features = pd.read_csv('dataset/train_features.csv')
test_features = pd.read_csv('dataset/test_features.csv')
print(f"✅ Train: {len(train):,} samples")
print(f"✅ Test: {len(test):,} samples")
print()

# TF-IDF features (100 dims - Phase 5's proven amount)
print("-"*80)
print("Step 2: Creating TF-IDF features (100 dims - Phase 5's proven)...")
print("-"*80)
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)
train_texts = train['catalog_content'].fillna('')
test_texts = test['catalog_content'].fillna('')

train_tfidf_full = tfidf.fit_transform(train_texts)
test_tfidf_full = tfidf.transform(test_texts)

svd = TruncatedSVD(n_components=100, random_state=42)
train_tfidf = svd.fit_transform(train_tfidf_full)
test_tfidf = svd.transform(test_tfidf_full)

train_tfidf_df = pd.DataFrame(train_tfidf, columns=[f'tfidf_{i}' for i in range(100)])
test_tfidf_df = pd.DataFrame(test_tfidf, columns=[f'tfidf_{i}' for i in range(100)])
print(f"✅ TF-IDF: 100 dimensions")
print()

# Prepare features
print("-"*80)
print("Step 3: Engineering NEW interaction features...")
print("-"*80)

train_features['ipq_unit'] = train_features['ipq_unit'].fillna('Count')
test_features['ipq_unit'] = test_features['ipq_unit'].fillna('Count')

# NEW: Interaction features
print("  → Creating interaction features...")
train_features['char_word_ratio'] = train_features['char_count'] / (train_features['word_count'] + 1)
test_features['char_word_ratio'] = test_features['char_count'] / (test_features['word_count'] + 1)

train_features['words_per_bullet'] = train_features['word_count'] / (train_features['bullet_points'] + 1)
test_features['words_per_bullet'] = test_features['word_count'] / (test_features['bullet_points'] + 1)

train_features['content_density'] = train_features['word_count'] * train_features['has_description']
test_features['content_density'] = test_features['word_count'] * test_features['has_description']

train_features['numeric_intensity'] = train_features['num_count'] / (train_features['word_count'] + 1)
test_features['numeric_intensity'] = test_features['num_count'] / (test_features['word_count'] + 1)

# Category combinations
train_features['is_food_bev'] = train_features['is_food'] * train_features['is_beverage']
test_features['is_food_bev'] = test_features['is_food'] * test_features['is_beverage']

train_features['is_health_care'] = train_features['is_health'] * train_features['is_personal_care']
test_features['is_health_care'] = test_features['is_health'] * test_features['is_personal_care']

numeric_features = [
    'ipq_value', 'char_count', 'word_count', 'bullet_points',
    'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
    'is_food', 'is_beverage', 'is_grocery', 'is_health',
    'is_personal_care', 'is_household',
    # NEW interaction features
    'char_word_ratio', 'words_per_bullet', 'content_density',
    'numeric_intensity', 'is_food_bev', 'is_health_care'
]

print(f"✅ Created {len(numeric_features)} numeric features (14 original + 6 new)")
print()

# NEW: Target encoding for ipq_unit
print("  → Applying target encoding for categorical...")
y_train = train['price'].values

# Calculate mean price per category
unit_means = train_features.copy()
unit_means['target'] = y_train
unit_encoding = unit_means.groupby('ipq_unit')['target'].mean().to_dict()
global_mean = y_train.mean()

train_features['unit_target_enc'] = train_features['ipq_unit'].map(unit_encoding).fillna(global_mean)
test_features['unit_target_enc'] = test_features['ipq_unit'].map(unit_encoding).fillna(global_mean)

numeric_features.append('unit_target_enc')
print(f"✅ Target encoding applied")
print()

# One-hot encoding (keep for additional signal)
train_encoded = pd.get_dummies(train_features[['ipq_unit']], prefix='unit', drop_first=True)
test_encoded = pd.get_dummies(test_features[['ipq_unit']], prefix='unit', drop_first=True)

for col in train_encoded.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
for col in test_encoded.columns:
    if col not in train_encoded.columns:
        train_encoded[col] = 0
test_encoded = test_encoded[train_encoded.columns]

# Combine all features
X_train = pd.concat([
    train_features[numeric_features].reset_index(drop=True),
    train_encoded.reset_index(drop=True),
    train_tfidf_df.reset_index(drop=True)
], axis=1)

X_test = pd.concat([
    test_features[numeric_features].reset_index(drop=True),
    test_encoded.reset_index(drop=True),
    test_tfidf_df.reset_index(drop=True)
], axis=1)

# Fix column names
X_train.columns = [str(col).replace('[', '(').replace(']', ')').replace('<', 'lt') for col in X_train.columns]
X_test.columns = [str(col).replace('[', '(').replace(']', ')').replace('<', 'lt') for col in X_test.columns]

print(f"✅ Total features: {X_train.shape[1]}")
print(f"   - Numeric (with interactions): {len(numeric_features)}")
print(f"   - Categorical one-hot: {train_encoded.shape[1]}")
print(f"   - TF-IDF: 100")
print()

# Train with optimized hyperparameters
print("="*80)
print("Step 4: Training 3 models with different random seeds...")
print("="*80)
print()

# Slightly better hyperparameters (learned from all experiments)
base_params = {
    'n_estimators': 600,        # More trees
    'learning_rate': 0.04,      # Slightly slower (more careful)
    'max_depth': 6,             # Slightly shallower (less overfit)
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.5,          # Slightly more regularization
    'tree_method': 'hist',
    'device': 'cpu'
}

print("Hyperparameters (optimized from all learnings):")
for key, val in base_params.items():
    print(f"  {key}: {val}")
print()

# Train 3 models with different seeds
seeds = [42, 123, 456]
all_cv_scores = []
all_test_predictions = []

for seed_idx, seed in enumerate(seeds, 1):
    print(f"{'='*80}")
    print(f"Model {seed_idx}/3 (seed={seed})")
    print(f"{'='*80}")
    
    model_params = base_params.copy()
    model_params['random_state'] = seed
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = []
    fold_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = XGBRegressor(**model_params)
        model.fit(X_tr, y_tr, verbose=False)
        
        val_pred = model.predict(X_val)
        score = smape(y_val, val_pred)
        cv_scores.append(score)
        
        test_pred = model.predict(X_test)
        fold_predictions.append(test_pred)
        
        print(f"  Fold {fold}: {score:.4f}%")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    all_cv_scores.append(cv_mean)
    
    # Average predictions from 5 folds
    model_test_pred = np.mean(fold_predictions, axis=0)
    all_test_predictions.append(model_test_pred)
    
    print(f"  Model {seed_idx} CV: {cv_mean:.4f}% (±{cv_std:.4f}%)")
    print()

# Average predictions from all 3 models
print("="*80)
print("Step 5: Ensembling 3 models...")
print("="*80)
final_predictions = np.mean(all_test_predictions, axis=0)

overall_cv = np.mean(all_cv_scores)
cv_std = np.std(all_cv_scores)

print(f"Individual model CVs:")
for i, cv in enumerate(all_cv_scores, 1):
    print(f"  Model {i}: {cv:.4f}%")
print()
print(f"Average CV: {overall_cv:.4f}% (±{cv_std:.4f}%)")
print()

# Results
print("="*80)
print("FINAL RESULTS")
print("="*80)
print()
print(f"Phase 5:  58.38% CV → 57.900% LB ✅")
print(f"Phase 11: {overall_cv:.2f}% CV → Expected {overall_cv-0.5:.2f}%-{overall_cv+0.5:.2f}% LB")
print()

if overall_cv < 58.0:
    improvement = 58.38 - overall_cv
    print(f"🎉 EXCELLENT! {improvement:.2f} points better than Phase 5!")
    print(f"✅ Expected LB: {overall_cv-0.5:.2f}%-{overall_cv+0.5:.2f}% (should beat 57.900%!)")
    print()
    print("✅ STRONGLY RECOMMEND: SUBMIT THIS!")
elif overall_cv < 58.38:
    improvement = 58.38 - overall_cv
    print(f"🎯 GOOD! {improvement:.2f} points better than Phase 5!")
    print(f"✅ Expected LB: {overall_cv-0.5:.2f}%-{overall_cv+0.5:.2f}% (could beat 57.900%)")
    print()
    print("✅ RECOMMEND: Submit!")
elif overall_cv < 58.8:
    print(f"⚠️  Marginal: {overall_cv-58.38:.2f} points worse than Phase 5")
    print(f"   But ensemble might help on LB")
    print()
    print("🤔 YOUR CHOICE")
else:
    print(f"❌ WORSE: {overall_cv-58.38:.2f} points worse than Phase 5")
    print()
    print("❌ DON'T SUBMIT")

print()
print("-"*80)

# Save predictions
output = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_predictions
})
output.to_csv('dataset/test_out.csv', index=False)
output.to_csv('dataset/submission_phase11.csv', index=False)

print("✅ Saved: dataset/test_out.csv")
print("✅ Saved backup: dataset/submission_phase11.csv")
print()

# Save results
results = {
    'phase': 'Phase 11: Final Aggressive Attempt',
    'strategies': [
        'Interaction features (6 new)',
        'Target encoding for categorical',
        '3 models with different seeds (42, 123, 456)',
        'Ensemble average predictions',
        'Optimized hyperparameters (lr=0.04, depth=6)'
    ],
    'cv_mean': float(overall_cv),
    'cv_std': float(cv_std),
    'model_cvs': [float(x) for x in all_cv_scores],
    'features': int(X_train.shape[1]),
    'hyperparameters': base_params,
    'comparison': {
        'phase5_cv': 58.38,
        'phase5_lb': 57.900,
        'phase11_cv': float(overall_cv),
        'improvement': float(58.38 - overall_cv)
    }
}

import json
with open('phase11_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved: phase11_results.json")
print("="*80)
print()
print("🎲 NEW FEATURES TRIED:")
print("  1. ✅ Interaction features (char/word ratio, words per bullet, etc.)")
print("  2. ✅ Target encoding (mean price per category)")
print("  3. ✅ Multiple random seeds for diversity")
print("  4. ✅ Ensemble averaging for stability")
print("  5. ✅ Optimized hyperparameters from all learnings")
print()
print("If this doesn't beat Phase 5, we've truly exhausted all options! 🎯")
print("="*80)
