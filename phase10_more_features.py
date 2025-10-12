"""
Phase 10: MORE Features (opposite of feature selection)
========================================================

Strategy:
- Keep ALL features (no selection)
- Use 120 TF-IDF dimensions (vs Phase 5's 100)
- Use Phase 5's PROVEN hyperparameters (lr=0.05, depth=7)
- Simple and focused test

Time: ~15 minutes
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 PHASE 10: MORE FEATURES (120 TF-IDF)")
print("="*80)
print("\nStrategy: Phase 5 hyperparameters + 120 TF-IDF (vs 100)")
print("Phase 5: 58.38% CV → 57.900% LB")
print("Target: Beat 58.38% CV with more TF-IDF dimensions")
print()

def smape(y_true, y_pred):
    """Correct SMAPE formula with 200*"""
    diff = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return 200 * np.mean(diff)

# Load data
print("-"*80)
print("Loading data...")
print("-"*80)
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
train_features = pd.read_csv('dataset/train_features.csv')
test_features = pd.read_csv('dataset/test_features.csv')
print(f"Train: {len(train):,} samples")
print(f"Test: {len(test):,} samples")
print()

# Create TF-IDF features (120 dimensions - MORE than Phase 5's 100)
print("-"*80)
print("Creating TF-IDF features (120 dims vs Phase 5's 100)...")
print("-"*80)
from sklearn.decomposition import TruncatedSVD

# Use same approach as Phase 5 but with 120 dims
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

# Reduce to 120 dims (vs Phase 5's 100)
svd = TruncatedSVD(n_components=120, random_state=42)
train_tfidf = svd.fit_transform(train_tfidf_full)
test_tfidf = svd.transform(test_tfidf_full)

train_tfidf_df = pd.DataFrame(
    train_tfidf, 
    columns=[f'tfidf_{i}' for i in range(120)]
)
test_tfidf_df = pd.DataFrame(
    test_tfidf, 
    columns=[f'tfidf_{i}' for i in range(120)]
)
print(f"✅ TF-IDF: 120 dimensions (Phase 5 had 100)")
print()

# Prepare features  
print("-"*80)
print("Preparing features...")
print("-"*80)

# Use engineered features from Phase 5 approach
train_features['ipq_unit'] = train_features['ipq_unit'].fillna('Count')
test_features['ipq_unit'] = test_features['ipq_unit'].fillna('Count')

numeric_features = [
    'ipq_value', 'char_count', 'word_count', 'bullet_points',
    'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
    'is_food', 'is_beverage', 'is_grocery', 'is_health',
    'is_personal_care', 'is_household'
]

# Encode categorical unit
train_encoded = pd.get_dummies(train_features[['ipq_unit']], prefix='unit', drop_first=True)
test_encoded = pd.get_dummies(test_features[['ipq_unit']], prefix='unit', drop_first=True)

# Align columns
for col in train_encoded.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
for col in test_encoded.columns:
    if col not in train_encoded.columns:
        train_encoded[col] = 0
test_encoded = test_encoded[train_encoded.columns]

# Combine all features
train_tfidf_df = pd.DataFrame(
    train_tfidf,
    columns=[f'tfidf_{i}' for i in range(120)]
)
test_tfidf_df = pd.DataFrame(
    test_tfidf,
    columns=[f'tfidf_{i}' for i in range(120)]
)

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

# Fix column names to avoid XGBoost error (no [, ], <)
X_train.columns = [str(col).replace('[', '(').replace(']', ')').replace('<', 'lt') for col in X_train.columns]
X_test.columns = [str(col).replace('[', '(').replace(']', ')').replace('<', 'lt') for col in X_test.columns]

y_train = train['price'].values

print(f"Features: {X_train.shape[1]} (Phase 5 had 114+100=214 or ~255)")
print(f"  - Numeric: {len(numeric_features)}")
print(f"  - Categorical: {train_encoded.shape[1]}")
print(f"  - TF-IDF: 120 (Phase 5 had 100)")
print()

# Train with Phase 5's PROVEN hyperparameters
print("="*80)
print("Training with Phase 5's PROVEN hyperparameters...")
print("="*80)
print()

model_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,      # Phase 5's proven lr
    'max_depth': 7,             # Phase 5's proven depth
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cpu'
}

print("Hyperparameters (same as Phase 5):")
for key, val in model_params.items():
    print(f"  {key}: {val}")
print()

# 5-Fold CV (same as Phase 5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
    
    # Predict on test
    test_pred = model.predict(X_test)
    fold_predictions.append(test_pred)
    
    print(f"  Fold {fold}: {score:.4f}%")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print()
print(f"  CV: {cv_mean:.4f}% (±{cv_std:.4f}%)")
print()

# Average predictions
final_predictions = np.mean(fold_predictions, axis=0)

# Save results
print("="*80)
print("FINAL RESULTS")
print("="*80)
print()
print(f"Phase 5:  58.38% CV → 57.900% LB ✅ (current best)")
print(f"Phase 10: {cv_mean:.2f}% CV → Expected {cv_mean-0.5:.2f}%-{cv_mean+0.5:.2f}% LB")
print()

if cv_mean < 58.38:
    improvement = 58.38 - cv_mean
    print(f"🎉 IMPROVEMENT: {improvement:.2f} points better than Phase 5!")
    print(f"✅ Expected LB: {cv_mean-0.5:.2f}%-{cv_mean+0.5:.2f}% (could beat 57.900%!)")
    print()
    print("✅ RECOMMEND: SUBMIT THIS!")
elif cv_mean < 58.8:
    print(f"⚠️  Marginal: {cv_mean-58.38:.2f} points worse than Phase 5")
    print(f"   But might still help due to CV/LB variance")
    print()
    print("🤔 YOUR CHOICE: Submit or keep Phase 5")
else:
    print(f"❌ WORSE: {cv_mean-58.38:.2f} points worse than Phase 5")
    print(f"   Expected LB: ~{cv_mean:.2f}% (worse than 57.900%)")
    print()
    print("❌ DON'T SUBMIT: Keep Phase 5")

print()
print("-"*80)

# Save predictions
output = pd.DataFrame({
    'id': test['id'],
    'price': final_predictions
})
output.to_csv('dataset/test_out.csv', index=False)
output.to_csv('dataset/submission_phase10.csv', index=False)

print("✅ Saved: dataset/test_out.csv")
print("✅ Saved backup: dataset/submission_phase10.csv")
print()

# Save results
results = {
    'phase': 'Phase 10: MORE Features (120 TF-IDF)',
    'cv_mean': float(cv_mean),
    'cv_std': float(cv_std),
    'cv_scores': [float(x) for x in cv_scores],
    'features': int(X_train.shape[1]),
    'tfidf_dims': 120,
    'hyperparameters': model_params,
    'comparison': {
        'phase5_cv': 58.38,
        'phase5_lb': 57.900,
        'phase10_cv': float(cv_mean),
        'improvement': float(58.38 - cv_mean)
    }
}

import json
with open('phase10_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved: phase10_results.json")
print("="*80)
