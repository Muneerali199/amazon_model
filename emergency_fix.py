"""
EMERGENCY FIX - Simple Baseline That Works
===========================================

Problem: Phase 7 showed 33% CV but got 66% on leaderboard = DATA LEAKAGE!

Solution: Use ONLY the proven Phase 5 approach (58% on leaderboard)
         Then improve it conservatively

Expected: 52-54% SMAPE (better than 58%, achievable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

print("=" * 80)
print("EMERGENCY FIX - CONSERVATIVE APPROACH")
print("=" * 80)
print()
print("Using proven Phase 4a approach (60.93% leaderboard)")
print("+ Slight improvements")
print("Target: 52-54% SMAPE")
print()

# ============================================================================
# PATHS
# ============================================================================

DATASET_PATH = Path('dataset')
TRAIN_CSV = DATASET_PATH / 'train.csv'
TEST_CSV = DATASET_PATH / 'test.csv'
OUTPUT_CSV = DATASET_PATH / 'test_out_fixed.csv'
RESULTS_JSON = Path('emergency_fix_results.json')

# ============================================================================
# LOAD DATA
# ============================================================================

print("-" * 80)
print("STEP 1: Load Data")
print("-" * 80)
print()

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
print()

# ============================================================================
# EXTRACT SAFE FEATURES (NO LEAKAGE!)
# ============================================================================

print("-" * 80)
print("STEP 2: Extract Basic Features (SAFE - NO LEAKAGE)")
print("-" * 80)
print()

def extract_safe_features(df):
    """Extract only the most basic, safe features"""
    features = pd.DataFrame()
    
    text = df['catalog_content'].fillna('')
    
    # Very basic text features
    features['text_length'] = text.str.len()
    features['word_count'] = text.str.split().str.len()
    features['avg_word_length'] = text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)
    features['digit_count'] = text.str.count(r'\d')
    features['upper_count'] = text.str.count(r'[A-Z]')
    
    # Category indicators (binary only)
    features['has_electronics'] = text.str.contains('electronic|battery|cable', case=False, regex=True).astype(int)
    features['has_beauty'] = text.str.contains('beauty|cosmetic|skin|hair', case=False, regex=True).astype(int)
    features['has_food'] = text.str.contains('food|snack|candy', case=False, regex=True).astype(int)
    features['has_home'] = text.str.contains('home|kitchen|furniture', case=False, regex=True).astype(int)
    features['has_clothing'] = text.str.contains('clothing|shirt|pants|dress', case=False, regex=True).astype(int)
    
    # Size indicators
    features['has_oz'] = text.str.contains(r'\d+\s*oz', case=False, regex=True).astype(int)
    features['has_lb'] = text.str.contains(r'\d+\s*lb', case=False, regex=True).astype(int)
    features['has_pack'] = text.str.contains('pack|count|set', case=False, regex=True).astype(int)
    
    features = features.fillna(0)
    return features

print("Extracting basic features...")
train_basic = extract_safe_features(train_df)
test_basic = extract_safe_features(test_df)
print(f"Basic features: {train_basic.shape[1]}")
print()

# ============================================================================
# TF-IDF FEATURES (PROVEN TO WORK - Phase 4a got 60.93%)
# ============================================================================

print("-" * 80)
print("STEP 3: Add TF-IDF Features (Proven Approach)")
print("-" * 80)
print()

print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=50,  # Reduced from 100 to avoid overfitting
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    strip_accents='unicode',
    lowercase=True
)

train_text = train_df['catalog_content'].fillna('')
test_text = test_df['catalog_content'].fillna('')

tfidf_train = tfidf.fit_transform(train_text)
tfidf_test = tfidf.transform(test_text)

train_tfidf_df = pd.DataFrame(
    tfidf_train.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_train.shape[1])]
)

test_tfidf_df = pd.DataFrame(
    tfidf_test.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_test.shape[1])]
)

print(f"TF-IDF features: {train_tfidf_df.shape[1]}")
print()

# ============================================================================
# COMBINE FEATURES
# ============================================================================

print("-" * 80)
print("STEP 4: Combine Features")
print("-" * 80)
print()

# Combine features
X_train = pd.concat([train_basic, train_tfidf_df], axis=1).values
X_test = pd.concat([test_basic, test_tfidf_df], axis=1).values
y_train = train_df['price'].values

print(f"Total features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print()

# ============================================================================
# TRAIN MODEL (CONSERVATIVE PARAMETERS)
# ============================================================================

print("-" * 80)
print("STEP 5: Train XGBoost (Conservative Settings)")
print("-" * 80)
print()

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# Conservative XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.05,  # Slower learning
    'max_depth': 6,  # Shallower trees (less overfitting)
    'min_child_weight': 5,  # More regularization
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.5,  # More L1 regularization
    'reg_lambda': 2.0,  # More L2 regularization
    'n_estimators': 500,  # Fewer trees
    'random_state': 42,
    'tree_method': 'hist'
}

# 5-fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), 5))
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    print(f"Fold {fold}/5...")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds[:, fold-1] = model.predict(X_test)
    
    fold_score = smape(y_val, oof_preds[val_idx])
    cv_scores.append(fold_score)
    print(f"   SMAPE: {fold_score:.4f}%")

print()
cv_score = smape(y_train, oof_preds)
print(f"Overall CV SMAPE: {cv_score:.4f}% (±{np.std(cv_scores):.4f}%)")
print()

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("-" * 80)
print("STEP 6: Generate Test Predictions")
print("-" * 80)
print()

# Average predictions across folds
final_preds = test_preds.mean(axis=1)

# Clip to reasonable range
final_preds = np.clip(final_preds, 0.1, 1000.0)

print(f"Predictions: {len(final_preds):,}")
print(f"Price range: ${final_preds.min():.2f} - ${final_preds.max():.2f}")
print(f"Mean: ${final_preds.mean():.2f}")
print(f"Median: ${np.median(final_preds):.2f}")
print()

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("-" * 80)
print("STEP 7: Save Submission")
print("-" * 80)
print()

submission_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_preds
})

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
print()

# Also save as test_out.csv
submission_df.to_csv(DATASET_PATH / 'test_out.csv', index=False)
print(f"✅ Saved: dataset/test_out.csv")
print()

# Save results
results = {
    'approach': 'Emergency Fix - Conservative',
    'cv_score': float(cv_score),
    'cv_std': float(np.std(cv_scores)),
    'fold_scores': [float(s) for s in cv_scores],
    'features': {
        'basic': 13,
        'tfidf': train_tfidf_df.shape[1],
        'total': X_train.shape[1]
    },
    'predictions': {
        'count': int(len(final_preds)),
        'min': float(final_preds.min()),
        'max': float(final_preds.max()),
        'mean': float(final_preds.mean()),
        'median': float(np.median(final_preds))
    }
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved: {RESULTS_JSON}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("EMERGENCY FIX COMPLETE!")
print("=" * 80)
print()
print(f"CV Score: {cv_score:.2f}% (±{np.std(cv_scores):.2f}%)")
print(f"Expected Leaderboard: {cv_score + 2:.1f}% - {cv_score + 5:.1f}%")
print()
print("This is more conservative and should match leaderboard better.")
print("Expected rank: TOP 20-30 (better than your 58.16%)")
print()
print(f"Submit: {OUTPUT_CSV}")
print()
print("=" * 80)
