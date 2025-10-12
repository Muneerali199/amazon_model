"""
ULTRA-SAFE BASELINE - GUARANTEED NO LEAKAGE
============================================

Strategy: Use ONLY TF-IDF (nothing else)
         Exactly replicate Phase 4 that got 60.93% CV = 60.93% LB

This is our safety net. It WILL work.
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
print("ULTRA-SAFE BASELINE - TF-IDF ONLY")
print("=" * 80)
print()
print("Replicating Phase 4 approach (60.93% CV = 60.93% LB)")
print("Expected: ~60% CV → ~60% LB")
print()

# ============================================================================
# PATHS
# ============================================================================

DATASET_PATH = Path('dataset')
TRAIN_CSV = DATASET_PATH / 'train.csv'
TEST_CSV = DATASET_PATH / 'test.csv'
OUTPUT_CSV = DATASET_PATH / 'test_out.csv'
RESULTS_JSON = Path('safe_baseline_results.json')

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
# TF-IDF FEATURES ONLY (PROVEN SAFE)
# ============================================================================

print("-" * 80)
print("STEP 2: Create TF-IDF Features")
print("-" * 80)
print()

print("Creating TF-IDF features (100 dimensions, ngram 1-2)...")
tfidf = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    strip_accents='unicode',
    lowercase=True
)

train_text = train_df['catalog_content'].fillna('')
test_text = test_df['catalog_content'].fillna('')

X_train = tfidf.fit_transform(train_text).toarray()
X_test = tfidf.transform(test_text).toarray()
y_train = train_df['price'].values

print(f"Features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print()

# ============================================================================
# TRAIN XGBOOST
# ============================================================================

print("-" * 80)
print("STEP 3: Train XGBoost")
print("-" * 80)
print()

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)  # CORRECT SMAPE: multiply by 200, not 100!

# Simple XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 300,
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
    
    # Train model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Validation predictions
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    
    # Test predictions
    test_preds[:, fold-1] = model.predict(X_test)
    
    # Calculate SMAPE
    fold_smape = smape(y_val, val_pred)
    cv_scores.append(fold_smape)
    print(f"   SMAPE: {fold_smape:.4f}%")

# Overall CV score
overall_cv = smape(y_train, oof_preds)
print()
print(f"Overall CV SMAPE: {overall_cv:.4f}% (±{np.std(cv_scores):.4f}%)")
print()

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("-" * 80)
print("STEP 4: Generate Test Predictions")
print("-" * 80)
print()

# Average test predictions
final_test_preds = test_preds.mean(axis=1)

print(f"Predictions: {len(final_test_preds):,}")
print(f"Price range: ${final_test_preds.min():.2f} - ${final_test_preds.max():.2f}")
print(f"Mean: ${final_test_preds.mean():.2f}")
print(f"Median: ${np.median(final_test_preds):.2f}")
print()

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("-" * 80)
print("STEP 5: Save Submission")
print("-" * 80)
print()

# Create submission DataFrame
submission_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],  # Correct column name
    'price': final_test_preds
})

# Save submission
submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
print()

# Save results
results = {
    'approach': 'Ultra-Safe Baseline - TF-IDF Only',
    'cv_score': overall_cv,
    'cv_std': np.std(cv_scores),
    'fold_scores': cv_scores,
    'features': {
        'tfidf': X_train.shape[1],
        'total': X_train.shape[1]
    },
    'predictions': {
        'count': len(final_test_preds),
        'min': float(final_test_preds.min()),
        'max': float(final_test_preds.max()),
        'mean': float(final_test_preds.mean()),
        'median': float(np.median(final_test_preds))
    }
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {RESULTS_JSON}")
print()

# ============================================================================
# FINAL STATUS
# ============================================================================

print("=" * 80)
print("ULTRA-SAFE BASELINE COMPLETE!")
print("=" * 80)
print()
print(f"CV Score: {overall_cv:.2f}% (±{np.std(cv_scores):.2f}%)")
print(f"Expected Leaderboard: ~{overall_cv:.0f}%")
print()
print("This is a SAFE baseline with NO leakage.")
print(f"CV score should be 58-62% (realistic range).")
print()
print(f"Submit: {OUTPUT_CSV}")
print()
print("=" * 80)
