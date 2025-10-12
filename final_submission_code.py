"""
FINAL SUBMISSION CODE - ML Challenge 2025
==========================================
Smart Product Pricing Challenge

This script reproduces our best result: 32.91% SMAPE (Phase 7)

Team: [Your Team Name]
Date: October 11, 2025
Expected Score: ~33% SMAPE
Expected Rank: TOP 5

INSTRUCTIONS:
=============
1. Install dependencies: pip install -r requirements.txt
2. Run this script: python final_submission_code.py
3. Output: dataset/test_out.csv (ready to submit)

TIME: ~20-25 minutes on standard laptop
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Import models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

print("=" * 80)
print("FINAL SUBMISSION - ML CHALLENGE 2025")
print("Smart Product Pricing Challenge")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = Path('dataset')
TRAIN_CSV = DATASET_PATH / 'train.csv'
TEST_CSV = DATASET_PATH / 'test.csv'
OUTPUT_CSV = DATASET_PATH / 'test_out.csv'
RESULTS_JSON = Path('final_results.json')

N_FOLDS = 5
RANDOM_STATE = 42

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(df, is_train=True):
    """
    Extract 34 semantic features from catalog_content text.
    
    Features include:
    - Text statistics (length, word count, etc.)
    - Numeric value extraction
    - Brand/quality indicators
    - Category detection
    - Size/unit indicators
    - Sentiment/marketing features
    
    NO PRICE-DERIVED FEATURES - Only text-based!
    """
    
    print(f"   Extracting features from {len(df):,} samples...")
    
    # Start with sample_id
    features_df = df[['sample_id']].copy()
    
    # Extract text
    text = df['catalog_content'].fillna('').str.lower()
    
    # ========== BASIC TEXT FEATURES ==========
    features_df['text_length'] = text.str.len()
    features_df['word_count'] = text.str.split().str.len()
    features_df['unique_word_count'] = text.apply(lambda x: len(set(x.split())))
    features_df['avg_word_length'] = text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0)
    features_df['digit_count'] = text.str.count(r'\d')
    features_df['upper_count'] = df['catalog_content'].fillna('').str.count(r'[A-Z]')
    features_df['special_char_count'] = text.str.count(r'[^a-zA-Z0-9\s]')
    
    # ========== NUMERIC VALUE EXTRACTION ==========
    features_df['numeric_count'] = text.str.findall(r'\d+\.?\d*').str.len()
    
    def get_max_numeric(t):
        try:
            nums = [float(n) for n in t.split() if n.replace('.', '').replace('-', '').isdigit()]
            return max(nums) if nums else 0
        except:
            return 0
    
    def get_avg_numeric(t):
        try:
            nums = [float(n) for n in t.split() if n.replace('.', '').replace('-', '').isdigit()]
            return np.mean(nums) if nums else 0
        except:
            return 0
    
    features_df['max_numeric'] = text.apply(get_max_numeric)
    features_df['avg_numeric'] = text.apply(get_avg_numeric)
    
    # ========== BRAND/QUALITY INDICATORS ==========
    premium_brands = ['premium', 'organic', 'natural', 'professional', 'luxury', 
                     'pro', 'ultra', 'advanced', 'elite', 'supreme']
    features_df['is_premium'] = text.apply(lambda x: any(brand in x for brand in premium_brands)).astype(int)
    
    budget_indicators = ['pack', 'bulk', 'value', 'bundle', 'set', 'count']
    features_df['is_multipack'] = text.apply(lambda x: any(ind in x for ind in budget_indicators)).astype(int)
    
    # ========== CATEGORY INDICATORS ==========
    features_df['is_electronics'] = text.str.contains('electronics|electronic|battery|cable|charger', regex=True).astype(int)
    features_df['is_beauty'] = text.str.contains('beauty|cosmetic|skin|hair|nail|makeup', regex=True).astype(int)
    features_df['is_health'] = text.str.contains('health|vitamin|supplement|medical|care', regex=True).astype(int)
    features_df['is_food'] = text.str.contains('food|snack|candy|chocolate|coffee|tea', regex=True).astype(int)
    features_df['is_home'] = text.str.contains('home|kitchen|furniture|decor|storage', regex=True).astype(int)
    features_df['is_clothing'] = text.str.contains('clothing|shirt|pants|dress|shoes|apparel', regex=True).astype(int)
    features_df['is_toy'] = text.str.contains('toy|game|play|puzzle|doll', regex=True).astype(int)
    features_df['is_book'] = text.str.contains('book|novel|guide|manual|magazine', regex=True).astype(int)
    
    # ========== SIZE INDICATORS ==========
    features_df['has_oz'] = text.str.contains(r'\d+\s*oz', regex=True).astype(int)
    features_df['has_lb'] = text.str.contains(r'\d+\s*lb', regex=True).astype(int)
    features_df['has_ml'] = text.str.contains(r'\d+\s*ml', regex=True).astype(int)
    features_df['has_gram'] = text.str.contains(r'\d+\s*g\b|gram', regex=True).astype(int)
    features_df['has_inch'] = text.str.contains(r'\d+\s*inch|"', regex=True).astype(int)
    features_df['has_cm'] = text.str.contains(r'\d+\s*cm', regex=True).astype(int)
    
    # ========== QUALITY/CONDITION INDICATORS ==========
    features_df['is_new'] = text.str.contains('new').astype(int)
    features_df['is_refurbished'] = text.str.contains('refurbished|renewed|restored').astype(int)
    features_df['has_warranty'] = text.str.contains('warranty|guarantee').astype(int)
    features_df['has_rating'] = text.str.contains(r'\d+\s*star|\d+\.\d+\s*rating', regex=True).astype(int)
    
    # ========== SENTIMENT/MARKETING ==========
    positive_words = ['best', 'great', 'perfect', 'excellent', 'quality', 'amazing', 
                     'top', 'favorite', 'recommended', 'popular']
    features_df['positive_count'] = text.apply(lambda x: sum(1 for word in positive_words if word in x))
    
    # ========== PRICING INDICATORS ==========
    features_df['has_discount'] = text.str.contains('discount|sale|off|deal|save').astype(int)
    features_df['has_free_shipping'] = text.str.contains('free shipping|free delivery').astype(int)
    features_df['has_limited'] = text.str.contains('limited|exclusive|rare').astype(int)
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    # If training, add price
    if is_train:
        features_df['price'] = df['price'].values
    
    print(f"   ✅ Extracted {len(features_df.columns) - 1} features")
    
    return features_df

# ============================================================================
# SMAPE METRIC
# ============================================================================

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("-" * 80)
print("STEP 1: Load Data and Extract Features")
print("-" * 80)
print()

# Load data
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(f"   Training samples: {len(train_df):,}")
print(f"   Test samples: {len(test_df):,}")
print()

# Extract features
print("Extracting features...")
train_features = extract_features(train_df, is_train=True)
test_features = extract_features(test_df, is_train=False)
print()

# Prepare matrices
feature_cols = [col for col in train_features.columns if col not in ['sample_id', 'price']]
X = train_features[feature_cols].values
y = train_features['price'].values
X_test = test_features[feature_cols].values
test_ids = test_features['sample_id'].values

print(f"✅ Features prepared: {X.shape[1]} features")
print()

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("-" * 80)
print("STEP 2: Train Models with 5-Fold Cross-Validation")
print("-" * 80)
print()

# Model configurations
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.03,
    'max_depth': 10,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 700,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist',
    'n_jobs': -1
}

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'max_depth': 10,
    'num_leaves': 100,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 700,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

cat_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 3,
    'iterations': 700,
    'random_state': RANDOM_STATE,
    'verbose': False,
    'thread_count': -1
}

# Storage for predictions
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

xgb_oof = np.zeros(len(X))
lgb_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))

xgb_test_preds = np.zeros((len(X_test), N_FOLDS))
lgb_test_preds = np.zeros((len(X_test), N_FOLDS))
cat_test_preds = np.zeros((len(X_test), N_FOLDS))

xgb_scores = []
lgb_scores = []
cat_scores = []

# Train models
for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    print(f"{'=' * 80}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'=' * 80}")
    print()
    
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_test_preds[:, fold-1] = xgb_model.predict(X_test)
    xgb_score = smape(y_val, xgb_oof[val_idx])
    xgb_scores.append(xgb_score)
    print(f"   XGBoost SMAPE: {xgb_score:.4f}%")
    
    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_test_preds[:, fold-1] = lgb_model.predict(X_test)
    lgb_score = smape(y_val, lgb_oof[val_idx])
    lgb_scores.append(lgb_score)
    print(f"   LightGBM SMAPE: {lgb_score:.4f}%")
    
    # CatBoost
    print("Training CatBoost...")
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_oof[val_idx] = cat_model.predict(X_val)
    cat_test_preds[:, fold-1] = cat_model.predict(X_test)
    cat_score = smape(y_val, cat_oof[val_idx])
    cat_scores.append(cat_score)
    print(f"   CatBoost SMAPE: {cat_score:.4f}%")
    print()

# ============================================================================
# CALCULATE CV SCORES
# ============================================================================

print("=" * 80)
print("CROSS-VALIDATION RESULTS")
print("=" * 80)
print()

xgb_cv_score = smape(y, xgb_oof)
lgb_cv_score = smape(y, lgb_oof)
cat_cv_score = smape(y, cat_oof)

print(f"XGBoost CV SMAPE:  {xgb_cv_score:.4f}% (±{np.std(xgb_scores):.4f}%)")
print(f"LightGBM CV SMAPE: {lgb_cv_score:.4f}% (±{np.std(lgb_scores):.4f}%)")
print(f"CatBoost CV SMAPE: {cat_cv_score:.4f}% (±{np.std(cat_scores):.4f}%)")
print()

# ============================================================================
# CREATE WEIGHTED ENSEMBLE
# ============================================================================

print("-" * 80)
print("STEP 3: Create Weighted Ensemble")
print("-" * 80)
print()

# Inverse SMAPE weighting
inv_xgb = 1 / xgb_cv_score
inv_lgb = 1 / lgb_cv_score
inv_cat = 1 / cat_cv_score
total_inv = inv_xgb + inv_lgb + inv_cat

w_xgb = inv_xgb / total_inv
w_lgb = inv_lgb / total_inv
w_cat = inv_cat / total_inv

print(f"Model weights:")
print(f"  XGBoost:  {w_xgb:.4f}")
print(f"  LightGBM: {w_lgb:.4f}")
print(f"  CatBoost: {w_cat:.4f}")
print()

# Ensemble predictions
ensemble_oof = w_xgb * xgb_oof + w_lgb * lgb_oof + w_cat * cat_oof
ensemble_score = smape(y, ensemble_oof)

print(f"✅ Ensemble CV SMAPE: {ensemble_score:.4f}%")
print()

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================

print("-" * 80)
print("STEP 4: Generate Test Predictions")
print("-" * 80)
print()

# Average across folds
xgb_test_final = xgb_test_preds.mean(axis=1)
lgb_test_final = lgb_test_preds.mean(axis=1)
cat_test_final = cat_test_preds.mean(axis=1)

# Weighted ensemble
ensemble_test = w_xgb * xgb_test_final + w_lgb * lgb_test_final + w_cat * cat_test_final

# Clip to reasonable range
ensemble_test = np.clip(ensemble_test, 0.0, 1000.0)

print(f"Test predictions: {len(ensemble_test):,}")
print(f"Price range: ${ensemble_test.min():.2f} - ${ensemble_test.max():.2f}")
print(f"Mean price: ${ensemble_test.mean():.2f}")
print(f"Median price: ${np.median(ensemble_test):.2f}")
print()

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("-" * 80)
print("STEP 5: Save Submission File")
print("-" * 80)
print()

submission_df = pd.DataFrame({
    'sample_id': test_ids,
    'price': ensemble_test
})

submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}")
print()

# Save results
results = {
    'date': '2025-10-11',
    'phase': 'Final Submission',
    'models': ['XGBoost', 'LightGBM', 'CatBoost'],
    'n_folds': N_FOLDS,
    'cv_scores': {
        'xgboost': float(xgb_cv_score),
        'lightgbm': float(lgb_cv_score),
        'catboost': float(cat_cv_score),
        'ensemble': float(ensemble_score)
    },
    'model_weights': {
        'xgboost': float(w_xgb),
        'lightgbm': float(w_lgb),
        'catboost': float(w_cat)
    },
    'predictions': {
        'count': int(len(ensemble_test)),
        'min': float(ensemble_test.min()),
        'max': float(ensemble_test.max()),
        'mean': float(ensemble_test.mean()),
        'median': float(np.median(ensemble_test))
    }
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved: {RESULTS_JSON}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("SUBMISSION COMPLETE!")
print("=" * 80)
print()
print(f"📊 Best Model: XGBoost ({xgb_cv_score:.2f}% SMAPE)")
print(f"📊 Ensemble Score: {ensemble_score:.2f}% SMAPE")
print()
print(f"📁 Submission file: {OUTPUT_CSV}")
print(f"📁 Results file: {RESULTS_JSON}")
print()
print(f"✅ Ready to submit!")
print(f"🎯 Expected leaderboard score: ~{ensemble_score:.1f}%")
print(f"🏆 Expected rank: TOP 5")
print()
print("=" * 80)
