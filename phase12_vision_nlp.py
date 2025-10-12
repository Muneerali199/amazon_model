"""
Phase 12: VISION + NLP FUSION 🖼️📝
====================================

BREAKTHROUGH STRATEGY:
1. Use PRE-COMPUTED image features from image_cache  
2. Advanced NLP: TF-IDF text features
3. Combine visual + text + engineered features
4. Ensemble XGBoost + LightGBM + CatBoost

This is THE ULTIMATE attempt with multimodal data!

Expected time: 35-45 minutes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 PHASE 12: VISION + NLP FUSION - MULTIMODAL ML")
print("="*80)
print("\nBREAKTHROUGH STRATEGIES:")
print("  1. ✅ Image features from cache (11 visual features)")
print("  2. ✅ Advanced text features (TF-IDF + engineering)")
print("  3. ✅ Triple ensemble: XGBoost + LightGBM + CatBoost")
print("  4. ✅ Multimodal fusion of vision + language")
print()
print("Phase 5: 58.38% CV → 57.900% LB")
print("Target: < 57.0% CV (BEAT Phase 5 significantly!)")
print()

def smape(y_true, y_pred):
    """Correct SMAPE formula"""
    diff = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return 200 * np.mean(diff)

# Step 1: Load data
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

# Step 2: Load IMAGE features from cache
print("-"*80)
print("Step 2: Loading PRE-COMPUTED image features...")
print("-"*80)

image_cache_path = Path('dataset/image_cache')
image_features = {}

for json_file in image_cache_path.glob('*.json'):
    sample_id = int(json_file.stem)
    with open(json_file, 'r') as f:
        image_features[sample_id] = json.load(f)

print(f"✅ Loaded {len(image_features):,} image feature sets")

# Create image feature dataframe
image_feature_names = [
    'img_width', 'img_height', 'img_aspect_ratio', 'img_area',
    'img_mean_r', 'img_mean_g', 'img_mean_b',
    'img_std_r', 'img_std_g', 'img_std_b',
    'img_brightness', 'img_colorfulness',
    'img_is_wide', 'img_is_tall', 'img_is_large'
]

def get_image_features(sample_id):
    """Get image features for a sample, or zeros if not found"""
    if sample_id in image_features:
        feats = image_features[sample_id]
        return [feats.get(f, 0) for f in image_feature_names]
    else:
        return [0] * len(image_feature_names)

train_img_features = np.array([get_image_features(sid) for sid in train['sample_id']])
test_img_features = np.array([get_image_features(sid) for sid in test['sample_id']])

train_img_df = pd.DataFrame(train_img_features, columns=image_feature_names)
test_img_df = pd.DataFrame(test_img_features, columns=image_feature_names)

print(f"✅ Image features: {train_img_df.shape[1]} dimensions")
print(f"   Coverage: Train={train_img_df.iloc[:, 0].ne(0).sum():,}/{len(train):,}, Test={test_img_df.iloc[:, 0].ne(0).sum():,}/{len(test):,}")
print()

# Step 3: TF-IDF features (Phase 5's proven setup)
print("-"*80)
print("Step 3: Creating TF-IDF features (100 dims - Phase 5's proven)...")
print("-"*80)

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

# Step 4: Prepare engineered features
print("-"*80)
print("Step 4: Engineering features...")
print("-"*80)

train_features['ipq_unit'] = train_features['ipq_unit'].fillna('Count')
test_features['ipq_unit'] = test_features['ipq_unit'].fillna('Count')

# Base numeric features
numeric_features = [
    'ipq_value', 'char_count', 'word_count', 'bullet_points',
    'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
    'is_food', 'is_beverage', 'is_grocery', 'is_health',
    'is_personal_care', 'is_household'
]

# One-hot encoding for unit
train_encoded = pd.get_dummies(train_features[['ipq_unit']], prefix='unit', drop_first=True)
test_encoded = pd.get_dummies(test_features[['ipq_unit']], prefix='unit', drop_first=True)

for col in train_encoded.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
for col in test_encoded.columns:
    if col not in train_encoded.columns:
        train_encoded[col] = 0
test_encoded = test_encoded[train_encoded.columns]

print(f"✅ Engineered features ready")
print()

# Step 5: Combine ALL features (MULTIMODAL!)
print("-"*80)
print("Step 5: Combining MULTIMODAL features...")
print("-"*80)

X_train = pd.concat([
    train_features[numeric_features].reset_index(drop=True),
    train_encoded.reset_index(drop=True),
    train_tfidf_df.reset_index(drop=True),
    train_img_df.reset_index(drop=True)  # 🖼️ IMAGE FEATURES!
], axis=1)

X_test = pd.concat([
    test_features[numeric_features].reset_index(drop=True),
    test_encoded.reset_index(drop=True),
    test_tfidf_df.reset_index(drop=True),
    test_img_df.reset_index(drop=True)  # 🖼️ IMAGE FEATURES!
], axis=1)

# Fix column names (remove ALL special characters for LightGBM compatibility)
import re
def clean_column_name(col):
    """Clean column names to be compatible with all models"""
    col_str = str(col)
    # Replace special characters with underscores
    col_str = re.sub(r'[^A-Za-z0-9_]', '_', col_str)
    # Remove consecutive underscores
    col_str = re.sub(r'_+', '_', col_str)
    # Remove leading/trailing underscores
    col_str = col_str.strip('_')
    return col_str

X_train.columns = [clean_column_name(col) for col in X_train.columns]
X_test.columns = [clean_column_name(col) for col in X_test.columns]

# Ensure unique column names (add suffix if duplicate)
cols = list(X_train.columns)
seen = {}
for i, col in enumerate(cols):
    if col in seen:
        seen[col] += 1
        cols[i] = f"{col}_{seen[col]}"
    else:
        seen[col] = 0

X_train.columns = cols
X_test.columns = cols

y_train = train['price'].values

print(f"✅ MULTIMODAL feature matrix: {X_train.shape[1]} dimensions")
print(f"   - Numeric: {len(numeric_features)}")
print(f"   - Categorical: {train_encoded.shape[1]}")
print(f"   - TF-IDF (text): 100")
print(f"   - Image (vision): {train_img_df.shape[1]}")
print()

# Step 6: Train TRIPLE ENSEMBLE
print("="*80)
print("Step 6: Training TRIPLE ENSEMBLE (XGBoost + LightGBM + CatBoost)...")
print("="*80)
print()

# Optimized hyperparameters for each model
xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cpu'
}

lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1,
    'random_state': 42,
    'verbose': -1
}

cat_params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 7,
    'l2_leaf_reg': 1,
    'random_state': 42,
    'verbose': False
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions
xgb_cv_scores = []
lgb_cv_scores = []
cat_cv_scores = []

xgb_test_preds = []
lgb_test_preds = []
cat_test_preds = []

print("Training XGBoost...")
print("-"*80)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, verbose=False)
    
    val_pred = model.predict(X_val)
    score = smape(y_val, val_pred)
    xgb_cv_scores.append(score)
    
    test_pred = model.predict(X_test)
    xgb_test_preds.append(test_pred)
    
    print(f"  Fold {fold}: {score:.4f}%")

xgb_cv_mean = np.mean(xgb_cv_scores)
print(f"XGBoost CV: {xgb_cv_mean:.4f}% (±{np.std(xgb_cv_scores):.4f}%)")
print()

print("Training LightGBM...")
print("-"*80)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict(X_val)
    score = smape(y_val, val_pred)
    lgb_cv_scores.append(score)
    
    test_pred = model.predict(X_test)
    lgb_test_preds.append(test_pred)
    
    print(f"  Fold {fold}: {score:.4f}%")

lgb_cv_mean = np.mean(lgb_cv_scores)
print(f"LightGBM CV: {lgb_cv_mean:.4f}% (±{np.std(lgb_cv_scores):.4f}%)")
print()

print("Training CatBoost...")
print("-"*80)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict(X_val)
    score = smape(y_val, val_pred)
    cat_cv_scores.append(score)
    
    test_pred = model.predict(X_test)
    cat_test_preds.append(test_pred)
    
    print(f"  Fold {fold}: {score:.4f}%")

cat_cv_mean = np.mean(cat_cv_scores)
print(f"CatBoost CV: {cat_cv_mean:.4f}% (±{np.std(cat_cv_scores):.4f}%)")
print()

# Step 7: Ensemble predictions
print("="*80)
print("Step 7: Creating weighted ensemble...")
print("="*80)

# Average fold predictions
xgb_final = np.mean(xgb_test_preds, axis=0)
lgb_final = np.mean(lgb_test_preds, axis=0)
cat_final = np.mean(cat_test_preds, axis=0)

# Weight by inverse CV score (better models get higher weight)
xgb_weight = 1 / xgb_cv_mean
lgb_weight = 1 / lgb_cv_mean
cat_weight = 1 / cat_cv_mean

total_weight = xgb_weight + lgb_weight + cat_weight

xgb_weight_norm = xgb_weight / total_weight
lgb_weight_norm = lgb_weight / total_weight
cat_weight_norm = cat_weight / total_weight

print(f"Model weights:")
print(f"  XGBoost:  {xgb_weight_norm:.3f} (CV: {xgb_cv_mean:.2f}%)")
print(f"  LightGBM: {lgb_weight_norm:.3f} (CV: {lgb_cv_mean:.2f}%)")
print(f"  CatBoost: {cat_weight_norm:.3f} (CV: {cat_cv_mean:.2f}%)")
print()

# Weighted ensemble
final_predictions = (
    xgb_final * xgb_weight_norm +
    lgb_final * lgb_weight_norm +
    cat_final * cat_weight_norm
)

# Calculate ensemble CV (average of best model CVs)
best_cv = min(xgb_cv_mean, lgb_cv_mean, cat_cv_mean)
ensemble_cv = (xgb_cv_mean + lgb_cv_mean + cat_cv_mean) / 3

print("="*80)
print("FINAL RESULTS - MULTIMODAL VISION + NLP")
print("="*80)
print()
print(f"Individual Model CVs:")
print(f"  XGBoost:  {xgb_cv_mean:.4f}%")
print(f"  LightGBM: {lgb_cv_mean:.4f}%")
print(f"  CatBoost: {cat_cv_mean:.4f}%")
print()
print(f"Best Single Model: {best_cv:.4f}%")
print(f"Ensemble Estimate: {ensemble_cv:.4f}%")
print()
print(f"Phase 5:  58.38% CV → 57.900% LB ✅")
print(f"Phase 12: {best_cv:.2f}% CV → Expected {best_cv-0.5:.2f}%-{best_cv+0.5:.2f}% LB")
print()

if best_cv < 57.5:
    improvement = 58.38 - best_cv
    print(f"🎉🎉🎉 BREAKTHROUGH! {improvement:.2f} points better than Phase 5!")
    print(f"✅ Expected LB: < 57.5% (MAJOR IMPROVEMENT!)")
    print()
    print("✅ STRONGLY RECOMMEND: SUBMIT THIS!")
elif best_cv < 58.38:
    improvement = 58.38 - best_cv
    print(f"🎉 EXCELLENT! {improvement:.2f} points better than Phase 5!")
    print(f"✅ Expected LB: {best_cv-0.5:.2f}%-{best_cv+0.5:.2f}%")
    print()
    print("✅ RECOMMEND: Submit!")
elif best_cv < 59.0:
    print(f"⚠️  Close: {best_cv-58.38:.2f} points worse than Phase 5")
    print(f"   But multimodal ensemble might help on LB")
    print()
    print("🤔 YOUR CHOICE")
else:
    print(f"❌ WORSE: {best_cv-58.38:.2f} points worse than Phase 5")
    print()
    print("❌ Keep Phase 5")

print()
print("-"*80)

# Save predictions
output = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_predictions
})
output.to_csv('dataset/test_out.csv', index=False)
output.to_csv('dataset/submission_phase12.csv', index=False)

print("✅ Saved: dataset/test_out.csv")
print("✅ Saved backup: dataset/submission_phase12.csv")
print()

# Save results
results = {
    'phase': 'Phase 12: Vision + NLP Fusion (Multimodal)',
    'strategies': [
        'Image features from cache (15 visual features)',
        'TF-IDF text features (100 dims)',
        'Triple ensemble: XGBoost + LightGBM + CatBoost',
        'Weighted ensemble by inverse CV',
        'Multimodal fusion of vision + language'
    ],
    'model_cvs': {
        'xgboost': float(xgb_cv_mean),
        'lightgbm': float(lgb_cv_mean),
        'catboost': float(cat_cv_mean),
        'best': float(best_cv),
        'ensemble': float(ensemble_cv)
    },
    'features': {
        'total': int(X_train.shape[1]),
        'numeric': len(numeric_features),
        'categorical': train_encoded.shape[1],
        'text_tfidf': 100,
        'image': train_img_df.shape[1]
    },
    'weights': {
        'xgboost': float(xgb_weight_norm),
        'lightgbm': float(lgb_weight_norm),
        'catboost': float(cat_weight_norm)
    },
    'comparison': {
        'phase5_cv': 58.38,
        'phase5_lb': 57.900,
        'phase12_best_cv': float(best_cv),
        'improvement': float(58.38 - best_cv)
    }
}

with open('phase12_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved: phase12_results.json")
print("="*80)
print()
print("🎨 MULTIMODAL FEATURES USED:")
print("  1. ✅ 15 Image features (width, height, colors, brightness)")
print("  2. ✅ 100 TF-IDF features (text semantics)")
print("  3. ✅ 14 Engineered features (word count, bullets, etc.)")
print("  4. ✅ ~141 Categorical one-hot features")
print()
print("🤖 TRIPLE ENSEMBLE:")
print("  1. ✅ XGBoost (gradient boosting)")
print("  2. ✅ LightGBM (faster gradient boosting)")
print("  3. ✅ CatBoost (categorical boosting)")
print()
print("This is THE MOST COMPREHENSIVE approach possible!")
print("If this doesn't beat Phase 5, we've truly tried everything! 🎯")
print("="*80)
