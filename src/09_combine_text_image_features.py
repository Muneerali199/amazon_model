"""
Phase 7b: Combine Text + Image Features and Train Model
========================================================

This script combines text features with image features (ResNet50)
and trains an XGBoost model on the combined feature set.

Expected Impact: -2 to -4% SMAPE improvement (59% → 55-57%)

Author: ML Challenge 2025 Team
Date: October 11, 2025
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 7b: COMBINE TEXT + IMAGE FEATURES & TRAIN MODEL")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
TRAIN_TEXT_FEATURES = 'dataset/train_features.csv'
TEST_TEXT_FEATURES = 'dataset/test_features.csv'
TRAIN_IMAGE_FEATURES = 'dataset/train_image_features_resnet50.npy'
TEST_IMAGE_FEATURES = 'dataset/test_image_features_resnet50.npy'
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'dataset/test.csv'

# Output paths
SUBMISSION_FILE = 'dataset/submission_phase7_text_image.csv'
RESULTS_FILE = 'dataset/phase7_results.json'

# Model configuration
N_FOLDS = 5
RANDOM_STATE = 42

# XGBoost hyperparameters (from Phase 4b optimization)
XGBOOST_PARAMS = {
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
    'n_jobs': -1,
    'tree_method': 'hist'
}

print("Configuration:")
print(f"  N-Fold CV: {N_FOLDS}")
print(f"  Random State: {RANDOM_STATE}")
print()

# ============================================================================
# SMAPE CALCULATION
# ============================================================================

def smape(y_true, y_pred):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# ============================================================================
# LOAD DATA
# ============================================================================

print("-" * 80)
print("STEP 1: Load All Features")
print("-" * 80)
print()

# Load text features
print("Loading text features...")
train_text = pd.read_csv(TRAIN_TEXT_FEATURES)
test_text = pd.read_csv(TEST_TEXT_FEATURES)

# Get sample IDs and prices
train_sample_ids = train_text['sample_id'].values
test_sample_ids = test_text['sample_id'].values

# Load prices
train_df = pd.read_csv(TRAIN_CSV)
prices = train_df['price'].values

# Remove non-feature columns from text features
text_feature_cols = [col for col in train_text.columns 
                     if col not in ['sample_id', 'price']]

train_text_features = train_text[text_feature_cols].values
test_text_features = test_text[text_feature_cols].values

print(f"  ✅ Text features loaded")
print(f"     Training: {train_text_features.shape}")
print(f"     Test: {test_text_features.shape}")
print()

# Load image features
print("Loading image features...")
train_image_features = np.load(TRAIN_IMAGE_FEATURES)
test_image_features = np.load(TEST_IMAGE_FEATURES)

print(f"  ✅ Image features loaded")
print(f"     Training: {train_image_features.shape}")
print(f"     Test: {test_image_features.shape}")
print()

# ============================================================================
# FEATURE SCALING & COMBINATION
# ============================================================================

print("-" * 80)
print("STEP 2: Scale and Combine Features")
print("-" * 80)
print()

# Scale image features (they have large values from ResNet)
print("Scaling image features...")
image_scaler = StandardScaler()
train_image_scaled = image_scaler.fit_transform(train_image_features)
test_image_scaled = image_scaler.transform(test_image_features)

print("  ✅ Image features scaled")
print()

# Combine text + image features
print("Combining text + image features...")
X_train = np.concatenate([train_text_features, train_image_scaled], axis=1)
X_test = np.concatenate([test_text_features, test_image_scaled], axis=1)

print(f"  ✅ Combined features created")
print(f"     Training shape: {X_train.shape}")
print(f"     Test shape: {X_test.shape}")
print(f"     Total features: {X_train.shape[1]:,}")
print(f"       - Text features: {train_text_features.shape[1]}")
print(f"       - Image features: {train_image_scaled.shape[1]}")
print()

# ============================================================================
# CROSS-VALIDATION TRAINING
# ============================================================================

print("-" * 80)
print("STEP 3: Train XGBoost with Cross-Validation")
print("-" * 80)
print()

print("Training XGBoost with combined text + image features...")
print(f"Using {N_FOLDS}-fold cross-validation")
print()

# Storage for results
fold_scores = []
fold_predictions = np.zeros(len(X_test))

# K-Fold cross-validation
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"Fold {fold}/{N_FOLDS}")
    print("-" * 40)
    
    # Split data
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = prices[train_idx], prices[val_idx]
    
    # Train model
    print("  Training XGBoost...")
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Validate
    val_pred = model.predict(X_val)
    val_pred = np.clip(val_pred, 0.0, 1000.0)  # Clip predictions
    
    val_score = smape(y_val, val_pred)
    fold_scores.append(val_score)
    
    print(f"  Validation SMAPE: {val_score:.4f}%")
    
    # Predict on test set
    test_pred = model.predict(X_test)
    test_pred = np.clip(test_pred, 0.0, 1000.0)
    fold_predictions += test_pred / N_FOLDS
    
    print()

# ============================================================================
# CROSS-VALIDATION SUMMARY
# ============================================================================

print("-" * 80)
print("STEP 4: Cross-Validation Summary")
print("-" * 80)
print()

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print("Cross-Validation Results:")
print(f"  Mean SMAPE: {mean_score:.4f}%")
print(f"  Std Dev: {std_score:.4f}%")
print(f"  Min SMAPE: {np.min(fold_scores):.4f}%")
print(f"  Max SMAPE: {np.max(fold_scores):.4f}%")
print()

print("Individual Fold Scores:")
for i, score in enumerate(fold_scores, 1):
    print(f"  Fold {i}: {score:.4f}%")
print()

# ============================================================================
# COMPARISON WITH PREVIOUS PHASES
# ============================================================================

print("-" * 80)
print("STEP 5: Performance Comparison")
print("-" * 80)
print()

# Previous best results
previous_results = {
    'Phase 3 (Baseline)': 66.44,
    'Phase 4a (TF-IDF)': 60.93,
    'Phase 4b (Optimized)': 58.94,
    'Phase 5 (Advanced)': 58.38,
    'Phase 6 (Ensemble)': 58.00
}

print("Performance Progression:")
for phase, score in previous_results.items():
    print(f"  {phase}: {score:.2f}%")
print(f"  Phase 7 (Text+Image): {mean_score:.2f}%")
print()

# Calculate improvement
best_previous = 58.00  # Phase 6 ensemble
improvement = best_previous - mean_score

print(f"Improvement over Phase 6:")
print(f"  Absolute: {improvement:.2f}%")
print(f"  Relative: {100*improvement/best_previous:.1f}%")
print()

if mean_score < best_previous:
    print(f"✅ SUCCESS! New best score: {mean_score:.2f}%")
    print(f"   Improvement: -{improvement:.2f}% SMAPE")
else:
    print(f"⚠️  No improvement from image features")
    print(f"   Difference: +{-improvement:.2f}% SMAPE")
print()

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================

print("-" * 80)
print("STEP 6: Generate Submission File")
print("-" * 80)
print()

# Create submission DataFrame
submission = pd.DataFrame({
    'sample_id': test_sample_ids,
    'price': fold_predictions
})

# Validate submission
print("Submission Validation:")
print(f"  Shape: {submission.shape}")
print(f"  Columns: {list(submission.columns)}")
print(f"  Missing values: {submission.isnull().sum().sum()}")
print(f"  Negative prices: {(submission['price'] < 0).sum()}")
print()

print("Price Statistics:")
print(f"  Mean: ${submission['price'].mean():.2f}")
print(f"  Median: ${submission['price'].median():.2f}")
print(f"  Std: ${submission['price'].std():.2f}")
print(f"  Min: ${submission['price'].min():.2f}")
print(f"  Max: ${submission['price'].max():.2f}")
print()

# Save submission
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"✅ Submission saved to: {SUBMISSION_FILE}")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'phase': '7b_text_image_features',
    'model': 'XGBoost',
    'features': {
        'text_features': int(train_text_features.shape[1]),
        'image_features': int(train_image_scaled.shape[1]),
        'total_features': int(X_train.shape[1])
    },
    'cross_validation': {
        'n_folds': N_FOLDS,
        'mean_smape': float(mean_score),
        'std_smape': float(std_score),
        'min_smape': float(np.min(fold_scores)),
        'max_smape': float(np.max(fold_scores)),
        'fold_scores': [float(s) for s in fold_scores]
    },
    'improvement': {
        'previous_best': float(best_previous),
        'current': float(mean_score),
        'absolute_improvement': float(improvement),
        'relative_improvement': float(100 * improvement / best_previous)
    },
    'hyperparameters': XGBOOST_PARAMS,
    'submission_file': SUBMISSION_FILE
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: {RESULTS_FILE}")
print()

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================

print("=" * 80)
print("PHASE 7 COMPLETE: TEXT + IMAGE FEATURES!")
print("=" * 80)
print()

print("📊 Final Results:")
print(f"  Current Score: {mean_score:.2f}% SMAPE")
print(f"  Previous Best: {best_previous:.2f}% SMAPE")
print(f"  Improvement: {improvement:.2f}%")
print()

print("🎯 Leaderboard Impact:")
print(f"  Current Rank: #437 (59% SMAPE)")
if mean_score < 57:
    print(f"  Expected New Rank: ~#200-250 (top 30%)")
elif mean_score < 58:
    print(f"  Expected New Rank: ~#300-350 (significant improvement)")
else:
    print(f"  Expected New Rank: ~#350-400 (modest improvement)")
print()

print("🚀 Next Steps to Reach Top 10 (Target: <47% SMAPE):")
print()

gap_to_top10 = mean_score - 47.0
print(f"Gap to Top 10: {gap_to_top10:.2f}%")
print()

print("Recommended Next Phases:")
print("  1. Phase 8: Advanced Ensemble (LightGBM + CatBoost)")
print("     Expected: -1 to -2% improvement")
print()
print("  2. Phase 9: EfficientNet Image Features")
print("     Expected: -1 to -2% improvement")
print()
print("  3. Phase 10: Deep Learning Multi-Modal")
print("     Expected: -2 to -4% improvement")
print()
print("  4. Phase 11: Meta-Ensemble & Optimization")
print("     Expected: -1 to -2% improvement")
print()

print("💡 Submit this result to the leaderboard now and check your new rank!")
print(f"   File to submit: {SUBMISSION_FILE}")
print()

print("=" * 80)
