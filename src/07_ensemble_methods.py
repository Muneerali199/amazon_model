"""
PHASE 6: ENSEMBLE METHODS

Goal: Combine multiple models to reduce SMAPE from 58.38% to < 55%
Strategy: Train multiple models and use weighted averaging/stacking

Models to ensemble:
1. XGBoost (current best)
2. Random Forest
3. LightGBM (if available)
4. Gradient Boosting

Ensemble approaches:
- Weighted averaging based on CV performance
- Stacking with meta-learner

Compliance: 100% - All models trained on provided data only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 6: ENSEMBLE METHODS")
print("=" * 80)
print("\n✅ COMPLIANCE: Using ONLY provided training data")
print("✅ All models trained on train.csv ONLY\n")

# Load the features prepared in Phase 5
print("[1/7] Loading Phase 5 features...")

# We'll reconstruct the features from Phase 5
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
train_feat = pd.read_csv('dataset/train_features.csv')
test_feat = pd.read_csv('dataset/test_features.csv')

# Load Phase 5 advanced features (we'll regenerate them quickly)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Quick feature extraction (simplified from Phase 5)
def extract_quick_features(text):
    """Extract key features quickly"""
    text_lower = str(text).lower()
    features = {}
    
    # Brand
    brands = ['amazon', 'apple', 'samsung', 'sony', 'nike', 'adidas']
    features['has_brand'] = int(any(b in text_lower for b in brands))
    
    # Categories
    features['is_electronics'] = int(any(w in text_lower for w in ['electronic', 'digital', 'wireless', 'battery']))
    features['is_clothing'] = int(any(w in text_lower for w in ['shirt', 'pants', 'dress', 'cloth', 'cotton']))
    features['is_food'] = int(any(w in text_lower for w in ['food', 'snack', 'organic']))
    
    # Quality
    features['is_premium'] = int(any(w in text_lower for w in ['premium', 'luxury', 'deluxe']))
    
    return features

# Extract features
print("   → Extracting quick features...")
train_quick = pd.DataFrame([extract_quick_features(text) for text in train['catalog_content']])
test_quick = pd.DataFrame([extract_quick_features(text) for text in test['catalog_content']])

# Prepare base features
exclude_cols = ['sample_id', 'price', 'log_price', 'price_per_unit', 'item_name', 'image_filename']
feature_cols = [col for col in train_feat.columns if col not in exclude_cols]
common_cols = [col for col in feature_cols if col in test_feat.columns]

X_train_base = train_feat[common_cols].fillna(0)
X_test_base = test_feat[common_cols].fillna(0)

# Add quick features
X_train_base = pd.concat([X_train_base.reset_index(drop=True), train_quick], axis=1)
X_test_base = pd.concat([X_test_base.reset_index(drop=True), test_quick], axis=1)

# Handle ipq_unit encoding
if 'ipq_unit' in X_train_base.columns:
    train_ipq = pd.get_dummies(train_feat['ipq_unit'], prefix='ipq_unit', dummy_na=True)
    test_ipq = pd.get_dummies(test_feat['ipq_unit'], prefix='ipq_unit', dummy_na=True)
    train_ipq, test_ipq = train_ipq.align(test_ipq, join='left', axis=1, fill_value=0)
    
    X_train_base = X_train_base.drop('ipq_unit', axis=1)
    X_test_base = X_test_base.drop('ipq_unit', axis=1)
    
    X_train_base = pd.concat([X_train_base.reset_index(drop=True), train_ipq.reset_index(drop=True)], axis=1)
    X_test_base = pd.concat([X_test_base.reset_index(drop=True), test_ipq.reset_index(drop=True)], axis=1)

# Add TF-IDF
print("   → Adding TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)
train_tfidf = tfidf.fit_transform(train['catalog_content'])
test_tfidf = tfidf.transform(test['catalog_content'])

svd = TruncatedSVD(n_components=100, random_state=42)
train_tfidf_reduced = svd.fit_transform(train_tfidf)
test_tfidf_reduced = svd.transform(test_tfidf)

tfidf_cols = [f'tfidf_{i}' for i in range(100)]
train_tfidf_df = pd.DataFrame(train_tfidf_reduced, columns=tfidf_cols)
test_tfidf_df = pd.DataFrame(test_tfidf_reduced, columns=tfidf_cols)

X_train = pd.concat([X_train_base.reset_index(drop=True), train_tfidf_df], axis=1)
X_test = pd.concat([X_test_base.reset_index(drop=True), test_tfidf_df], axis=1)

# Convert to numpy arrays to avoid DataFrame indexing issues
X_train_np = X_train.values
X_test_np = X_test.values

y_train = train['price'].values
train_ids = train['sample_id'].values
test_ids = test['sample_id'].values

print(f"   ✓ Features ready: {X_train_np.shape}")

# SMAPE metric
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Initialize models
print("\n[2/7] Initializing ensemble models...")

models = {
    'xgboost': xgb.XGBRegressor(
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.9,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=0.5,
        n_estimators=700,
        random_state=42,
        n_jobs=-1
    ),
    'random_forest': RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
}

print(f"   ✓ Models initialized: {len(models)}")

# Train and evaluate each model
print("\n[3/7] Training individual models with 5-fold CV...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_results = {}

for model_name, model in models.items():
    print(f"\n   → Training {model_name}...")
    fold_scores = []
    fold_predictions_val = []
    fold_predictions_test = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np), 1):
        X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Clone and train model
        if model_name == 'xgboost':
            fold_model = xgb.XGBRegressor(**model.get_params())
        elif model_name == 'random_forest':
            fold_model = RandomForestRegressor(**model.get_params())
        else:
            fold_model = GradientBoostingRegressor(**model.get_params())
        
        fold_model.fit(X_tr, y_tr)
        
        # Validation predictions
        y_pred_val = fold_model.predict(X_val)
        fold_smape = smape(y_val, y_pred_val)
        fold_scores.append(fold_smape)
        
        # Test predictions
        y_pred_test = fold_model.predict(X_test_np)
        fold_predictions_test.append(y_pred_test)
        
        print(f"      Fold {fold}: SMAPE = {fold_smape:.4f}%")
    
    mean_smape = np.mean(fold_scores)
    std_smape = np.std(fold_scores)
    
    # Average test predictions across folds
    avg_test_pred = np.mean(fold_predictions_test, axis=0)
    
    model_results[model_name] = {
        'smape': mean_smape,
        'std': std_smape,
        'fold_scores': fold_scores,
        'test_predictions': avg_test_pred
    }
    
    print(f"   ✓ {model_name}: {mean_smape:.4f}% (±{std_smape:.4f}%)")

# Display individual model results
print("\n[4/7] Individual model performance:")
for model_name, results in model_results.items():
    print(f"   {model_name:20} SMAPE: {results['smape']:.4f}% (±{results['std']:.4f}%)")

# Simple weighted average ensemble
print("\n[5/7] Creating weighted average ensemble...")

# Weights inversely proportional to SMAPE (better models get higher weights)
smapes = np.array([results['smape'] for results in model_results.values()])
weights = 1 / smapes
weights = weights / weights.sum()  # Normalize to sum to 1

print("   Ensemble weights:")
for (model_name, results), weight in zip(model_results.items(), weights):
    print(f"      {model_name:20} weight: {weight:.4f}")

# Combine predictions
ensemble_predictions = np.zeros(len(test_ids))
for (model_name, results), weight in zip(model_results.items(), weights):
    ensemble_predictions += weight * results['test_predictions']

# Evaluate ensemble with cross-validation
print("\n[6/7] Evaluating ensemble with CV...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
ensemble_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np), 1):
    X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train each model and combine predictions
    fold_predictions = []
    for model_name, model in models.items():
        if model_name == 'xgboost':
            fold_model = xgb.XGBRegressor(**model.get_params())
        elif model_name == 'random_forest':
            fold_model = RandomForestRegressor(**model.get_params())
        else:
            fold_model = GradientBoostingRegressor(**model.get_params())
        
        fold_model.fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_val)
        fold_predictions.append(fold_pred)
    
    # Weighted average
    ensemble_pred = np.zeros(len(y_val))
    for pred, weight in zip(fold_predictions, weights):
        ensemble_pred += weight * pred
    
    fold_smape = smape(y_val, ensemble_pred)
    ensemble_cv_scores.append(fold_smape)
    print(f"   Fold {fold}: SMAPE = {fold_smape:.4f}%")

ensemble_mean = np.mean(ensemble_cv_scores)
ensemble_std = np.std(ensemble_cv_scores)

print(f"\n   ✓ Ensemble Mean SMAPE: {ensemble_mean:.4f}% (±{ensemble_std:.4f}%)")

# Compare results
print("\n[7/7] Final comparison:")
print(f"   Phase 3 (Baseline):         66.4390%")
print(f"   Phase 4a (TF-IDF):          60.9300%")
print(f"   Phase 4b (Optimized):       58.9426%")
print(f"   Phase 5 (Advanced):         58.3761%")
print(f"   Phase 6 (Ensemble):         {ensemble_mean:.4f}%")
print()
print(f"   📊 Best individual model:    {min(results['smape'] for results in model_results.values()):.4f}%")
print(f"   📊 Ensemble improvement:     {58.3761 - ensemble_mean:+.4f}%")
print(f"   📊 Total improvement:        {66.4390 - ensemble_mean:.4f}%")
print(f"   🎯 Gap to target (55%):      {ensemble_mean - 55.0:.4f}%")

# Check if target achieved
target_achieved = ensemble_mean < 55.0

if target_achieved:
    print(f"\n   🎉 TARGET ACHIEVED! {ensemble_mean:.4f}% < 55%")
else:
    print(f"\n   ⚠️  Target not yet achieved. Gap: {ensemble_mean - 55.0:.4f}%")

# Clip predictions to valid range
ensemble_predictions = np.clip(ensemble_predictions, 0.0, 1000.0)

print(f"\nGenerating final predictions...")
print(f"   ✓ Predictions: {len(ensemble_predictions):,}")
print(f"   ✓ Range: ${ensemble_predictions.min():.2f} - ${ensemble_predictions.max():.2f}")
print(f"   ✓ Mean: ${ensemble_predictions.mean():.2f}")

# Save ensemble submission
submission = pd.DataFrame({
    'sample_id': test_ids,
    'price': ensemble_predictions
})

submission.to_csv('dataset/submission_xgboost_ensemble.csv', index=False)
print(f"   ✓ Saved: dataset/submission_xgboost_ensemble.csv")

# Save results
results = {
    'phase': 'Phase 6 - Ensemble Methods',
    'baseline_smape': 66.4390,
    'tfidf_smape': 60.9300,
    'optimized_smape': 58.9426,
    'phase5_smape': 58.3761,
    'ensemble_smape': float(ensemble_mean),
    'ensemble_std': float(ensemble_std),
    'total_improvement': float(66.4390 - ensemble_mean),
    'improvement_from_phase5': float(58.3761 - ensemble_mean),
    'target_achieved': bool(target_achieved),
    'gap_to_target': float(ensemble_mean - 55.0),
    'individual_models': {
        name: {
            'smape': float(results['smape']),
            'std': float(results['std']),
            'weight': float(weight)
        }
        for (name, results), weight in zip(model_results.items(), weights)
    },
    'fold_scores': [float(s) for s in ensemble_cv_scores],
    'submission_file': 'dataset/submission_xgboost_ensemble.csv'
}

with open('phase6_ensemble_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✓ Saved: phase6_ensemble_results.json")

print("\n" + "=" * 80)
print("PHASE 6 COMPLETE!")
print("=" * 80)
print(f"\n✅ Ensemble SMAPE: {ensemble_mean:.4f}%")
print(f"✅ Total improvement: {66.4390 - ensemble_mean:.4f}%")

if target_achieved:
    print(f"✅ TARGET ACHIEVED! 🎉🎉🎉")
else:
    print(f"⚠️  Gap remaining: {ensemble_mean - 55.0:.4f}%")
    if ensemble_mean < 56.0:
        print(f"💡 Very close! Consider: Fine-tuning ensemble weights or image features")
    else:
        print(f"💡 Recommend: Add image features to close remaining gap")

print("\n" + "=" * 80)
