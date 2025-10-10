"""
PHASE 5: ADVANCED FEATURE ENGINEERING

Goal: Extract advanced features from catalog_content to close the remaining 3.94% gap
Current: 58.94% SMAPE → Target: < 55% SMAPE

Features to extract:
1. Brand names (e.g., "Apple", "Samsung", "Nike")
2. Product categories (electronics, clothing, food, etc.)
3. Material types (cotton, plastic, metal, etc.)
4. Size indicators (small, medium, large, XXL, etc.)
5. Color mentions
6. Interaction features (ipq_value × category)
7. Price range indicators based on text patterns

Compliance: 100% - All features extracted from provided train.csv only
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import json

print("=" * 80)
print("PHASE 5: ADVANCED FEATURE ENGINEERING")
print("=" * 80)
print("\n✅ COMPLIANCE: Using ONLY provided training data")
print("✅ Source: catalog_content from train.csv ONLY\n")

# Load data
print("[1/8] Loading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
train_feat = pd.read_csv('dataset/train_features.csv')
test_feat = pd.read_csv('dataset/test_features.csv')

print(f"   ✓ Training: {train.shape}")
print(f"   ✓ Test: {test.shape}")

# Feature extraction functions
def extract_brand_features(text):
    """Extract brand-related features from text"""
    text_lower = str(text).lower()
    
    # Common brand indicators
    brand_keywords = [
        'amazon', 'apple', 'samsung', 'sony', 'lg', 'microsoft', 'dell', 
        'hp', 'lenovo', 'asus', 'acer', 'nike', 'adidas', 'puma',
        'panasonic', 'philips', 'canon', 'nikon', 'bose', 'jbl'
    ]
    
    features = {}
    features['has_known_brand'] = int(any(brand in text_lower for brand in brand_keywords))
    features['brand_mentions'] = sum(1 for brand in brand_keywords if brand in text_lower)
    
    # Generic brand indicators
    features['has_brand_text'] = int(bool(re.search(r'\bbrand[:\s]', text_lower)))
    
    return features

def extract_category_features(text):
    """Extract product category features from text"""
    text_lower = str(text).lower()
    
    # Category keywords
    categories = {
        'electronics': ['electronic', 'digital', 'wireless', 'bluetooth', 'usb', 'hdmi', 
                       'battery', 'charger', 'cable', 'adapter', 'tech'],
        'clothing': ['shirt', 'pants', 'dress', 'jacket', 'shoe', 'sock', 'cloth', 
                    'cotton', 'polyester', 'fabric', 'wear', 'apparel'],
        'food': ['food', 'snack', 'drink', 'beverage', 'organic', 'nutrition', 
                'vitamin', 'supplement', 'grocery'],
        'home': ['home', 'kitchen', 'bathroom', 'furniture', 'decor', 'storage',
                'organizer', 'household'],
        'beauty': ['beauty', 'skin', 'care', 'cosmetic', 'lotion', 'shampoo',
                  'makeup', 'hair'],
        'toys': ['toy', 'game', 'play', 'kids', 'children', 'puzzle'],
        'health': ['health', 'medical', 'fitness', 'wellness', 'exercise'],
        'automotive': ['car', 'auto', 'vehicle', 'automotive', 'motor']
    }
    
    features = {}
    for category, keywords in categories.items():
        features[f'cat_{category}'] = int(any(kw in text_lower for kw in keywords))
    
    features['num_categories'] = sum(features.values())
    
    return features

def extract_material_features(text):
    """Extract material/composition features"""
    text_lower = str(text).lower()
    
    materials = {
        'metal': ['metal', 'steel', 'aluminum', 'iron', 'brass', 'copper'],
        'plastic': ['plastic', 'polymer', 'acrylic'],
        'fabric': ['cotton', 'polyester', 'nylon', 'silk', 'wool', 'fabric'],
        'wood': ['wood', 'wooden', 'bamboo', 'oak'],
        'glass': ['glass', 'crystal'],
        'leather': ['leather', 'suede']
    }
    
    features = {}
    for material, keywords in materials.items():
        features[f'mat_{material}'] = int(any(kw in text_lower for kw in keywords))
    
    features['has_material'] = int(sum(features.values()) > 0)
    
    return features

def extract_size_features(text):
    """Extract size-related features"""
    text_lower = str(text).lower()
    
    size_patterns = {
        'has_size_small': r'\b(small|sm|mini|tiny|compact)\b',
        'has_size_medium': r'\b(medium|med|md|standard)\b',
        'has_size_large': r'\b(large|lg|big|jumbo|xl|xxl)\b',
        'has_dimensions': r'\d+\s*(x|×)\s*\d+',
        'has_weight': r'\d+\s*(lb|kg|oz|gram|pound)',
        'has_volume': r'\d+\s*(ml|liter|gallon|oz|fl oz)',
    }
    
    features = {}
    for feat_name, pattern in size_patterns.items():
        features[feat_name] = int(bool(re.search(pattern, text_lower)))
    
    return features

def extract_color_features(text):
    """Extract color mentions"""
    text_lower = str(text).lower()
    
    colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 
             'purple', 'orange', 'brown', 'gray', 'grey', 'silver', 'gold']
    
    features = {}
    features['has_color'] = int(any(color in text_lower for color in colors))
    features['num_colors'] = sum(1 for color in colors if color in text_lower)
    
    return features

def extract_quality_features(text):
    """Extract quality/premium indicators"""
    text_lower = str(text).lower()
    
    quality_keywords = {
        'premium': ['premium', 'deluxe', 'luxury', 'high-end', 'professional'],
        'economy': ['economy', 'budget', 'value', 'affordable', 'cheap'],
        'quality': ['quality', 'durable', 'sturdy', 'heavy-duty', 'reliable']
    }
    
    features = {}
    for qual_type, keywords in quality_keywords.items():
        features[f'qual_{qual_type}'] = int(any(kw in text_lower for kw in keywords))
    
    return features

def extract_quantity_features(text):
    """Extract pack/set/bundle information"""
    text_lower = str(text).lower()
    
    features = {}
    features['has_pack'] = int(bool(re.search(r'\b(pack|ct|count)\b', text_lower)))
    features['has_set'] = int(bool(re.search(r'\bset\b', text_lower)))
    features['has_bundle'] = int(bool(re.search(r'\bbundle\b', text_lower)))
    
    # Extract pack numbers (e.g., "24 pack", "12 ct")
    pack_match = re.search(r'(\d+)\s*(pack|ct|count)', text_lower)
    features['pack_size'] = int(pack_match.group(1)) if pack_match else 0
    
    return features

# Extract features
print("\n[2/8] Extracting advanced features...")
print("   → Brand features...")

advanced_features_train = []
advanced_features_test = []

for idx, row in train.iterrows():
    text = row['catalog_content']
    features = {}
    
    features.update(extract_brand_features(text))
    features.update(extract_category_features(text))
    features.update(extract_material_features(text))
    features.update(extract_size_features(text))
    features.update(extract_color_features(text))
    features.update(extract_quality_features(text))
    features.update(extract_quantity_features(text))
    
    advanced_features_train.append(features)
    
    if idx % 10000 == 0:
        print(f"      Processed {idx:,}/75,000 training samples...")

print("   ✓ Training features extracted")

print("   → Extracting test features...")
for idx, row in test.iterrows():
    text = row['catalog_content']
    features = {}
    
    features.update(extract_brand_features(text))
    features.update(extract_category_features(text))
    features.update(extract_material_features(text))
    features.update(extract_size_features(text))
    features.update(extract_color_features(text))
    features.update(extract_quality_features(text))
    features.update(extract_quantity_features(text))
    
    advanced_features_test.append(features)
    
    if idx % 10000 == 0:
        print(f"      Processed {idx:,}/75,000 test samples...")

print("   ✓ Test features extracted")

# Convert to DataFrames
train_adv = pd.DataFrame(advanced_features_train)
test_adv = pd.DataFrame(advanced_features_test)

print(f"\n   ✓ Advanced features: {train_adv.shape[1]} new features")
print(f"   Feature names: {list(train_adv.columns)[:5]}...")

# Combine with existing features
print("\n[3/8] Combining features...")

# Merge with existing features
train_combined = train_feat.merge(train_adv, left_index=True, right_index=True)
test_combined = test_feat.merge(test_adv, left_index=True, right_index=True)

# Add interaction features
print("   → Creating interaction features...")

# IPQ value × category interactions
for cat_col in ['cat_electronics', 'cat_clothing', 'cat_food', 'cat_home']:
    if cat_col in train_combined.columns:
        train_combined[f'ipq_x_{cat_col}'] = train_combined['ipq_value'].fillna(0) * train_combined[cat_col]
        test_combined[f'ipq_x_{cat_col}'] = test_combined['ipq_value'].fillna(0) * test_combined[cat_col]

# Word count × quality
if 'qual_premium' in train_combined.columns:
    train_combined['words_x_premium'] = train_combined['word_count'] * train_combined['qual_premium']
    test_combined['words_x_premium'] = test_combined['word_count'] * test_combined['qual_premium']

# Price range indicators from text patterns
train_combined['high_value_indicator'] = (
    train_combined['qual_premium'].fillna(0) + 
    train_combined['has_known_brand'].fillna(0) +
    train_combined['mat_metal'].fillna(0)
) / 3.0

test_combined['high_value_indicator'] = (
    test_combined['qual_premium'].fillna(0) + 
    test_combined['has_known_brand'].fillna(0) +
    test_combined['mat_metal'].fillna(0)
) / 3.0

print(f"   ✓ Total features: {train_combined.shape[1]}")

# Prepare features for modeling
print("\n[4/8] Preparing features for modeling...")

# Drop sample_id and target, and also drop price-derived features that don't exist in test
exclude_cols = ['sample_id', 'price', 'log_price', 'price_per_unit', 'item_name', 'image_filename']
feature_cols = [col for col in train_combined.columns if col not in exclude_cols]

# Only use columns that exist in both train and test
common_cols = [col for col in feature_cols if col in test_combined.columns]

X_train_base = train_combined[common_cols].fillna(0)
X_test_base = test_combined[common_cols].fillna(0)

# Handle ipq_unit encoding (if exists)
if 'ipq_unit' in X_train_base.columns:
    # Get dummy variables
    train_ipq = pd.get_dummies(train_combined['ipq_unit'], prefix='ipq_unit', dummy_na=True)
    test_ipq = pd.get_dummies(test_combined['ipq_unit'], prefix='ipq_unit', dummy_na=True)
    
    # Align columns
    train_ipq, test_ipq = train_ipq.align(test_ipq, join='left', axis=1, fill_value=0)
    
    # Drop original ipq_unit and add encoded
    X_train_base = X_train_base.drop('ipq_unit', axis=1)
    X_test_base = X_test_base.drop('ipq_unit', axis=1)
    
    X_train_base = pd.concat([X_train_base.reset_index(drop=True), train_ipq.reset_index(drop=True)], axis=1)
    X_test_base = pd.concat([X_test_base.reset_index(drop=True), test_ipq.reset_index(drop=True)], axis=1)

print(f"   ✓ Base features prepared: {X_train_base.shape[1]}")

# Add TF-IDF features (from Phase 4)
print("\n[5/8] Adding TF-IDF features...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# TF-IDF on catalog_content
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

train_tfidf = tfidf.fit_transform(train['catalog_content'])
test_tfidf = tfidf.transform(test['catalog_content'])

# Reduce dimensionality
svd = TruncatedSVD(n_components=100, random_state=42)
train_tfidf_reduced = svd.fit_transform(train_tfidf)
test_tfidf_reduced = svd.transform(test_tfidf)

# Convert to DataFrame
tfidf_cols = [f'tfidf_{i}' for i in range(100)]
train_tfidf_df = pd.DataFrame(train_tfidf_reduced, columns=tfidf_cols)
test_tfidf_df = pd.DataFrame(test_tfidf_reduced, columns=tfidf_cols)

# Combine all features
X_train = pd.concat([X_train_base.reset_index(drop=True), train_tfidf_df], axis=1)
X_test = pd.concat([X_test_base.reset_index(drop=True), test_tfidf_df], axis=1)

y_train = train['price'].values
train_ids = train['sample_id'].values
test_ids = test['sample_id'].values

print(f"   ✓ Total features: {X_train.shape[1]}")
print(f"   ✓ Training: {X_train.shape}, Test: {X_test.shape}")

# Train model with best parameters from Phase 4
print("\n[6/8] Training model with Phase 5 features...")

best_params = {
    'learning_rate': 0.03,
    'max_depth': 10,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'n_estimators': 700,
    'random_state': 42,
    'n_jobs': -1
}

# Custom SMAPE metric
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
smape_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_tr, y_tr, verbose=False)
    
    y_pred = model.predict(X_val)
    fold_smape = smape(y_val, y_pred)
    smape_scores.append(fold_smape)
    models.append(model)
    
    print(f"   Fold {fold}: SMAPE = {fold_smape:.4f}%")

mean_smape = np.mean(smape_scores)
std_smape = np.std(smape_scores)

print(f"   ✓ Mean SMAPE: {mean_smape:.4f}% (±{std_smape:.4f}%)")

# Compare with previous phases
print("\n[7/8] Comparing results...")
baseline_smape = 66.4390
tfidf_smape = 60.9300
optimized_smape = 58.9426
phase5_smape = mean_smape

print(f"   Phase 3 (Baseline):    {baseline_smape:.4f}%")
print(f"   Phase 4a (TF-IDF):     {tfidf_smape:.4f}%")
print(f"   Phase 4b (Optimized):  {optimized_smape:.4f}%")
print(f"   Phase 5 (Advanced):    {phase5_smape:.4f}%")
print()
print(f"   📊 Improvement from optimized: {optimized_smape - phase5_smape:+.4f}%")
print(f"   📊 Total improvement: {baseline_smape - phase5_smape:.4f}%")
print(f"   🎯 Gap to target (55%): {phase5_smape - 55.0:.4f}%")

# Check if target achieved
target_achieved = phase5_smape < 55.0

if target_achieved:
    print(f"\n   🎉 TARGET ACHIEVED! {phase5_smape:.4f}% < 55%")
else:
    print(f"\n   ⚠️  Target not yet achieved. Gap: {phase5_smape - 55.0:.4f}%")

# Generate predictions
print("\n[8/8] Generating predictions...")

# Train final model on all data
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train, verbose=False)

predictions = final_model.predict(X_test)

print(f"   ✓ Predictions: {len(predictions):,}")
print(f"   ✓ Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
print(f"   ✓ Mean: ${predictions.mean():.2f}")

# Save submission
submission = pd.DataFrame({
    'sample_id': test_ids,
    'price': predictions
})

submission.to_csv('dataset/submission_xgboost_phase5.csv', index=False)
print(f"   ✓ Saved: dataset/submission_xgboost_phase5.csv")

# Save results
results = {
    'phase': 'Phase 5 - Advanced Feature Engineering',
    'baseline_smape': baseline_smape,
    'tfidf_smape': tfidf_smape,
    'optimized_smape': optimized_smape,
    'phase5_smape': float(phase5_smape),
    'total_improvement': float(baseline_smape - phase5_smape),
    'improvement_from_phase4b': float(optimized_smape - phase5_smape),
    'target_achieved': bool(target_achieved),
    'gap_to_target': float(phase5_smape - 55.0),
    'feature_count': X_train.shape[1],
    'advanced_features_added': train_adv.shape[1],
    'fold_scores': [float(s) for s in smape_scores],
    'submission_file': 'dataset/submission_xgboost_phase5.csv'
}

with open('phase5_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✓ Saved: phase5_results.json")

print("\n" + "=" * 80)
print("PHASE 5 COMPLETE!")
print("=" * 80)
print(f"\n✅ Final SMAPE: {phase5_smape:.4f}%")
print(f"✅ Total improvement: {baseline_smape - phase5_smape:.4f}%")

if target_achieved:
    print(f"✅ TARGET ACHIEVED! 🎉")
else:
    print(f"⚠️  Gap remaining: {phase5_smape - 55.0:.4f}%")
    print(f"💡 Consider: Ensemble methods or image features")

print("\n" + "=" * 80)
