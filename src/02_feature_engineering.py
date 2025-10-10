"""
Phase 2: Feature Engineering
ML Challenge 2025 - Smart Product Pricing Challenge

This script extracts features from:
1. Text (catalog_content) - IPQ, brand, categories, text embeddings
2. Images (image_link) - Will be prepared for download
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 2: FEATURE ENGINEERING")
print("ML Challenge 2025 - Smart Product Pricing Challenge")
print("="*80)
print("\n")

# =============================================================================
# Load Data
# =============================================================================
print("📊 Loading datasets...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
print(f"✅ Loaded {len(train_df):,} training samples")
print(f"✅ Loaded {len(test_df):,} test samples")
print("\n")

# =============================================================================
# TASK 2.1: Extract Item Pack Quantity (IPQ)
# =============================================================================
print("="*80)
print("📦 TASK 2.1: Extracting Item Pack Quantity (IPQ)")
print("-"*80)

def extract_ipq(text):
    """
    Extract Item Pack Quantity from catalog content
    Pattern: Value: 72.0\nUnit: Fl Oz
    """
    try:
        # Extract value
        value_match = re.search(r'Value:\s*(\d+\.?\d*)', str(text))
        value = float(value_match.group(1)) if value_match else 1.0
        
        # Extract unit
        unit_match = re.search(r'Unit:\s*([^\n]+)', str(text))
        unit = unit_match.group(1).strip() if unit_match else 'Count'
        
        return value, unit
    except:
        return 1.0, 'Count'

# Extract IPQ for training data
print("Extracting IPQ from training data...")
train_df[['ipq_value', 'ipq_unit']] = train_df['catalog_content'].apply(
    lambda x: pd.Series(extract_ipq(x))
)

# Extract IPQ for test data
print("Extracting IPQ from test data...")
test_df[['ipq_value', 'ipq_unit']] = test_df['catalog_content'].apply(
    lambda x: pd.Series(extract_ipq(x))
)

print(f"\n✅ IPQ extracted successfully!")
print(f"\nIPQ Statistics (Training Data):")
print(f"  Mean value: {train_df['ipq_value'].mean():.2f}")
print(f"  Median value: {train_df['ipq_value'].median():.2f}")
print(f"  Min value: {train_df['ipq_value'].min():.2f}")
print(f"  Max value: {train_df['ipq_value'].max():.2f}")
print(f"\nTop 5 most common units:")
print(train_df['ipq_unit'].value_counts().head())
print("\n")

# =============================================================================
# TASK 2.2: Basic Text Features
# =============================================================================
print("="*80)
print("📝 TASK 2.2: Creating Basic Text Features")
print("-"*80)

def extract_text_features(df):
    """Extract basic text features from catalog content"""
    print("  - Character count")
    df['char_count'] = df['catalog_content'].str.len()
    
    print("  - Word count")
    df['word_count'] = df['catalog_content'].str.split().str.len()
    
    print("  - Number of bullet points")
    df['bullet_points'] = df['catalog_content'].str.count('Bullet Point')
    
    print("  - Has product description")
    df['has_description'] = df['catalog_content'].str.contains('Product Description', na=False).astype(int)
    
    print("  - Number of numbers in text")
    df['num_count'] = df['catalog_content'].apply(lambda x: len(re.findall(r'\d+', str(x))))
    
    print("  - Number of uppercase words (potential brands)")
    df['uppercase_words'] = df['catalog_content'].apply(
        lambda x: len([w for w in str(x).split() if w.isupper() and len(w) > 2])
    )
    
    print("  - Average word length")
    df['avg_word_length'] = df['catalog_content'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    return df

print("\nExtracting text features from training data...")
train_df = extract_text_features(train_df)

print("\nExtracting text features from test data...")
test_df = extract_text_features(test_df)

print("\n✅ Basic text features created!")
print(f"\nFeature Statistics (Training Data):")
print(f"  Avg char count: {train_df['char_count'].mean():.2f}")
print(f"  Avg word count: {train_df['word_count'].mean():.2f}")
print(f"  Avg bullet points: {train_df['bullet_points'].mean():.2f}")
print(f"  Products with description: {train_df['has_description'].sum():,} ({train_df['has_description'].mean()*100:.1f}%)")
print("\n")

# =============================================================================
# TASK 2.3: Extract Brand and Category Keywords
# =============================================================================
print("="*80)
print("🏷️ TASK 2.3: Extracting Brand and Category Keywords")
print("-"*80)

def extract_item_name(text):
    """Extract the item name from catalog content"""
    match = re.search(r'Item Name:\s*([^\n]+)', str(text))
    return match.group(1).strip() if match else ''

# Extract item names
print("Extracting item names...")
train_df['item_name'] = train_df['catalog_content'].apply(extract_item_name)
test_df['item_name'] = test_df['catalog_content'].apply(extract_item_name)

print("✅ Item names extracted!")

# Common brand keywords (you can expand this list)
common_brands = [
    'Amazon', 'Organic', 'Premium', 'Natural', 'Classic', 'Original',
    'Fresh', 'Pure', 'Whole', 'Real', 'Best', 'Great', 'Simply'
]

# Category keywords
category_keywords = {
    'food': ['Food', 'Snack', 'Candy', 'Cookie', 'Chocolate', 'Cheese', 'Meat', 'Fruit', 'Vegetable'],
    'beverage': ['Juice', 'Water', 'Coffee', 'Tea', 'Drink', 'Soda', 'Beer', 'Wine'],
    'grocery': ['Sauce', 'Oil', 'Flour', 'Sugar', 'Salt', 'Spice', 'Seasoning'],
    'health': ['Organic', 'Natural', 'Gluten Free', 'Vegan', 'Non-GMO'],
    'personal_care': ['Soap', 'Shampoo', 'Lotion', 'Deodorant', 'Cream'],
    'household': ['Clean', 'Detergent', 'Paper', 'Towel', 'Bag']
}

def extract_category_features(text):
    """Extract category features from text"""
    text_lower = str(text).lower()
    features = {}
    
    for category, keywords in category_keywords.items():
        features[f'is_{category}'] = int(any(kw.lower() in text_lower for kw in keywords))
    
    return features

print("\nExtracting category features...")
# For training data
category_features_train = train_df['catalog_content'].apply(extract_category_features).apply(pd.Series)
train_df = pd.concat([train_df, category_features_train], axis=1)

# For test data
category_features_test = test_df['catalog_content'].apply(extract_category_features).apply(pd.Series)
test_df = pd.concat([test_df, category_features_test], axis=1)

print("✅ Category features extracted!")
print(f"\nCategory Distribution (Training Data):")
for category in category_keywords.keys():
    count = train_df[f'is_{category}'].sum()
    pct = (count / len(train_df)) * 100
    print(f"  {category}: {count:,} ({pct:.1f}%)")
print("\n")

# =============================================================================
# TASK 2.4: Price-Related Features (Training Only)
# =============================================================================
print("="*80)
print("💰 TASK 2.4: Creating Price-Related Features")
print("-"*80)

# Log transformation of price
print("Creating log-transformed price...")
train_df['log_price'] = np.log1p(train_df['price'])
print(f"✅ Log price created (mean: {train_df['log_price'].mean():.2f})")

# Price per unit (using IPQ)
print("Creating price per unit...")
train_df['price_per_unit'] = train_df['price'] / train_df['ipq_value'].replace(0, 1)
print(f"✅ Price per unit created (mean: ${train_df['price_per_unit'].mean():.2f})")
print("\n")

# =============================================================================
# TASK 2.5: Prepare Image Download List
# =============================================================================
print("="*80)
print("🖼️ TASK 2.5: Preparing Image Download Information")
print("-"*80)

# Create directories for images
train_img_dir = Path('dataset/train_images')
test_img_dir = Path('dataset/test_images')

train_img_dir.mkdir(exist_ok=True, parents=True)
test_img_dir.mkdir(exist_ok=True, parents=True)

print(f"✅ Image directories created:")
print(f"   {train_img_dir}")
print(f"   {test_img_dir}")

# Add image filename feature
train_df['image_filename'] = train_df['image_link'].apply(lambda x: Path(x).name if pd.notna(x) else '')
test_df['image_filename'] = test_df['image_link'].apply(lambda x: Path(x).name if pd.notna(x) else '')

print(f"\n✅ Image filenames extracted")
print(f"   Training images: {train_df['image_filename'].notna().sum():,}")
print(f"   Test images: {test_df['image_filename'].notna().sum():,}")
print("\n")

# =============================================================================
# TASK 2.6: Feature Summary and Statistics
# =============================================================================
print("="*80)
print("📊 TASK 2.6: Feature Summary")
print("-"*80)

# List all numeric features (excluding target and derived features)
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in ['sample_id', 'price', 'log_price', 'price_per_unit']]

print(f"\n✅ Created {len(numeric_features)} numeric features:")
for i, feat in enumerate(numeric_features, 1):
    print(f"   {i}. {feat}")

print(f"\n📈 Feature Correlations with Price (Top 10):")
correlations = train_df[numeric_features + ['price']].corr()['price'].sort_values(ascending=False)
print(correlations.head(10))
print("\n")

# =============================================================================
# TASK 2.7: Save Processed Features
# =============================================================================
print("="*80)
print("💾 TASK 2.7: Saving Processed Features")
print("-"*80)

# Select features to save
feature_columns = ['sample_id'] + numeric_features + ['ipq_unit', 'item_name', 'image_filename']

# Save training features
train_features = train_df[feature_columns + ['price', 'log_price', 'price_per_unit']]
train_features.to_csv('dataset/train_features.csv', index=False)
print(f"✅ Training features saved: dataset/train_features.csv")
print(f"   Shape: {train_features.shape}")

# Save test features (without price-related columns)
test_features = test_df[feature_columns]
test_features.to_csv('dataset/test_features.csv', index=False)
print(f"✅ Test features saved: dataset/test_features.csv")
print(f"   Shape: {test_features.shape}")

# Save feature names for later use
feature_info = {
    'numeric_features': numeric_features,
    'categorical_features': ['ipq_unit'],
    'text_features': ['item_name'],
    'image_features': ['image_filename'],
    'target': 'price',
    'log_target': 'log_price'
}

with open('dataset/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"✅ Feature info saved: dataset/feature_info.json")
print("\n")

# =============================================================================
# SUMMARY
# =============================================================================
print("="*80)
print("📊 PHASE 2 SUMMARY: FEATURE ENGINEERING COMPLETE")
print("="*80)

print("\n✅ FEATURES CREATED:")
print(f"   1. IPQ Features: ipq_value, ipq_unit")
print(f"   2. Text Features: {train_df[['char_count', 'word_count', 'bullet_points']].shape[1]} basic features")
print(f"   3. Category Features: {len(category_keywords)} category indicators")
print(f"   4. Price Features: log_price, price_per_unit")
print(f"   5. Image Features: image_filename (ready for download)")

print(f"\n📊 FEATURE STATISTICS:")
print(f"   Total numeric features: {len(numeric_features)}")
print(f"   Training samples: {len(train_features):,}")
print(f"   Test samples: {len(test_features):,}")

print(f"\n🔗 TOP CORRELATIONS WITH PRICE:")
top_corr = correlations.head(5)
for feat, corr in top_corr.items():
    if feat != 'price':
        print(f"   {feat}: {corr:.3f}")

print(f"\n⏭️ NEXT STEPS FOR PHASE 3:")
print(f"   1. Download product images (optional - 75K training + 75K test)")
print(f"   2. Extract image features using pre-trained models (ResNet/EfficientNet)")
print(f"   3. Create text embeddings using sentence-transformers (local)")
print(f"   4. Build baseline models (XGBoost, LightGBM)")
print(f"   5. Train and evaluate models")

print("\n" + "="*80)
print("✅ PHASE 2 COMPLETE: Ready for Phase 3 (Modeling)")
print("="*80)
print("\n")

print("💡 TIP: You can now:")
print("   - Review: dataset/train_features.csv")
print("   - Check: dataset/feature_info.json")
print("   - Start modeling with the extracted features!")
print("\n")
