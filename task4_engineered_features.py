"""
TASK 4: Engineered Features (INSTANT - 30 seconds!)
Uses: Fast regex + string operations
Output: engineered_features.npz
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 4: ENGINEERED FEATURES (30 sec)")
print("=" * 70)

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")

def create_features(df):
    """Lightning-fast feature engineering"""
    feats = pd.DataFrame()
    text = df['catalog_content'].fillna('')
    
    # Basic stats (FAST)
    feats['text_len'] = text.str.len()
    feats['word_count'] = text.str.split().str.len()
    feats['avg_word_len'] = feats['text_len'] / (feats['word_count'] + 1)
    
    # Price indicators (HIGH IMPACT)
    feats['has_price'] = text.str.contains(r'\$|price', case=False, na=False).astype(int)
    feats['has_discount'] = text.str.contains(r'off|discount|save', case=False, na=False).astype(int)
    
    # Quality signals
    feats['has_premium'] = text.str.contains(r'premium|luxury|pro', case=False, na=False).astype(int)
    feats['has_budget'] = text.str.contains(r'cheap|budget|affordable', case=False, na=False).astype(int)
    
    # Product info
    feats['has_size'] = text.str.contains(r'\d+\s*(inch|cm|mm)', case=False, na=False).astype(int)
    feats['has_weight'] = text.str.contains(r'\d+\s*(kg|lb|oz)', case=False, na=False).astype(int)
    feats['has_brand'] = text.str.contains(r'brand|™|®', case=False, na=False).astype(int)
    
    # Text patterns
    feats['upper_ratio'] = text.str.count(r'[A-Z]') / (feats['text_len'] + 1)
    feats['digit_ratio'] = text.str.count(r'[0-9]') / (feats['text_len'] + 1)
    
    # Amazon specific
    feats['bullet_count'] = text.str.count(r'Bullet Point')
    
    return feats.fillna(0).values

print("\n🔥 Creating features...")
train_features = create_features(train_df)
test_features = create_features(test_df)

# Save
print(f"\n💾 Saving features: {train_features.shape}")
np.savez_compressed('engineered_features.npz',
                    train=train_features,
                    test=test_features)

print("\n" + "=" * 70)
print("✅ TASK 4 COMPLETE! File: engineered_features.npz")
print("=" * 70)
