"""
TASK 3: TF-IDF Features (SUPER FAST - 2 minutes!)
Uses: TF-IDF on text
Output: tfidf_features.npz
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 3: TF-IDF FEATURES (2 min)")
print("=" * 70)

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")

# TF-IDF
print("\n📝 Creating TF-IDF features...")
train_text = train_df['catalog_content'].fillna('').astype(str)
test_text = test_df['catalog_content'].fillna('').astype(str)

tfidf = TfidfVectorizer(
    max_features=300,  # Compact but powerful
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

print("🔥 Fitting TF-IDF...")
train_features = tfidf.fit_transform(train_text).toarray()
test_features = tfidf.transform(test_text).toarray()

# Save
print(f"\n💾 Saving features: {train_features.shape}")
np.savez_compressed('tfidf_features.npz',
                    train=train_features,
                    test=test_features)

print("\n" + "=" * 70)
print("✅ TASK 3 COMPLETE! File: tfidf_features.npz")
print("=" * 70)
