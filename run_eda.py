"""
Phase 1: Exploratory Data Analysis (EDA)
ML Challenge 2025 - Smart Product Pricing Challenge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configure
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

print("="*80)
print("PHASE 1: EXPLORATORY DATA ANALYSIS")
print("ML Challenge 2025 - Smart Product Pricing Challenge")
print("="*80)
print("\n")

# =============================================================================
# TASK 1.1: Load and Explore Training Data
# =============================================================================
print("📊 TASK 1.1: Loading Training Data...")
print("-"*80)

try:
    train_df = pd.read_csv('dataset/train.csv')
    print(f"✅ Training data loaded successfully!")
    print(f"   Rows: {train_df.shape[0]:,}")
    print(f"   Columns: {train_df.shape[1]}")
    print("\n")
    
    # Display first few rows
    print("📋 First 5 rows of training data:")
    print(train_df.head())
    print("\n")
    
    # Dataset info
    print("📊 Dataset Information:")
    print("="*80)
    print(train_df.info())
    print("\n")
    
    # Check for missing values
    print("🔍 Missing Values Analysis:")
    print("="*80)
    missing = train_df.isnull().sum()
    missing_pct = (missing / len(train_df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': missing_pct.values
    })
    print(missing_df)
    print(f"\n✅ Total missing values: {missing.sum():,}")
    print("\n")
    
except FileNotFoundError:
    print("❌ Error: train.csv not found!")
    print("   Make sure you're running this script from the student_resource directory")
    exit(1)

# =============================================================================
# TASK 1.2: Price Distribution Analysis
# =============================================================================
print("="*80)
print("💰 TASK 1.2: Price Distribution Analysis")
print("-"*80)

print("\n📈 Price Statistics:")
print("="*80)
print(f"Mean Price:        ${train_df['price'].mean():,.2f}")
print(f"Median Price:      ${train_df['price'].median():,.2f}")
print(f"Std Dev:           ${train_df['price'].std():,.2f}")
print(f"Min Price:         ${train_df['price'].min():,.2f}")
print(f"Max Price:         ${train_df['price'].max():,.2f}")
print(f"\nPrice Range:       ${train_df['price'].min():,.2f} - ${train_df['price'].max():,.2f}")
print(f"\nQuartiles:")
print(f"  25th percentile: ${train_df['price'].quantile(0.25):,.2f}")
print(f"  50th percentile: ${train_df['price'].quantile(0.50):,.2f}")
print(f"  75th percentile: ${train_df['price'].quantile(0.75):,.2f}")
print(f"  95th percentile: ${train_df['price'].quantile(0.95):,.2f}")
print(f"  99th percentile: ${train_df['price'].quantile(0.99):,.2f}")
print("\n")

# Outlier analysis using IQR
Q1 = train_df['price'].quantile(0.25)
Q3 = train_df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_low = train_df[train_df['price'] < lower_bound]
outliers_high = train_df[train_df['price'] > upper_bound]

print("🔍 Outlier Analysis (IQR Method):")
print("="*80)
print(f"IQR: ${IQR:.2f}")
print(f"Lower Bound: ${lower_bound:.2f}")
print(f"Upper Bound: ${upper_bound:.2f}")
print(f"\nLow Outliers:  {len(outliers_low):,} ({len(outliers_low)/len(train_df)*100:.2f}%)")
print(f"High Outliers: {len(outliers_high):,} ({len(outliers_high)/len(train_df)*100:.2f}%)")
print(f"\nTotal Outliers: {len(outliers_low) + len(outliers_high):,} ({(len(outliers_low) + len(outliers_high))/len(train_df)*100:.2f}%)")
print("\n")

# =============================================================================
# TASK 1.3: Text Analysis
# =============================================================================
print("="*80)
print("📝 TASK 1.3: Text Analysis (catalog_content)")
print("-"*80)

# Calculate text features
train_df['text_length'] = train_df['catalog_content'].str.len()
train_df['word_count'] = train_df['catalog_content'].str.split().str.len()

print("\n📊 Text Statistics:")
print("="*80)
print(f"\nText Length (characters):")
print(f"  Mean:   {train_df['text_length'].mean():.2f}")
print(f"  Median: {train_df['text_length'].median():.2f}")
print(f"  Min:    {train_df['text_length'].min()}")
print(f"  Max:    {train_df['text_length'].max()}")

print(f"\nWord Count:")
print(f"  Mean:   {train_df['word_count'].mean():.2f}")
print(f"  Median: {train_df['word_count'].median():.2f}")
print(f"  Min:    {train_df['word_count'].min()}")
print(f"  Max:    {train_df['word_count'].max()}")
print("\n")

# Sample catalog content
print("📝 Sample Catalog Content (first 3 samples):")
print("="*80)
for i in range(min(3, len(train_df))):
    print(f"\n[Sample {i+1}]")
    print(f"ID: {train_df.iloc[i]['sample_id']}")
    print(f"Price: ${train_df.iloc[i]['price']:.2f}")
    print(f"Content: {train_df.iloc[i]['catalog_content'][:300]}...")
    print("-"*80)
print("\n")

# Correlation with price
correlation = train_df[['text_length', 'word_count', 'price']].corr()
print("🔗 Correlation with Price:")
print("="*80)
print(correlation['price'].sort_values(ascending=False))
print("\n")

# =============================================================================
# TASK 1.4: Image Link Analysis
# =============================================================================
print("="*80)
print("🖼️ TASK 1.4: Image Link Analysis")
print("-"*80)

train_df['has_image'] = train_df['image_link'].notna()

print(f"\nTotal samples: {len(train_df):,}")
print(f"Samples with image links: {train_df['image_link'].notna().sum():,}")
print(f"Samples without image links: {train_df['image_link'].isna().sum():,}")
print(f"\nPercentage with images: {train_df['image_link'].notna().sum() / len(train_df) * 100:.2f}%")

if train_df['image_link'].notna().sum() > 0:
    print("\n📝 Sample image links (first 3):")
    for i, link in enumerate(train_df['image_link'].dropna().head(3)):
        print(f"{i+1}. {link}")

# Price comparison
price_with_image = train_df[train_df['has_image']]['price']
price_without_image = train_df[~train_df['has_image']]['price']

print("\n💰 Price Comparison (With vs Without Images):")
print("="*80)
print(f"\nWith Images:")
print(f"  Count: {len(price_with_image):,}")
print(f"  Mean Price: ${price_with_image.mean():.2f}")
print(f"  Median Price: ${price_with_image.median():.2f}")

if len(price_without_image) > 0:
    print(f"\nWithout Images:")
    print(f"  Count: {len(price_without_image):,}")
    print(f"  Mean Price: ${price_without_image.mean():.2f}")
    print(f"  Median Price: ${price_without_image.median():.2f}")
else:
    print(f"\n✅ All samples have image links!")
print("\n")

# =============================================================================
# TASK 1.5: Test Data Exploration
# =============================================================================
print("="*80)
print("📊 TASK 1.5: Test Data Exploration")
print("-"*80)

try:
    test_df = pd.read_csv('dataset/test.csv')
    print(f"✅ Test data loaded successfully!")
    print(f"   Rows: {test_df.shape[0]:,}")
    print(f"   Columns: {test_df.shape[1]}")
    print("\n")
    
    # Test data stats
    test_df['text_length'] = test_df['catalog_content'].str.len()
    test_df['word_count'] = test_df['catalog_content'].str.split().str.len()
    
    print("📊 Train vs Test Comparison:")
    print("="*80)
    print(f"\nText Length:")
    print(f"  Train Mean: {train_df['text_length'].mean():.2f}")
    print(f"  Test Mean:  {test_df['text_length'].mean():.2f}")
    
    print(f"\nWord Count:")
    print(f"  Train Mean: {train_df['word_count'].mean():.2f}")
    print(f"  Test Mean:  {test_df['word_count'].mean():.2f}")
    
    print(f"\nImage Availability:")
    print(f"  Train: {train_df['image_link'].notna().sum() / len(train_df) * 100:.2f}%")
    print(f"  Test:  {test_df['image_link'].notna().sum() / len(test_df) * 100:.2f}%")
    print("\n")
    
except FileNotFoundError:
    print("⚠️ Warning: test.csv not found!")
    print("\n")

# =============================================================================
# SUMMARY
# =============================================================================
print("="*80)
print("📊 PHASE 1 SUMMARY: KEY FINDINGS")
print("="*80)

print("\n1️⃣ DATASET SIZE:")
print(f"   ✓ Training samples: {len(train_df):,}")
if 'test_df' in locals():
    print(f"   ✓ Test samples: {len(test_df):,}")

print("\n2️⃣ PRICE DISTRIBUTION:")
print(f"   ✓ Price range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
print(f"   ✓ Mean price: ${train_df['price'].mean():.2f}")
print(f"   ✓ Median price: ${train_df['price'].median():.2f}")
print(f"   ✓ Outliers: {(len(outliers_low) + len(outliers_high))/len(train_df)*100:.2f}% of data")

print("\n3️⃣ TEXT FEATURES:")
print(f"   ✓ Average text length: {train_df['text_length'].mean():.0f} characters")
print(f"   ✓ Average word count: {train_df['word_count'].mean():.0f} words")
print(f"   ✓ Text-price correlation: {train_df[['text_length', 'price']].corr().iloc[0, 1]:.3f}")

print("\n4️⃣ IMAGE AVAILABILITY:")
print(f"   ✓ Train images: {train_df['image_link'].notna().sum() / len(train_df) * 100:.2f}%")
if 'test_df' in locals():
    print(f"   ✓ Test images: {test_df['image_link'].notna().sum() / len(test_df) * 100:.2f}%")

print("\n5️⃣ MISSING VALUES:")
print(f"   ✓ Training data: {train_df.isnull().sum().sum()} missing values")
if 'test_df' in locals():
    print(f"   ✓ Test data: {test_df.isnull().sum().sum()} missing values")

print("\n6️⃣ RECOMMENDATIONS FOR NEXT PHASE:")
print("   ✓ Consider log-transformation for price (check if right-skewed)")
print("   ✓ Extract Item Pack Quantity (IPQ) from catalog_content")
print("   ✓ Extract brand names and product categories")
print("   ✓ Download and process product images")
print("   ✓ Handle price outliers carefully (cap or transform)")
print("   ✓ Create text embeddings for better feature representation")

print("\n" + "="*80)
print("✅ PHASE 1 COMPLETE: Ready for Phase 2 (Feature Engineering)")
print("="*80)
print("\n")

# Save basic stats to file
print("💾 Saving EDA results to 'phase1_results.txt'...")
with open('phase1_results.txt', 'w') as f:
    f.write("PHASE 1 EDA RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset Size: {len(train_df):,} training samples\n")
    f.write(f"Price Range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}\n")
    f.write(f"Mean Price: ${train_df['price'].mean():.2f}\n")
    f.write(f"Median Price: ${train_df['price'].median():.2f}\n")
    f.write(f"Outliers: {(len(outliers_low) + len(outliers_high))/len(train_df)*100:.2f}%\n")
    f.write(f"Avg Text Length: {train_df['text_length'].mean():.0f} chars\n")
    f.write(f"Avg Word Count: {train_df['word_count'].mean():.0f} words\n")
    f.write(f"Image Availability: {train_df['image_link'].notna().sum() / len(train_df) * 100:.2f}%\n")

print("✅ Results saved successfully!")
print("\n🎉 Phase 1 EDA completed! Review the output above for insights.")
