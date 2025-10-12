"""
Quick check: What image data do we have?
"""
import pandas as pd
from pathlib import Path

print("="*80)
print("🔍 CHECKING IMAGE DATA")
print("="*80)
print()

# Load data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print()
print("Columns:", train.columns.tolist())
print()

# Check for image column
if 'image_link' in train.columns:
    print("✅ Found 'image_link' column!")
    print()
    print("Sample image links:")
    print(train['image_link'].head(5).values)
    print()
    print(f"Non-null images in train: {train['image_link'].notna().sum():,}/{len(train):,}")
    print(f"Non-null images in test: {test['image_link'].notna().sum():,}/{len(test):,}")
else:
    print("❌ No image_link column")

# Check if images folder exists
images_path = Path('dataset/images')
if images_path.exists():
    images = list(images_path.glob('*'))
    print(f"\n✅ Found images folder with {len(images)} files")
else:
    print("\n❌ No images folder found")

print("="*80)
