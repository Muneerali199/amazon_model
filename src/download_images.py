"""
Download Product Images
ML Challenge 2025 - Smart Product Pricing Challenge

This script downloads images for training and test datasets.
Note: This may take a while (75K + 75K images)
"""

import pandas as pd
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import download_images

print("="*80)
print("🖼️ IMAGE DOWNLOAD UTILITY")
print("ML Challenge 2025 - Smart Product Pricing Challenge")
print("="*80)
print("\n")

print("⚠️ WARNING:")
print("   - This will download 150,000 images (75K train + 75K test)")
print("   - This may take several hours depending on your internet speed")
print("   - Images will take ~10-15 GB of disk space")
print("   - Some images may fail due to throttling (you can retry later)")
print("\n")

response = input("Do you want to proceed? (yes/no): ")

if response.lower() not in ['yes', 'y']:
    print("❌ Download cancelled.")
    exit(0)

print("\n" + "="*80)
print("Starting image downloads...")
print("="*80)
print("\n")

# Load datasets
print("📊 Loading datasets...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
print(f"✅ Loaded {len(train_df):,} training samples")
print(f"✅ Loaded {len(test_df):,} test samples")
print("\n")

# Download training images
print("="*80)
print("📥 DOWNLOADING TRAINING IMAGES")
print("="*80)
print(f"Downloading {len(train_df):,} images to dataset/train_images/")
print("This will take a while... Please be patient.")
print("\n")

try:
    download_images(train_df['image_link'].tolist(), 'dataset/train_images/')
    print("\n✅ Training images download complete!")
except Exception as e:
    print(f"\n⚠️ Error during training image download: {e}")
    print("You can retry later. Some images may have been downloaded successfully.")

# Download test images
print("\n" + "="*80)
print("📥 DOWNLOADING TEST IMAGES")
print("="*80)
print(f"Downloading {len(test_df):,} images to dataset/test_images/")
print("This will take a while... Please be patient.")
print("\n")

try:
    download_images(test_df['image_link'].tolist(), 'dataset/test_images/')
    print("\n✅ Test images download complete!")
except Exception as e:
    print(f"\n⚠️ Error during test image download: {e}")
    print("You can retry later. Some images may have been downloaded successfully.")

print("\n" + "="*80)
print("✅ IMAGE DOWNLOAD PROCESS COMPLETE")
print("="*80)
print("\n")

# Check downloaded images
import os
train_imgs = len(os.listdir('dataset/train_images/')) if os.path.exists('dataset/train_images/') else 0
test_imgs = len(os.listdir('dataset/test_images/')) if os.path.exists('dataset/test_images/') else 0

print("📊 Download Summary:")
print(f"   Training images downloaded: {train_imgs:,} / {len(train_df):,}")
print(f"   Test images downloaded: {test_imgs:,} / {len(test_df):,}")

if train_imgs < len(train_df) or test_imgs < len(test_df):
    print("\n⚠️ Some images failed to download (likely due to throttling)")
    print("   You can run this script again to retry failed downloads.")
else:
    print("\n✅ All images downloaded successfully!")

print("\n💡 Next: Extract image features using pre-trained models")
