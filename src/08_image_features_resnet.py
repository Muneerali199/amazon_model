"""
Phase 7a: Image Feature Extraction using ResNet50
==================================================

This script extracts deep learning features from product images using ResNet50
pre-trained on ImageNet. These features capture visual patterns that are highly
correlated with product prices.

Expected Impact: -2 to -4% SMAPE improvement (59% → 55-57%)

Author: ML Challenge 2025 Team
Date: October 11, 2025
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tqdm import tqdm

print("=" * 80)
print("PHASE 7a: IMAGE FEATURE EXTRACTION (ResNet50)")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TRAIN_IMAGES_DIR = 'dataset/train_images'
TEST_IMAGES_DIR = 'dataset/test_images'
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'dataset/test.csv'

# Output paths
TRAIN_IMAGE_FEATURES = 'dataset/train_image_features_resnet50.npy'
TEST_IMAGE_FEATURES = 'dataset/test_image_features_resnet50.npy'
TRAIN_IMAGE_METADATA = 'dataset/train_image_metadata.csv'
TEST_IMAGE_METADATA = 'dataset/test_image_metadata.csv'

# ResNet50 config
IMG_SIZE = (224, 224)  # ResNet50 input size
BATCH_SIZE = 32        # Process images in batches
FEATURE_DIM = 2048     # ResNet50 output dimension

print("Configuration:")
print(f"  Image size: {IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Feature dimension: {FEATURE_DIM}")
print()

# ============================================================================
# LOAD PRE-TRAINED RESNET50
# ============================================================================

print("Loading ResNet50 model (pre-trained on ImageNet)...")
print("  This may take a minute on first run (downloading weights)...")

# Load ResNet50 without top layer (classifier)
# We want feature extraction, not classification
base_model = ResNet50(
    weights='imagenet',
    include_top=False,  # Remove classification layer
    pooling='avg'       # Global average pooling → 2048-dim vector
)

print("  ✅ Model loaded successfully!")
print(f"  Output shape: {base_model.output_shape}")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_preprocess_image(img_path, target_size=IMG_SIZE):
    """
    Load image and preprocess for ResNet50
    
    Args:
        img_path: Path to image file
        target_size: Target size (height, width)
    
    Returns:
        Preprocessed image array (1, 224, 224, 3)
    """
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to array
        img_array = keras_image.img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for ResNet50
        img_array = preprocess_input(img_array)
        
        return img_array, True
    
    except Exception as e:
        # Return zero array if image fails to load
        return np.zeros((1, *target_size, 3)), False


def extract_features_batch(image_paths, model):
    """
    Extract features from a batch of images
    
    Args:
        image_paths: List of image paths
        model: ResNet50 model
    
    Returns:
        Feature matrix (batch_size, 2048)
        Success flags (batch_size,)
    """
    batch_images = []
    success_flags = []
    
    for img_path in image_paths:
        img_array, success = load_and_preprocess_image(img_path)
        batch_images.append(img_array[0])  # Remove batch dimension
        success_flags.append(success)
    
    # Stack into batch
    batch_images = np.array(batch_images)
    
    # Extract features
    features = model.predict(batch_images, verbose=0)
    
    return features, np.array(success_flags)


def extract_all_features(df, images_dir, model, desc="Extracting features"):
    """
    Extract features from all images in dataframe
    
    Args:
        df: DataFrame with sample_id
        images_dir: Directory containing images
        model: ResNet50 model
        desc: Progress bar description
    
    Returns:
        features: Feature matrix (N, 2048)
        metadata: DataFrame with image statistics
    """
    n_samples = len(df)
    all_features = np.zeros((n_samples, FEATURE_DIM), dtype=np.float32)
    
    # Metadata tracking
    success_count = 0
    failed_images = []
    
    # Process in batches
    sample_ids = df['sample_id'].values
    
    # Create progress bar
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=n_batches, desc=desc)
    
    for i in range(0, n_samples, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, n_samples)
        batch_ids = sample_ids[i:batch_end]
        
        # Build image paths
        batch_paths = [
            os.path.join(images_dir, f"{sid}.jpg")
            for sid in batch_ids
        ]
        
        # Extract features
        batch_features, success_flags = extract_features_batch(batch_paths, model)
        
        # Store features
        all_features[i:batch_end] = batch_features
        
        # Track success
        success_count += success_flags.sum()
        
        # Track failures
        for j, (sid, success) in enumerate(zip(batch_ids, success_flags)):
            if not success:
                failed_images.append(sid)
        
        pbar.update(1)
    
    pbar.close()
    
    # Create metadata
    metadata = pd.DataFrame({
        'sample_id': sample_ids,
        'image_loaded': [sid not in failed_images for sid in sample_ids],
        'feature_norm': np.linalg.norm(all_features, axis=1)
    })
    
    print(f"  ✅ Processed {n_samples} images")
    print(f"  ✅ Successfully loaded: {success_count}/{n_samples} ({100*success_count/n_samples:.1f}%)")
    
    if failed_images:
        print(f"  ⚠️  Failed to load: {len(failed_images)} images")
        print(f"     (These will have zero features)")
    
    print()
    
    return all_features, metadata


# ============================================================================
# EXTRACT TRAINING IMAGE FEATURES
# ============================================================================

print("-" * 80)
print("STEP 1: Extract Training Image Features")
print("-" * 80)
print()

# Load training data
train_df = pd.read_csv(TRAIN_CSV)
print(f"Training samples: {len(train_df):,}")
print()

# Check if images exist
if not os.path.exists(TRAIN_IMAGES_DIR):
    print(f"❌ ERROR: Training images directory not found: {TRAIN_IMAGES_DIR}")
    print("   Please ensure images are downloaded.")
    exit(1)

# Count available images
train_images = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith('.jpg')]
print(f"Available training images: {len(train_images):,}")
print()

# Extract features
print("Extracting features (this will take 15-30 minutes)...")
train_features, train_metadata = extract_all_features(
    train_df,
    TRAIN_IMAGES_DIR,
    base_model,
    desc="Training images"
)

# Save features
print(f"Saving training features to {TRAIN_IMAGE_FEATURES}...")
np.save(TRAIN_IMAGE_FEATURES, train_features)
print(f"  ✅ Saved shape: {train_features.shape}")
print()

# Save metadata
train_metadata.to_csv(TRAIN_IMAGE_METADATA, index=False)
print(f"  ✅ Saved metadata to {TRAIN_IMAGE_METADATA}")
print()

# Feature statistics
print("Training Image Feature Statistics:")
print(f"  Shape: {train_features.shape}")
print(f"  Mean: {train_features.mean():.4f}")
print(f"  Std: {train_features.std():.4f}")
print(f"  Min: {train_features.min():.4f}")
print(f"  Max: {train_features.max():.4f}")
print(f"  Non-zero features: {(train_features != 0).sum():,} / {train_features.size:,}")
print()

# ============================================================================
# EXTRACT TEST IMAGE FEATURES
# ============================================================================

print("-" * 80)
print("STEP 2: Extract Test Image Features")
print("-" * 80)
print()

# Load test data
test_df = pd.read_csv(TEST_CSV)
print(f"Test samples: {len(test_df):,}")
print()

# Check if images exist
if not os.path.exists(TEST_IMAGES_DIR):
    print(f"❌ ERROR: Test images directory not found: {TEST_IMAGES_DIR}")
    print("   Please ensure images are downloaded.")
    exit(1)

# Count available images
test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
print(f"Available test images: {len(test_images):,}")
print()

# Extract features
print("Extracting features (this will take 15-30 minutes)...")
test_features, test_metadata = extract_all_features(
    test_df,
    TEST_IMAGES_DIR,
    base_model,
    desc="Test images"
)

# Save features
print(f"Saving test features to {TEST_IMAGE_FEATURES}...")
np.save(TEST_IMAGE_FEATURES, test_features)
print(f"  ✅ Saved shape: {test_features.shape}")
print()

# Save metadata
test_metadata.to_csv(TEST_IMAGE_METADATA, index=False)
print(f"  ✅ Saved metadata to {TEST_IMAGE_METADATA}")
print()

# Feature statistics
print("Test Image Feature Statistics:")
print(f"  Shape: {test_features.shape}")
print(f"  Mean: {test_features.mean():.4f}")
print(f"  Std: {test_features.std():.4f}")
print(f"  Min: {test_features.min():.4f}")
print(f"  Max: {test_features.max():.4f}")
print(f"  Non-zero features: {(test_features != 0).sum():,} / {test_features.size:,}")
print()

# ============================================================================
# FEATURE CORRELATION ANALYSIS
# ============================================================================

print("-" * 80)
print("STEP 3: Feature Correlation Analysis")
print("-" * 80)
print()

# Calculate correlation with price (training set only)
prices = train_df['price'].values

# Find top correlated features
correlations = []
for i in range(FEATURE_DIM):
    feature_values = train_features[:, i]
    
    # Skip if all zeros
    if feature_values.std() < 1e-6:
        correlations.append(0.0)
        continue
    
    # Calculate Pearson correlation
    corr = np.corrcoef(feature_values, prices)[0, 1]
    correlations.append(abs(corr))  # Absolute correlation

correlations = np.array(correlations)

# Top correlated features
top_k = 10
top_indices = np.argsort(correlations)[-top_k:][::-1]

print("Top 10 Image Features Correlated with Price:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. Feature {idx:4d}: |correlation| = {correlations[idx]:.4f}")
print()

# Overall correlation statistics
print("Overall Correlation Statistics:")
print(f"  Mean |correlation|: {correlations.mean():.4f}")
print(f"  Max |correlation|: {correlations.max():.4f}")
print(f"  Features with |corr| > 0.1: {(correlations > 0.1).sum()} / {FEATURE_DIM}")
print(f"  Features with |corr| > 0.05: {(correlations > 0.05).sum()} / {FEATURE_DIM}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 7a COMPLETE: IMAGE FEATURES EXTRACTED!")
print("=" * 80)
print()

print("✅ Generated Files:")
print(f"  1. {TRAIN_IMAGE_FEATURES} - Training image features ({train_features.shape})")
print(f"  2. {TEST_IMAGE_FEATURES} - Test image features ({test_features.shape})")
print(f"  3. {TRAIN_IMAGE_METADATA} - Training metadata")
print(f"  4. {TEST_IMAGE_METADATA} - Test metadata")
print()

print("📊 Feature Summary:")
print(f"  • Feature dimension: {FEATURE_DIM}")
print(f"  • Total features extracted: {train_features.shape[0] + test_features.shape[0]:,}")
print(f"  • Top correlation with price: {correlations.max():.4f}")
print()

print("🚀 Next Steps:")
print("  1. Run Phase 7b: Combine image + text features")
print("  2. Train XGBoost on combined features")
print("  3. Generate new submission")
print("  4. Expected improvement: -2 to -4% SMAPE")
print("  5. Target: 55-57% SMAPE (from current 59%)")
print()

print("💡 To continue:")
print("  python src/09_combine_text_image_features.py")
print()

print("=" * 80)
