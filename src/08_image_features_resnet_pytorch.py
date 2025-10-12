"""
Phase 7a: Image Feature Extraction using ResNet50 (PyTorch)
============================================================

This script extracts deep learning features from product images using ResNet50
pre-trained on ImageNet via PyTorch (more stable on Windows than TensorFlow).

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

# PyTorch imports
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("=" * 80)
print("PHASE 7a: IMAGE FEATURE EXTRACTION (ResNet50 - PyTorch)")
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
IMG_SIZE = 224         # ResNet50 input size
BATCH_SIZE = 64        # Process images in batches (increased for speed)
FEATURE_DIM = 2048     # ResNet50 output dimension
NUM_WORKERS = 0        # DataLoader workers (0 for Windows compatibility)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Configuration:")
print(f"  Device: {device}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Feature dimension: {FEATURE_DIM}")
print()

# ============================================================================
# LOAD PRE-TRAINED RESNET50
# ============================================================================

print("Loading ResNet50 model (pre-trained on ImageNet)...")

# Load ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Remove final classification layer
# We want features from the last convolutional layer
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# Move to device and set to evaluation mode
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

print("  ✅ Model loaded successfully!")
print(f"  ✅ Model on device: {device}")
print()

# ============================================================================
# IMAGE TRANSFORMATIONS
# ============================================================================

# ImageNet normalization (ResNet50 was trained on these stats)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class ImageDataset(Dataset):
    """Dataset for loading product images"""
    
    def __init__(self, sample_ids, images_dir, transform=None):
        self.sample_ids = sample_ids
        self.images_dir = images_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.images_dir, f"{sample_id}.jpg")
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            success = True
            
        except Exception as e:
            # Return black image if loading fails
            image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            success = False
        
        return image, success, sample_id

# ============================================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================================

@torch.no_grad()
def extract_features(dataset, model, batch_size=BATCH_SIZE, desc="Extracting"):
    """
    Extract features from all images in dataset
    
    Args:
        dataset: ImageDataset instance
        model: Feature extractor model
        batch_size: Batch size for processing
        desc: Progress bar description
    
    Returns:
        features: Feature matrix (N, 2048)
        metadata: DataFrame with image statistics
    """
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda')
    )
    
    # Storage
    all_features = []
    all_success = []
    all_sample_ids = []
    
    # Extract features
    for images, success_flags, sample_ids in tqdm(dataloader, desc=desc):
        # Move images to device
        images = images.to(device)
        
        # Extract features
        features = model(images)
        
        # Flatten features (batch_size, 2048, 1, 1) → (batch_size, 2048)
        features = features.squeeze(-1).squeeze(-1)
        
        # Move to CPU and convert to numpy
        features = features.cpu().numpy()
        
        # Store
        all_features.append(features)
        all_success.extend(success_flags.numpy())
        all_sample_ids.extend(sample_ids.numpy())
    
    # Concatenate all features
    features_matrix = np.concatenate(all_features, axis=0)
    
    # Create metadata
    metadata = pd.DataFrame({
        'sample_id': all_sample_ids,
        'image_loaded': all_success,
        'feature_norm': np.linalg.norm(features_matrix, axis=1)
    })
    
    return features_matrix, metadata

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

# Check if images exist
if not os.path.exists(TRAIN_IMAGES_DIR):
    print(f"❌ ERROR: Training images directory not found: {TRAIN_IMAGES_DIR}")
    print("   Please ensure images are downloaded.")
    exit(1)

# Count available images
train_images = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.endswith('.jpg')]
print(f"Available training images: {len(train_images):,}")
print()

# Create dataset
train_dataset = ImageDataset(
    sample_ids=train_df['sample_id'].values,
    images_dir=TRAIN_IMAGES_DIR,
    transform=transform
)

print(f"Extracting features from {len(train_dataset):,} images...")
print("(This will take 10-20 minutes depending on your hardware)")
print()

# Extract features
train_features, train_metadata = extract_features(
    train_dataset,
    feature_extractor,
    batch_size=BATCH_SIZE,
    desc="Training images"
)

# Save features
print()
print(f"Saving training features to {TRAIN_IMAGE_FEATURES}...")
np.save(TRAIN_IMAGE_FEATURES, train_features.astype(np.float32))
print(f"  ✅ Saved shape: {train_features.shape}")

# Save metadata
train_metadata.to_csv(TRAIN_IMAGE_METADATA, index=False)
print(f"  ✅ Saved metadata to {TRAIN_IMAGE_METADATA}")
print()

# Statistics
success_rate = train_metadata['image_loaded'].sum() / len(train_metadata) * 100
print("Training Image Feature Statistics:")
print(f"  Shape: {train_features.shape}")
print(f"  Successfully loaded: {train_metadata['image_loaded'].sum():,} / {len(train_metadata):,} ({success_rate:.1f}%)")
print(f"  Mean: {train_features.mean():.4f}")
print(f"  Std: {train_features.std():.4f}")
print(f"  Min: {train_features.min():.4f}")
print(f"  Max: {train_features.max():.4f}")
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

# Check if images exist
if not os.path.exists(TEST_IMAGES_DIR):
    print(f"❌ ERROR: Test images directory not found: {TEST_IMAGES_DIR}")
    print("   Please ensure images are downloaded.")
    exit(1)

# Count available images
test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
print(f"Available test images: {len(test_images):,}")
print()

# Create dataset
test_dataset = ImageDataset(
    sample_ids=test_df['sample_id'].values,
    images_dir=TEST_IMAGES_DIR,
    transform=transform
)

print(f"Extracting features from {len(test_dataset):,} images...")
print("(This will take 10-20 minutes depending on your hardware)")
print()

# Extract features
test_features, test_metadata = extract_features(
    test_dataset,
    feature_extractor,
    batch_size=BATCH_SIZE,
    desc="Test images"
)

# Save features
print()
print(f"Saving test features to {TEST_IMAGE_FEATURES}...")
np.save(TEST_IMAGE_FEATURES, test_features.astype(np.float32))
print(f"  ✅ Saved shape: {test_features.shape}")

# Save metadata
test_metadata.to_csv(TEST_IMAGE_METADATA, index=False)
print(f"  ✅ Saved metadata to {TEST_IMAGE_METADATA}")
print()

# Statistics
success_rate = test_metadata['image_loaded'].sum() / len(test_metadata) * 100
print("Test Image Feature Statistics:")
print(f"  Shape: {test_features.shape}")
print(f"  Successfully loaded: {test_metadata['image_loaded'].sum():,} / {len(test_metadata):,} ({success_rate:.1f}%)")
print(f"  Mean: {test_features.mean():.4f}")
print(f"  Std: {test_features.std():.4f}")
print(f"  Min: {test_features.min():.4f}")
print(f"  Max: {test_features.max():.4f}")
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

print("Analyzing correlation between image features and prices...")

# Calculate correlations
correlations = []
for i in tqdm(range(FEATURE_DIM), desc="Computing correlations"):
    feature_values = train_features[:, i]
    
    # Skip if all zeros or constant
    if feature_values.std() < 1e-6:
        correlations.append(0.0)
        continue
    
    # Calculate Pearson correlation
    corr = np.corrcoef(feature_values, prices)[0, 1]
    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

correlations = np.array(correlations)

# Top correlated features
top_k = 10
top_indices = np.argsort(correlations)[-top_k:][::-1]

print()
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

# Save correlations for later use
correlations_df = pd.DataFrame({
    'feature_idx': range(FEATURE_DIM),
    'abs_correlation': correlations
})
correlations_df.to_csv('dataset/image_feature_correlations.csv', index=False)
print("  ✅ Saved correlations to dataset/image_feature_correlations.csv")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 7a COMPLETE: IMAGE FEATURES EXTRACTED!")
print("=" * 80)
print()

print("✅ Generated Files:")
print(f"  1. {TRAIN_IMAGE_FEATURES}")
print(f"     Shape: {train_features.shape}, Size: {os.path.getsize(TRAIN_IMAGE_FEATURES)/1024/1024:.1f} MB")
print(f"  2. {TEST_IMAGE_FEATURES}")
print(f"     Shape: {test_features.shape}, Size: {os.path.getsize(TEST_IMAGE_FEATURES)/1024/1024:.1f} MB")
print(f"  3. {TRAIN_IMAGE_METADATA}")
print(f"  4. {TEST_IMAGE_METADATA}")
print(f"  5. dataset/image_feature_correlations.csv")
print()

print("📊 Feature Summary:")
print(f"  • Feature dimension: {FEATURE_DIM}")
print(f"  • Training images processed: {len(train_features):,}")
print(f"  • Test images processed: {len(test_features):,}")
print(f"  • Top correlation with price: {correlations.max():.4f}")
print(f"  • Device used: {device}")
print()

print("🚀 Next Steps:")
print("  1. Run Phase 7b: Combine image + text features")
print("     python src/09_combine_text_image_features.py")
print()
print("  2. Expected improvement: -2 to -4% SMAPE")
print("  3. Target: 55-57% SMAPE (from current 59%)")
print("  4. This should move you from rank #437 to ~#200-300!")
print()

print("=" * 80)
