"""
TASK 2: Extract Vision Features ONLY (8 minutes)
Uses: MobileNetV3 (FASTEST vision model!)
Output: vision_features.npz
"""

import pandas as pd
import numpy as np
import torch
import timm
from PIL import Image
import requests
from io import BytesIO
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 2: VISION FEATURES (8 min)")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Use MobileNetV3 - FASTEST model!
print("\n🖼️ Loading MobileNetV3-Large (FASTEST!)...")
model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
model = model.to(device)
model.eval()

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_size = config['input_size']

print(f"✅ Model loaded! Output: {model.num_features} features")

def download_image(url, timeout=3):
    """Fast download with short timeout"""
    try:
        response = requests.get(url, timeout=timeout)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except:
        return None

def extract_vision_features(image_urls, batch_size=64):
    """Extract with LARGE batches and parallel download"""
    features = []
    failed = 0
    
    for i in tqdm(range(0, len(image_urls), batch_size), desc='Processing'):
        batch_urls = image_urls[i:i+batch_size]
        
        # Parallel download (FAST!)
        with ThreadPoolExecutor(max_workers=32) as executor:
            images = list(executor.map(download_image, batch_urls))
        
        batch_images = []
        for img in images:
            if img is not None:
                batch_images.append(transform(img))
            else:
                batch_images.append(torch.zeros(input_size))
                failed += 1
        
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_features = model(batch_tensor).cpu().numpy()
            
            features.append(batch_features)
    
    print(f"⚠️ Failed: {failed}/{len(image_urls)} ({100*failed/len(image_urls):.1f}%)")
    return np.vstack(features)

# Process
train_images = train_df['image_link'].fillna('').values
test_images = test_df['image_link'].fillna('').values

print("\n🔥 Extracting TRAIN vision features...")
train_features = extract_vision_features(train_images, batch_size=64)

print("🔥 Extracting TEST vision features...")
test_features = extract_vision_features(test_images, batch_size=64)

# Save
print(f"\n💾 Saving features: {train_features.shape}")
np.savez_compressed('vision_features.npz',
                    train=train_features,
                    test=test_features)

print("\n" + "=" * 70)
print("✅ TASK 2 COMPLETE! File: vision_features.npz")
print("=" * 70)
