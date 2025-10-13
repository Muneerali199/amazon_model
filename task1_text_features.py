"""
TASK 1: Extract Text Features ONLY (6 minutes)
Uses: DistilBERT (fast & accurate)
Output: text_features.npz
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TASK 1: TEXT FEATURES (6 min)")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")

# Use DistilBERT - 40% faster than BERT, 97% accuracy!
print("\n📝 Loading DistilBERT (FAST!)...")
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def extract_text_features(texts, batch_size=64):
    """Extract features with LARGE batches for speed"""
    features = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc='Processing'):
        batch = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch.tolist(),
            padding=True,
            truncation=True,
            max_length=128,  # Shorter for SPEED
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        features.append(batch_features)
    
    return np.vstack(features)

# Process
train_text = train_df['catalog_content'].fillna('').astype(str)
test_text = test_df['catalog_content'].fillna('').astype(str)

print("\n🔥 Extracting TRAIN text features...")
train_features = extract_text_features(train_text, batch_size=64)

print("🔥 Extracting TEST text features...")
test_features = extract_text_features(test_text, batch_size=64)

# Save
print(f"\n💾 Saving features: {train_features.shape}")
np.savez_compressed('text_features.npz', 
                    train=train_features,
                    test=test_features)

print("\n" + "=" * 70)
print("✅ TASK 1 COMPLETE! File: text_features.npz")
print("=" * 70)
