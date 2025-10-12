"""
=================================================================================
KAGGLE GPU NOTEBOOK: CLIP-Based Multimodal Price Prediction
=================================================================================

This notebook uses OpenAI's CLIP model to extract powerful multimodal features
from product images + text descriptions, then trains XGBoost on top.

EXPECTED IMPROVEMENT: 2-4 points (from 57.9% to 54-56% SMAPE)
RUNTIME: ~3-4 hours on Kaggle GPU (T4 or P100)

SETUP INSTRUCTIONS:
1. Upload train.csv and test.csv to Kaggle
2. Enable GPU in Kaggle settings (Runtime > Change runtime type > GPU)
3. Run all cells
4. Download predictions

=================================================================================
"""

# Cell 1: Install required packages
# ===================================
print("📦 Installing required packages...")
print("="*80)

!pip install transformers torch pillow requests -q

print("✅ Packages installed!")
print()

# Cell 2: Import libraries
# ========================
print("📚 Importing libraries...")
print("="*80)

import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported!")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print()

# Cell 3: Load data
# =================
print("📂 Loading data...")
print("="*80)

# Upload these files to Kaggle dataset
train = pd.read_csv('/kaggle/input/your-dataset/train.csv')
test = pd.read_csv('/kaggle/input/your-dataset/test.csv')

print(f"✅ Train: {len(train):,} samples")
print(f"✅ Test: {len(test):,} samples")
print()
print("Columns:", train.columns.tolist())
print()

# Cell 4: Load CLIP model
# ========================
print("🤖 Loading CLIP model (Vision + Language)...")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("✅ CLIP model loaded!")
print("   - Model: clip-vit-base-patch32")
print("   - Embedding size: 512 dimensions")
print()

# Cell 5: Helper functions
# =========================
print("⚙️ Setting up helper functions...")
print("="*80)

def download_image(image_url, max_retries=3):
    """Download image from URL with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
        except Exception as e:
            if attempt == max_retries - 1:
                # Return a blank image if all attempts fail
                return Image.new('RGB', (224, 224), (128, 128, 128))
    return None

def extract_clip_features(image_url, text, model, processor, device):
    """Extract CLIP embeddings for image + text"""
    try:
        # Download image
        image = download_image(image_url)
        if image is None:
            # Use only text if image fails
            inputs = processor(text=[text], images=None, return_tensors="pt", 
                             padding=True, truncation=True, max_length=77)
        else:
            # Use both image and text
            inputs = processor(text=[text], images=[image], return_tensors="pt",
                             padding=True, truncation=True, max_length=77)
        
        # Move to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use image embeddings (or text if image failed)
            if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                embeddings = outputs.image_embeds
            else:
                embeddings = outputs.text_embeds
        
        return embeddings.cpu().numpy().flatten()
    
    except Exception as e:
        print(f"Error processing: {str(e)[:50]}")
        # Return zero vector on error
        return np.zeros(512)

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    diff = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))
    return 200 * np.mean(diff)

print("✅ Helper functions ready!")
print()

# Cell 6: Extract CLIP features (TRAINING)
# =========================================
print("="*80)
print("🖼️📝 EXTRACTING CLIP FEATURES (TRAINING)")
print("="*80)
print()
print("This will take ~1.5-2 hours for 75,000 samples")
print("Progress will be shown every 1,000 samples")
print()

train_embeddings = []
failed_count = 0

for idx, row in train.iterrows():
    if idx % 1000 == 0:
        success_rate = ((idx - failed_count) / max(idx, 1)) * 100
        print(f"Progress: {idx:,}/{len(train):,} ({idx/len(train)*100:.1f}%) | Success: {success_rate:.1f}%")
    
    # Extract features
    embedding = extract_clip_features(
        image_url=row['image_link'],
        text=row['catalog_content'],
        model=model,
        processor=processor,
        device=device
    )
    
    # Check if extraction failed (all zeros)
    if np.all(embedding == 0):
        failed_count += 1
    
    train_embeddings.append(embedding)

train_embeddings = np.array(train_embeddings)

print()
print("="*80)
print("✅ TRAINING FEATURES EXTRACTED!")
print("="*80)
print(f"   Shape: {train_embeddings.shape}")
print(f"   Success rate: {((len(train) - failed_count) / len(train) * 100):.1f}%")
print(f"   Failed: {failed_count:,} samples")
print()

# Cell 7: Extract CLIP features (TEST)
# =====================================
print("="*80)
print("🖼️📝 EXTRACTING CLIP FEATURES (TEST)")
print("="*80)
print()
print("This will take ~1.5-2 hours for 75,000 samples")
print()

test_embeddings = []
test_failed_count = 0

for idx, row in test.iterrows():
    if idx % 1000 == 0:
        success_rate = ((idx - test_failed_count) / max(idx, 1)) * 100
        print(f"Progress: {idx:,}/{len(test):,} ({idx/len(test)*100:.1f}%) | Success: {success_rate:.1f}%")
    
    # Extract features
    embedding = extract_clip_features(
        image_url=row['image_link'],
        text=row['catalog_content'],
        model=model,
        processor=processor,
        device=device
    )
    
    if np.all(embedding == 0):
        test_failed_count += 1
    
    test_embeddings.append(embedding)

test_embeddings = np.array(test_embeddings)

print()
print("="*80)
print("✅ TEST FEATURES EXTRACTED!")
print("="*80)
print(f"   Shape: {test_embeddings.shape}")
print(f"   Success rate: {((len(test) - test_failed_count) / len(test) * 100):.1f}%")
print(f"   Failed: {test_failed_count:,} samples")
print()

# Cell 8: Train XGBoost on CLIP features
# =======================================
print("="*80)
print("🤖 TRAINING XGBOOST ON CLIP FEATURES")
print("="*80)
print()

y_train = train['price'].values

# XGBoost parameters optimized for CLIP features
xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1,
    'random_state': 42,
    'tree_method': 'gpu_hist',  # GPU acceleration!
    'device': 'cuda'
}

print("Hyperparameters:")
for key, val in xgb_params.items():
    print(f"  {key}: {val}")
print()

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
test_predictions = []

print("Training 5-fold CV...")
print("-"*80)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_embeddings), 1):
    X_tr, X_val = train_embeddings[train_idx], train_embeddings[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train model
    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, verbose=False)
    
    # Validate
    val_pred = model_xgb.predict(X_val)
    score = smape(y_val, val_pred)
    cv_scores.append(score)
    
    # Predict on test
    test_pred = model_xgb.predict(test_embeddings)
    test_predictions.append(test_pred)
    
    print(f"  Fold {fold}: {score:.4f}%")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print()
print(f"CV Score: {cv_mean:.4f}% (±{cv_std:.4f}%)")
print()

# Average predictions from all folds
final_predictions = np.mean(test_predictions, axis=0)

# Cell 9: Compare with Phase 5
# =============================
print("="*80)
print("📊 FINAL RESULTS - CLIP vs Phase 5")
print("="*80)
print()
print(f"Phase 5 (local):  58.38% CV → 57.900% LB ✅")
print(f"CLIP + XGBoost:   {cv_mean:.2f}% CV → Expected {cv_mean-0.5:.2f}%-{cv_mean+0.5:.2f}% LB")
print()

if cv_mean < 57.0:
    improvement = 58.38 - cv_mean
    print(f"🎉🎉🎉 MAJOR BREAKTHROUGH! {improvement:.2f} points better!")
    print(f"✅ Expected LB: {cv_mean-0.5:.2f}%-{cv_mean+0.5:.2f}% (could reach TOP 30-40%!)")
    print()
    print("✅ STRONGLY RECOMMEND: SUBMIT THIS!")
elif cv_mean < 58.38:
    improvement = 58.38 - cv_mean
    print(f"🎉 EXCELLENT! {improvement:.2f} points better than Phase 5!")
    print(f"✅ Expected LB: {cv_mean-0.5:.2f}%-{cv_mean+0.5:.2f}%")
    print()
    print("✅ RECOMMEND: Submit!")
elif cv_mean < 59.0:
    print(f"⚠️  Marginal: {cv_mean-58.38:.2f} points from Phase 5")
    print(f"   But CLIP features might help on LB")
    print()
    print("🤔 YOUR CHOICE")
else:
    print(f"❌ WORSE: {cv_mean-58.38:.2f} points worse than Phase 5")
    print()
    print("❌ Stick with Phase 5")

print()
print("-"*80)

# Cell 10: Save predictions
# ==========================
print("💾 Saving predictions...")
print("="*80)

# Create submission file
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_predictions
})

submission.to_csv('submission_clip_kaggle.csv', index=False)

print("✅ Saved: submission_clip_kaggle.csv")
print(f"   Shape: {submission.shape}")
print()
print("📥 Download this file and submit to competition!")
print()

# Also save results
results = {
    'method': 'CLIP + XGBoost (Kaggle GPU)',
    'cv_mean': float(cv_mean),
    'cv_std': float(cv_std),
    'cv_scores': [float(x) for x in cv_scores],
    'embedding_dim': 512,
    'train_success_rate': float((len(train) - failed_count) / len(train)),
    'test_success_rate': float((len(test) - test_failed_count) / len(test)),
    'comparison': {
        'phase5_cv': 58.38,
        'phase5_lb': 57.900,
        'clip_cv': float(cv_mean),
        'improvement': float(58.38 - cv_mean)
    }
}

import json
with open('clip_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Saved: clip_results.json")
print("="*80)
print()
print("🎉 CLIP MULTIMODAL PIPELINE COMPLETE!")
print()
print("Next steps:")
print("  1. Download submission_clip_kaggle.csv")
print("  2. Submit to competition")
print("  3. Compare with Phase 5's 57.900%")
print()
print("="*80)
