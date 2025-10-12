"""
KAGGLE ENSEMBLE - INTERNET-FRIENDLY VERSION
Pre-downloads all models first, then processes data
Handles connection issues more gracefully
"""

print("="*80)
print("ADVANCED ENSEMBLE MODEL - INTERNET-FRIENDLY VERSION")
print("="*80)

import sys
import subprocess
import time

# ============================================
# SECTION 1: SETUP & INSTALLATIONS WITH RETRY
# ============================================
print("\n[1/8] Installing packages with retry logic...")

def install_with_retry(package, max_retries=3):
    """Install package with retry on failure"""
    for attempt in range(max_retries):
        try:
            print(f"  Installing {package}... (attempt {attempt + 1}/{max_retries})")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✓ {package} installed!")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Failed, retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  ✗ Failed to install {package}: {e}")
                return False

packages = [
    'transformers',
    'torch',
    'torchvision', 
    'pillow',
    'xgboost',
    'lightgbm',
    'catboost',
    'scikit-learn',
    'pandas',
    'numpy',
    'requests',
    'tqdm'
]

print("\nInstalling packages (this may take 3-5 minutes)...")
for package in packages:
    install_with_retry(package, max_retries=3)

print("\n✓ Package installation complete!")

# Import libraries
print("\nImporting libraries...")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import gc

print("✓ All libraries imported!")

# Check internet connectivity
print("\nChecking internet connection...")
try:
    response = requests.get('https://www.google.com', timeout=5)
    print("✓ Internet connection: ACTIVE")
    INTERNET_ENABLED = True
except:
    print("✗ Internet connection: NOT AVAILABLE")
    print("⚠ WARNING: Image download will fail without internet!")
    INTERNET_ENABLED = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# SECTION 2: PRE-DOWNLOAD MODELS
# ============================================
print("\n[2/8] Pre-downloading models...")

# Download ResNet50
print("  Downloading ResNet50 (may take 1-2 minutes)...")
try:
    resnet_model = models.resnet50(pretrained=True)
    print("  ✓ ResNet50 downloaded successfully!")
except Exception as e:
    print(f"  ✗ ResNet50 download failed: {e}")
    print("  ⚠ Please enable Internet in Kaggle Settings!")
    raise

# Download BERT
print("  Downloading BERT (may take 2-3 minutes)...")
try:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    print("  ✓ BERT downloaded successfully!")
except Exception as e:
    print(f"  ✗ BERT download failed: {e}")
    print("  ⚠ Please enable Internet in Kaggle Settings!")
    raise

print("\n✓ All models pre-downloaded and cached!")

# ============================================
# SECTION 3: LOAD DATA
# ============================================
print("\n[3/8] Loading data...")

train_df = pd.read_csv('/kaggle/input/your-dataset-name/sample_train.csv')
test_df = pd.read_csv('/kaggle/input/your-dataset-name/sample_test.csv')

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape: {test_df.shape}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nTarget stats:\n{train_df['price'].describe()}")

# ============================================
# SECTION 4: IMAGE FEATURES (ResNet50)
# ============================================
print("\n[4/8] Extracting ResNet50 image features...")

class ResNetFeatureExtractor:
    def __init__(self, device, model):
        self.device = device
        # Use pre-downloaded model
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.model.to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.failed_downloads = 0
        self.successful_downloads = 0
    
    def download_image(self, url, timeout=5):
        try:
            response = requests.get(url, timeout=timeout)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            self.successful_downloads += 1
            return img
        except:
            self.failed_downloads += 1
            return Image.new('RGB', (224, 224), color='gray')
    
    def extract_features(self, image_url):
        img = self.download_image(image_url)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features

resnet_extractor = ResNetFeatureExtractor(device, resnet_model)

def extract_resnet_batch(df, name="train"):
    features_list = []
    print(f"  Extracting ResNet features for {name} ({len(df)} samples)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = resnet_extractor.extract_features(row['image_link'])
        features_list.append(features)
        
        if idx % 1000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if idx > 0:
                success_rate = resnet_extractor.successful_downloads / (idx + 1) * 100
                print(f"    Progress: {idx}/{len(df)} | Success rate: {success_rate:.1f}%")
    
    feature_df = pd.DataFrame(
        features_list,
        columns=[f'resnet_{i}' for i in range(2048)]
    )
    
    total = resnet_extractor.successful_downloads + resnet_extractor.failed_downloads
    success_rate = resnet_extractor.successful_downloads / total * 100
    print(f"  ✓ Final success rate: {success_rate:.1f}% ({resnet_extractor.successful_downloads}/{total})")
    
    return feature_df

train_resnet = extract_resnet_batch(train_df, "train")
test_resnet = extract_resnet_batch(test_df, "test")

print(f"✓ ResNet features extracted: {train_resnet.shape}")

# ============================================
# SECTION 5: TEXT FEATURES (BERT)
# ============================================
print("\n[5/8] Extracting BERT text features...")

class BERTFeatureExtractor:
    def __init__(self, device, tokenizer, model):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.model.to(device)
    
    def extract_features(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return features

bert_extractor = BERTFeatureExtractor(device, bert_tokenizer, bert_model)

def extract_bert_batch(df, name="train"):
    features_list = []
    print(f"  Extracting BERT features for {name} ({len(df)} samples)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = bert_extractor.extract_features(row['catalog_content'])
        features_list.append(features)
        
        if idx % 1000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    feature_df = pd.DataFrame(
        features_list,
        columns=[f'bert_{i}' for i in range(768)]
    )
    return feature_df

train_bert = extract_bert_batch(train_df, "train")
test_bert = extract_bert_batch(test_df, "test")

print(f"✓ BERT features extracted: {train_bert.shape}")

# ============================================
# SECTION 6: TRADITIONAL FEATURES
# ============================================
print("\n[6/8] Creating traditional text features...")

def create_text_features(df):
    features = pd.DataFrame()
    
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['upper_count'] = df['catalog_content'].str.findall(r'[A-Z]').str.len()
    features['upper_ratio'] = features['upper_count'] / (features['text_length'] + 1)
    features['number_count'] = df['catalog_content'].str.findall(r'\d').str.len()
    features['has_numbers'] = (features['number_count'] > 0).astype(int)
    features['special_char_count'] = df['catalog_content'].str.findall(r'[^\w\s]').str.len()
    
    brand_keywords = ['sony', 'samsung', 'apple', 'lg', 'hp', 'dell', 'lenovo', 'nike', 'adidas']
    for brand in brand_keywords:
        features[f'has_{brand}'] = df['catalog_content'].str.lower().str.contains(brand).astype(int)
    
    categories = ['electronic', 'clothing', 'book', 'home', 'toy', 'sport', 'beauty', 'food']
    for cat in categories:
        features[f'cat_{cat}'] = df['catalog_content'].str.lower().str.contains(cat).astype(int)
    
    return features

train_text_feat = create_text_features(train_df)
test_text_feat = create_text_features(test_df)

print(f"✓ Text features created: {train_text_feat.shape}")

# ============================================
# SECTION 7: COMBINE & TRAIN
# ============================================
print("\n[7/8] Combining features and training models...")

X_train = pd.concat([train_resnet, train_bert, train_text_feat], axis=1)
X_test = pd.concat([test_resnet, test_bert, test_text_feat], axis=1)
y_train = train_df['price'].values

print(f"✓ Total features: {X_train.shape[1]}")

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

models = {
    'xgboost': xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='gpu_hist', gpu_id=0, random_state=42
    ),
    'lightgbm': lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        device='gpu', random_state=42
    ),
    'catboost': CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=8,
        task_type='GPU', verbose=False, random_state=42
    )
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = {name: np.zeros(len(X_train)) for name in models.keys()}
test_predictions = {name: np.zeros(len(X_test)) for name in models.keys()}
cv_scores = {name: [] for name in models.keys()}

print("\nTraining with 5-fold CV:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n  Fold {fold + 1}/5")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    for name, model in models.items():
        print(f"    Training {name}...")
        
        if name == 'xgboost':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, verbose=False)
        elif name == 'lightgbm':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        else:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                     early_stopping_rounds=50, verbose=False)
        
        val_pred = model.predict(X_val)
        oof_predictions[name][val_idx] = val_pred
        test_predictions[name] += model.predict(X_test) / 5
        
        score = smape(y_val, val_pred)
        cv_scores[name].append(score)
        print(f"      SMAPE: {score:.4f}%")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================
# SECTION 8: CREATE SUBMISSION
# ============================================
print("\n[8/8] Creating ensemble submission...")

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
for name in models.keys():
    mean_score = np.mean(cv_scores[name])
    oof_score = smape(y_train, oof_predictions[name])
    print(f"{name.upper()}: CV={mean_score:.4f}% | OOF={oof_score:.4f}%")

ensemble_test = np.zeros(len(X_test))
for name, preds in test_predictions.items():
    ensemble_test += preds / len(models)

ensemble_oof = np.zeros(len(X_train))
for name, preds in oof_predictions.items():
    ensemble_oof += preds / len(models)

ensemble_cv = smape(y_train, ensemble_oof)

print("\n" + "="*80)
print("ENSEMBLE RESULTS")
print("="*80)
print(f"Ensemble CV Score: {ensemble_cv:.4f}%")
print(f"Your Phase 5: 57.900%")
print(f"Improvement: {57.900 - ensemble_cv:.2f}%")

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': ensemble_test
})

submission.to_csv('submission_ensemble_simple.csv', index=False)

print("\n✓ Submission created: submission_ensemble_simple.csv")
print("\n" + "="*80)
print("EXECUTION COMPLETE!")
print("="*80)
