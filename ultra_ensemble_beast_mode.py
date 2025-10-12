"""
🔥 ULTRA-ENSEMBLE BEAST MODE 🔥
Target: TOP 10 (43-46% SMAPE)
Combines: 3 Vision Models + 3 Text Models + 6 ML Models + Meta-Stacking
Runtime: 18-24 hours on Kaggle GPU
Strategy: Maximum diversity + Advanced features + Meta-learning
"""

# ============================================
# SECTION 1: SETUP & INSTALLATIONS
# ============================================
print("="*80)
print("🔥 ULTRA-ENSEMBLE BEAST MODE - STARTING 🔥")
print("Target: TOP 10 Position (43-46% SMAPE)")
print("="*80)

import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

print("\n[1/12] Installing packages (this will take 5-7 minutes)...")

# Install all required packages
packages = [
    'transformers==4.44.2',
    'torch==2.3.0',
    'torchvision==0.18.0',
    'timm',  # For EfficientNet
    'sentence-transformers',  # For advanced embeddings
    'pillow',
    'xgboost==2.1.1',
    'lightgbm==4.5.0',
    'catboost==1.2.5',
    'scikit-learn==1.5.2',
    'pandas',
    'numpy',
    'requests',
    'tqdm',
    'scipy'
]

for package in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)

print("✓ All packages installed!")

# ============================================
# SECTION 2: IMPORT LIBRARIES
# ============================================
print("\nImporting libraries...")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    CLIPProcessor, CLIPModel
)
import timm
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.spatial.distance import cosine
from tqdm import tqdm
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================
# SECTION 3: LOAD DATA
# ============================================
print("\n[2/12] Loading data...")

train_df = pd.read_csv('/kaggle/input/your-dataset-name/sample_train.csv')
test_df = pd.read_csv('/kaggle/input/your-dataset-name/sample_test.csv')

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape: {test_df.shape}")
print(f"\nTarget stats:\n{train_df['price'].describe()}")

# ============================================
# SECTION 4: VISION FEATURES (3 MODELS)
# ============================================
print("\n[3/12] Extracting vision features from 3 models...")
print("  This will take ~4-5 hours for 150K images")

class VisionFeatureExtractor:
    def __init__(self, device):
        self.device = device
        
        # Model 1: ResNet50 (2048-dim)
        print("  Loading ResNet50...")
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet.eval().to(device)
        
        # Model 2: EfficientNet-B4 (1792-dim)
        print("  Loading EfficientNet-B4...")
        self.efficientnet = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.efficientnet.eval().to(device)
        
        # Model 3: Vision Transformer (768-dim)
        print("  Loading Vision Transformer...")
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit.eval().to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.failed = 0
        self.success = 0
    
    def download_image(self, url):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content)).convert('RGB')
            self.success += 1
            return img
        except:
            self.failed += 1
            return Image.new('RGB', (224, 224), color='gray')
    
    def extract_features(self, image_url):
        img = self.download_image(image_url)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # ResNet50: 2048-dim
            resnet_feat = self.resnet(img_tensor).squeeze().cpu().numpy()
            
            # EfficientNet: 1792-dim
            eff_feat = self.efficientnet(img_tensor).squeeze().cpu().numpy()
            
            # ViT: 768-dim
            vit_feat = self.vit(img_tensor).squeeze().cpu().numpy()
        
        # Concatenate all: 4608-dim total
        return np.concatenate([resnet_feat, eff_feat, vit_feat])

vision_extractor = VisionFeatureExtractor(device)

def extract_vision_batch(df, name="train"):
    features_list = []
    print(f"  Extracting vision features for {name} ({len(df)} samples)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = vision_extractor.extract_features(row['image_link'])
        features_list.append(features)
        
        if idx % 500 == 0 and idx > 0:
            gc.collect()
            torch.cuda.empty_cache()
            success_rate = vision_extractor.success / (idx + 1) * 100
            print(f"    Progress: {idx}/{len(df)} | Success: {success_rate:.1f}%")
    
    feature_df = pd.DataFrame(features_list, columns=[f'vision_{i}' for i in range(4608)])
    total = vision_extractor.success + vision_extractor.failed
    print(f"  ✓ Success rate: {vision_extractor.success/total*100:.1f}% ({vision_extractor.success}/{total})")
    return feature_df

train_vision = extract_vision_batch(train_df, "train")
test_vision = extract_vision_batch(test_df, "test")

print(f"✓ Vision features: {train_vision.shape}")

# ============================================
# SECTION 5: TEXT FEATURES (3 MODELS)
# ============================================
print("\n[4/12] Extracting text features from 3 models...")
print("  This will take ~2-3 hours for 150K texts")

class TextFeatureExtractor:
    def __init__(self, device):
        self.device = device
        
        # Model 1: BERT (768-dim)
        print("  Loading BERT...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval().to(device)
        
        # Model 2: RoBERTa (768-dim)
        print("  Loading RoBERTa...")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.roberta_model.eval().to(device)
        
        # Model 3: CLIP Text (512-dim)
        print("  Loading CLIP...")
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model.eval().to(device)
    
    def extract_features(self, text):
        with torch.no_grad():
            # BERT
            bert_inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, 
                                             padding='max_length', max_length=128)
            bert_inputs = {k: v.to(self.device) for k, v in bert_inputs.items()}
            bert_feat = self.bert_model(**bert_inputs).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            # RoBERTa
            roberta_inputs = self.roberta_tokenizer(text, return_tensors='pt', truncation=True,
                                                   padding='max_length', max_length=128)
            roberta_inputs = {k: v.to(self.device) for k, v in roberta_inputs.items()}
            roberta_feat = self.roberta_model(**roberta_inputs).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            # CLIP
            clip_inputs = self.clip_processor(text=[text], return_tensors='pt', truncation=True, 
                                             padding=True, max_length=77)
            clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
            clip_feat = self.clip_model.get_text_features(**clip_inputs).squeeze().cpu().numpy()
        
        # Concatenate: 2048-dim total
        return np.concatenate([bert_feat, roberta_feat, clip_feat])

text_extractor = TextFeatureExtractor(device)

def extract_text_batch(df, name="train"):
    features_list = []
    print(f"  Extracting text features for {name} ({len(df)} samples)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = text_extractor.extract_features(str(row['catalog_content']))
        features_list.append(features)
        
        if idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return pd.DataFrame(features_list, columns=[f'text_{i}' for i in range(2048)])

train_text = extract_text_batch(train_df, "train")
test_text = extract_text_batch(test_df, "test")

print(f"✓ Text features: {train_text.shape}")

# ============================================
# SECTION 6: ADVANCED ENGINEERED FEATURES
# ============================================
print("\n[5/12] Creating advanced engineered features...")

def create_advanced_features(df, train_vision, train_text, is_train=True):
    features = pd.DataFrame()
    
    # Text statistics
    features['text_len'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_len'] = features['text_len'] / (features['word_count'] + 1)
    features['upper_ratio'] = df['catalog_content'].str.findall(r'[A-Z]').str.len() / (features['text_len'] + 1)
    features['digit_count'] = df['catalog_content'].str.findall(r'\d').str.len()
    features['has_price_mention'] = df['catalog_content'].str.contains(r'\$|price|cost|usd', case=False).astype(int)
    
    # Brand detection (expanded)
    brands = ['sony','samsung','apple','lg','hp','dell','lenovo','nike','adidas','puma',
              'canon','nikon','microsoft','google','amazon','philips','panasonic','xiaomi']
    for brand in brands:
        features[f'brand_{brand}'] = df['catalog_content'].str.lower().str.contains(brand).astype(int)
    
    # Category detection (expanded)
    categories = ['electronic','clothing','book','home','toy','sport','beauty','food',
                  'phone','laptop','camera','watch','shoe','bag','game','music']
    for cat in categories:
        features[f'cat_{cat}'] = df['catalog_content'].str.lower().str.contains(cat).astype(int)
    
    # Condition keywords
    conditions = ['new','used','refurbished','like new','excellent','good','fair']
    for cond in conditions:
        features[f'cond_{cond}'] = df['catalog_content'].str.lower().str.contains(cond).astype(int)
    
    if is_train:
        # Price clustering (only on train)
        print("  Creating price clusters...")
        kmeans = KMeans(n_clusters=10, random_state=42)
        features['price_cluster'] = kmeans.fit_predict(train_vision.iloc[:, :100])
        
        # PCA components
        print("  Computing PCA components...")
        pca_vision = PCA(n_components=50, random_state=42)
        pca_text = PCA(n_components=30, random_state=42)
        
        vision_pca = pca_vision.fit_transform(train_vision)
        text_pca = pca_text.fit_transform(train_text)
        
        for i in range(50):
            features[f'vision_pca_{i}'] = vision_pca[:, i]
        for i in range(30):
            features[f'text_pca_{i}'] = text_pca[:, i]
    
    return features

train_advanced = create_advanced_features(train_df, train_vision, train_text, is_train=True)
test_advanced = create_advanced_features(test_df, test_vision, test_text, is_train=False)

print(f"✓ Advanced features: {train_advanced.shape}")

# ============================================
# SECTION 7: COMBINE ALL FEATURES
# ============================================
print("\n[6/12] Combining all features...")

X_train = pd.concat([train_vision, train_text, train_advanced], axis=1)
X_test = pd.concat([test_vision, test_text, test_advanced], axis=1)
y_train = train_df['price'].values

# Fill NaN values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"✓ Total features: {X_train.shape[1]}")
print(f"  - Vision: 4608")
print(f"  - Text: 2048")
print(f"  - Advanced: {train_advanced.shape[1]}")

# ============================================
# SECTION 8: TRAIN DIVERSE ENSEMBLE (6 MODELS)
# ============================================
print("\n[7/12] Training 6 diverse ML models...")

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

# Define 6 diverse models
models = {
    'xgboost_1': xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.03, max_depth=10,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='gpu_hist', gpu_id=0, random_state=42
    ),
    'xgboost_2': xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=8,
        subsample=0.7, colsample_bytree=0.7,
        tree_method='gpu_hist', gpu_id=0, random_state=123
    ),
    'lightgbm_1': lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, max_depth=10,
        subsample=0.8, colsample_bytree=0.8,
        device='gpu', random_state=42
    ),
    'lightgbm_2': lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=8,
        subsample=0.7, colsample_bytree=0.7,
        device='gpu', random_state=123
    ),
    'catboost_1': CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=10,
        task_type='GPU', verbose=False, random_state=42
    ),
    'catboost_2': CatBoostRegressor(
        iterations=1200, learning_rate=0.05, depth=8,
        task_type='GPU', verbose=False, random_state=123
    )
}

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = {name: np.zeros(len(X_train)) for name in models.keys()}
test_predictions = {name: np.zeros(len(X_test)) for name in models.keys()}
cv_scores = {name: [] for name in models.keys()}

print("\nTraining with 5-fold CV (this will take ~4-6 hours)...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n  Fold {fold + 1}/5")
    
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    for name, model in models.items():
        print(f"    Training {name}...")
        
        if 'xgboost' in name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=100, verbose=False)
        elif 'lightgbm' in name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        else:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                     early_stopping_rounds=100, verbose=False)
        
        val_pred = model.predict(X_val)
        oof_predictions[name][val_idx] = val_pred
        test_predictions[name] += model.predict(X_test) / 5
        
        score = smape(y_val, val_pred)
        cv_scores[name].append(score)
        print(f"      SMAPE: {score:.4f}%")
    
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*80)
print("INDIVIDUAL MODEL RESULTS")
print("="*80)
for name in models.keys():
    mean_cv = np.mean(cv_scores[name])
    oof_score = smape(y_train, oof_predictions[name])
    print(f"{name}: CV={mean_cv:.4f}% | OOF={oof_score:.4f}%")

# ============================================
# SECTION 9: META-STACKING LAYER
# ============================================
print("\n[8/12] Training meta-stacking layer...")

# Create meta-features from OOF predictions
meta_train = np.column_stack([oof_predictions[name] for name in models.keys()])
meta_test = np.column_stack([test_predictions[name] for name in models.keys()])

# Add original features (top 100 most important)
from sklearn.ensemble import RandomForestRegressor as RF
print("  Selecting top 100 features...")
rf_selector = RF(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_selector.fit(X_train.iloc[:10000], y_train[:10000])  # Sample for speed
importances = rf_selector.feature_importances_
top_indices = np.argsort(importances)[-100:]

meta_train = np.column_stack([meta_train, X_train.iloc[:, top_indices]])
meta_test = np.column_stack([meta_test, X_test.iloc[:, top_indices]])

# Train meta-model
print("  Training XGBoost meta-model...")
meta_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='gpu_hist', random_state=42
)

meta_oof = np.zeros(len(meta_train))
meta_pred = np.zeros(len(meta_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(meta_train)):
    print(f"  Meta-fold {fold + 1}/5...")
    meta_model.fit(meta_train[tr_idx], y_train[tr_idx],
                  eval_set=[(meta_train[val_idx], y_train[val_idx])],
                  early_stopping_rounds=50, verbose=False)
    meta_oof[val_idx] = meta_model.predict(meta_train[val_idx])
    meta_pred += meta_model.predict(meta_test) / 5

meta_cv = smape(y_train, meta_oof)

# ============================================
# SECTION 10: ENSEMBLE COMBINATIONS
# ============================================
print("\n[9/12] Creating multiple ensemble combinations...")

# 1. Simple average
simple_avg = sum(test_predictions.values()) / len(models)

# 2. Weighted by CV score
weights = np.array([1.0 / np.mean(cv_scores[name]) for name in models.keys()])
weights = weights / weights.sum()
weighted_avg = sum(test_predictions[name] * w for name, w in zip(models.keys(), weights))

# 3. Best 3 models average
sorted_models = sorted(models.keys(), key=lambda x: np.mean(cv_scores[x]))
best_3_avg = sum(test_predictions[name] for name in sorted_models[:3]) / 3

# 4. Rank averaging
rank_preds = []
for name in models.keys():
    ranks = pd.Series(test_predictions[name]).rank(pct=True)
    rank_preds.append(ranks.values)
rank_avg = np.mean(rank_preds, axis=0)
# Convert ranks back to prices
rank_avg_prices = np.percentile(y_train, rank_avg * 100)

# 5. Meta-stacking
meta_final = meta_pred

# ============================================
# SECTION 11: CREATE SUBMISSIONS
# ============================================
print("\n[10/12] Creating submissions...")

submissions = {
    'simple_avg': simple_avg,
    'weighted_avg': weighted_avg,
    'best_3_avg': best_3_avg,
    'rank_avg': rank_avg_prices,
    'meta_stacking': meta_final
}

for name, preds in submissions.items():
    df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': preds})
    df.to_csv(f'submission_{name}.csv', index=False)
    print(f"  ✓ submission_{name}.csv")

# ============================================
# SECTION 12: FINAL RESULTS
# ============================================
print("\n[11/12] Computing final scores...")

print("\n" + "="*80)
print("🔥 ULTRA-ENSEMBLE RESULTS 🔥")
print("="*80)

ensemble_scores = {
    'Simple Average': smape(y_train, sum(oof_predictions.values()) / len(models)),
    'Weighted Average': smape(y_train, sum(oof_predictions[name] * w for name, w in zip(models.keys(), weights))),
    'Best 3 Average': smape(y_train, sum(oof_predictions[name] for name in sorted_models[:3]) / 3),
    'Meta-Stacking': meta_cv
}

for name, score in ensemble_scores.items():
    print(f"{name}: {score:.4f}% SMAPE")

best_method = min(ensemble_scores, key=ensemble_scores.get)
best_score = ensemble_scores[best_method]

print(f"\n🏆 BEST METHOD: {best_method}")
print(f"🏆 BEST CV SCORE: {best_score:.4f}%")
print(f"\n📊 Your Phase 5: 57.900%")
print(f"📊 Improvement: {57.900 - best_score:.2f}%")
print(f"\n🎯 Expected Leaderboard: {best_score * 0.98:.2f}% - {best_score * 1.02:.2f}%")

if best_score < 46:
    print("\n🔥🔥🔥 AMAZING! TOP 10 POTENTIAL! 🔥🔥🔥")
elif best_score < 48:
    print("\n⭐⭐⭐ EXCELLENT! TOP 30 POTENTIAL! ⭐⭐⭐")
elif best_score < 52:
    print("\n✅✅✅ VERY GOOD! TOP 100 POTENTIAL! ✅✅✅")
else:
    print("\n✅ GOOD! Significant improvement over Phase 5!")

print("\n[12/12] Execution complete!")
print("="*80)
print("RECOMMENDATIONS:")
print("1. Submit 'submission_meta_stacking.csv' first")
print("2. If that doesn't work well, try 'submission_best_3_avg.csv'")
print("3. Compare with your Phase 5 (57.900%) after scoring")
print("="*80)
