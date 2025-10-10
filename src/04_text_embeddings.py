"""
Phase 4: Model Optimization - Text Embeddings
ML Challenge 2025 - Smart Product Pricing Challenge

This script:
1. Loads features from Phase 2
2. Generates text embeddings using sentence-transformers (LOCAL)
3. Combines embeddings with existing features
4. Retrains XGBoost with enhanced features
5. Compares performance with baseline

Goal: Reduce SMAPE from 66.44% to < 60% using text embeddings
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import sentence transformers for text embeddings
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow to avoid DLL issues

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Installing now...")

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)

def load_data():
    """Load data from Phase 2"""
    print("=" * 80)
    print("PHASE 4: MODEL OPTIMIZATION - TEXT EMBEDDINGS")
    print("=" * 80)
    print("\n[1/8] Loading data from Phase 2...")
    
    train_df = pd.read_csv('dataset/train_features.csv')
    test_df = pd.read_csv('dataset/test_features.csv')
    
    # Also load original data for full text
    train_orig = pd.read_csv('dataset/train.csv')
    test_orig = pd.read_csv('dataset/test.csv')
    
    print(f"   ✓ Training data: {train_df.shape}")
    print(f"   ✓ Test data: {test_df.shape}")
    print(f"   ✓ Original train: {train_orig.shape}")
    print(f"   ✓ Original test: {test_orig.shape}")
    
    return train_df, test_df, train_orig, test_orig

def generate_embeddings(train_orig, test_orig, model_name='all-MiniLM-L6-v2'):
    """
    Generate text embeddings using sentence-transformers
    
    Model: all-MiniLM-L6-v2
    - Size: 22.7 MB (< 8B parameter limit)
    - Dimensions: 384
    - Speed: Fast
    - Quality: Good for product descriptions
    - License: Apache 2.0 ✅
    """
    print("\n[2/8] Generating text embeddings...")
    print(f"   → Loading model: {model_name}")
    
    # Load pre-trained model (downloads automatically if not cached)
    model = SentenceTransformer(model_name)
    
    print(f"   → Model loaded: {model_name}")
    print(f"   → Embedding dimensions: 384")
    print(f"   → Processing catalog content...")
    
    # Generate embeddings for training data
    print(f"   → Encoding training data (75,000 samples)...")
    train_texts = train_orig['catalog_content'].fillna('').tolist()
    train_embeddings = model.encode(train_texts, show_progress_bar=True, batch_size=128)
    
    # Generate embeddings for test data
    print(f"   → Encoding test data (75,000 samples)...")
    test_texts = test_orig['catalog_content'].fillna('').tolist()
    test_embeddings = model.encode(test_texts, show_progress_bar=True, batch_size=128)
    
    print(f"   ✓ Training embeddings shape: {train_embeddings.shape}")
    print(f"   ✓ Test embeddings shape: {test_embeddings.shape}")
    
    # Convert to DataFrames
    train_emb_df = pd.DataFrame(
        train_embeddings,
        columns=[f'emb_{i}' for i in range(train_embeddings.shape[1])]
    )
    test_emb_df = pd.DataFrame(
        test_embeddings,
        columns=[f'emb_{i}' for i in range(test_embeddings.shape[1])]
    )
    
    return train_emb_df, test_emb_df

def prepare_features(train_df, test_df, train_emb_df, test_emb_df):
    """Combine original features with embeddings"""
    print("\n[3/8] Combining features with embeddings...")
    
    # Handle missing values in original features
    train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
    test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')
    train_df['item_name'] = train_df['item_name'].fillna('')
    test_df['item_name'] = test_df['item_name'].fillna('')
    
    # Select numeric features
    numeric_features = [
        'ipq_value', 'char_count', 'word_count', 'bullet_points',
        'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
        'is_food', 'is_beverage', 'is_grocery', 'is_health',
        'is_personal_care', 'is_household'
    ]
    
    # One-hot encode ipq_unit
    train_encoded = pd.get_dummies(train_df[['ipq_unit']], prefix='unit', drop_first=True)
    test_encoded = pd.get_dummies(test_df[['ipq_unit']], prefix='unit', drop_first=True)
    
    # Align columns
    all_columns = list(set(train_encoded.columns) | set(test_encoded.columns))
    for col in all_columns:
        if col not in train_encoded.columns:
            train_encoded[col] = 0
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    
    train_encoded = train_encoded[sorted(all_columns)]
    test_encoded = test_encoded[sorted(all_columns)]
    
    # Combine: numeric + encoded + embeddings
    X_train = pd.concat([
        train_df[numeric_features].reset_index(drop=True),
        train_encoded.reset_index(drop=True),
        train_emb_df.reset_index(drop=True)
    ], axis=1)
    
    X_test = pd.concat([
        test_df[numeric_features].reset_index(drop=True),
        test_encoded.reset_index(drop=True),
        test_emb_df.reset_index(drop=True)
    ], axis=1)
    
    # Clean column names
    for char in ['[', ']', '<', '>', '{', '}', '"', ':', ',']:
        X_train.columns = X_train.columns.str.replace(char, '_', regex=False)
        X_test.columns = X_test.columns.str.replace(char, '_', regex=False)
    
    # Make unique
    def make_unique_columns(df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index
            cols.iloc[dup_indices] = [f"{dup}_{i}" for i in range(len(dup_indices))]
        return cols.tolist()
    
    X_train.columns = make_unique_columns(X_train)
    X_test.columns = make_unique_columns(X_test)
    
    # Target variables
    y_train = train_df['price'].values
    train_ids = train_df['sample_id'].values
    test_ids = test_df['sample_id'].values
    
    print(f"   ✓ Original features: {len(numeric_features) + len(all_columns)}")
    print(f"   ✓ Embedding features: 384")
    print(f"   ✓ Total features: {X_train.shape[1]}")
    print(f"   ✓ Missing values: {X_train.isnull().sum().sum()}")
    
    return X_train, X_test, y_train, train_ids, test_ids

def train_xgboost_with_embeddings(X_train, y_train, n_folds=5):
    """Train XGBoost with embeddings"""
    print("\n[4/8] Training XGBoost with text embeddings...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**params, n_estimators=500, early_stopping_rounds=50)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        val_pred = np.maximum(val_pred, 0)
        score = smape(y_val, val_pred)
        cv_scores.append(score)
        models.append(model)
        
        print(f"   Fold {fold}: SMAPE = {score:.4f}%")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f"   ✓ Mean SMAPE: {mean_score:.4f}% (±{std_score:.4f}%)")
    
    return models, mean_score

def make_predictions(models, X_test):
    """Make predictions using ensemble"""
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)
        predictions.append(pred)
    
    final_pred = np.mean(predictions, axis=0)
    return final_pred

def compare_with_baseline(new_score, baseline_score=66.4390):
    """Compare new score with baseline"""
    print("\n[5/8] Comparing with baseline...")
    print(f"   Baseline (Phase 3):  {baseline_score:.4f}%")
    print(f"   With Embeddings:     {new_score:.4f}%")
    
    improvement = baseline_score - new_score
    improvement_pct = (improvement / baseline_score) * 100
    
    if improvement > 0:
        print(f"   ✅ Improvement: -{improvement:.4f}% ({improvement_pct:.2f}% better)")
    else:
        print(f"   ⚠️ Degradation: +{abs(improvement):.4f}% ({abs(improvement_pct):.2f}% worse)")
    
    return improvement

def save_results(test_ids, predictions, new_score, baseline_score, improvement):
    """Save results and update documentation"""
    print("\n[6/8] Saving results...")
    
    # Save predictions
    submission_df = pd.DataFrame({
        'sample_id': test_ids,
        'price': predictions
    })
    filename = 'dataset/submission_xgboost_embeddings.csv'
    submission_df.to_csv(filename, index=False)
    print(f"   ✓ Predictions saved: {filename}")
    
    # Save results JSON
    results = {
        "phase": "Phase 4 - Text Embeddings",
        "baseline_smape": float(baseline_score),
        "new_smape": float(new_score),
        "improvement": float(improvement),
        "improvement_percentage": float((improvement / baseline_score) * 100),
        "feature_count": 539,  # 155 original + 384 embeddings
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimensions": 384,
        "submission_file": filename
    }
    
    with open('phase4_embeddings_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Results saved: phase4_embeddings_results.json")
    
    return filename

def analyze_improvement(improvement, new_score):
    """Analyze improvement and provide recommendations"""
    print("\n[7/8] Analysis and Recommendations...")
    
    if improvement >= 5:
        print("   🎉 EXCELLENT: Large improvement achieved!")
        print("   → Text embeddings are very effective")
        print("   → Ready to proceed with hyperparameter tuning")
    elif improvement >= 2:
        print("   ✅ GOOD: Moderate improvement achieved")
        print("   → Text embeddings provide value")
        print("   → Consider combining with other strategies")
    elif improvement > 0:
        print("   ⚠️ MARGINAL: Small improvement")
        print("   → Embeddings help but limited")
        print("   → Focus on other optimization strategies")
    else:
        print("   ❌ NO IMPROVEMENT: Embeddings didn't help")
        print("   → May need different embedding model")
        print("   → Or embeddings not suitable for this task")
    
    print(f"\n   Current SMAPE: {new_score:.4f}%")
    print(f"   Target SMAPE: < 55.00%")
    remaining = new_score - 55.0
    print(f"   Gap to target: {remaining:.4f}%")

def main():
    # Check if sentence-transformers is installed
    if not EMBEDDINGS_AVAILABLE:
        print("\n❌ ERROR: sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return
    
    # Load data
    train_df, test_df, train_orig, test_orig = load_data()
    
    # Generate embeddings
    train_emb_df, test_emb_df = generate_embeddings(train_orig, test_orig)
    
    # Prepare features
    X_train, X_test, y_train, train_ids, test_ids = prepare_features(
        train_df, test_df, train_emb_df, test_emb_df
    )
    
    # Train model with embeddings
    models, new_score = train_xgboost_with_embeddings(X_train, y_train)
    
    # Compare with baseline
    baseline_score = 66.4390
    improvement = compare_with_baseline(new_score, baseline_score)
    
    # Make predictions
    print("\n[8/8] Generating final predictions...")
    predictions = make_predictions(models, X_test)
    print(f"   ✓ Predictions: {len(predictions)}")
    print(f"   ✓ Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    
    # Save results
    filename = save_results(test_ids, predictions, new_score, baseline_score, improvement)
    
    # Analyze
    analyze_improvement(improvement, new_score)
    
    print("\n" + "=" * 80)
    print("PHASE 4 (TEXT EMBEDDINGS) COMPLETE!")
    print("=" * 80)
    print(f"\n✅ New SMAPE: {new_score:.4f}%")
    print(f"✅ Improvement: {improvement:.4f}%")
    print(f"✅ Submission ready: {filename}")
    print("\n📊 Next Steps:")
    print("   → Hyperparameter optimization")
    print("   → Feature engineering v2")
    print("   → Ensemble methods")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
