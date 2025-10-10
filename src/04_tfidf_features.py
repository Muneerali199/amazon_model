"""
Phase 4: Model Optimization - TF-IDF Text Features
ML Challenge 2025 - Smart Product Pricing Challenge

Alternative approach using TF-IDF instead of embeddings (avoids TensorFlow issues)

This script:
1. Loads features from Phase 2
2. Generates TF-IDF features from product text
3. Combines with existing features
4. Retrains XGBoost with enhanced features
5. Compares performance with baseline

Goal: Reduce SMAPE from 66.44%
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)

def load_data():
    """Load data"""
    print("=" * 80)
    print("PHASE 4: MODEL OPTIMIZATION - TF-IDF TEXT FEATURES")
    print("=" * 80)
    print("\n[1/7] Loading data...")
    
    train_df = pd.read_csv('dataset/train_features.csv')
    test_df = pd.read_csv('dataset/test_features.csv')
    train_orig = pd.read_csv('dataset/train.csv')
    test_orig = pd.read_csv('dataset/test.csv')
    
    print(f"   ✓ Training: {train_df.shape}")
    print(f"   ✓ Test: {test_df.shape}")
    
    return train_df, test_df, train_orig, test_orig

def generate_tfidf_features(train_orig, test_orig, n_components=100):
    """
    Generate TF-IDF features and reduce dimensions with SVD
    
    TF-IDF captures word importance
    SVD reduces to manageable dimensions
    """
    print("\n[2/7] Generating TF-IDF features...")
    print(f"   → Extracting TF-IDF from catalog content...")
    
    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,  # Top 5000 words
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5,  # Word must appear in at least 5 documents
        max_df=0.8,  # Ignore words in >80% of documents
        strip_accents='unicode',
        lowercase=True
    )
    
    # Fit on training text
    train_texts = train_orig['catalog_content'].fillna('')
    test_texts = test_orig['catalog_content'].fillna('')
    
    print(f"   → Fitting TF-IDF on {len(train_texts)} samples...")
    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)
    
    print(f"   ✓ TF-IDF shape: {train_tfidf.shape}")
    print(f"   ✓ Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # Reduce dimensions with Truncated SVD
    print(f"   → Reducing to {n_components} dimensions with SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"   ✓ Explained variance: {explained_var:.2%}")
    print(f"   ✓ Final shape: {train_svd.shape}")
    
    # Convert to DataFrame
    train_feat_df = pd.DataFrame(
        train_svd,
        columns=[f'tfidf_{i}' for i in range(n_components)]
    )
    test_feat_df = pd.DataFrame(
        test_svd,
        columns=[f'tfidf_{i}' for i in range(n_components)]
    )
    
    return train_feat_df, test_feat_df

def prepare_features(train_df, test_df, train_tfidf_df, test_tfidf_df):
    """Combine original features with TF-IDF"""
    print("\n[3/7] Combining features...")
    
    # Handle missing values
    train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
    test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')
    train_df['item_name'] = train_df['item_name'].fillna('')
    test_df['item_name'] = test_df['item_name'].fillna('')
    
    # Numeric features
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
    
    # Combine all features
    X_train = pd.concat([
        train_df[numeric_features].reset_index(drop=True),
        train_encoded.reset_index(drop=True),
        train_tfidf_df.reset_index(drop=True)
    ], axis=1)
    
    X_test = pd.concat([
        test_df[numeric_features].reset_index(drop=True),
        test_encoded.reset_index(drop=True),
        test_tfidf_df.reset_index(drop=True)
    ], axis=1)
    
    # Clean column names
    for char in ['[', ']', '<', '>', '{', '}', '"', ':', ',']:
        X_train.columns = X_train.columns.str.replace(char, '_', regex=False)
        X_test.columns = X_test.columns.str.replace(char, '_', regex=False)
    
    # Target variables
    y_train = train_df['price'].values
    train_ids = train_df['sample_id'].values
    test_ids = test_df['sample_id'].values
    
    print(f"   ✓ Total features: {X_train.shape[1]}")
    print(f"   ✓ Missing values: {X_train.isnull().sum().sum()}")
    
    return X_train, X_test, y_train, train_ids, test_ids

def train_xgboost_enhanced(X_train, y_train, n_folds=5):
    """Train XGBoost with TF-IDF features"""
    print("\n[4/7] Training XGBoost with TF-IDF features...")
    
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
    """Make predictions"""
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)
        predictions.append(pred)
    
    return np.mean(predictions, axis=0)

def compare_results(new_score, baseline_score=66.4390):
    """Compare with baseline"""
    print("\n[5/7] Comparing with baseline...")
    print(f"   Baseline (Phase 3):  {baseline_score:.4f}%")
    print(f"   With TF-IDF:         {new_score:.4f}%")
    
    improvement = baseline_score - new_score
    improvement_pct = (improvement / baseline_score) * 100
    
    if improvement > 0:
        print(f"   ✅ Improvement: -{improvement:.4f}% ({improvement_pct:.2f}% better)")
    else:
        print(f"   ⚠️ Degradation: +{abs(improvement):.4f}% ({abs(improvement_pct):.2f}% worse)")
    
    return improvement

def save_results(test_ids, predictions, new_score, baseline_score, improvement, feature_count):
    """Save results"""
    print("\n[6/7] Saving results...")
    
    # Save predictions
    submission_df = pd.DataFrame({
        'sample_id': test_ids,
        'price': predictions
    })
    filename = 'dataset/submission_xgboost_tfidf.csv'
    submission_df.to_csv(filename, index=False)
    print(f"   ✓ Saved: {filename}")
    
    # Save results JSON
    results = {
        "phase": "Phase 4 - TF-IDF Features",
        "baseline_smape": float(baseline_score),
        "new_smape": float(new_score),
        "improvement": float(improvement),
        "improvement_percentage": float((improvement / baseline_score) * 100),
        "feature_count": feature_count,
        "tfidf_dimensions": 100,
        "submission_file": filename
    }
    
    with open('phase4_tfidf_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Saved: phase4_tfidf_results.json")
    
    return filename

def main():
    # Load data
    train_df, test_df, train_orig, test_orig = load_data()
    
    # Generate TF-IDF features
    train_tfidf_df, test_tfidf_df = generate_tfidf_features(train_orig, test_orig, n_components=100)
    
    # Prepare features
    X_train, X_test, y_train, train_ids, test_ids = prepare_features(
        train_df, test_df, train_tfidf_df, test_tfidf_df
    )
    
    # Train model
    models, new_score = train_xgboost_enhanced(X_train, y_train)
    
    # Compare
    baseline_score = 66.4390
    improvement = compare_results(new_score, baseline_score)
    
    # Make predictions
    print("\n[7/7] Generating predictions...")
    predictions = make_predictions(models, X_test)
    print(f"   ✓ Count: {len(predictions)}")
    print(f"   ✓ Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"   ✓ Mean: ${predictions.mean():.2f}")
    
    # Save
    filename = save_results(test_ids, predictions, new_score, baseline_score, improvement, X_train.shape[1])
    
    print("\n" + "=" * 80)
    print("PHASE 4 (TF-IDF) COMPLETE!")
    print("=" * 80)
    print(f"\n✅ New SMAPE: {new_score:.4f}%")
    print(f"✅ Improvement: {improvement:.4f}%")
    print(f"✅ Submission: {filename}")
    print("\n📊 Next Steps:")
    print("   → Try hyperparameter tuning")
    print("   → Add more feature engineering")
    print("   → Consider ensemble methods")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
