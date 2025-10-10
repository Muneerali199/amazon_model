"""
Phase 4 Continued: Hyperparameter Optimization
ML Challenge 2025 - Smart Product Pricing Challenge

This script:
1. Loads TF-IDF enhanced features
2. Uses RandomizedSearchCV to find optimal XGBoost parameters
3. Trains model with best parameters
4. Compares with previous results

Goal: Reduce SMAPE from 60.93% to < 55%
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return 200 * np.mean(diff)

def smape_scorer(y_true, y_pred):
    """SMAPE scorer for sklearn (negative because sklearn maximizes)"""
    return -smape(y_true, y_pred)

def load_data():
    """Load data"""
    print("=" * 80)
    print("PHASE 4 CONTINUED: HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\n[1/6] Loading data...")
    
    train_df = pd.read_csv('dataset/train_features.csv')
    test_df = pd.read_csv('dataset/test_features.csv')
    train_orig = pd.read_csv('dataset/train.csv')
    test_orig = pd.read_csv('dataset/test.csv')
    
    print(f"   ✓ Training: {train_df.shape}")
    print(f"   ✓ Test: {test_df.shape}")
    
    return train_df, test_df, train_orig, test_orig

def generate_tfidf_features(train_orig, test_orig, n_components=100):
    """Generate TF-IDF features (same as before)"""
    print("\n[2/6] Generating TF-IDF features...")
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        strip_accents='unicode',
        lowercase=True
    )
    
    train_texts = train_orig['catalog_content'].fillna('')
    test_texts = test_orig['catalog_content'].fillna('')
    
    print(f"   → Fitting TF-IDF...")
    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)
    
    print(f"   → Reducing to {n_components} dimensions...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    
    train_feat_df = pd.DataFrame(train_svd, columns=[f'tfidf_{i}' for i in range(n_components)])
    test_feat_df = pd.DataFrame(test_svd, columns=[f'tfidf_{i}' for i in range(n_components)])
    
    print(f"   ✓ TF-IDF features ready: {train_feat_df.shape}")
    
    return train_feat_df, test_feat_df

def prepare_features(train_df, test_df, train_tfidf_df, test_tfidf_df):
    """Combine features"""
    print("\n[3/6] Preparing features...")
    
    train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
    test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')
    train_df['item_name'] = train_df['item_name'].fillna('')
    test_df['item_name'] = test_df['item_name'].fillna('')
    
    numeric_features = [
        'ipq_value', 'char_count', 'word_count', 'bullet_points',
        'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
        'is_food', 'is_beverage', 'is_grocery', 'is_health',
        'is_personal_care', 'is_household'
    ]
    
    train_encoded = pd.get_dummies(train_df[['ipq_unit']], prefix='unit', drop_first=True)
    test_encoded = pd.get_dummies(test_df[['ipq_unit']], prefix='unit', drop_first=True)
    
    all_columns = list(set(train_encoded.columns) | set(test_encoded.columns))
    for col in all_columns:
        if col not in train_encoded.columns:
            train_encoded[col] = 0
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    
    train_encoded = train_encoded[sorted(all_columns)]
    test_encoded = test_encoded[sorted(all_columns)]
    
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
    
    for char in ['[', ']', '<', '>', '{', '}', '"', ':', ',']:
        X_train.columns = X_train.columns.str.replace(char, '_', regex=False)
        X_test.columns = X_test.columns.str.replace(char, '_', regex=False)
    
    y_train = train_df['price'].values
    train_ids = train_df['sample_id'].values
    test_ids = test_df['sample_id'].values
    
    print(f"   ✓ Total features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, train_ids, test_ids

def optimize_hyperparameters(X_train, y_train):
    """Optimize XGBoost hyperparameters using RandomizedSearchCV"""
    print("\n[4/6] Optimizing hyperparameters...")
    print("   → This may take 15-20 minutes...")
    
    # Define parameter search space
    param_distributions = {
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'max_depth': [4, 6, 8, 10],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5],
        'n_estimators': [300, 500, 700]
    }
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Custom scorer
    from sklearn.metrics import make_scorer
    smape_scorer_func = make_scorer(smape_scorer, greater_is_better=True)
    
    # RandomizedSearchCV
    print(f"   → Testing {param_distributions}")
    print(f"   → Using 3-fold CV, 30 iterations")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,  # Try 30 random combinations
        scoring=smape_scorer_func,
        cv=3,  # 3-fold CV (faster than 5-fold)
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit
    random_search.fit(X_train, y_train)
    
    print(f"\n   ✓ Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"      {param}: {value}")
    
    print(f"\n   ✓ Best CV score: {-random_search.best_score_:.4f}% SMAPE")
    
    return random_search.best_params_, -random_search.best_score_

def train_with_best_params(X_train, y_train, best_params, n_folds=5):
    """Train model with optimized parameters"""
    print("\n[5/6] Training with optimized parameters...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Create model with best params
        model = xgb.XGBRegressor(**best_params, early_stopping_rounds=50)
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

def compare_results(new_score, tfidf_score=60.9300, baseline_score=66.4390):
    """Compare with previous results"""
    print("\n[6/6] Comparing results...")
    print(f"   Phase 3 (Baseline):    {baseline_score:.4f}%")
    print(f"   Phase 4 (TF-IDF):      {tfidf_score:.4f}%")
    print(f"   Phase 4 (Optimized):   {new_score:.4f}%")
    
    improvement_from_baseline = baseline_score - new_score
    improvement_from_tfidf = tfidf_score - new_score
    
    print(f"\n   📊 Total improvement from baseline: -{improvement_from_baseline:.4f}%")
    print(f"   📊 Additional from tuning: -{improvement_from_tfidf:.4f}%")
    
    if new_score < 55.0:
        print(f"\n   🎉 TARGET ACHIEVED! SMAPE < 55%")
    else:
        remaining = new_score - 55.0
        print(f"\n   🎯 Gap to target (55%): {remaining:.4f}%")
    
    return improvement_from_baseline, improvement_from_tfidf

def make_predictions(models, X_test):
    """Make predictions"""
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

def save_results(test_ids, predictions, new_score, best_params, baseline_score, tfidf_score):
    """Save results"""
    print("\n[7/7] Saving results...")
    
    # Save predictions
    submission_df = pd.DataFrame({
        'sample_id': test_ids,
        'price': predictions
    })
    filename = 'dataset/submission_xgboost_optimized.csv'
    submission_df.to_csv(filename, index=False)
    print(f"   ✓ Saved: {filename}")
    
    # Save results
    results = {
        "phase": "Phase 4 - Hyperparameter Optimization",
        "baseline_smape": float(baseline_score),
        "tfidf_smape": float(tfidf_score),
        "optimized_smape": float(new_score),
        "total_improvement": float(baseline_score - new_score),
        "tuning_improvement": float(tfidf_score - new_score),
        "target_achieved": bool(new_score < 55.0),
        "best_parameters": best_params,
        "submission_file": filename
    }
    
    with open('phase4_optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Saved: phase4_optimized_results.json")
    
    return filename

def main():
    # Load data
    train_df, test_df, train_orig, test_orig = load_data()
    
    # Generate TF-IDF
    train_tfidf_df, test_tfidf_df = generate_tfidf_features(train_orig, test_orig)
    
    # Prepare features
    X_train, X_test, y_train, train_ids, test_ids = prepare_features(
        train_df, test_df, train_tfidf_df, test_tfidf_df
    )
    
    # Optimize hyperparameters
    best_params, cv_score = optimize_hyperparameters(X_train, y_train)
    
    # Train with best params (full 5-fold CV)
    models, final_score = train_with_best_params(X_train, y_train, best_params)
    
    # Compare
    improvement_from_baseline, improvement_from_tfidf = compare_results(final_score)
    
    # Predict
    print("\nGenerating predictions...")
    predictions = make_predictions(models, X_test)
    print(f"   ✓ Predictions: {len(predictions)}")
    print(f"   ✓ Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"   ✓ Mean: ${predictions.mean():.2f}")
    
    # Save
    filename = save_results(test_ids, predictions, final_score, best_params, 66.4390, 60.9300)
    
    print("\n" + "=" * 80)
    print("PHASE 4 (OPTIMIZATION) COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Final SMAPE: {final_score:.4f}%")
    print(f"✅ Total improvement: {improvement_from_baseline:.4f}%")
    print(f"✅ From tuning: {improvement_from_tfidf:.4f}%")
    print(f"✅ Submission: {filename}")
    
    if final_score < 55.0:
        print(f"\n🎉 TARGET ACHIEVED! SMAPE < 55%")
    else:
        print(f"\n🎯 Close to target! Gap: {final_score - 55.0:.4f}%")
    
    print("\n📊 Next steps:")
    print("   → Feature engineering v2 (if needed)")
    print("   → Ensemble methods")
    print("   → Final submission preparation")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
