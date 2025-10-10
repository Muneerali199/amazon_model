"""
Phase 3: Baseline Model Training
ML Challenge 2025 - Smart Product Pricing Challenge

This script:
1. Loads engineered features from Phase 2
2. Handles missing values
3. Trains baseline models (XGBoost, LightGBM, Random Forest)
4. Evaluates using SMAPE metric
5. Saves best model and predictions
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Custom SMAPE metric (Symmetric Mean Absolute Percentage Error)
def smape(y_true, y_pred):
    """
    Calculate SMAPE metric
    Range: 0-200%, lower is better
    """
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # Handle zero division
    return 200 * np.mean(diff)

def load_data():
    """Load training and test features"""
    print("=" * 80)
    print("PHASE 3: BASELINE MODEL TRAINING")
    print("=" * 80)
    print("\n[1/7] Loading data...")
    
    train_df = pd.read_csv('dataset/train_features.csv')
    test_df = pd.read_csv('dataset/test_features.csv')
    
    print(f"   ✓ Training data: {train_df.shape}")
    print(f"   ✓ Test data: {test_df.shape}")
    print(f"   ✓ Missing values - Train: {train_df.isnull().sum().sum()}, Test: {test_df.isnull().sum().sum()}")
    
    return train_df, test_df

def prepare_features(train_df, test_df):
    """Prepare features for modeling"""
    print("\n[2/7] Preparing features...")
    
    # Handle missing values
    print("   → Handling missing values...")
    train_df['ipq_unit'] = train_df['ipq_unit'].fillna('Count')
    test_df['ipq_unit'] = test_df['ipq_unit'].fillna('Count')
    train_df['item_name'] = train_df['item_name'].fillna('')
    test_df['item_name'] = test_df['item_name'].fillna('')
    
    # Select numeric features for baseline
    numeric_features = [
        'ipq_value', 'char_count', 'word_count', 'bullet_points',
        'has_description', 'num_count', 'uppercase_words', 'avg_word_length',
        'is_food', 'is_beverage', 'is_grocery', 'is_health',
        'is_personal_care', 'is_household'
    ]
    
    # One-hot encode ipq_unit
    print("   → Encoding categorical feature (ipq_unit)...")
    train_encoded = pd.get_dummies(train_df[['ipq_unit']], prefix='unit', drop_first=True)
    test_encoded = pd.get_dummies(test_df[['ipq_unit']], prefix='unit', drop_first=True)
    
    # Align columns (ensure train and test have same features)
    all_columns = list(set(train_encoded.columns) | set(test_encoded.columns))
    for col in all_columns:
        if col not in train_encoded.columns:
            train_encoded[col] = 0
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    
    train_encoded = train_encoded[sorted(all_columns)]
    test_encoded = test_encoded[sorted(all_columns)]
    
    # Combine numeric and encoded features
    X_train = pd.concat([train_df[numeric_features], train_encoded], axis=1)
    X_test = pd.concat([test_df[numeric_features], test_encoded], axis=1)
    
    # Clean column names for XGBoost/LightGBM (remove special characters)
    for char in ['[', ']', '<', '>', '{', '}', '"', ':', ',']:
        X_train.columns = X_train.columns.str.replace(char, '_', regex=False)
        X_test.columns = X_test.columns.str.replace(char, '_', regex=False)
    
    # Make column names unique using pandas
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
    y_train_log = train_df['log_price'].values
    
    # Store sample IDs
    train_ids = train_df['sample_id'].values
    test_ids = test_df['sample_id'].values
    
    print(f"   ✓ Final feature count: {X_train.shape[1]}")
    print(f"   ✓ Numeric features: {len(numeric_features)}")
    print(f"   ✓ Encoded features: {len(all_columns)}")
    print(f"   ✓ Missing values after prep: Train={X_train.isnull().sum().sum()}, Test={X_test.isnull().sum().sum()}")
    
    return X_train, X_test, y_train, y_train_log, train_ids, test_ids

def train_xgboost(X_train, y_train, n_folds=5):
    """Train XGBoost with cross-validation"""
    print("\n[3/7] Training XGBoost...")
    
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
        val_pred = np.maximum(val_pred, 0)  # Ensure non-negative prices
        score = smape(y_val, val_pred)
        cv_scores.append(score)
        models.append(model)
        
        print(f"   Fold {fold}: SMAPE = {score:.4f}%")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f"   ✓ Mean SMAPE: {mean_score:.4f}% (±{std_score:.4f}%)")
    
    return models, mean_score

def train_lightgbm(X_train, y_train, n_folds=5):
    """Train LightGBM with cross-validation"""
    print("\n[4/7] Training LightGBM...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**params, n_estimators=500)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
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

def train_random_forest(X_train, y_train, n_folds=5):
    """Train Random Forest with cross-validation"""
    print("\n[5/7] Training Random Forest...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        
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
    """Make predictions using ensemble of models"""
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)  # Ensure non-negative
        predictions.append(pred)
    
    # Average predictions from all folds
    final_pred = np.mean(predictions, axis=0)
    return final_pred

def save_results(test_ids, predictions, model_name, cv_score):
    """Save predictions and model info"""
    # Save predictions
    submission_df = pd.DataFrame({
        'sample_id': test_ids,
        'price': predictions
    })
    filename = f'dataset/submission_{model_name.lower()}.csv'
    submission_df.to_csv(filename, index=False)
    print(f"   ✓ Saved to: {filename}")
    
    return filename

def main():
    # Load data
    train_df, test_df = load_data()
    
    # Prepare features
    X_train, X_test, y_train, y_train_log, train_ids, test_ids = prepare_features(train_df, test_df)
    
    # Train models
    xgb_models, xgb_score = train_xgboost(X_train, y_train)
    
    print("\n[4/7] Skipping LightGBM (feature name issues)...")
    lgb_score = None
    
    rf_models, rf_score = train_random_forest(X_train, y_train)
    
    # Compare models
    print("\n[6/7] Model Comparison...")
    print(f"   XGBoost:       {xgb_score:.4f}%")
    print(f"   Random Forest: {rf_score:.4f}%")
    
    # Select best model
    best_models = None
    best_name = None
    best_score = float('inf')
    
    if xgb_score < best_score:
        best_score = xgb_score
        best_models = xgb_models
        best_name = "XGBoost"
    
    if rf_score < best_score:
        best_score = rf_score
        best_models = rf_models
        best_name = "Random Forest"
    
    print(f"\n   🏆 Best Model: {best_name} (SMAPE: {best_score:.4f}%)")
    
    # Make predictions with best model
    print("\n[7/7] Generating predictions...")
    predictions = make_predictions(best_models, X_test)
    
    print(f"   ✓ Predictions generated: {len(predictions)}")
    print(f"   ✓ Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"   ✓ Mean price: ${predictions.mean():.2f}")
    
    # Save results
    submission_file = save_results(test_ids, predictions, best_name, best_score)
    
    # Save model comparison results
    results = {
        "phase": "Phase 3 - Baseline Models",
        "feature_count": X_train.shape[1],
        "training_samples": len(X_train),
        "models": {
            "xgboost": {"smape": float(xgb_score)},
            "random_forest": {"smape": float(rf_score)}
        },
        "best_model": {
            "name": best_name,
            "smape": float(best_score)
        },
        "submission_file": submission_file,
        "features_used": "14 numeric + encoded ipq_unit"
    }
    
    with open('phase3_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Baseline established: {best_score:.4f}% SMAPE")
    print(f"✅ Submission file ready: {submission_file}")
    print(f"✅ Results saved: phase3_baseline_results.json")
    print("\n📊 Next Steps:")
    print("   → Analyze feature importance")
    print("   → Add text embeddings (sentence-transformers)")
    print("   → Try hyperparameter optimization")
    print("   → Consider image features (optional)")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
