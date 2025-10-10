# ML Challenge 2025 - Smart Product Pricing Challenge
## Project Plan & Implementation Guide

---

## 📋 Problem Statement

### Objective
Develop an ML solution that analyzes product details (text and images) to predict optimal product prices for e-commerce platforms.

### Challenge Overview
- **Task Type**: Regression (Price Prediction)
- **Training Data**: 75,000 products with prices
- **Test Data**: 75,000 products (no prices)
- **Evaluation Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Target**: Minimize SMAPE score

---

## 📊 Data Description

### Available Features

| Column | Description | Type | Availability |
|--------|-------------|------|--------------|
| `sample_id` | Unique identifier for each sample | Integer | Train & Test |
| `catalog_content` | Title + Description + Item Pack Quantity (IPQ) | Text | Train & Test |
| `image_link` | Public URL to product image | URL | Train & Test |
| `price` | Product price (TARGET VARIABLE) | Float | Train only |

### Dataset Files
- `dataset/train.csv` - Training data with prices (75K samples)
- `dataset/test.csv` - Test data without prices (75K samples)
- `dataset/sample_test.csv` - Sample test input
- `dataset/sample_test_out.csv` - Sample output format reference

---

## 🎯 Evaluation Metric: SMAPE

### Formula
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2) * 100%
```

### Properties
- **Range**: 0% to 200%
- **Lower is Better**: 0% = perfect prediction
- **Symmetric**: Treats over-prediction and under-prediction equally

### Example
- Actual Price: $100
- Predicted Price: $120
- SMAPE = |100-120| / ((100+120)/2) = 20/110 = 18.18%

---

## 🚀 Implementation Roadmap

### Phase 1: Data Exploration & Understanding (Week 1)
- [ ] **Task 1.1**: Load and explore training data
  - Check data shape, types, and basic statistics
  - Analyze price distribution (mean, median, outliers)
  - Identify missing values
  
- [ ] **Task 1.2**: Text Analysis
  - Analyze `catalog_content` structure
  - Extract common patterns (brands, categories, IPQ)
  - Calculate text statistics (length, word count)
  
- [ ] **Task 1.3**: Image Analysis
  - Check image availability and quality
  - Download sample images using `src/utils.py`
  - Analyze image dimensions and formats

- [ ] **Task 1.4**: Price Distribution Analysis
  - Visualize price distribution
  - Check for outliers (very high/low prices)
  - Log transformation analysis

**Deliverable**: EDA notebook with insights

---

### Phase 2: Feature Engineering (Week 2)

#### Text Features
- [ ] **Task 2.1**: Basic Text Features
  - Text length, word count, character count
  - Number of uppercase words (brands often capitalized)
  - Punctuation density
  
- [ ] **Task 2.2**: Extract Structured Information
  - Item Pack Quantity (IPQ) extraction using regex
  - Brand name extraction
  - Product category/type keywords
  - Measurement units (oz, kg, ml, etc.)
  
- [ ] **Task 2.3**: Advanced Text Embeddings
  - TF-IDF vectorization (baseline)
  - Word2Vec/GloVe embeddings
  - Transformer embeddings (sentence-transformers)
  - Model options: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

#### Image Features
- [ ] **Task 2.4**: Download Images
  - Use `download_images()` from `src/utils.py`
  - Implement retry logic for failed downloads
  - Store images organized by train/test
  
- [ ] **Task 2.5**: Image Preprocessing
  - Resize to standard dimensions (224x224 or 384x384)
  - Normalize pixel values
  - Handle missing images (use default/average features)
  
- [ ] **Task 2.6**: Feature Extraction from Images
  - Use pre-trained CNN models:
    - ResNet-50/101
    - EfficientNet-B0/B3
    - MobileNetV2 (faster)
  - Extract features from penultimate layer
  - Dimensionality reduction (PCA if needed)

**Deliverable**: Feature engineering pipeline

---

### Phase 3: Model Development (Week 3)

#### Baseline Models
- [ ] **Task 3.1**: Text-Only Baseline
  - Linear Regression with TF-IDF
  - Random Forest with basic text features
  - XGBoost with text features
  
- [ ] **Task 3.2**: Image-Only Baseline
  - Random Forest with image features
  - XGBoost with image features
  
- [ ] **Task 3.3**: Combined Features Baseline
  - Concatenate text + image features
  - Train tree-based models

#### Advanced Models
- [ ] **Task 3.4**: Gradient Boosting Models
  - XGBoost (hyperparameter tuning)
  - LightGBM (faster training)
  - CatBoost (handles categorical features)
  
- [ ] **Task 3.5**: Neural Network Approaches
  - Multi-input neural network (text + image branches)
  - Fine-tune pre-trained models (if within 8B param limit)
  
- [ ] **Task 3.6**: Ensemble Methods
  - Weighted average of top models
  - Stacking ensemble
  - Blending different model types

**Deliverable**: Trained models with validation SMAPE scores

---

### Phase 4: Model Optimization (Week 4)

- [ ] **Task 4.1**: Cross-Validation
  - Implement 5-fold cross-validation
  - Calculate mean and std of SMAPE
  
- [ ] **Task 4.2**: Hyperparameter Tuning
  - Grid Search / Random Search
  - Bayesian Optimization (Optuna)
  - Focus on SMAPE optimization
  
- [ ] **Task 4.3**: Feature Selection
  - Remove low-importance features
  - Test feature combinations
  
- [ ] **Task 4.4**: Error Analysis
  - Identify samples with high prediction error
  - Analyze patterns in misclassifications
  - Adjust model or features accordingly

**Deliverable**: Optimized model with best SMAPE score

---

### Phase 5: Final Prediction & Submission (Week 5)

- [ ] **Task 5.1**: Generate Test Predictions
  - Load test data from `dataset/test.csv`
  - Apply same preprocessing pipeline
  - Generate predictions using best model
  
- [ ] **Task 5.2**: Post-Processing
  - Ensure all predictions are positive floats
  - Handle any edge cases (NaN, infinity)
  - Clip extreme values if necessary
  
- [ ] **Task 5.3**: Format Submission File
  - Create CSV with columns: `sample_id`, `price`
  - Match format of `dataset/sample_test_out.csv`
  - Verify 75,000 predictions (no missing IDs)
  
- [ ] **Task 5.4**: Documentation
  - Fill out `Documentation_template.md`
  - Describe methodology, models, features
  - Include performance metrics

**Deliverable**: `test_out.csv` + Documentation

---

## 🛠️ Technical Implementation Details

### Required Libraries
```python
# Data Processing
pandas
numpy
scikit-learn

# Text Processing
nltk
spacy
sentence-transformers
transformers

# Image Processing
PIL / Pillow
opencv-python
torchvision

# Machine Learning Models
xgboost
lightgbm
catboost
torch (PyTorch)

# Visualization
matplotlib
seaborn
plotly

# Utilities
tqdm
requests
```

### Model Constraints
- **License**: MIT or Apache 2.0 only
- **Model Size**: Maximum 8 Billion parameters
- **No External Data**: Cannot lookup prices from internet

### Recommended Models (Within Constraints)

#### Text Models (< 8B params)
- ✅ `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- ✅ `sentence-transformers/all-mpnet-base-v2` (110M params)
- ✅ `microsoft/deberta-v3-base` (184M params)
- ✅ `microsoft/mpnet-base` (110M params)
- ⚠️ `meta-llama/Llama-2-7b` (7B params) - check license
- ⚠️ `mistralai/Mistral-7B-v0.1` (7B params) - Apache 2.0

#### Image Models (< 8B params)
- ✅ ResNet-50/101/152 (25-60M params)
- ✅ EfficientNet-B0 to B7 (5-66M params)
- ✅ MobileNetV2/V3 (3-5M params)
- ✅ ViT-Base (86M params)
- ✅ ConvNeXt-Base (89M params)

---

## 📁 Project Structure

```
student_resource/
│
├── dataset/
│   ├── train.csv                    # Training data with prices
│   ├── test.csv                     # Test data (predict prices)
│   ├── sample_test.csv              # Sample test input
│   ├── sample_test_out.csv          # Sample output format
│   ├── train_images/                # Downloaded training images
│   └── test_images/                 # Downloaded test images
│
├── src/
│   ├── utils.py                     # Helper functions (download_images)
│   ├── example.ipynb                # Example notebook
│   ├── 01_eda.ipynb                 # Data exploration (Phase 1)
│   ├── 02_feature_engineering.ipynb # Feature creation (Phase 2)
│   ├── 03_modeling.ipynb            # Model training (Phase 3)
│   ├── 04_prediction.ipynb          # Final predictions (Phase 5)
│   ├── feature_extractor.py         # Feature extraction module
│   ├── text_processor.py            # Text processing functions
│   ├── image_processor.py           # Image processing functions
│   └── model_trainer.py             # Model training pipeline
│
├── models/                          # Saved trained models
│   ├── best_model.pkl
│   └── ensemble_models/
│
├── submissions/                     # Generated submission files
│   └── test_out.csv
│
├── Documentation_template.md        # Documentation template
├── PROJECT_PLAN.md                  # This file
├── README.md                        # Project README
└── sample_code.py                   # Sample dummy code
```

---

## 🎓 Key Strategies for Success

### 1. **Multi-Modal Learning**
- Combine text and image features effectively
- Test different fusion strategies (early/late fusion)

### 2. **Robust Feature Engineering**
- Extract domain-specific features (brands, pack quantities)
- Create interaction features (e.g., brand × category)

### 3. **Ensemble Approach**
- Combine predictions from multiple models
- Weight models based on validation performance

### 4. **Handle Outliers**
- Log-transform prices for training
- Clip extreme predictions
- Use robust metrics during validation

### 5. **Cross-Validation**
- Use stratified K-fold based on price ranges
- Ensure consistent SMAPE across folds

### 6. **Error Analysis**
- Focus on high-error samples
- Identify patterns (e.g., specific categories hard to predict)
- Iterate on features/models

---

## ⚠️ Important Constraints & Rules

### 🚨 CRITICAL DISCLAIMER - Amazon ML Challenge
**As part of the Amazon ML Challenge hackathon event, you are STRICTLY PROHIBITED from using publicly or commercially available large language model (LLM) APIs such as:**
- ❌ OpenAI APIs (GPT-3, GPT-4, ChatGPT, etc.)
- ❌ Anthropic APIs (Claude, etc.)
- ❌ Microsoft APIs (Azure OpenAI, etc.)
- ❌ Facebook/Meta APIs (Llama API services, etc.)
- ❌ Google APIs (PaLM, Gemini, Bard, etc.)
- ❌ Any other commercial AI company APIs

**⚠️ Submissions using ANY LLM APIs will be DISCARDED.**

**Allowed Alternative**: You may use **locally hosted open-source models** (e.g., download and run Llama-2, Mistral locally) as long as they meet the license and parameter constraints.

---

### ✅ ALLOWED
- Use provided training data only
- Use **locally hosted** pre-trained models (MIT/Apache 2.0 license, <8B params)
- Download and run open-source models locally (no API calls)
- Feature engineering from text and images
- Ensemble multiple models
- Data augmentation (text paraphrasing, image transformations)
- Traditional ML models (XGBoost, LightGBM, Random Forest, etc.)
- Pre-trained embeddings (word2vec, GloVe, sentence-transformers)
- Pre-trained vision models (ResNet, EfficientNet, ViT)

### ❌ PROHIBITED
- **ANY LLM API calls** (OpenAI, Anthropic, Microsoft, Google, Meta, etc.)
- **Web scraping for prices**
- **Using external price databases**
- **API calls to fetch current market prices**
- **Manual price lookup from online sources**
- **Models > 8B parameters**
- **Non-MIT/Apache 2.0 licensed models**

### Violation = Immediate Disqualification & Submission Discarded

---

## 📈 Success Metrics

### Target SMAPE Scores (Estimated)
- **Baseline (text-only)**: 35-45%
- **Good (text + basic ML)**: 25-35%
- **Very Good (text + images + ensemble)**: 15-25%
- **Excellent (optimized multi-modal)**: <15%

### Validation Strategy
1. Split training data: 80% train, 20% validation
2. Use 5-fold cross-validation for final model
3. Track both validation SMAPE and overfitting

---

## 📝 Submission Checklist

- [ ] `test_out.csv` with exactly 75,000 predictions
- [ ] All `sample_id` values match `test.csv`
- [ ] All prices are positive float values
- [ ] File format matches `sample_test_out.csv` exactly
- [ ] Documentation completed using `Documentation_template.md`
- [ ] Methodology clearly described
- [ ] Model architecture documented
- [ ] Feature engineering techniques explained
- [ ] No external price lookups used

---

## 🔄 Iteration Plan

### Week 1: Baseline
- Simple text features + Linear/RF model
- **Target**: Establish baseline SMAPE

### Week 2: Improvement v1
- Add image features
- Try XGBoost/LightGBM
- **Target**: 10% SMAPE improvement

### Week 3: Improvement v2
- Advanced text embeddings
- Ensemble models
- **Target**: Another 10% improvement

### Week 4: Optimization
- Hyperparameter tuning
- Feature selection
- **Target**: Fine-tune for best performance

### Week 5: Final
- Generate submission
- Complete documentation
- **Target**: Submit best solution

---

## 🤝 Collaboration & Version Control

### Git Workflow (if using)
1. Create branches for each phase
2. Commit frequently with clear messages
3. Tag model versions
4. Document experiments in notebooks

### Experiment Tracking
- Log all model configurations
- Track validation SMAPE for each experiment
- Keep notes on what works/doesn't work

---

## 📚 Resources & References

### Helpful Libraries
- **Sentence Transformers**: https://www.sbert.net/
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **PyTorch Vision Models**: https://pytorch.org/vision/stable/models.html

### Similar Competitions
- Kaggle: Mercari Price Suggestion Challenge
- Kaggle: Home Depot Product Search Relevance

### Papers
- Multi-modal learning for e-commerce
- Price prediction using deep learning
- BERT for text regression tasks

---

## 🎯 Next Steps

1. **Start with Phase 1**: Run data exploration in Jupyter notebook
2. **Quick baseline**: Train simple model to understand data
3. **Iterate rapidly**: Test ideas quickly, keep what works
4. **Document everything**: Track experiments and results
5. **Focus on SMAPE**: Optimize specifically for this metric

---

## 📞 Notes & Questions

### Questions to Answer During EDA
- What's the price range? (min, max, median)
- Are there price outliers? How many?
- What's the typical catalog_content structure?
- How many images fail to download?
- Are there any patterns in expensive vs cheap products?

### Potential Challenges
- Image download throttling (retry logic needed)
- Large dataset size (75K samples)
- Multi-modal fusion complexity
- SMAPE optimization (different from MSE/MAE)

---

**Last Updated**: October 11, 2025  
**Challenge Deadline**: [Add deadline when known]  
**Team Members**: [Add team members]

---

*Good luck! 🚀 Remember: Start simple, iterate quickly, and focus on SMAPE optimization!*
