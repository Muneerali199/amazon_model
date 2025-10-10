# 🚀 Quick Start Guide - Phase 1
## ML Challenge 2025 - Smart Product Pricing

---

## ✅ What We've Created

1. **`PROJECT_PLAN.md`** - Complete 5-phase project plan with detailed tasks
2. **`PHASE1_PROGRESS.md`** - Phase 1 checklist and progress tracker  
3. **`src/01_eda.ipynb`** - Comprehensive EDA notebook ready to run
4. **`requirements.txt`** - All required Python libraries

---

## 🎯 Next Steps - Get Started with Phase 1

### Step 1: Install Required Libraries

Open PowerShell terminal and run:

```powershell
pip install pandas numpy matplotlib seaborn
```

**Note**: Don't install all libraries at once. Start with basics for EDA, then add more as needed.

### Step 2: Open the EDA Notebook

1. Navigate to: `src/01_eda.ipynb`
2. Open it in VS Code or Jupyter
3. Run cells one by one from top to bottom

### Step 3: Explore Your Data

The notebook will help you:
- ✅ Load and inspect the training data (75K samples)
- ✅ Analyze price distribution and outliers
- ✅ Explore text features (catalog_content)
- ✅ Check image link availability
- ✅ Compare train vs test data

### Step 4: Document Findings

As you run the notebook, fill in the metrics table in `PHASE1_PROGRESS.md`:
- Dataset sizes
- Price statistics
- Text characteristics
- Image availability

---

## 📊 What You'll Learn

### About the Data

The dataset has:
- **75,000 training samples** with prices
- **75,000 test samples** (need to predict prices)
- **3 columns** in training data:
  - `sample_id` - Unique identifier
  - `catalog_content` - Product title, description, Item Pack Quantity
  - `image_link` - URL to product image
  - `price` - Target variable (only in training data)

### Sample Data Structure

From `sample_test.csv`, here's what the data looks like:

```
sample_id: 217392
catalog_content: "Item Name: Gift Basket Village Gourmet Meat and Cheese...
                  Product Description: Elevate your gifting experience...
                  Value: 1.0
                  Unit: Count"
image_link: https://m.media-amazon.com/images/I/91GB1wC6ObL.jpg
price: $62.08 (in training data only)
```

---

## 🔍 Key Things to Look For

### 1. Price Distribution
- Is the price distribution skewed?
- Are there outliers (very cheap or expensive products)?
- Do you need log transformation?

### 2. Text Patterns
- How long is typical catalog_content?
- Can you extract Item Pack Quantity (IPQ)?
- Are there brand names or categories mentioned?

### 3. Image Availability
- Do all products have images?
- Are image URLs accessible?
- Do products with images have different prices?

### 4. Train/Test Similarity
- Are train and test distributions similar?
- Any data leakage concerns?

---

## ⚠️ Important Constraints (Remember!)

### 🚨 STRICTLY PROHIBITED:
- ❌ No OpenAI, Anthropic, Microsoft, Google, Meta APIs
- ❌ No external price lookups from websites
- ❌ No web scraping for prices

### ✅ ALLOWED:
- ✅ Locally hosted open-source models
- ✅ Traditional ML (XGBoost, LightGBM, Random Forest)
- ✅ Pre-trained embeddings (sentence-transformers, running locally)
- ✅ Vision models (ResNet, EfficientNet, running locally)

---

## 📝 Tips for Success

### For EDA:
1. **Start Small**: Load a subset of data first (e.g., 1000 samples) to test
2. **Visualize Everything**: Charts help you spot patterns
3. **Document Surprises**: Note anything unexpected
4. **Think About Features**: While exploring, think what features you'll create

### For This Challenge:
1. **Text is Important**: Catalog content contains rich information
2. **Images Add Value**: Products with images may have different pricing
3. **IPQ Matters**: Item Pack Quantity affects price significantly
4. **Brands Matter**: Brand names influence pricing

---

## 🐛 Common Issues & Solutions

### Issue 1: Dataset too large
**Problem**: 75K rows might be slow to load or plot

**Solution**:
```python
# Load only first 10,000 rows for testing
train_df = pd.read_csv('dataset/train.csv', nrows=10000)

# Or sample random 10,000 rows
train_df = pd.read_csv('dataset/train.csv')
train_df = train_df.sample(10000, random_state=42)
```

### Issue 2: Missing libraries
**Problem**: ImportError when running notebook

**Solution**:
```powershell
pip install pandas numpy matplotlib seaborn
```

### Issue 3: Plots not showing
**Problem**: Visualizations don't appear

**Solution**: Make sure you have this in the notebook:
```python
%matplotlib inline
```

---

## 📋 Phase 1 Completion Checklist

Mark these off as you complete them:

- [ ] Installed required libraries
- [ ] Opened `src/01_eda.ipynb` notebook
- [ ] Loaded training data successfully
- [ ] Analyzed price distribution (mean, median, outliers)
- [ ] Explored text features (length, word count)
- [ ] Checked image link availability
- [ ] Loaded and compared test data
- [ ] Documented key findings in `PHASE1_PROGRESS.md`
- [ ] Identified patterns for feature engineering
- [ ] Ready to move to Phase 2

---

## 🎓 After Phase 1

Once you complete EDA, you'll move to **Phase 2: Feature Engineering**, where you'll:
1. Extract Item Pack Quantity (IPQ) from text
2. Create text embeddings
3. Download and process images
4. Extract image features using pre-trained models
5. Prepare dataset for modeling

---

## 📚 Useful Resources

### Documentation:
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/contents.html
- Seaborn: https://seaborn.pydata.org/

### For Later Phases:
- sentence-transformers: https://www.sbert.net/
- XGBoost: https://xgboost.readthedocs.io/
- PyTorch Vision: https://pytorch.org/vision/stable/models.html

---

## 💡 Pro Tips

1. **Run Cells Incrementally**: Don't run all cells at once. Run one, check output, then next.
2. **Save Insights**: Keep a separate notes file for observations
3. **Think Ahead**: While exploring, think about what features to create
4. **Ask Questions**: If something seems odd, investigate deeper
5. **Have Fun**: Data exploration can be fun! Look for interesting patterns

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify file paths are correct
3. Make sure libraries are installed
4. Try with smaller data sample first

---

## ✨ You're Ready!

You have everything you need to start Phase 1. Open `src/01_eda.ipynb` and begin your data exploration journey!

**Good luck! 🚀**

---

**Created**: October 11, 2025  
**Phase**: 1 of 5  
**Status**: Ready to Execute
