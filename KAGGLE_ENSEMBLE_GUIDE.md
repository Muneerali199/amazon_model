# 🚀 KAGGLE ENSEMBLE MODEL - SETUP GUIDE

## Expected Results
- **Current Best**: 57.900% (Phase 5)
- **Expected CV**: 52-56% SMAPE
- **Expected Improvement**: 2-6 percentage points
- **Runtime**: 6-8 hours on Kaggle GPU
- **Expected Rank**: 200-600 (from current 1000)

---

## 📋 STEP 1: PREPARE YOUR DATASET

### Option A: Use Existing Dataset (if you already created one)
1. Go to: https://www.kaggle.com/datasets
2. Find your dataset (the one you used for CLIP)
3. Note the dataset name (e.g., `yourusername/amazon-ml-dataset`)

### Option B: Create New Dataset
1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload these files from your computer:
   - `sample_train.csv` (75,000 rows)
   - `sample_test.csv` (75,000 rows)
4. Name it: `amazon-ml-challenge`
5. Make it **Public**
6. Click **"Create"**
7. Note the full path: `yourusername/amazon-ml-challenge`

---

## 📝 STEP 2: CREATE NEW KAGGLE NOTEBOOK

1. **Go to Kaggle Notebooks**:
   - Visit: https://www.kaggle.com/code
   - Click **"New Notebook"**

2. **Configure Notebook Settings**:
   - Click **"File"** → **"Notebook Settings"** (or ⚙️ icon top-right)
   - **Accelerator**: Select **"GPU T4 x2"** or **"GPU P100"**
   - **Internet**: Turn **ON** (required for downloading images)
   - Click **"Save"**

3. **Add Your Dataset**:
   - Click **"+ Add Data"** (right sidebar)
   - Search for your dataset name
   - Click **"Add"** next to your dataset
   - Verify it appears under "Input" section

---

## 💻 STEP 3: COPY CODE TO NOTEBOOK

1. **Open the file**: `kaggle_ensemble_advanced.py` (on your computer)

2. **Copy ALL the code** (Ctrl+A, Ctrl+C)

3. **In Kaggle Notebook**:
   - Delete the default cell
   - Click **"+ Code"** to add new code cell
   - Paste all the code (Ctrl+V)

4. **IMPORTANT: Update Dataset Path**:
   - Find line 66-67:
     ```python
     train_df = pd.read_csv('/kaggle/input/your-dataset-name/sample_train.csv')
     test_df = pd.read_csv('/kaggle/input/your-dataset-name/sample_test.csv')
     ```
   - Replace `your-dataset-name` with your actual dataset name
   - Example:
     ```python
     train_df = pd.read_csv('/kaggle/input/amazon-ml-challenge/sample_train.csv')
     test_df = pd.read_csv('/kaggle/input/amazon-ml-challenge/sample_test.csv')
     ```

---

## ▶️ STEP 4: RUN THE NOTEBOOK

1. **Save the notebook**:
   - Click **"Save Version"** (top-right)
   - Select **"Save & Run All"**
   - Add comment: "Advanced ensemble with ResNet50 + BERT + XGBoost/LightGBM/CatBoost"
   - Click **"Save"**

2. **Monitor Progress**:
   - Notebook will start running automatically
   - You'll see 8 phases executing:
     - [1/8] Installing packages (~3 minutes)
     - [2/8] Loading data (~30 seconds)
     - [3/8] ResNet50 features (~2 hours)
     - [4/8] BERT features (~1.5 hours)
     - [5/8] Text features (~1 minute)
     - [6/8] Combining features (~30 seconds)
     - [7/8] Training models (~2.5 hours)
     - [8/8] Creating submissions (~1 minute)

3. **Total Runtime**: ~6-8 hours

---

## 📊 STEP 5: INTERPRET RESULTS

### While Running:
Check the output for each fold:
```
Fold 1/5
  Training xgboost...
    Fold 1 SMAPE: 54.2345%
  Training lightgbm...
    Fold 1 SMAPE: 53.8912%
  Training catboost...
    Fold 1 SMAPE: 54.5621%
```

### Final Results (After Completion):
Look for the **"ENSEMBLE RESULTS"** section:
```
ENSEMBLE RESULTS
================================================================================
Simple Average Ensemble OOF: 53.4567%
Expected Leaderboard: 52.38% - 54.53%
```

### Score Interpretation:

| CV Score | What It Means | Action |
|----------|---------------|--------|
| **< 52%** | 🔥 AMAZING! Beat Phase 5 by 5%+ | Submit immediately! Expected Top 300 |
| **52-54%** | ⭐ EXCELLENT! Beat Phase 5 by 3-5% | Submit with confidence! Expected Top 500 |
| **54-56%** | ✅ GOOD! Beat Phase 5 by 1-3% | Worth submitting! Expected Top 700 |
| **56-58%** | ⚠️ MODEST! Similar to Phase 5 | Consider submitting |
| **> 58%** | ❌ WORSE than Phase 5 | Use Phase 5 backup (57.900%) |

---

## 📥 STEP 6: DOWNLOAD SUBMISSIONS

1. **Wait for notebook to finish** (status shows "100% Complete")

2. **Download the CSV files**:
   - Click **"Output"** section (right sidebar)
   - You'll see two files:
     - `submission_ensemble_simple.csv` ← **Use this one**
     - `submission_ensemble_weighted.csv` ← Alternative
   - Click ⬇️ to download `submission_ensemble_simple.csv`

3. **Verify the file**:
   - Open in Excel or text editor
   - Check: 75,001 rows (1 header + 75,000 data)
   - Columns: `sample_id`, `price`
   - Prices should be reasonable (not negative, not extreme)

---

## 🏆 STEP 7: SUBMIT TO COMPETITION

1. **Go to Competition Page**:
   - Visit: https://www.kaggle.com/competitions/ml-challenge-2025

2. **Submit Predictions**:
   - Click **"Submit Predictions"** button
   - Upload `submission_ensemble_simple.csv`
   - Description: "Advanced Ensemble: ResNet50 + BERT + XGBoost/LightGBM/CatBoost"
   - Click **"Make Submission"**

3. **Wait for Scoring** (2-5 minutes)

4. **Check Your Score**:
   - Look at the leaderboard
   - Compare with your Phase 5: **57.900%**
   - **If better**: Celebrate! 🎉
   - **If worse**: You still have Phase 5 as backup

---

## 🔧 TROUBLESHOOTING

### Problem: "Dataset not found"
**Solution**: Update the file paths in lines 66-67 with your exact dataset name

### Problem: "Out of memory"
**Solution**: 
- Make sure you selected GPU accelerator
- Restart the notebook
- Try GPU T4 x2 instead of P100 (or vice versa)

### Problem: "Image download errors"
**Solution**: This is normal! The code handles failed downloads by using blank images. Check the success rate in output.

### Problem: Notebook taking too long (> 10 hours)
**Solution**: 
- Normal for 75,000 images
- Be patient, feature extraction takes time
- Check if internet is enabled in settings

### Problem: CV Score worse than Phase 5 (> 58%)
**Solution**: 
- Don't worry! This can happen
- Use your Phase 5 backup: `submission_xgboost_phase5.csv` at 57.900%
- That's still a solid score

---

## 📈 WHAT THIS MODEL DOES DIFFERENTLY

**vs Your Phase 5 (57.900%)**:
- ✅ Adds **ResNet50 image features** (2048-dim) - captures visual patterns
- ✅ Adds **BERT text features** (768-dim) - better text understanding
- ✅ Uses **3 different models** (XGBoost, LightGBM, CatBoost) - ensemble reduces variance
- ✅ Uses **GPU acceleration** - trains faster and better
- ✅ Uses **5-fold CV** - more reliable score estimation

**vs Your CLIP Notebook**:
- ✅ Separates vision and text - more features
- ✅ Adds traditional ML models - proven effective
- ✅ Uses ensemble - combines strengths of multiple models
- ✅ More robust - less dependent on single model

---

## ⏱️ TIMELINE

| Time | Phase | What's Happening |
|------|-------|------------------|
| **0:00** | Start | Installing packages |
| **0:05** | Setup | Loading data |
| **0:05-2:05** | Images | ResNet50 extracting features from 150,000 images |
| **2:05-3:35** | Text | BERT processing 150,000 product descriptions |
| **3:35-3:40** | Features | Creating traditional text features |
| **3:40-6:10** | Training | Training 3 models × 5 folds = 15 model fits |
| **6:10-6:15** | Ensemble | Combining predictions |
| **6:15** | Done! | Download submissions |

---

## 🎯 SUCCESS CRITERIA

### Minimum Success (Worth submitting):
- CV < 57.9% (beats Phase 5)
- Expected rank improvement: 1000 → 700

### Good Success:
- CV < 56% 
- Expected rank: 500-700

### Excellent Success:
- CV < 54%
- Expected rank: 300-500

### Amazing Success:
- CV < 52%
- Expected rank: 200-300

---

## 💡 PRO TIPS

1. **Run while you sleep**: This takes 6-8 hours, start it before bed

2. **Keep track**: Take screenshots of the CV scores from each fold

3. **Download immediately**: Once done, download the CSV right away (Kaggle can be glitchy)

4. **Submit both versions**: Try both `simple` and `weighted` submissions to see which is better

5. **Don't panic if slow**: Feature extraction from 150,000 images takes time

6. **Check progress**: Kaggle shows "elapsed time" - if it's progressing, it's working

7. **Have backup**: Keep your Phase 5 submission ready (57.900%)

---

## ❓ FAQ

**Q: Can I close my browser while it runs?**
A: YES! Kaggle runs in the cloud. Close browser, come back later.

**Q: How do I know if it's still running?**
A: Check the notebook - if status is "Running" and time is increasing, it's working.

**Q: What if it fails midway?**
A: Click "Save Version" → "Save & Run All" again to restart.

**Q: Should I use Simple or Weighted submission?**
A: Start with Simple. If that works, try Weighted too.

**Q: What if both submissions are worse than Phase 5?**
A: Submit your Phase 5 (`submission_xgboost_phase5.csv`) - it's proven at 57.900%.

---

## 🚨 IMPORTANT REMINDERS

1. ✅ **GPU Must Be ON** - Check settings before running
2. ✅ **Internet Must Be ON** - Required for image downloads
3. ✅ **Update Dataset Path** - Lines 66-67 with your dataset name
4. ✅ **Wait for Completion** - Don't stop midway, let it finish
5. ✅ **Download CSV** - Get submission file when done
6. ✅ **Keep Phase 5 Backup** - In case this doesn't beat 57.900%

---

## 📞 NEXT STEPS AFTER RUNNING

1. **Share your results**: Tell me the CV score when done
2. **Compare with Phase 5**: Did it beat 57.900%?
3. **Submit to leaderboard**: Upload the CSV
4. **Report back**: What was your LB score?
5. **Decide next move**: Based on improvement, plan Phase 11

---

**Good luck! Let this run while your CLIP notebook finishes, then you'll have 2 strong submissions! 🚀**
