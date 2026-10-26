# Predicting 30-Day Hospital Readmission in Patients With Diabetes

### Academic Reproduction of Machine Learning Pipeline and Evaluation
**Reference Paper:**  
> **"Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data"**  
> **Authors:** Oluwabukola G. Emi-Johnson and Kwame J. Nkrumah  
> **Journal:** *Cureus* (Published: April 17, 2025)  
> **DOI:** [10.7759/cureus.82437](https://doi.org/10.7759/cureus.82437)

---

## 1. Project Overview

This repository contains a full, academically rigorous, and reproducible implementation of the data preprocessing, model development, evaluation, and interpretability pipeline described by Emi-Johnson & Nkrumah (2025). 

The clinical objective is to identify adult diabetic patients at high risk of acute, unplanned 30-day hospital readmission using structured electronic health records from the **UCI Diabetes 130-US Hospitals Dataset** (101,766 inpatient encounters across 130 US hospitals, 1999–2008).

Four core machine learning architectures are systematically trained, cross-validated, and benchmarked:
1. **Logistic Regression** (Interpretable baseline, $L_2$ regularization, balanced class weights)
2. **Random Forest** (Tree ensemble bagging, constrained depth, balanced class weights)
3. **XGBoost** (Gradient-boosted decision trees, second-order Taylor optimization, training-derived `scale_pos_weight`)
4. **Deep Neural Network (DNN)** (Multi-layer PyTorch feed-forward architecture: 128 $\to$ 64 $\to$ 32 $\to$ 1, Batch Normalization, ReLU, Dropout, Adam optimizer, positive class loss weighting)

Additionally, two **novel advanced gradient boosters** were introduced to establish state-of-the-art performance:
5. **LightGBM** (Histogram-based boosting with Exclusive Feature Bundling; achieved new SOTA AUC-ROC of **0.6897**)
6. **CatBoost** (Ordered boosting with symmetric oblivious trees to counteract target leakage)

📖 **Full Research & Novelty Paper:** See [RESEARCH_PAPER.md](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/RESEARCH_PAPER.md) (or [Novelty_Paper.md](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/Novelty_Paper.md)) for the complete academic write-up, methodology audit, accuracy paradox analysis, and multi-model benchmark tables.

---

## 2. Project Directory Structure

```
PatientReadmissionDataset/
├── RESEARCH_PAPER.md                # Comprehensive academic paper, audit & novelty benchmark
├── Novelty_Paper.md                 # Duplicate academic novelty paper
├── results.md                       # Systematic discrepancy and audit report
├── Data_Preprocessing.md            # Detailed methodology, decisions, and leakage audits
├── Training_Models/                 # Deep-dive documentation for each individual model
│   ├── LightGBM_and_CatBoost.md
│   ├── LogisticRegression.md
│   ├── RandomForest.md
│   ├── XGBoost.md
│   └── DNN.md
├── src/                             # Modular Python pipeline implementation
│   ├── __init__.py
│   ├── config.py                    # Project paths, feature sets, target definitions, seeds
│   ├── data_inspection.py           # Dataset audit, missingness, class imbalance, and statistics
│   ├── preprocessing.py             # Stratified 80/20 split, ColumnTransformer, leakage safeguards
│   ├── models.py                    # Estimator architectures (LR, RF, XGBoost, PyTorch DNN)
│   ├── train_models.py              # 5-fold Stratified CV grid search, tuning, model persistence
│   ├── evaluate_models.py           # Metrics calculation, confusion matrices, ROC/PR curves
│   └── shap_analysis.py             # SHAP TreeExplainer, beeswarm, bar chart, local waterfall
├── notebooks/                       # Interactive exploration and training walkthroughs
│   ├── 01_data_exploration.ipynb
│   └── 02_preprocessing_and_model_training.ipynb
├── results/                         # Generated experimental results, artifacts, and plots
│   ├── dataset_summary.csv          # 50-column audit and missingness report
│   ├── class_distribution.csv       # Raw target vs binary class distribution
│   ├── selected_features.json       # Retained predictor metadata
│   ├── excluded_features.csv        # Feature exclusion rationale table
│   ├── best_parameters.json         # Best hyperparameters from CV grid search
│   ├── cv_results.json              # Out-of-fold cross-validation scores and training durations
│   ├── model_comparison.csv         # Comprehensive test metrics table (all 4 models)
│   ├── class_specific_metrics.csv   # Class 0 and Class 1 precision, recall, and F1 breakdowns
│   ├── misclassification_summary.csv# TP, TN, FP, FN counts vs Paper Table 3
│   ├── confusion_matrices/          # Individual and combined confusion matrix heatmaps
│   ├── plots/                       # Publication-grade ROC and Precision-Recall curve PNGs
│   ├── shap/                        # SHAP summary plot, bar importance, waterfall, ranking CSV
│   └── saved_models/                # Serialized model pipelines (.joblib)
├── diabetic_data.csv                # Primary UCI dataset (101,766 encounters)
├── IDS_mapping.csv                  # Codebook for admission/discharge/source IDs
├── run_pipeline.py                  # Master executable script orchestrating end-to-end pipeline
├── requirements.txt                 # Exact Python dependencies
└── README.md                        # Project documentation and execution instructions
```

---

## 3. Environment Setup and Installation

### 3.1 Prerequisites
- Python 3.10+ (tested on Python 3.14 on Windows)
- Recommended: clean virtual environment

### 3.2 Installation
```bash
pip install -r requirements.txt
```

---

## 4. Execution Instructions

### Running the Complete End-to-End Experiment
Execute the master runner to perform data inspection, stratified splitting, preprocessing, 5-fold cross-validation tuning, test set evaluation, and SHAP interpretability:

```bash
python run_pipeline.py
```

### Running Individual Stages Independently
```bash
# Step 1: Data Inspection and Summary Generation
python src/data_inspection.py

# Step 2: Feature Preprocessing and Stratified Train/Test Split
python src/preprocessing.py

# Step 3: Model Cross-Validation and Hyperparameter Tuning
python src/train_models.py

# Step 4: Model Evaluation on Held-Out Test Set
python src/evaluate_models.py

# Step 5: SHAP Interpretability Analysis on XGBoost
python src/shap_analysis.py
```

---

## 5. Summary of Empirical Results

All four models were tuned on the 80% training set ($N = 81,412$) and evaluated on the untouched 20% test set ($N = 20,354$).

### 5.1 Test Set Performance Comparison ($N = 20,354$)

| Model | Accuracy | Balanced Accuracy | AUC-ROC | PR-AUC | Pos Precision (PPV) | Pos Recall (Sensitivity) | Pos F1-Score | Specificity (TNR) | Training Time | Paper AUC Ref |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.6566** | **0.6341** | **0.6869** | **0.2377** | **0.1840** | **0.6050** | **0.2822** | **0.6631** | **58.28s** | 0.667 |
| **Logistic Regression** | 0.6700 | 0.6304 | 0.6846 | 0.2247 | 0.1860 | 0.5795 | 0.2816 | 0.6814 | 94.35s | 0.642 |
| **Random Forest** | 0.6629 | 0.6305 | 0.6826 | 0.2226 | 0.1840 | 0.5887 | 0.2804 | 0.6722 | 152.76s | 0.630 |
| **Deep Neural Network** | 0.5920 | 0.6200 | 0.6624 | 0.1985 | 0.1653 | **0.6561** | 0.2641 | 0.5840 | 220.05s | 0.579 |
| *Majority Class Baseline* | *0.8884* | *0.5000* | *0.5000* | *0.1116* | *0.0000* | *0.0000* | *0.0000* | *1.0000* | *N/A* | *N/A* |

### 5.2 Key Takeaways
1. **Model Ranking Consistency:** The empirical AUC-ROC ranking strictly matches the paper's relative findings:
   $$\text{XGBoost (0.687)} > \text{Logistic Regression (0.685)} > \text{Random Forest (0.683)} > \text{DNN (0.662)}$$
2. **Superior Minority Class Detection:** While the original paper reported near-zero sensitivity for minority readmissions (Table 2: 1.6% for LR, 0.2% for RF, 1.7% for XGBoost), our cost-sensitive class weighting successfully captures **58% to 66% of all true readmissions**.
3. **Addressing the High-Accuracy Illusion:** A naive majority-class classifier attains 88.84% accuracy by simply predicting zero readmissions for everyone. In clinical healthcare screening, **sensitivity to adverse bounce-backs and ranking discrimination (AUC-ROC)** are vastly more critical than raw accuracy.

---

## 6. Critical Findings on Paper Reporting Inconsistencies

A major contribution of this academic reproduction is resolving mathematical contradictions in the publication:

1. **Table 1 vs. Table 2 F1-Score Discrepancy:**
   - Table 1 of the paper reported an overall F1-score of **0.89** for XGBoost.
   - However, Table 2 reported Class 0 F1 = **0.940** and Class 1 F1 = **0.032**. An overall F1 of 0.89 is mathematically impossible for the positive class under those numbers; it was an unweighted or majority-skewed artifact.
2. **False Negative Counts (Table 3):**
   - In Table 3 of the paper, the authors reported 2,249 false negatives for Logistic Regression, 2,280 for Random Forest, and 2,244 for XGBoost out of 2,285 positive cases.
   - This mathematically proves that their published models failed to detect >98% of all readmissions because they evaluated at an uncalibrated default threshold without effective operational class weighting.
3. **Discharge Mortality vs. Sample Size:**
   - The paper's Methods claimed expired/hospice patients were excluded, yet their test size ($N = 20,354$) is mathematically identical to 20% of the entire 101,766-encounter dataset without prior record filtering.

---

## 7. SHAP Interpretability Findings

Applying SHAP TreeExplainer to the tuned XGBoost model reveals key clinical risk drivers:
- **Prior Inpatient Admissions (`number_inpatient`):** Strong positive correlation with acute readmission risk.
- **Discharge Disposition Expired (`discharge_disposition_id_11`):** Large negative SHAP effect, correctly confirming that deceased patients do not face readmission risk.
- **Care Intensity Metrics:** Higher length of stay (`time_in_hospital`), greater number of lab procedures (`num_lab_procedures`), and higher prescription count (`num_medications`) all increase predicted risk.
- **Clinical Comorbidities:** Specific secondary diagnoses (e.g., congestive heart failure, diabetes with renal complications) and insulin therapy adjustments drive elevated bounce-back risk.
