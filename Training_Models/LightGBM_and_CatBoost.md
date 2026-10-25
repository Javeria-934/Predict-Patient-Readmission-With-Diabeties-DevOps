# Advanced Gradient Boosters: LightGBM and CatBoost

This document details the implementation, 5-fold cross-validation, and held-out test evaluation of **LightGBM** and **CatBoost** on the Diabetes 30-Day Hospital Readmission dataset, directly benchmarked against the research paper (*Emi-Johnson & Nkrumah, Cureus 2025*) and the four core models.

---

## 1. Why Implement LightGBM and CatBoost?

While the original paper explored Logistic Regression, Random Forest, XGBoost, and a DNN, modern tabular healthcare machine learning relies heavily on **LightGBM** and **CatBoost**:

1. **LightGBM (Light Gradient Boosting Machine):**
   - Implements **Histogram-Based Decision Tree Algorithms** and **Exclusive Feature Bundling (EFB)**.
   - Drastically accelerates training across the **2,414 sparse one-hot encoded columns**, training the entire 81,412-encounter cohort in **11.39 seconds** (vs. ~117s for Random Forest).
2. **CatBoost (Categorical Boosting):**
   - Implements **Ordered Boosting** to combat target leakage and prediction shift.
   - Natively designed to minimize overfitting on high-cardinality medical diagnosis codes and clinical department features.

---

## 2. Model Configurations and Hyperparameters

Both models were integrated into the pipeline factory ([`src/models.py`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/src/models.py)) using identical random seeds (`seed=42`) and evaluated on the exact same held-out test set ($N_{\text{test}} = 20,354$ encounters):

### LightGBM Configuration
```python
LGBMClassifier(
    n_estimators=120,
    learning_rate=0.08,
    num_leaves=31,
    scale_pos_weight=1.0,  # 1.0 for Paper-Matching; 7.96 for Cost-Sensitive
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
```

### CatBoost Configuration
```python
CatBoostClassifier(
    iterations=150,
    learning_rate=0.08,
    depth=5,
    scale_pos_weight=1.0,  # 1.0 for Paper-Matching; 7.96 for Cost-Sensitive
    random_state=42,
    verbose=0,
    thread_count=-1
)
```

---

## 3. Benchmark Results: All Models vs. Published Paper Target

| Model Architecture | Model Family | Paper Accuracy Target | Reached Accuracy *(Paper-Matching)* | Accuracy Delta | Paper AUC-ROC Ref | Reached AUC-ROC | Positive Recall | Positive Precision | False Negatives (FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (New)** | **Advanced GBDT** | **0.8900** | **0.8893** (88.93%) | **-0.0007** | **0.667** | **0.6897** | **0.0167** (1.7%) | **0.6441** (64.4%) | **2,233** |
| **XGBoost** | Paper Model | 0.8900 | **0.8889** (88.89%) | -0.0011 | 0.667 | **0.6884** | 0.0110 (1.1%) | 0.6098 (61.0%) | 2,246 |
| **Logistic Regression**| Paper Model | 0.8900 | **0.8886** (88.86%) | -0.0014 | 0.642 | **0.6820** | 0.0172 (1.7%) | 0.5270 (52.7%) | 2,232 |
| **CatBoost (New)** | **Advanced GBDT** | **0.8900** | **0.8886** (88.86%) | **-0.0014** | **0.667** | **0.6857** | **0.0079** (0.8%) | **0.5625** (56.3%) | **2,253** |
| **Random Forest** | Paper Model | 0.8900 | **0.8884** (88.84%) | -0.0016 | 0.630 | **0.6836** | 0.0000 (0.0%) | 0.0000 (0.0%) | 2,271 |
| **Deep Neural Net** | Paper Neural Net | 0.8400 | **0.8802** (88.02%) | +0.0402 | 0.579 | **0.6519** | 0.0762 (7.6%) | 0.3372 (33.7%) | 2,098 |

---

## 4. Key Performance Insights

1. **LightGBM Achieved the Highest Discriminative AUC-ROC (0.6897):**
   - Outperformed all four original models (XGBoost: 0.6884, RF: 0.6836, LR: 0.6820, DNN: 0.6519).
   - Significantly exceeded the published paper's reported XGBoost AUC benchmark of **0.667** (+0.0227 higher).
   - Achieved **88.93% accuracy**, matching the paper's 89% target within 0.07%.
2. **Computational Speed:**
   - LightGBM completed 5-fold cross-validation and full dataset refitting in just **11.39 seconds**, compared to 42.52s for XGBoost and 117.19s for Random Forest.
3. **CatBoost Precision & Robustness:**
   - CatBoost reached **88.86% accuracy** and an AUC-ROC of **0.6857**, with positive precision of **56.25%** and low variance across CV folds.

---

## 5. Cost-Sensitive Weighted Clinical Evaluation

When trained with class weighting (`scale_pos_weight = 7.96`) to actively prioritize catching patients at risk of 30-day readmission:

| Model | Evaluation Mode | Accuracy | Sensitivity (Positive Recall) | Specificity (TNR) | AUC-ROC | True Readmissions Caught |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | Cost-Sensitive | **0.6655** (66.6%) | **59.80%** | **67.40%** | **0.6882** | **1,358 patients** |
| **CatBoost** | Cost-Sensitive | **0.6603** (66.0%) | **59.71%** | **66.82%** | **0.6839** | **1,356 patients** |

---

## 6. Artifacts and How to Run

- **Standalone Script:** Run [`src/train_and_evaluate_advanced_models.py`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/src/train_and_evaluate_advanced_models.py) to train and evaluate LightGBM and CatBoost independently:
  ```powershell
  python src/train_and_evaluate_advanced_models.py
  ```
- **Generated CSV Reports:**
  - [`results/advanced_models/advanced_models_comparison.csv`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/results/advanced_models/advanced_models_comparison.csv)
  - [`results/advanced_models/extended_benchmark_table.csv`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/results/advanced_models/extended_benchmark_table.csv)
- **Saved Model Objects:**
  - `results/advanced_models/saved_models/lightgbm_unweighted.joblib`
  - `results/advanced_models/saved_models/catboost_unweighted.joblib`
- **Plots:**
  - ROC Curves: [`results/advanced_models/plots/advanced_roc_comparison.png`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/results/advanced_models/plots/advanced_roc_comparison.png)
  - PR Curves: [`results/advanced_models/plots/advanced_pr_curves.png`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/results/advanced_models/plots/advanced_pr_curves.png)
  - Confusion Matrices: [`results/advanced_models/confusion_matrices/`](file:///c:/Users/javer/OneDrive/Desktop/PatientReadmissionDataset/results/advanced_models/confusion_matrices/)
