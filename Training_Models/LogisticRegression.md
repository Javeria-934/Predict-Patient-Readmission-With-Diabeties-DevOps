# Model Documentation: Logistic Regression Baseline

## 1. Model Overview and Rationale

In this research project, **Logistic Regression** serves as the primary linear baseline model for predicting 30-day hospital readmissions in diabetic patients, reproducing the methodology of Emi-Johnson & Nkrumah (*Cureus*, 2025).

Logistic Regression was selected for the following methodological reasons:
1. **Clinical Interpretability:** The log-odds coefficients directly reflect the directional risk associated with each clinical feature, providing an established clinical benchmark.
2. **Standard Healthcare Baseline:** In clinical risk scoring (e.g., LACE index, HOSPITAL score), logistic models represent the traditional standard against which complex non-linear ensemble models and neural networks are benchmarked.
3. **Calibrated Probabilistic Output:** The sigmoid link function directly maps linear predictor combinations to well-calibrated posterior probabilities $P(Y=1|X) \in [0, 1]$.

---

## 2. Mathematical Formulation and Loss Function

For an input feature vector $x \in \mathbb{R}^D$ and binary outcome $y \in \{0, 1\}$ (where $1 = \text{Readmitted } <30\text{ days}$ and $0 = \text{Not Readmitted}$):

$$P(y = 1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

### 2.1 Class-Weighted Objective Function
To counteract severe class imbalance (88.84% negative vs. 11.16% positive encounters), balanced class weighting is incorporated into the negative log-likelihood objective:

$$\min_{w, b} \left[ \frac{1}{2} w^T w + C \sum_{i=1}^N w_{y_i} \left( - y_i \log \hat{p}_i - (1 - y_i) \log(1 - \hat{p}_i) \right) \right]$$

where sample weights $w_{y_i}$ are computed inversely proportional to class frequencies:
$$w_0 = \frac{N}{2 \times N_0} = \frac{81,412}{2 \times 72,326} \approx 0.5628$$
$$w_1 = \frac{N}{2 \times N_1} = \frac{81,412}{2 \times 9,086} \approx 4.4801$$

The ratio $w_1 / w_0 \approx 7.96$ penalizes false negatives nearly 8 times more heavily than false positives during gradient updates.

---

## 3. Preprocessing and Pipeline Integration

The model is embedded inside a strict scikit-learn `Pipeline` to prevent data leakage:
- **Continuous Features (8):** Imputed with training-fold mean values (`SimpleImputer(strategy='mean')`) and scaled to zero mean and unit variance (`StandardScaler()`). Standardization is essential for Logistic Regression to ensure the $L_2$ regularization penalty treats all features equitably.
- **Categorical Features (36):** Missing values imputed as `'Missing'`, followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`.
- **Transformed Feature Space:** 2,414 sparse binary and continuous features.
- **Solver:** Limited-memory BFGS (`solver='lbfgs'`), which efficiently optimizes smooth convex objectives with thousands of sparse inputs.

---

## 4. Hyperparameter Tuning and 5-Fold Cross-Validation

Hyperparameter optimization was conducted strictly within the 80% training set ($N = 81,412$) using **5-Fold Stratified Cross-Validation**:
- **Candidate Search Space:** Inverse regularization strength $C \in \{0.01, 0.1, 1.0, 10.0\}$.
- **Primary Tuning Metric:** Area Under the ROC Curve (`roc_auc`), prioritizing rank-order discrimination across the class distribution over default threshold accuracy.
- **Cross-Validation Results:**
  - $C = 0.01$ achieved the highest cross-validation discriminative performance (**CV ROC-AUC = 0.6730**).
  - Stronger regularization ($C=0.01$) prevented overfitting on high-cardinality sparse diagnosis indicators.
- **Final Fit:** The optimal model was refit on all 81,412 training records prior to held-out test evaluation.

---

## 5. Performance on Held-Out Test Set ($N = 20,354$)

The model was evaluated on the isolated 20% test set using both global and positive-class-specific metrics:

| Performance Metric | Experimental Result | Paper Reference (Emi-Johnson & Nkrumah) | Notes & Clinical Context |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~0.63 – 0.65 (balanced threshold) | 0.89 | Paper used default 0.5 threshold which predicts nearly all majority class. |
| **ROC-AUC** | **0.673** | **0.642** | Reproduction exceeds paper AUC by capturing full feature set. |
| **PR-AUC (Avg Precision)** | **0.198** | Not Reported | Substantially above random prevalence baseline (0.112). |
| **Positive Class Recall (Sensitivity)** | **0.618** | 0.016 (Table 2) | Balanced weighting correctly identifies ~62% of actual readmissions. |
| **Positive Class Precision (PPV)** | **0.175** | 0.414 (Table 2) | Trade-off from increased sensitivity under 8:1 class imbalance. |
| **Positive Class F1-Score** | **0.272** | 0.030 (Table 2) | Significant improvement in minority class F1 over paper's Table 2. |
| **Specificity (TNR)** | **0.634** | 0.997 (Table 2) | True negative detection rate under balanced weighting. |
| **False Positives (FP)** | ~6,600 | 1,276 (Table 3) | Reflects decision threshold shift prioritizing clinical sensitivity. |
| **False Negatives (FN)** | ~860 | 2,249 (Table 3) | Significantly fewer missed readmissions than the paper's 2,249 missed cases. |

---

## 6. Discrepancy Analysis and Clinical Interpretation

### 6.1 The High Accuracy Illusion
In Table 1 of the original paper, Logistic Regression was reported with **Accuracy = 0.89** and **ROC-AUC = 0.642**. However, examining Table 2 of the paper reveals:
- **Positive Class Recall = 0.016 (1.6%)**: Out of 2,285 readmitted patients, the paper's model detected only ~36 patients, missing 2,249 patients (Table 3: False Negatives = 2,249).
- **The model achieved 89% accuracy merely by predicting that almost every patient would not be readmitted**, matching the 88.8% majority class frequency.

### 6.2 Value of Calibrated Class Weighting
In our reproduction, incorporating cost-sensitive class weighting shifts the decision threshold so that:
- Over **60% of true readmissions are flagged** for post-discharge intervention.
- The model functions as an actionable clinical screening tool rather than a passive majority-class predictor.
- The ROC-AUC of **0.673** demonstrates solid ranking capability across all classification thresholds.
