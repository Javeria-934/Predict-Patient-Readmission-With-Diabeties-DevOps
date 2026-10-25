# Model Documentation: Random Forest Ensemble

## 1. Model Overview and Rationale

In this research reproduction, **Random Forest** represents the bagging-based decision tree ensemble method for predicting 30-day hospital readmissions in diabetic patients, based on Emi-Johnson & Nkrumah (*Cureus*, 2025).

Random Forest was selected for the following methodological reasons:
1. **Non-Linear Interactions:** Clinical risks often stem from combinatorial effects (e.g., specific age groups taking complex multi-drug regimens alongside high prior inpatient admissions) that are poorly captured by linear models.
2. **Robustness to Overfitting:** Averaging over a forest of de-correlated trees reduces the variance inherent in deep decision trees.
3. **Inherent Handling of Mixed Data Types:** Tree-based split criteria naturally accommodate binary one-hot indicators alongside continuous clinical scales.

---

## 2. Mathematical Formulation and Ensemble Architecture

A Random Forest constructs an ensemble of $B$ de-correlated decision trees $\{T_b\}_{b=1}^B$. For an encounter input vector $x$:

1. **Bootstrap Aggregation (Bagging):** Each tree $T_b$ is trained on a distinct bootstrap sample $\mathcal{D}_b$ drawn with replacement from the training set ($|\mathcal{D}_b| = N_{\text{train}} = 81,412$).
2. **Feature Subspace Randomization:** At each candidate split node within a tree, a random subset of $m \approx \sqrt{D}$ features is evaluated rather than the full feature space, reducing inter-tree correlation $\rho$.
3. **Ensemble Probability Estimation:** The final predicted probability of 30-day readmission is computed by averaging class probabilities across all individual trees:
   $$P(y = 1 \mid x) = \frac{1}{B} \sum_{b=1}^B P_{T_b}(y = 1 \mid x)$$

### 2.1 Balanced Class Weighting
Under severe class imbalance (7.96 : 1), standard Gini impurity splitting biases splits toward the dominant non-readmission class. To correct this, `class_weight='balanced'` was applied:
- Leaf impurity calculations weigh minority class instances proportionally higher:
  $$w_j = \frac{N}{2 \times N_j}$$
- This penalizes misclassifying readmission encounters during tree construction, encouraging deeper branches dedicated to minority-class risk patterns.

---

## 3. Preprocessing and Pipeline Integration

- **Continuous Features (8):** Imputed with training-fold means and standardized (`StandardScaler()`).
- **Categorical Features (36):** Missing values imputed as `'Missing'`, followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`.
- **Transformed Feature Matrix:** 2,414 sparse binary and continuous features.
- Trees efficiently partition sparse dummy variables, evaluating presence versus absence of specific diagnostic codes (e.g., congestive heart failure, diabetes with renal manifestations).

---

## 4. Hyperparameter Tuning and 5-Fold Cross-Validation

Hyperparameter tuning was conducted using **5-Fold Stratified Cross-Validation** on the training partition ($N = 81,412$):
- **Candidate Grid:**
  - `max_depth`: `[10, 15]`
  - `min_samples_split`: `[5, 10]`
  - `n_estimators`: Fixed at `100` trees to balance statistical stability and computational budget.
- **Scoring Objective:** ROC-AUC (`roc_auc`), prioritizing risk discrimination over accuracy.
- **Cross-Validation Outcome:**
  - **Optimal Configuration:** `{'max_depth': 15, 'min_samples_split': 5}`
  - **Best 5-Fold CV ROC-AUC:** **0.6697**
  - Restricting maximum tree depth to 15 prevented individual trees from memorizing rare ICD-9 dummy codes.
- **Tuning and Refit Duration:** 152.76 seconds across all 20 cross-validation fits.

---

## 5. Performance on Held-Out Test Set ($N = 20,354$)

The tuned model was evaluated on the untouched test partition:

| Performance Metric | Experimental Result | Paper Reference (Emi-Johnson & Nkrumah) | Notes & Clinical Context |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~0.63 – 0.65 | 0.89 | Paper used default 0.5 threshold resulting in near-total majority class prediction. |
| **ROC-AUC** | **0.675** | **0.630** | Substantially outperforms paper's reported AUC (0.630). |
| **PR-AUC (Avg Precision)** | **0.201** | Not Reported | Nearly double the 11.2% random baseline. |
| **Positive Class Recall (Sensitivity)** | **0.605** | 0.002 (Table 2) | Identifies >60% of true readmissions vs. paper's 0.2%. |
| **Positive Class Precision (PPV)** | **0.178** | 0.364 (Table 2) | Expected clinical trade-off for capturing 300x more true positives. |
| **Positive Class F1-Score** | **0.275** | 0.003 (Table 2) | Drastic improvement in minority class F1 over published Table 2. |
| **Specificity (TNR)** | **0.648** | 0.999 (Table 2) | Preserves healthy specificity while catching high-risk bounce-backs. |
| **False Positives (FP)** | ~6,360 | 1,224 (Table 3) | Clinically acceptable false alarms to prevent missed early deaths/admissions. |
| **False Negatives (FN)** | ~890 | 2,280 (Table 3) | Misses fewer than 900 readmissions compared to the paper's 2,280 missed cases. |

---

## 6. Discrepancy Analysis and Clinical Interpretation

### 6.1 Catastrophic Recall Collapse in Published Model
In Table 2 of Emi-Johnson & Nkrumah, Random Forest demonstrated a near-total recall collapse:
- **Recall for Class 1 = 0.002 (0.2%)**: Out of 2,285 readmitted patients, the paper's Random Forest detected only **5 patients**, failing on **2,280 patients** (Table 3: False Negatives = 2,280).
- Despite an apparent Accuracy of 0.89, the model was practically useless as a screening tool because it missed 99.8% of high-risk patients.

### 6.2 Benefit of Controlled Tree Depth and Class Balancing
By constraining `max_depth=15` and enforcing `class_weight='balanced'`:
- The ensemble avoids getting trapped in pure leaves dominated by the 88.8% majority class.
- The model achieves a competitive **ROC-AUC of 0.675**, ranking patients reliably from highest to lowest probability of acute readmission.
