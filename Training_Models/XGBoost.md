# Model Documentation: XGBoost Classifier

## 1. Model Overview and Rationale

In this research project, **Extreme Gradient Boosting (XGBoost)** represents the gradient-boosted decision tree framework for predicting 30-day hospital readmissions in diabetic patients, following the methodology of Emi-Johnson & Nkrumah (*Cureus*, 2025).

XGBoost was selected for the following methodological reasons:
1. **Empirical Superiority on Tabular Data:** Gradient boosted trees consistently achieve state-of-the-art predictive performance on structured EHR datasets by sequentially correcting predecessor errors.
2. **Paper's Leading Model:** In the reference publication, XGBoost achieved the highest reported discriminative ability among all evaluated models (**AUC-ROC = 0.667**).
3. **Exact Integration with SHAP:** XGBoost's tree structure allows exact polynomial-time SHAP value computation via TreeSHAP (Lundberg et al., 2020), enabling rigorous clinical interpretability.

---

## 2. Mathematical Formulation and Optimization

XGBoost minimizes a regularized objective function across $K$ additive boosting rounds:

$$\mathcal{L}^{(t)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

where the tree complexity penalty is defined as:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Here, $T$ is the number of terminal leaves and $w_j$ represents leaf weights.

### 2.1 Second-Order Taylor Approximation
At step $t$, the objective is expanded via a second-order Taylor series around the previous prediction:
$$\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
where the first-order gradient $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and second-order Hessian $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$.

### 2.2 Cost-Sensitive Class Weighting (`scale_pos_weight`)
To resolve the 7.96 : 1 class imbalance without artificial synthetic oversampling (SMOTE), `scale_pos_weight` was derived strictly from the training partition:

$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{72,326}{9,086} \approx 7.9602$$

In the binary cross-entropy loss, this scales the gradient contribution of positive (readmitted) encounters by approximately $8\times$:
$$g_i = \begin{cases} \hat{p}_i - 1, & \text{if } y_i = 1 \text{ (scaled by } 7.9602) \\ \hat{p}_i, & \text{if } y_i = 0 \end{cases}$$

---

## 3. Preprocessing and Feature Matrix Compatibility

- **Numerical Features (8):** Imputed with training means and standardized.
- **Categorical Features (36):** Imputed with `'Missing'` category and one-hot encoded (`handle_unknown='ignore'`).
- **Feature Matrix:** 2,414 sparse binary and continuous features.
- XGBoost natively handles sparse CSR matrices (`scipy.sparse.csr_matrix`), storing only non-zero entries. This enables sub-second boosting iterations without dense memory inflation.

---

## 4. Hyperparameter Tuning and 5-Fold Cross-Validation

Hyperparameter optimization was conducted via **5-Fold Stratified Cross-Validation** on the training partition ($N = 81,412$):
- **Candidate Parameter Grid:**
  - `max_depth`: `[3, 5]` (controls tree complexity and high-order feature interactions).
  - `learning_rate` ($\eta$): `[0.05, 0.1]` (shrinkage parameter mitigating overfitting).
  - `n_estimators`: Fixed at `100` boosting rounds.
  - `subsample`: Fixed at `0.8` (stochastic row subsampling).
  - `scale_pos_weight`: Fixed at `7.9602` (derived from training labels).
- **Scoring Objective:** ROC-AUC (`roc_auc`).
- **Cross-Validation Outcome:**
  - **Optimal Hyperparameters:** `{'learning_rate': 0.1, 'max_depth': 5, 'scale_pos_weight': 7.9602}`
  - **Best 5-Fold CV ROC-AUC:** **0.6757**
  - **Tuning and Refit Duration:** 58.28 seconds across all 20 cross-validation fits.

---

## 5. Performance on Held-Out Test Set ($N = 20,354$)

The tuned XGBoost model was evaluated on the untouched test partition:

| Performance Metric | Experimental Result | Paper Reference (Emi-Johnson & Nkrumah) | Notes & Clinical Context |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~0.64 – 0.66 | 0.89 | Paper used default 0.5 threshold resulting in near-total majority class prediction. |
| **ROC-AUC** | **0.677** | **0.667** | Confirms XGBoost as the top-ranking discriminator among all models. |
| **PR-AUC (Avg Precision)** | **0.203** | Not Reported | Highest PR-AUC among all evaluated architectures. |
| **Positive Class Recall (Sensitivity)** | **0.598** | 0.017 (Table 2) | Successfully detects ~60% of true readmissions vs. paper's 1.7%. |
| **Positive Class Precision (PPV)** | **0.183** | 0.437 (Table 2) | Highest positive precision among class-weighted models. |
| **Positive Class F1-Score** | **0.280** | 0.032 (Table 2) | Nearly 9x higher than the paper's reported Table 2 F1 of 0.032. |
| **Specificity (TNR)** | **0.661** | 0.997 (Table 2) | Strongest true negative rate among sensitive models. |
| **False Positives (FP)** | ~6,130 | 1,172 (Table 3) | Controlled false alarm volume while ensuring early risk detection. |
| **False Negatives (FN)** | ~910 | 2,244 (Table 3) | Misses fewer than 920 readmissions vs. paper's 2,244 missed cases. |

---

## 6. Discrepancy Analysis and Clinical Interpretation

### 6.1 Discrepancies in the Published Paper
1. **The F1-Score Contradiction:** In Table 1 of the paper, the authors reported an F1-score of **0.89** for XGBoost. However, in Table 2, the exact same model's class-specific performance is reported as:
   - Class 0 F1-score = 0.940
   - Class 1 F1-score = **0.032**
   The overall 0.89 figure was either an erroneous reporting of accuracy or an unweighted metric heavily skewed by the negative majority class.
2. **False Negative Volume:** In Table 3 of the paper, XGBoost recorded **2,244 False Negatives** out of 2,285 positive cases (missing 98.2% of readmitted patients).
3. **Inconsistency in False Positives:** In Table 3, False Positives are reported as 1,172, whereas Table 2 specificity of 0.997 on 18,069 negative cases implies only ~54 false positives ($18,069 \times 0.003 \approx 54$).

### 6.2 Superiority of Gradient Boosting on Clinical Tabular Data
Our reproduction confirms the core scientific finding of Emi-Johnson & Nkrumah: **XGBoost achieves the highest discriminative ability (AUC-ROC = 0.677)** among all evaluated classifiers. Its capacity to perform second-order gradient descent across thousands of sparse indicator variables makes it exceptionally well-suited for high-dimensional EHR risk stratification.
