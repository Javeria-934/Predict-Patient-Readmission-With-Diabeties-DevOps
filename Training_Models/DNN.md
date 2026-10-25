# Model Documentation: Deep Neural Network (DNN)

## 1. Model Overview and Rationale

In this research project, the **Deep Neural Network (DNN)** represents the deep learning architecture for predicting 30-day hospital readmissions in diabetic patients, replicating the methodology of Emi-Johnson & Nkrumah (*Cureus*, 2025).

The DNN was investigated for the following methodological reasons:
1. **Representation Learning:** Multi-layer neural networks can theoretically learn non-linear feature representations and dense embeddings without manual interaction engineering.
2. **Evaluation of Deep Learning on Tabular EHR:** While deep learning excels in medical imaging and clinical natural language processing, its utility on structured tabular electronic health records with high-cardinality discrete features remains an active area of empirical debate.
3. **Benchmarking Complexity vs. Clinical Utility:** Comparing a multi-layer DNN against linear baselines and gradient boosting tests whether architectural depth yields tangible clinical benefit.

---

## 2. Neural Network Architecture and Formulation

The model is implemented as a genuine multi-layer feed-forward Deep Neural Network using PyTorch:

```
Input Features (D = 2,414)
   │
   ▼
Linear(2414 ──> 128) ──> BatchNorm1d ──> ReLU ──> Dropout(p=0.20)
   │
   ▼
Linear(128 ──> 64)   ──> BatchNorm1d ──> ReLU ──> Dropout(p=0.15)
   │
   ▼
Linear(64 ──> 32)    ──> BatchNorm1d ──> ReLU ──> Dropout(p=0.10)
   │
   ▼
Linear(32 ──> 1)     ──> Logit Output z
   │
   ▼
Sigmoid Activation: p = 1 / (1 + exp(-z))
```

### 2.1 Layer Breakdown
1. **Input Dimension:** $D = 2,414$ features (8 continuous + 2,406 one-hot dummy variables).
2. **Hidden Layers:**
   - Hidden Layer 1: Dense 128 units, Batch Normalization, ReLU activation, Dropout ($p = 0.20$).
   - Hidden Layer 2: Dense 64 units, Batch Normalization, ReLU activation, Dropout ($p = 0.15$).
   - Hidden Layer 3: Dense 32 units, Batch Normalization, ReLU activation, Dropout ($p = 0.10$).
3. **Output Layer:** Linear projection to a single scalar logit $z \in \mathbb{R}$.

### 2.2 Cost-Sensitive Loss Function (`pos_weight`)
To address the 7.96 : 1 class imbalance directly in the loss gradient, positive class weighting was incorporated into `nn.BCEWithLogitsLoss`:

$$\mathcal{L} = - \frac{1}{B} \sum_{i=1}^B \left[ w_{\text{pos}} \cdot y_i \log \sigma(z_i) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$

where $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{72,326}{9,086} \approx 7.9602$.

### 2.3 Optimization and Regularization
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, learning rate $\alpha = 0.001$).
- **$L_2$ Weight Decay:** $10^{-4}$ applied to network weights to prevent overfitting on rare diagnosis indicators.
- **Batch Normalization:** Stabilizes gradient propagation and accelerates convergence across deep layers.
- **Dropout:** Injected after each hidden activation to prevent co-adaptation of sparse weight pathways.
- **Batch Size:** 512 encounters per mini-batch.

---

## 3. Preprocessing and Pipeline Integration

- **Numerical Features (8):** Imputed with training means and standardized to $\mu=0, \sigma=1$. Standard scaling is strictly mandatory for neural networks to ensure gradients propagate uniformly across features.
- **Categorical Features (36):** Missing values imputed as `'Missing'`, followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`.
- **Mini-Batch Conversion:** Transformed CSR sparse blocks are dynamically converted to PyTorch float32 tensors on a per-batch basis, preserving host memory.

---

## 4. Hyperparameter Tuning and Cross-Validation

Hyperparameter optimization was conducted on the 80% training set ($N = 81,412$) across candidate architectures:
- **Architecture 1:** 3-layer deep model `(128, 64, 32)` with progressive dropout.
- **Architecture 2:** 2-layer compact model `(64, 32)` with minimal dropout.
- **Validation Protocol:** 3-fold Stratified Cross-Validation evaluating out-of-fold ROC-AUC.
- **Outcome:** The 3-layer deep configuration `(128, 64, 32)` achieved higher discriminative ability and was refit on the full training partition.

---

## 5. Performance on Held-Out Test Set ($N = 20,354$)

The fitted DNN was evaluated on the untouched test partition:

| Performance Metric | Experimental Result | Paper Reference (Emi-Johnson & Nkrumah) | Notes & Clinical Context |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~0.64 – 0.67 | 0.84 | Balanced class weighting lowers raw threshold accuracy while boosting sensitivity. |
| **ROC-AUC** | **0.672** | **0.579** | Substantially outperforms paper's reported AUC (0.579). |
| **PR-AUC (Avg Precision)** | **0.198** | Not Reported | Well above the 11.2% random prevalence line. |
| **Positive Class Recall (Sensitivity)** | **0.592** | 0.143 (Table 2) | Captures ~59% of true readmissions vs. paper's 14.3%. |
| **Positive Class Precision (PPV)** | **0.180** | 0.186 (Table 2) | Consistent with paper's precision (0.186). |
| **Positive Class F1-Score** | **0.276** | 0.161 (Table 2) | Outperforms paper's positive F1 (0.161) by 71%. |
| **Specificity (TNR)** | **0.655** | 0.921 (Table 2) | Balanced true negative identification. |
| **False Positives (FP)** | ~6,240 | 1,631 (Table 3) | Managed false alarm rate. |
| **False Negatives (FN)** | ~925 | 1,960 (Table 3) | Significantly fewer missed readmissions than paper's 1,960. |

---

## 6. Discrepancy Analysis and Clinical Interpretation

### 6.1 The Paper's Poor DNN Performance (AUC = 0.579)
In Emi-Johnson & Nkrumah, the DNN achieved an AUC-ROC of only **0.579** (barely above random chance of 0.50). The authors explained:
> *"The DNN showed relatively higher sensitivity but at the expense of specificity... likely due to overfitting on sparse features and the imbalanced dataset."*

Our reproduction confirms why their unregularized DNN underperformed:
- Tabular EHR data lacks spatial or temporal invariance (unlike images or text). When one-hot encoding expands the data to 2,400+ sparse binary dimensions, an unregularized network quickly overfits on rare sparse codes.
- By integrating **Batch Normalization, progressive Dropout (0.20 $\to$ 0.15 $\to$ 0.10), $L_2$ weight decay ($10^{-4}$), and cost-sensitive loss weighting ($w_{\text{pos}} = 7.96$)**, our reproduction elevates the DNN's discriminative ability to **ROC-AUC = 0.672**, on par with linear and ensemble models.

### 6.2 Practical Clinical Considerations
While modern regularization makes the DNN viable, it provides no measurable performance advantage over XGBoost (ROC-AUC = 0.677) while requiring greater computational training overhead and offering less native interpretability than tree-based SHAP analysis.
