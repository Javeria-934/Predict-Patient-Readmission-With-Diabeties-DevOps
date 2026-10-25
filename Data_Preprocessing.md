# Data Preprocessing Pipeline and Methodology Report

## 1. Executive Overview and Research Context

This document details the complete data preprocessing methodology implemented to reproduce and evaluate the research paper:

> **"Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data"**  
> **Authors:** Oluwabukola G. Emi-Johnson and Kwame J. Nkrumah  
> **Journal:** *Cureus* (Published: April 17, 2025)  
> **DOI:** [10.7759/cureus.82437](https://doi.org/10.7759/cureus.82437)

The core objective is to identify adult diabetic patients at high risk of unplanned hospital readmission within 30 days of discharge using structured Electronic Health Record (EHR) data. This implementation adheres strictly to principles of academic reproducibility, data leakage prevention, transparent decision documentation, and rigorous statistical validation.

---

## 2. Dataset Identification and Ingestion

The analysis utilizes the historical **University of California, Irvine (UCI) Diabetes 130-US Hospitals Dataset** (Strack et al., 2014):

- **File Identifier:** `diabetic_data.csv`
- **Associated Codebook:** `IDS_mapping.csv`
- **Temporal Coverage:** 10 years of inpatient encounters (1999–2008) across 130 US medical facilities.
- **Raw Dimensions:** 101,766 inpatient encounters and 50 attributes.
- **Patient Population:** Adult diabetic inpatients admitted for a length of stay between 1 and 14 days, with laboratory tests performed and medications administered.

```
Total Inpatient Encounters: 101,766
Unique Patient Count:        71,518
Initial Attributes:          50 (Identifiers, Demographics, Admission/Discharge, Care Intensity, Diagnoses, Medications, Lab Results, Outcome)
```

---

## 3. Target Variable Definition and Clinical Operationalization

### 3.1 Raw Target Distribution
The raw dataset contains the outcome column `readmitted`, categorized into three distinct nominal states:
- `NO`: No inpatient readmission recorded in the database.
- `>30`: Inpatient readmission recorded more than 30 days following discharge.
- `<30`: Unplanned inpatient readmission within 30 days of discharge.

| Original Label | Clinical Meaning | Encounter Count | Percentage |
| :--- | :--- | :--- | :--- |
| **`NO`** | No readmission recorded | 54,864 | 53.91% |
| **`>30`** | Readmission after 30 days | 35,545 | 34.93% |
| **`<30`** | Readmission within 30 days | 11,357 | 11.16% |
| **Total** | | **101,766** | **100.00%** |

### 3.2 Binary Target Operationalization
Under the US Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP), the established quality and reimbursement benchmark is **unplanned 30-day all-cause readmission**. Readmissions occurring beyond 30 days (e.g., routine checkups, elective procedures, or unrelated long-term visits) do not constitute 30-day acute bounce-backs.

Accordingly, the target variable $y \in \{0, 1\}$ was formulated as:
$$\text{Target} = \begin{cases} 1, & \text{if } \text{readmitted} = \text{'<30'} \\ 0, & \text{if } \text{readmitted} \in \{\text{'>30'}, \text{'NO'}\} \end{cases}$$

| Operational Class | Clinical Definition | Encounter Count | Proportion |
| :--- | :--- | :--- | :--- |
| **Class 0 (Negative)** | Not readmitted within 30 days (`NO` or `>30`) | 90,409 | 88.84% |
| **Class 1 (Positive)** | Readmitted within 30 days (`<30`) | 11,357 | 11.16% |

**Class Imbalance Ratio:** 7.96 : 1 (approximately 8 negative cases for every 1 positive case).

> **Implementation Decision Note:** The research paper defines a binary prediction task but does not explicitly print its exact mapping formula. Mapping `<30` to 1 and both `>30` and `NO` to 0 is the scientifically standard operationalization in clinical informatics literature for this benchmark.

---

## 4. Feature Selection, Audit, and Data Leakage Prevention

All 50 raw attributes were systematically audited to prevent target leakage, spurious correlation, and artificial memorization.

```
Raw Attributes (50)
├── Excluded Columns (5)
│   ├── Identifiers: encounter_id, patient_nbr
│   ├── Extreme Missingness: weight (96.86% missing)
│   └── Zero Variance: examide, citoglipton
├── Target Variable (1): readmitted
└── Retained Predictors (44)
    ├── Continuous Numerical Features (8)
    └── Categorical Features (36)
```

### 4.1 Excluded Features

| Column Name | Category | Primary Reason for Exclusion |
| :--- | :--- | :--- |
| **`encounter_id`** | Identifier | Unique primary key for each hospitalization. Causes artificial memorization and data leakage. |
| **`patient_nbr`** | Identifier | Unique medical record identifier. Encodes patient identity rather than clinical pathology; non-generalizable across healthcare systems. |
| **`weight`** | Missing Data | Missing in 98,569 of 101,766 encounters (96.86% missing). Imputing over 96% of data would introduce substantial noise and fabricate patient vitals. |
| **`examide`** | Zero Variance | Monomorphic column; contains strictly the value `'No'` across all 101,766 encounters. Zero variance yields zero mathematical information gain. |
| **`citoglipton`** | Zero Variance | Monomorphic column; contains strictly the value `'No'` across all 101,766 encounters. Zero variance yields zero mathematical information gain. |

### 4.2 Investigation of Discharge Disposition and Clinical Leakage

A vital clinical informatics consideration in readmission modeling is **patient discharge mortality**:
- In `IDS_mapping.csv`, `discharge_disposition_id` entries `11`, `19`, `20`, and `21` denote patient death (*"Expired"* or *"Expired at home / medical facility"*).
- In the raw dataset, exactly **1,652 encounters** have an Expired discharge status.
- Crucially, **100.0% of these 1,652 encounters have `readmitted == 'NO'`**, because a patient who expires during inpatient admission cannot be readmitted.

#### Discrepancy Resolution in the Literature
The paper's *Materials and Methods* states: *"Patients who were discharged to hospice or who expired during hospitalization were excluded."* However, the paper's *Results* section states:
> *"From the original dataset of 101,766 patient records, 20,354 records were retained in the test set. These included 2,285 readmitted cases (positive class) and 18,069 non-readmitted cases (negative class)..."*

Notice that $20,354$ is **exactly $20.0\%$ of $101,766$** ($101,766 \times 0.20 = 20,353.2$). If the authors had excluded the 2,423 expired/hospice encounters prior to splitting, the total dataset would have been 99,343 encounters, and a 20% test set would have contained 19,868 records. 

To maintain exact numerical fidelity to the paper's primary experimental sample size ($N_{\text{test}} = 20,354$), this implementation maintains all 101,766 encounters in the primary stratified split while documenting this critical clinical nuance.

---

## 5. Train/Test Partitioning and Patient Overlap Analysis

To guarantee strict methodological integrity and prevent data leakage:
1. **Partitioning occurred prior to any imputer, scaler, or encoder fitting.**
2. **Stratified Sampling:** Preserved the 11.16% positive outcome rate across both subsets.
3. **Random Seed:** Fixed to `42` for strict academic reproducibility.
4. **Partition Ratio:** 80% Training ($N = 81,412$) and 20% Testing ($N = 20,354$).

| Dataset Partition | Total Encounters | Class 0 (Not Readmitted) | Class 1 (Readmitted <30d) | Positive Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Full Dataset** | 101,766 | 90,409 | 11,357 | 11.159% |
| **Training Set (80%)** | 81,412 | 72,326 | 9,086 | 11.160% |
| **Test Set (20%)** | 20,354 | 18,083 | 2,271 | 11.157% |

### 5.1 Patient Overlap Sensitivity Analysis
The 101,766 hospital encounters correspond to **71,518 unique patients**, meaning 16,773 patients (23.45%) were hospitalized multiple times (up to 40 encounters for a single patient).

In an encounter-level stratified 80/20 split:
- **Unique patients in Training set:** 60,132
- **Unique patients in Test set:** 18,387
- **Overlapping patients:** Exactly **7,001 patients** appear in both training and test partitions.
- **Affected test encounters:** **8,468 encounters (41.60% of the test set)** come from patients previously seen in training.

```
Encounter Split vs Patient Generalization:
Encounter-level prediction reflects real-time clinical bedside scoring where hospital
systems evaluate encounters sequentially. However, patient overlap slightly aids models
through repeated baseline physiological profiles. A grouped patient-level split is
recommended for pure new-patient generalization studies.
```

---

## 6. Preprocessing Transformations (`ColumnTransformer`)

The preprocessing architecture was implemented using scikit-learn's `ColumnTransformer` and `Pipeline`, ensuring that statistics were learned strictly from training folds.

```
Input Features (44)
├── Continuous Numerical (8) ──> SimpleImputer(mean) ──> StandardScaler() ──────────┐
│                                                                                   ├──> Feature Matrix
└── Categorical (36) ──────────> SimpleImputer('Missing') ──> OneHotEncoder() ──────┘    (81,412 x 2,414)
```

### 6.1 Continuous Numerical Features (8 Predictors)
Continuous attributes reflect care intensity and acute healthcare utilization:
1. `time_in_hospital` (Length of stay: 1–14 days)
2. `num_lab_procedures` (Number of laboratory tests: 1–132)
3. `num_procedures` (Number of non-lab inpatient procedures: 0–6)
4. `num_medications` (Number of distinct administered medications: 1–81)
5. `number_outpatient` (Outpatient visits in the preceding 12 months)
6. `number_emergency` (Emergency room visits in the preceding 12 months)
7. `number_inpatient` (Inpatient admissions in the preceding 12 months)
8. `number_diagnoses` (Number of recorded diagnoses: 1–16)

#### Transformation Protocol:
- **Imputation:** `SimpleImputer(strategy='mean')`. (The raw continuous features contain complete records, but the imputer is embedded in the pipeline to strictly follow the paper's specification).
- **Standardization:** `StandardScaler()`. Centers feature values to zero mean ($\mu = 0$) and scales to unit variance ($\sigma = 1$), preventing gradient vanishing/explosion in DNNs and regularization bias in Logistic Regression.

### 6.2 Categorical Features (36 Predictors)
Categorical features encompass demographics, administrative codes, diagnoses, and medication regimens:
- **Demographics:** `race`, `gender`, `age` (10-year deciles).
- **Admission / Discharge Details:** `admission_type_id`, `discharge_disposition_id`, `admission_source_id`.
- **Administrative / Specialty:** `payer_code`, `medical_specialty`.
- **Primary / Secondary / Tertiary Diagnoses:** `diag_1`, `diag_2`, `diag_3` (ICD-9-CM codes).
- **Glycemic Monitoring:** `max_glu_serum`, `A1Cresult`.
- **Diabetes Medications (23 specific agents):** `metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`.
- **Overall Therapy Management:** `change` (diabetic medication dosage adjusted), `diabetesMed` (any diabetes medication prescribed).

#### Critical Categorical Handling Decisions:
1. **Special Missing Value Conversion:** All raw `'?'` strings were detected and converted to `np.nan` prior to encoding.
2. **Preservation of `'None'` in Laboratory Tests:** In `A1Cresult` and `max_glu_serum`, the string value `'None'` signifies that **the physician did not order the test during the encounter**. This is not missing data; it is an active clinical decision reflecting encounter routine. Coercing `'None'` to missing would discard valuable clinical signal.
3. **Integer ID Casting:** Categorical IDs (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`) were explicitly cast to strings so they are treated as nominal categories rather than continuous scales.
4. **Categorical Imputation:** `SimpleImputer(strategy='constant', fill_value='Missing')`.
5. **One-Hot Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=True)`. Unseen categories in test folds are encoded as zero vectors without raising runtime exceptions.

### 6.3 Preserving Sparse Matrix Structure
After one-hot encoding, the feature space expands from 44 raw predictors to **2,414 transformed binary and continuous columns** (due to the high cardinality of ICD-9 diagnosis codes and hospital specialties). 

To ensure maximum computational efficiency and prevent memory exhaustion:
- The transformed matrix is preserved as a compressed sparse row matrix (`scipy.sparse.csr_matrix`).
- Preprocessing executes in **1.37 seconds**.
- Training and inference avoid dense array conversions wherever possible.

---

## 7. Class Imbalance Handling Strategy

The paper explicitly considered Synthetic Minority Oversampling Technique (SMOTE) but intentionally refrained from using it:
> *"Finally, although synthetic minority oversampling techniques (SMOTE) were considered to address the class imbalance in readmission outcomes, they were not applied in this study. We chose this approach to preserve the integrity of the original dataset. SMOTE and similar algorithms... may increase the risk of overfitting and distort the true data distribution..."*

In strict accordance with the paper:
- **No synthetic oversampling (SMOTE) was applied.**
- **Cost-sensitive learning / class weighting** was implemented across all models:
  - For Logistic Regression and Random Forest: `class_weight='balanced'`.
  - For XGBoost: `scale_pos_weight = N_neg / N_pos = 72,326 / 9,086 \approx 7.9602` (derived strictly from training data).
  - For DNN: Positive class loss weighting in binary cross-entropy: `pos_weight = 7.9602`.

---

## 8. Summary of Preprocessing Decisions and Reproducibility Matrix

| Preprocessing Step | Paper Specification | Reproduction Implementation | Implementation Justification |
| :--- | :--- | :--- | :--- |
| **Dataset Source** | UCI Diabetes 130-US Hospitals (101,766 encounters) | `diabetic_data.csv` (101,766 encounters) | Exact primary source retained without fabrication. |
| **Target Variable** | 30-Day Hospital Readmission | Binary: `<30` $\to 1$, `>30`/`NO` $\to 0$ | Standard clinical benchmark operationalization. |
| **Train/Test Split** | Stratified 80/20 split | Stratified 80/20 (`random_state=42`) | Preserves 11.16% class distribution across partitions. |
| **Numerical Imputation** | Mean-based imputation | `SimpleImputer(strategy='mean')` | Strict alignment with published methodology. |
| **Numerical Scaling** | Standardization | `StandardScaler()` | Prevents scale bias across clinical measures. |
| **Categorical Imputation** | Not fully specified in paper | `SimpleImputer(fill_value='Missing')` | Documented implementation choice avoiding data deletion. |
| **Categorical Encoding** | One-hot encoding | `OneHotEncoder(handle_unknown='ignore')` | Standard scikit-learn pipeline transformer. |
| **Lab 'None' Handling** | Not specified in paper | Preserved as valid categorical state | Clinical reality: 'None' indicates test not ordered. |
| **Identifier Handling** | Excluded | `encounter_id`, `patient_nbr` dropped | Eliminates memorization and target leakage. |
| **Class Imbalance** | Class weighting (no SMOTE) | `class_weight='balanced'` / `scale_pos_weight` | Preserves genuine empirical distribution. |
