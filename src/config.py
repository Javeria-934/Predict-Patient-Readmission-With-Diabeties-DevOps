"""
Configuration settings for Diabetes 30-Day Readmission Reproduction Pipeline.
Academic reproduction of Emi-Johnson & Nkrumah (Cureus, April 2025).
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
DATASET_PATH = DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_PATH = DATA_DIR / "IDS_mapping.csv"

# Output Directories
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
SHAP_DIR = RESULTS_DIR / "shap"
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models"

for p in [RESULTS_DIR, PLOTS_DIR, CONFUSION_MATRICES_DIR, SHAP_DIR, SAVED_MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Definition
TARGET_COLUMN = "readmitted"
POSITIVE_CLASS_VALUE = "<30"
NEGATIVE_CLASS_VALUES = [">30", "NO"]
TARGET_NAME = "readmission_30d"

# Columns to Exclude
IDENTIFIER_COLUMNS = ["encounter_id", "patient_nbr"]
HIGH_MISSING_COLUMNS = ["weight"]  # >96% missing
ZERO_VARIANCE_COLUMNS = ["examide", "citoglipton"]  # Only single value ('No')

# Features realistically known at discharge
NUMERICAL_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Primary experiment categorical features
CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

# Hyperparameter grids for 5-fold CV Grid Search
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__max_iter": [500],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 10],
        "classifier__min_samples_leaf": [1, 4],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
    },
    "dnn": {
        "classifier__hidden_layer_sizes": [(128, 64, 32), (64, 32)],
        "classifier__alpha": [1e-4, 1e-3, 1e-2],
        "classifier__learning_rate_init": [0.001, 0.005],
    }
}
