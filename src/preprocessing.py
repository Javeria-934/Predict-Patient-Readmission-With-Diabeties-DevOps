"""
Preprocessing Pipeline Module for Diabetes Readmission Reproduction.
Implements data leakage prevention, stratified 80/20 splitting,
mean numerical imputation, standard scaling, categorical imputation,
and one-hot encoding using scikit-learn ColumnTransformer and Pipeline.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATASET_PATH,
    RESULTS_DIR,
    RANDOM_STATE,
    IDENTIFIER_COLUMNS,
    HIGH_MISSING_COLUMNS,
    ZERO_VARIANCE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def load_and_prepare_data(data_path=DATASET_PATH):
    """
    Loads dataset, extracts binary target, removes unneeded columns,
    and converts '?' into np.nan for categorical imputation.
    Preserves 'None' in lab tests as meaningful categories.
    """
    # keep_default_na=False ensures 'None' in A1Cresult/max_glu_serum is NOT coerced to NaN
    df = pd.read_csv(data_path, keep_default_na=False)

    # 1. Define Binary Target Variable
    # <30 -> 1 (Readmission within 30 days)
    # >30 -> 0 (Readmission after 30 days)
    # NO  -> 0 (No readmission recorded)
    y = (df["readmitted"] == "<30").astype(int).values

    # 2. Exclude identifiers, high missingness, and zero variance columns
    excluded_cols = IDENTIFIER_COLUMNS + HIGH_MISSING_COLUMNS + ZERO_VARIANCE_COLUMNS + ["readmitted"]
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    X = df[feature_cols].copy()

    # 3. Detect and convert '?' into np.nan
    # Note: 'None' in max_glu_serum and A1Cresult is retained as valid clinical category
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            # Cast integer-coded categorical features (IDs) to string to prevent numeric misinterpretation
            if X[col].dtype != object:
                X[col] = X[col].astype(str)
            X[col] = X[col].replace("?", np.nan)

    # Ensure numerical features are properly typed as float
    for col in NUMERICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Save selected features metadata
    selected_features_info = {
        "total_predictors": len(feature_cols),
        "numerical_features_count": len(NUMERICAL_FEATURES),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features_count": len(CATEGORICAL_FEATURES),
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": excluded_cols,
    }
    with open(RESULTS_DIR / "selected_features.json", "w") as f:
        json.dump(selected_features_info, f, indent=4)

    return X, y, df


def build_preprocessor(num_cols=NUMERICAL_FEATURES, cat_cols=CATEGORICAL_FEATURES):
    """
    Builds the ColumnTransformer containing:
    1. Numerical pipeline: Mean Imputer -> StandardScaler
    2. Categorical pipeline: Constant 'Missing' Imputer -> OneHotEncoder(handle_unknown='ignore')
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # Ensure output is sparse matrix to preserve memory
    )
    return preprocessor


def split_and_preprocess_data(data_path=DATASET_PATH, test_size=0.20, random_state=RANDOM_STATE):
    """
    Performs stratified 80/20 train/test split, calculates patient overlap,
    and fits preprocessor strictly on the training set to prevent data leakage.
    """
    print("=" * 80)
    print("STEP 2: FEATURE PREPROCESSING AND STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 80)

    X, y, df_raw = load_and_prepare_data(data_path)

    # Stratified 80/20 Train/Test Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Total encounters: {len(y):,d}")
    print(f"Training set: {len(y_train):,d} encounters ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Class 1 (readmitted <30d): {(y_train == 1).sum():,d} ({(y_train == 1).mean()*100:.2f}%)")
    print(f"  Class 0 (not readmitted): {(y_train == 0).sum():,d} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"Test set: {len(y_test):,d} encounters ({len(y_test)/len(y)*100:.1f}%)")
    print(f"  Class 1 (readmitted <30d): {(y_test == 1).sum():,d} ({(y_test == 1).mean()*100:.2f}%)")
    print(f"  Class 0 (not readmitted): {(y_test == 0).sum():,d} ({(y_test == 0).mean()*100:.2f}%)")

    # Patient Overlap Sensitivity Inspection
    train_patients = set(df_raw.loc[idx_train, "patient_nbr"])
    test_patients = set(df_raw.loc[idx_test, "patient_nbr"])
    overlapping_patients = train_patients.intersection(test_patients)
    test_overlap_encounters = df_raw.loc[idx_test, "patient_nbr"].isin(overlapping_patients).sum()
    print("\n--- Patient Overlap Analysis (Repeated Encounters) ---")
    print(f"  Unique patients in training set: {len(train_patients):,d}")
    print(f"  Unique patients in test set: {len(test_patients):,d}")
    print(f"  Patients present in both train and test: {len(overlapping_patients):,d} ({len(overlapping_patients)/len(test_patients)*100:.2f}% of test patients)")
    print(f"  Test encounters from overlapping patients: {test_overlap_encounters:,d} ({test_overlap_encounters/len(y_test)*100:.2f}% of test encounters)")

    # Save train and test encounter IDs for audit and reproducibility
    pd.DataFrame({"encounter_id": df_raw.loc[idx_train, "encounter_id"], "split": "train"}).to_csv(
        RESULTS_DIR / "train_split_encounter_ids.csv", index=False
    )
    pd.DataFrame({"encounter_id": df_raw.loc[idx_test, "encounter_id"], "split": "test"}).to_csv(
        RESULTS_DIR / "test_split_encounter_ids.csv", index=False
    )

    # Fit preprocessor strictly on training data
    preprocessor = build_preprocessor()
    print("\nFitting ColumnTransformer on training data only...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Retrieve feature names from OneHotEncoder
    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    print(f"Transformed Training feature matrix shape: {X_train_trans.shape}")
    print(f"Transformed Test feature matrix shape: {X_test_trans.shape}")
    print(f"Total engineered features after one-hot encoding: {len(all_feature_names)}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "preprocessor": preprocessor,
        "feature_names": all_feature_names,
        "patient_overlap_info": {
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "overlapping_patients": len(overlapping_patients),
            "test_overlap_encounters": int(test_overlap_encounters),
        }
    }


def patient_grouped_split(test_size=0.20, random_state=RANDOM_STATE):
    """
    Performs patient-grouped sensitivity split using GroupShuffleSplit on patient_nbr.
    Ensures 0% patient overlap between train and test to measure potential performance inflation.
    """
    X, y, df_raw = load_and_prepare_data()
    groups = df_raw["patient_nbr"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return {
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "train_patients": len(set(groups[train_idx])),
        "test_patients": len(set(groups[test_idx])),
        "n_train_encounters": len(y_train),
        "n_test_encounters": len(y_test),
    }


if __name__ == "__main__":
    split_and_preprocess_data()
