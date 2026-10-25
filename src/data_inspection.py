"""
Data inspection and preliminary analysis module.
Inspects UCI Diabetes 130-US Hospitals dataset, analyzes missingness,
target distribution, patient repeats, and feature properties.
Outputs summary CSVs to the results directory.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATASET_PATH,
    IDS_MAPPING_PATH,
    RESULTS_DIR,
    IDENTIFIER_COLUMNS,
    HIGH_MISSING_COLUMNS,
    ZERO_VARIANCE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def inspect_dataset(data_path=DATASET_PATH):
    """Performs comprehensive data inspection on the raw dataset."""
    print("=" * 80)
    print("STEP 1: DATASET INSPECTION AND INITIAL ANALYSIS")
    print("=" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {data_path}")

    # Read dataset keeping 'None' as a string category rather than default NaN
    df = pd.read_csv(data_path, keep_default_na=False)
    n_rows, n_cols = df.shape
    print(f"Loaded raw dataset from: {data_path}")
    print(f"Dataset dimensions: {n_rows:,} rows, {n_cols} columns")

    # 1. Target Variable Analysis
    print("\n--- 1. Target Variable Distribution (Raw) ---")
    raw_target_counts = df["readmitted"].value_counts()
    raw_target_pcts = df["readmitted"].value_counts(normalize=True) * 100
    for val in raw_target_counts.index:
        print(f"  '{val}': {raw_target_counts[val]:,d} records ({raw_target_pcts[val]:.2f}%)")

    # Binary Target Mapping: <30 -> 1, >30 -> 0, NO -> 0
    binary_target = (df["readmitted"] == "<30").astype(int)
    pos_count = (binary_target == 1).sum()
    neg_count = (binary_target == 0).sum()
    pos_pct = (pos_count / n_rows) * 100
    neg_pct = (neg_count / n_rows) * 100
    print("\n--- Binary Target Mapping (<30 -> 1, >=30 / NO -> 0) ---")
    print(f"  Class 1 (Readmitted <30 days): {pos_count:,d} ({pos_pct:.2f}%)")
    print(f"  Class 0 (Not Readmitted <30 days): {neg_count:,d} ({neg_pct:.2f}%)")
    print(f"  Class Imbalance Ratio (Neg:Pos): {neg_count / pos_count:.2f}:1")

    # Save class distribution
    class_dist_df = pd.DataFrame({
        "Category": ["NO", ">30", "<30", "Binary_Class_0 (No / >30)", "Binary_Class_1 (<30)"],
        "Count": [
            raw_target_counts.get("NO", 0),
            raw_target_counts.get(">30", 0),
            raw_target_counts.get("<30", 0),
            neg_count,
            pos_count,
        ],
        "Percentage": [
            raw_target_pcts.get("NO", 0.0),
            raw_target_pcts.get(">30", 0.0),
            raw_target_pcts.get("<30", 0.0),
            neg_pct,
            pos_pct,
        ],
        "Role": [
            "Original Negative Subcategory",
            "Original Negative Subcategory (>30d readmission)",
            "Original Positive Subcategory (<30d readmission)",
            "Aggregated Negative Class (0)",
            "Target Positive Class (1)"
        ]
    })
    class_dist_path = RESULTS_DIR / "class_distribution.csv"
    class_dist_df.to_csv(class_dist_path, index=False)
    print(f"Saved class distribution to: {class_dist_path}")

    # 2. Patient-Level vs Encounter-Level Analysis
    n_encounters = df["encounter_id"].nunique()
    n_unique_patients = df["patient_nbr"].nunique()
    multi_encounter_patients = (df["patient_nbr"].value_counts() > 1).sum()
    max_encounters = df["patient_nbr"].value_counts().max()
    print("\n--- 2. Patient Encounters & Potential Overlap ---")
    print(f"  Total encounters: {n_encounters:,d}")
    print(f"  Unique patients: {n_unique_patients:,d}")
    print(f"  Patients with multiple encounters: {multi_encounter_patients:,d} ({(multi_encounter_patients/n_unique_patients)*100:.2f}%)")
    print(f"  Maximum encounters for a single patient: {max_encounters}")

    # 3. Missing Value Analysis
    print("\n--- 3. Missing Value & Representation Analysis ---")
    col_summary = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        question_count = (df[col] == "?").sum()
        empty_count = (df[col] == "").sum()
        # Distinguish 'None' in lab tests vs actual missing
        is_lab_none = col in ["max_glu_serum", "A1Cresult"]
        actual_missing_count = question_count + empty_count
        actual_missing_pct = (actual_missing_count / n_rows) * 100
        n_unique = df[col].nunique()
        sample_vals = str(list(df[col].unique()[:3]))

        # Categorization
        if col in IDENTIFIER_COLUMNS:
            feature_role = "Identifier (Excluded)"
        elif col in HIGH_MISSING_COLUMNS:
            feature_role = "High Missingness (Excluded)"
        elif col in ZERO_VARIANCE_COLUMNS:
            feature_role = "Zero Variance (Excluded)"
        elif col == "readmitted":
            feature_role = "Prediction Target"
        elif col in NUMERICAL_FEATURES:
            feature_role = "Continuous Numerical Predictor"
        else:
            feature_role = "Categorical Predictor"

        col_summary.append({
            "Column_Name": col,
            "Data_Type": dtype,
            "Role": feature_role,
            "Question_Mark_Count": question_count,
            "Actual_Missing_Count": actual_missing_count,
            "Missing_Percentage": round(actual_missing_pct, 2),
            "Unique_Values": n_unique,
            "Special_Categories_Preserved": "Yes ('None' means unmeasured)" if is_lab_none else "N/A",
            "Sample_Values": sample_vals,
        })

    summary_df = pd.DataFrame(col_summary)
    summary_path = RESULTS_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved dataset summary to: {summary_path}")

    # 4. Excluded Features Documentation
    print("\n--- 4. Feature Selection & Exclusion Rationale ---")
    excluded_features = [
        {
            "Feature": "encounter_id",
            "Reason": "Unique encounter identifier. Memorization/leakage risk; contains no clinical information.",
            "Category": "Identifier"
        },
        {
            "Feature": "patient_nbr",
            "Reason": "Unique patient identifier. Memorization/leakage risk; non-generalizable across healthcare facilities.",
            "Category": "Identifier"
        },
        {
            "Feature": "weight",
            "Reason": "Missing in 98,569 of 101,766 encounters (96.86% missing). Imputation would fabricate nearly all data.",
            "Category": "Extreme Missingness"
        },
        {
            "Feature": "examide",
            "Reason": "Constant single value ('No') across all 101,766 encounters. Zero variance / zero predictive utility.",
            "Category": "Zero Variance"
        },
        {
            "Feature": "citoglipton",
            "Reason": "Constant single value ('No') across all 101,766 encounters. Zero variance / zero predictive utility.",
            "Category": "Zero Variance"
        },
    ]
    excluded_df = pd.DataFrame(excluded_features)
    excluded_path = RESULTS_DIR / "excluded_features.csv"
    excluded_df.to_csv(excluded_path, index=False)
    for ef in excluded_features:
        print(f"  [EXCLUDED] {ef['Feature']:15s} ({ef['Category']}): {ef['Reason']}")
    print(f"Saved feature exclusions to: {excluded_path}")

    # 5. Clinical Leakage & Discharge Disposition Check
    print("\n--- 5. Discharge Disposition & Clinical Leakage Investigation ---")
    # Expired IDs from IDS_mapping.csv: 11, 19, 20, 21
    expired_ids = [11, 19, 20, 21]
    expired_records = df[df["discharge_disposition_id"].isin(expired_ids)]
    print(f"  Encounter records with discharge status 'Expired' (IDs 11, 19, 20, 21): {len(expired_records):,d}")
    print(f"  Target distribution for expired patients: {dict(expired_records['readmitted'].value_counts())}")
    print("  Note: All expired patients have readmitted=='NO' because deceased patients cannot be readmitted.")
    print("  The paper stated in Methods that expired/hospice patients were excluded, but their test set N=20,354")
    print("  corresponds to exactly 20% of 101,766 (unfiltered). Both aspects are documented for transparent reporting.")

    return df, summary_df, class_dist_df, excluded_df


if __name__ == "__main__":
    inspect_dataset()
