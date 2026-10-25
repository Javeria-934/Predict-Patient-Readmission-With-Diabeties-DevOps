"""
Master Pipeline Runner for Diabetes 30-Day Hospital Readmission ML Reproduction & Audit.
Paper: Emi-Johnson & Nkrumah (Cureus, April 17, 2025, DOI: 10.7759/cureus.82437).

Orchestrates:
  Step 1: Dataset Inspection & Data Leakage Review
  Step 2: Preprocessing & Stratified 80/20 Train/Test Split (saves encounter IDs)
  Step 3: Primary Reproduction Experiment (Unweighted 5-Fold CV tuning & 0.5 threshold evaluation matching Paper Tables 1-3)
  Step 4: Secondary Sensitivity Experiment (Cost-Sensitive Weighted Clinical Detection Mode)
  Step 5: Patient-Grouped Sensitivity Analysis (GroupShuffleSplit on patient_nbr; 0% patient overlap)
  Step 6: SHAP Interpretability Analysis on Primary XGBoost
  Step 7: 3-Way Discrepancy & Systematic Comparison Table Generation
"""

import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import RESULTS_DIR, SAVED_MODELS_DIR, RANDOM_STATE
from src.data_inspection import inspect_dataset
from src.preprocessing import split_and_preprocess_data, patient_grouped_split
from src.train_models import train_and_tune_models
from src.evaluate_models import evaluate_all_models
from src.shap_analysis import run_shap_analysis
from src.models import get_model


def run_patient_grouped_sensitivity():
    """
    Executes sensitivity analysis with GroupShuffleSplit on patient_nbr to measure
    the impact of repeated patient hospitalizations (38% test patient overlap).
    """
    print("\n" + "=" * 80)
    print("STEP 5: PATIENT-GROUPED SENSITIVITY ANALYSIS (0% Patient Overlap)")
    print("=" * 80)

    pg_dir = RESULTS_DIR / "patient_grouped"
    pg_dir.mkdir(parents=True, exist_ok=True)

    pg_dict = patient_grouped_split()
    X_train_trans = pg_dict["X_train_trans"]
    X_test_trans = pg_dict["X_test_trans"]
    y_train = pg_dict["y_train"]
    y_test = pg_dict["y_test"]

    print(f"  Training encounters: {len(y_train):,d} (Patients: {pg_dict['train_patients']:,d})")
    print(f"  Test encounters: {len(y_test):,d} (Patients: {pg_dict['test_patients']:,d})")
    print("  Patient overlap: 0 patients (Strict patient separation)")

    # Train unweighted XGBoost
    t0 = time.time()
    xgb_pg = get_model("xgboost", y_train=y_train, weighted=False, max_depth=5, learning_rate=0.1, n_estimators=100)
    xgb_pg.fit(X_train_trans, y_train)
    t_train = time.time() - t0

    probs = xgb_pg.predict_proba(X_test_trans)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
    prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)

    pg_results = pd.DataFrame([{
        "Split_Type": "Patient-Grouped Split (Zero Patient Overlap)",
        "Model": "XGBoost (Unweighted, Threshold 0.5)",
        "Train_Encounters": len(y_train),
        "Test_Encounters": len(y_test),
        "Accuracy": round(acc, 4),
        "AUC_ROC": round(auc, 4),
        "Positive_Recall": round(rec, 4),
        "Positive_Precision": round(prec, 4),
        "Positive_F1": round(f1, 4),
        "Training_Time_s": round(t_train, 2),
    }])

    pg_csv_path = pg_dir / "patient_grouped_comparison.csv"
    pg_results.to_csv(pg_csv_path, index=False)
    print(f"  Patient-Grouped XGBoost AUC-ROC: {auc:.4f} | Accuracy: {acc:.4f} | Recall: {rec:.4f}")
    print(f"  Saved patient-grouped results to: {pg_csv_path}")

    return pg_results


def build_three_way_comparison():
    """
    Compiles the 3-Way Systematic Discrepancy Table comparing:
      1. Original Implementation (Initial Run from results_backup_v1)
      2. Corrected Implementation (Primary Paper-Matching Run from results)
      3. Published Paper Reference Values
    """
    print("\n" + "=" * 80)
    print("STEP 7: COMPILING 3-WAY SYSTEMATIC DISCREPANCY & COMPARISON TABLE")
    print("=" * 80)

    paper_data = {
        "Logistic Regression": {"Paper_Accuracy": 0.8900, "Paper_AUC_ROC": 0.6420, "Paper_Recall": 0.0160, "Paper_Precision": 0.4140, "Paper_F1": 0.0300, "Paper_FN": 2249, "Paper_FP": 1276},
        "Random Forest": {"Paper_Accuracy": 0.8900, "Paper_AUC_ROC": 0.6300, "Paper_Recall": 0.0020, "Paper_Precision": 0.3640, "Paper_F1": 0.0040, "Paper_FN": 2280, "Paper_FP": 1224},
        "XGBoost": {"Paper_Accuracy": 0.8900, "Paper_AUC_ROC": 0.6670, "Paper_Recall": 0.0170, "Paper_Precision": 0.4370, "Paper_F1": 0.0330, "Paper_FN": 2244, "Paper_FP": 1172},
        "Deep Neural Network (DNN)": {"Paper_Accuracy": 0.8400, "Paper_AUC_ROC": 0.5790, "Paper_Recall": 0.1430, "Paper_Precision": 0.1860, "Paper_F1": 0.1620, "Paper_FN": 1960, "Paper_FP": 1631},
    }

    # Load corrected primary results
    prim_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv").set_index("Model")
    prim_misc = pd.read_csv(RESULTS_DIR / "misclassification_summary.csv").set_index("Model")

    # Load original results from backup
    orig_path = Path("results_backup_v1") / "model_comparison.csv"
    orig_df = pd.read_csv(orig_path).set_index("Model") if orig_path.exists() else None

    records = []
    for model_name, pref in paper_data.items():
        prim_row = prim_df.loc[model_name] if model_name in prim_df.index else {}
        misc_row = prim_misc.loc[model_name] if model_name in prim_misc.index else {}
        orig_row = orig_df.loc[model_name] if (orig_df is not None and model_name in orig_df.index) else {}

        p_acc = pref["Paper_Accuracy"]
        c_acc = prim_row.get("Accuracy", np.nan)
        o_acc = orig_row.get("Accuracy", np.nan)

        records.append({
            "Model": model_name,
            # Paper Reference
            "Paper_Accuracy": p_acc,
            "Paper_AUC_ROC": pref["Paper_AUC_ROC"],
            "Paper_Recall": pref["Paper_Recall"],
            "Paper_FN": pref["Paper_FN"],
            # Original Implementation (Initial Weighted Run)
            "Orig_Accuracy": o_acc,
            "Orig_AUC_ROC": orig_row.get("AUC_ROC", np.nan),
            "Orig_Recall": orig_row.get("Positive_Recall (Sensitivity)", np.nan),
            # Corrected Implementation (Primary Paper-Matching Mode)
            "Corrected_Accuracy": c_acc,
            "Corrected_AUC_ROC": prim_row.get("AUC_ROC", np.nan),
            "Corrected_Recall": prim_row.get("Positive_Recall (Sensitivity)", np.nan),
            "Corrected_FN": misc_row.get("False_Negatives (FN)", np.nan),
            # Discrepancy (Corrected vs Paper)
            "Delta_Accuracy (Corrected - Paper)": round(c_acc - p_acc, 4) if not np.isnan(c_acc) else np.nan,
            "Delta_AUC_ROC (Corrected - Paper)": round(prim_row.get("AUC_ROC", 0) - pref["Paper_AUC_ROC"], 4),
        })

    comp_df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "three_way_comparison.csv"
    comp_df.to_csv(out_csv, index=False)
    print(f"Saved 3-Way Systematic Discrepancy Table to: {out_csv}")
    print("\n--- 3-WAY COMPARISON TABLE ---")
    print(comp_df[["Model", "Paper_Accuracy", "Orig_Accuracy", "Corrected_Accuracy", "Paper_AUC_ROC", "Corrected_AUC_ROC", "Paper_Recall", "Corrected_Recall"]].to_string(index=False))

    return comp_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce & Audit Diabetes Readmission Pipeline")
    parser.add_argument("--include-advanced", action="store_true", help="Also train and evaluate LightGBM and CatBoost")
    args, _ = parser.parse_known_args()

    start_time = time.time()
    print("=" * 80)
    print("REPRODUCING & AUDITING 30-DAY HOSPITAL READMISSION PREDICTION PIPELINE")
    print("Based on Emi-Johnson & Nkrumah (Cureus, 2025, DOI: 10.7759/cureus.82437)")
    print("=" * 80)

    # 1. Inspect Dataset & Audit Leakage
    df, summary_df, class_dist_df, excluded_df = inspect_dataset()

    # 2. Preprocess & Stratified 80/20 Split (saves encounter IDs)
    data_dict = split_and_preprocess_data()

    # 3. Primary Reproduction Experiment: 5-Fold Stratified CV Grid Search (Unweighted, threshold 0.5)
    train_dict = train_and_tune_models(data_dict, weighted=False, save_dir=RESULTS_DIR)

    # 4. Primary Evaluation on Held-Out Test Set (Paper-Matching Mode)
    eval_dict = evaluate_all_models(
        models_dict=train_dict["trained_models"],
        data_dict=data_dict,
        training_times=train_dict["training_times"],
        output_dir=RESULTS_DIR,
        title_suffix="[Primary Reproduction - Paper Matching]"
    )

    # 5. Secondary Evaluation: Cost-Sensitive Weighted Clinical Detection Mode
    sec_dir = RESULTS_DIR / "secondary_weighted"
    print("\n" + "=" * 80)
    print("EVALUATING SECONDARY SENSITIVITY EXPERIMENT (Cost-Sensitive Weighted)")
    print("=" * 80)
    eval_sec = evaluate_all_models(
        data_dict=data_dict,
        output_dir=sec_dir,
        title_suffix="[Secondary Sensitivity - Cost-Sensitive]"
    )

    # 6. Patient-Grouped Sensitivity Analysis (0% Patient Overlap)
    pg_res = run_patient_grouped_sensitivity()

    # 7. SHAP Interpretability Analysis on Primary XGBoost
    shap_dict = run_shap_analysis(
        xgb_model=train_dict["trained_models"]["xgboost"],
        data_dict=data_dict
    )

    # 8. 3-Way Systematic Discrepancy & Comparison Summary
    three_way_df = build_three_way_comparison()

    # 9. Advanced Models: LightGBM & CatBoost (Optional)
    if args.include_advanced:
        print("\n" + "=" * 80)
        print("STEP 9: ADVANCED MODELS BENCHMARK (LIGHTGBM & CATBOOST)")
        print("=" * 80)
        from src.train_and_evaluate_advanced_models import train_and_evaluate_advanced
        train_and_evaluate_advanced(data_dict)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("FULL AUDIT & REPRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("All results, tables, and publication plots saved to 'results/' directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
