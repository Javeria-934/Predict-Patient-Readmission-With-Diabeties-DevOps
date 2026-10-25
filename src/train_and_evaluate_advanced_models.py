"""
Module to Train, Evaluate, and Benchmark LightGBM and CatBoost
on the Diabetes 30-Day Hospital Readmission Dataset.
Compares performance directly against the Paper and the reproduced XGBoost baseline.
"""

import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, RANDOM_STATE
from src.preprocessing import split_and_preprocess_data
from src.models import get_model


def train_and_evaluate_advanced(data_dict=None):
    """
    Trains and benchmarks LightGBM and CatBoost.
    Evaluates both unweighted (paper-matching mode) and weighted (cost-sensitive).
    """
    print("=" * 80)
    print("TRAINING & BENCHMARKING ADVANCED MODELS: LIGHTGBM & CATBOOST")
    print("=" * 80)

    adv_dir = RESULTS_DIR / "advanced_models"
    plots_dir = adv_dir / "plots"
    cm_dir = adv_dir / "confusion_matrices"
    models_dir = adv_dir / "saved_models"
    for d in [adv_dir, plots_dir, cm_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if data_dict is None:
        data_dict = split_and_preprocess_data()

    X_train_trans = data_dict["X_train_trans"]
    X_test_trans = data_dict["X_test_trans"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    advanced_models_configs = {
        "lightgbm": {
            "name": "LightGBM",
            "model_builder": lambda w: get_model("lightgbm", y_train=y_train, weighted=w, n_estimators=120, learning_rate=0.08, num_leaves=31)
        },
        "catboost": {
            "name": "CatBoost",
            "model_builder": lambda w: get_model("catboost", y_train=y_train, weighted=w, iterations=150, learning_rate=0.08, depth=5)
        }
    }

    results_records = []
    predictions_dict = {}

    for key, cfg in advanced_models_configs.items():
        disp_name = cfg["name"]
        print(f"\n>>> [1/2] Training & Evaluating {disp_name} (Unweighted, Paper-Matching Mode)...")
        model_unw = cfg["model_builder"](False)

        # 5-fold CV score estimation
        t0 = time.time()
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_trans, y_train):
            m_fold = cfg["model_builder"](False)
            m_fold.fit(X_train_trans[train_idx], y_train[train_idx])
            probs_val = m_fold.predict_proba(X_train_trans[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train[val_idx], probs_val))

        mean_cv_auc = np.mean(cv_scores)
        print(f"  5-Fold CV Mean ROC-AUC: {mean_cv_auc:.4f}")

        # Fit on full training set
        model_unw.fit(X_train_trans, y_train)
        train_time = time.time() - t0
        print(f"  Trained & refitted in {train_time:.2f}s")

        # Save model
        joblib.dump(model_unw, models_dir / f"{key}_unweighted.joblib")

        # Test evaluation
        t_infer = time.time()
        probs = model_unw.predict_proba(X_test_trans)[:, 1]
        infer_time = time.time() - t_infer
        preds = (probs >= 0.5).astype(int)

        predictions_dict[f"{key}_unw"] = {"probs": probs, "preds": preds, "name": f"{disp_name} (Unweighted)"}

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        assert (tn + fp + fn + tp) == len(y_test)
        acc = accuracy_score(y_test, preds)
        auc_roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results_records.append({
            "Model": disp_name,
            "Mode": "Unweighted (Paper-Matching, Thresh 0.5)",
            "Accuracy": round(acc, 4),
            "AUC_ROC": round(auc_roc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Positive_Recall (Sensitivity)": round(rec, 4),
            "Positive_Precision (PPV)": round(prec, 4),
            "Positive_F1": round(f1, 4),
            "Specificity (TNR)": round(spec, 4),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "5Fold_CV_AUC": round(mean_cv_auc, 4),
            "Training_Time_s": round(train_time, 2),
            "Inference_Time_s": round(infer_time, 4),
        })

        # --- Also evaluate Weighted Mode for clinical comparison ---
        print(f"\n>>> [2/2] Training & Evaluating {disp_name} (Cost-Sensitive Weighted Mode)...")
        model_wt = cfg["model_builder"](True)
        t0 = time.time()
        model_wt.fit(X_train_trans, y_train)
        wt_train_time = time.time() - t0
        joblib.dump(model_wt, models_dir / f"{key}_weighted.joblib")

        probs_wt = model_wt.predict_proba(X_test_trans)[:, 1]
        preds_wt = (probs_wt >= 0.5).astype(int)
        predictions_dict[f"{key}_wt"] = {"probs": probs_wt, "preds": preds_wt, "name": f"{disp_name} (Weighted)"}

        tn_w, fp_w, fn_w, tp_w = confusion_matrix(y_test, preds_wt).ravel()
        acc_w = accuracy_score(y_test, preds_wt)
        auc_w = roc_auc_score(y_test, probs_wt)
        pr_w = average_precision_score(y_test, probs_wt)
        rec_w = recall_score(y_test, preds_wt, pos_label=1, zero_division=0)
        prec_w = precision_score(y_test, preds_wt, pos_label=1, zero_division=0)
        f1_w = f1_score(y_test, preds_wt, pos_label=1, zero_division=0)
        spec_w = tn_w / (tn_w + fp_w) if (tn_w + fp_w) > 0 else 0.0

        results_records.append({
            "Model": disp_name,
            "Mode": "Cost-Sensitive Weighted",
            "Accuracy": round(acc_w, 4),
            "AUC_ROC": round(auc_w, 4),
            "PR_AUC": round(pr_w, 4),
            "Positive_Recall (Sensitivity)": round(rec_w, 4),
            "Positive_Precision (PPV)": round(prec_w, 4),
            "Positive_F1": round(f1_w, 4),
            "Specificity (TNR)": round(spec_w, 4),
            "TP": int(tp_w),
            "TN": int(tn_w),
            "FP": int(fp_w),
            "FN": int(fn_w),
            "5Fold_CV_AUC": np.nan,
            "Training_Time_s": round(wt_train_time, 2),
            "Inference_Time_s": round(0.0, 4),
        })

    # Save results
    adv_df = pd.DataFrame(results_records)
    adv_csv = adv_dir / "advanced_models_comparison.csv"
    adv_df.to_csv(adv_csv, index=False)
    print(f"\nSaved advanced models comparison to: {adv_csv}")

    # Generate Confusion Matrix Plots
    for k in ["lightgbm_unw", "catboost_unw"]:
        item = predictions_dict[k]
        cm = confusion_matrix(y_test, item["preds"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Readmitted (0)", "Readmitted <30d (1)"],
                    yticklabels=["Not Readmitted (0)", "Readmitted <30d (1)"])
        plt.title(f"Confusion Matrix: {item['name']}", fontsize=12, pad=12)
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.savefig(cm_dir / f"{k}_cm.png", dpi=300)
        plt.close()

    # Load baseline XGBoost probabilities if available
    xgb_path = RESULTS_DIR / "saved_models" / "xgboost.joblib"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        probs_xgb = xgb_model.predict_proba(X_test_trans)[:, 1]
    else:
        probs_xgb = None

    # ROC Curves Comparison Plot
    plt.figure(figsize=(9, 7))
    if probs_xgb is not None:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, probs_xgb)
        auc_xgb = roc_auc_score(y_test, probs_xgb)
        plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost Baseline (AUC = {auc_xgb:.3f})", color="#2ca02c", linewidth=2.0)

    colors = {"lightgbm_unw": "#00a86b", "catboost_unw": "#e65100"}
    for k in ["lightgbm_unw", "catboost_unw"]:
        item = predictions_dict[k]
        fpr, tpr, _ = roc_curve(y_test, item["probs"])
        auc = roc_auc_score(y_test, item["probs"])
        plt.plot(fpr, tpr, label=f"{item['name']} (AUC = {auc:.3f})", color=colors[k], linewidth=2.2)

    # Reference paper line
    plt.axhline(0.667, color="gray", linestyle=":", label="Paper XGBoost AUC Ref (0.667)", alpha=0.7)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", linewidth=1.2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curves: Advanced Gradient Boosters (LightGBM & CatBoost vs. XGBoost)", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "advanced_roc_comparison.png", dpi=300)
    plt.close()

    # Precision-Recall Curves Plot
    plt.figure(figsize=(9, 7))
    baseline_pr = (y_test == 1).mean()
    plt.axhline(baseline_pr, color="k", linestyle="--", label=f"Baseline Prevalence ({baseline_pr*100:.1f}%)", linewidth=1.2)
    for k in ["lightgbm_unw", "catboost_unw"]:
        item = predictions_dict[k]
        prec, rec, _ = precision_recall_curve(y_test, item["probs"])
        pr_auc = average_precision_score(y_test, item["probs"])
        plt.plot(rec, prec, label=f"{item['name']} (PR-AUC = {pr_auc:.3f})", color=colors[k], linewidth=2.0)

    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall / Sensitivity", fontsize=12)
    plt.ylabel("Precision / PPV", fontsize=12)
    plt.title("Precision-Recall Curves: LightGBM & CatBoost", fontsize=13, pad=12)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "advanced_pr_curves.png", dpi=300)
    plt.close()

    # Build Extended Benchmark Table comparing Paper vs All Models
    build_extended_benchmark(adv_df)

    return adv_df


def build_extended_benchmark(adv_df):
    """
    Combines primary models and new advanced models into an all-inclusive benchmark table.
    """
    primary_csv = RESULTS_DIR / "model_comparison.csv"
    if not primary_csv.exists():
        return

    prim_df = pd.read_csv(primary_csv)

    paper_map = {
        "Logistic Regression": 0.89,
        "Random Forest": 0.89,
        "XGBoost": 0.89,
        "Deep Neural Network (DNN)": 0.84,
        "LightGBM": 0.89,
        "CatBoost": 0.89,
    }

    paper_auc_map = {
        "Logistic Regression": 0.642,
        "Random Forest": 0.630,
        "XGBoost": 0.667,
        "Deep Neural Network (DNN)": 0.579,
        "LightGBM": 0.667,
        "CatBoost": 0.667,
    }

    benchmark_rows = []

    # Add primary models
    for _, row in prim_df.iterrows():
        m_name = row["Model"]
        benchmark_rows.append({
            "Model": m_name,
            "Family": "Paper Model" if "DNN" not in m_name else "Paper Neural Net",
            "Paper_Accuracy_Target": paper_map.get(m_name, 0.89),
            "Reached_Accuracy (Unweighted)": row["Accuracy"],
            "Accuracy_Delta": round(row["Accuracy"] - paper_map.get(m_name, 0.89), 4),
            "Paper_AUC_Ref": paper_auc_map.get(m_name, np.nan),
            "Reached_AUC_ROC": row["AUC_ROC"],
            "PR_AUC": row["PR_AUC"],
            "Positive_Recall": row["Positive_Recall (Sensitivity)"],
            "Positive_Precision": row["Positive_Precision (PPV)"],
            "False_Negatives (FN)": row.get("False_Negatives (FN)", np.nan),
        })

    # Add advanced unweighted models
    unw_adv = adv_df[adv_df["Mode"].str.contains("Unweighted")]
    for _, row in unw_adv.iterrows():
        m_name = row["Model"]
        benchmark_rows.append({
            "Model": f"{m_name} (New)",
            "Family": "Advanced GBDT",
            "Paper_Accuracy_Target": 0.89,
            "Reached_Accuracy (Unweighted)": row["Accuracy"],
            "Accuracy_Delta": round(row["Accuracy"] - 0.89, 4),
            "Paper_AUC_Ref": 0.667,
            "Reached_AUC_ROC": row["AUC_ROC"],
            "PR_AUC": row["PR_AUC"],
            "Positive_Recall": row["Positive_Recall (Sensitivity)"],
            "Positive_Precision": row["Positive_Precision (PPV)"],
            "False_Negatives (FN)": row["FN"],
        })

    ext_df = pd.DataFrame(benchmark_rows)
    ext_csv = RESULTS_DIR / "advanced_models" / "extended_benchmark_table.csv"
    ext_df.to_csv(ext_csv, index=False)
    print(f"Saved extended benchmark table to: {ext_csv}")
    print("\n--- EXTENDED BENCHMARK TABLE (All Models vs Paper) ---")
    print(ext_df[["Model", "Paper_Accuracy_Target", "Reached_Accuracy (Unweighted)", "Accuracy_Delta", "Paper_AUC_Ref", "Reached_AUC_ROC", "Positive_Recall"]].to_string(index=False))


if __name__ == "__main__":
    train_and_evaluate_advanced()
