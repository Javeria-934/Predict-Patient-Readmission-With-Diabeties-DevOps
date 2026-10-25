"""
Model Evaluation and Comparison Module.
Evaluates all four trained models on the held-out test set (20,354 encounters).
Computes overall metrics, class-specific metrics, confusion matrices,
misclassification counts, ROC curves, and Precision-Recall curves.
Saves CSV tables and publication-quality plots.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

import joblib
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RESULTS_DIR,
    PLOTS_DIR,
    CONFUSION_MATRICES_DIR,
    SAVED_MODELS_DIR,
)
from src.preprocessing import split_and_preprocess_data
from src.train_models import train_and_tune_models
from src.models import PyTorchDNNClassifier


def evaluate_all_models(models_dict=None, data_dict=None, training_times=None, output_dir=None, title_suffix=""):
    """
    Evaluates each model on the untouched test set, generates evaluation tables,
    and produces publication-quality comparison figures.
    """
    target_results_dir = RESULTS_DIR if output_dir is None else Path(output_dir)
    target_plots_dir = target_results_dir / "plots"
    target_cm_dir = target_results_dir / "confusion_matrices"
    target_models_dir = target_results_dir / "saved_models"
    for d in [target_results_dir, target_plots_dir, target_cm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STEP 4: MODEL EVALUATION ON HELD-OUT TEST SET (N=20,354) {title_suffix}")
    print("=" * 80)

    if data_dict is None:
        data_dict = split_and_preprocess_data()

    if models_dict is None:
        model_files = {
            "logistic_regression": target_models_dir / "logistic_regression.joblib",
            "random_forest": target_models_dir / "random_forest.joblib",
            "xgboost": target_models_dir / "xgboost.joblib",
            "dnn": target_models_dir / "dnn.joblib",
        }
        if all(p.exists() for p in model_files.values()):
            print(f"Loading pre-trained models from {target_models_dir} ...")
            models_dict = {k: joblib.load(p) for k, p in model_files.items()}
        else:
            train_res = train_and_tune_models(data_dict, save_dir=target_results_dir)
            models_dict = train_res["trained_models"]
            training_times = train_res["training_times"]
    if training_times is None:
        import json
        cv_path = target_results_dir / "cv_results.json"
        if cv_path.exists():
            with open(cv_path) as f:
                cv_data = json.load(f)
                training_times = cv_data.get("training_times_seconds", {})
        else:
            training_times = {m: 0.0 for m in models_dict.keys()}

    X_test_trans = data_dict["X_test_trans"]
    y_test = data_dict["y_test"]

    model_names_map = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "dnn": "Deep Neural Network (DNN)",
    }

    # Reference values reported in Emi-Johnson & Nkrumah (Cureus, 2025)
    paper_reference = {
        "Logistic Regression": {"accuracy": 0.89, "roc_auc": 0.642, "class1_f1": 0.030, "fp": 1276, "fn": 2249},
        "Random Forest": {"accuracy": 0.89, "roc_auc": 0.630, "class1_f1": 0.003, "fp": 1224, "fn": 2280},
        "XGBoost": {"accuracy": 0.89, "roc_auc": 0.667, "class1_f1": 0.032, "fp": 1172, "fn": 2244},
        "Deep Neural Network (DNN)": {"accuracy": 0.84, "roc_auc": 0.579, "class1_f1": 0.161, "fp": 1631, "fn": 1960},
    }

    overall_metrics = []
    class_specific_metrics = []
    misclassification_records = []
    predictions_dict = {}

    # Majority class baseline
    majority_preds = np.zeros_like(y_test)
    majority_acc = accuracy_score(y_test, majority_preds)
    print(f"\n--- Baseline: Majority-Class Classifier (Always Predict 0) ---")
    print(f"  Accuracy: {majority_acc:.4f} (Accuracy artifact from 88.84% negative class)")
    print(f"  Recall on Positive Class: 0.0000 (Detects 0 readmissions)")

    # Evaluation loop
    for key, display_name in model_names_map.items():
        model = models_dict[key]
        print(f"\nEvaluating: {display_name}...")

        # Measure inference latency
        t_start = time.time()
        probs = model.predict_proba(X_test_trans)[:, 1]
        inference_time = time.time() - t_start

        preds = (probs >= 0.5).astype(int)
        predictions_dict[key] = {"probs": probs, "preds": preds}

        # Confusion Matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        # Rates and Metrics
        acc = accuracy_score(y_test, preds)
        # Independent Verification: Confusion matrix totals and accuracy
        assert (tn + fp + fn + tp) == len(y_test), f"CM sum {tn+fp+fn+tp} != {len(y_test)}"
        calc_acc = (tp + tn) / len(y_test)
        assert abs(calc_acc - acc) < 1e-6, f"CM accuracy {calc_acc} != sklearn {acc}"

        bal_acc = balanced_accuracy_score(y_test, preds)
        auc_roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        # Binary (Positive Class 1: Readmission <30d)
        pos_prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        pos_rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        pos_f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)

        # Negative Class (Class 0)
        neg_prec = precision_score(y_test, preds, pos_label=0, zero_division=0)
        neg_rec = recall_score(y_test, preds, pos_label=0, zero_division=0)
        neg_f1 = f1_score(y_test, preds, pos_label=0, zero_division=0)

        # Macro and Weighted Averages
        macro_prec = precision_score(y_test, preds, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, preds, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        weighted_prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        weighted_rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Clinical Rates
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        ref = paper_reference.get(display_name, {})

        overall_metrics.append({
            "Model": display_name,
            "Accuracy": round(acc, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
            "AUC_ROC": round(auc_roc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Positive_Precision (PPV)": round(pos_prec, 4),
            "Positive_Recall (Sensitivity)": round(pos_rec, 4),
            "Positive_F1": round(pos_f1, 4),
            "Specificity (TNR)": round(specificity, 4),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Macro_Precision": round(macro_prec, 4),
            "Macro_Recall": round(macro_rec, 4),
            "Macro_F1": round(macro_f1, 4),
            "Weighted_F1": round(weighted_f1, 4),
            "Training_Time_s": training_times.get(key, 0.0),
            "Inference_Time_s": round(inference_time, 4),
            "Paper_Accuracy_Ref": ref.get("accuracy", np.nan),
            "Paper_AUC_ROC_Ref": ref.get("roc_auc", np.nan),
        })

        # Class Specific Breakdown (matching Paper Table 2)
        class_specific_metrics.extend([
            {
                "Model": display_name,
                "Class": 0,
                "Class_Name": "Non-Readmitted (>=30d / NO)",
                "Support": int(tn + fp),
                "Precision": round(neg_prec, 4),
                "Recall": round(neg_rec, 4),
                "F1_Score": round(neg_f1, 4),
            },
            {
                "Model": display_name,
                "Class": 1,
                "Class_Name": "Readmitted <30d",
                "Support": int(tp + fn),
                "Precision": round(pos_prec, 4),
                "Recall": round(pos_rec, 4),
                "F1_Score": round(pos_f1, 4),
            }
        ])

        # Misclassification Summary (matching Paper Table 3)
        misclassification_records.append({
            "Model": display_name,
            "True_Positives (TP)": int(tp),
            "True_Negatives (TN)": int(tn),
            "False_Positives (FP)": int(fp),
            "False_Negatives (FN)": int(fn),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Paper_FP_Ref": ref.get("fp", np.nan),
            "Paper_FN_Ref": ref.get("fn", np.nan),
        })

    # Save CSVs
    overall_df = pd.DataFrame(overall_metrics)
    overall_path = target_results_dir / "model_comparison.csv"
    overall_df.to_csv(overall_path, index=False)

    class_df = pd.DataFrame(class_specific_metrics)
    class_path = target_results_dir / "class_specific_metrics.csv"
    class_df.to_csv(class_path, index=False)

    misc_df = pd.DataFrame(misclassification_records)
    misc_path = target_results_dir / "misclassification_summary.csv"
    misc_df.to_csv(misc_path, index=False)

    print(f"\nSaved overall model comparison to: {overall_path}")
    print(f"Saved class-specific metrics to: {class_path}")
    print(f"Saved misclassification summary to: {misc_path}")

    # Display tables in console
    print("\n--- MODEL COMPARISON TABLE ---")
    print(overall_df[["Model", "Accuracy", "AUC_ROC", "Positive_Precision (PPV)", "Positive_Recall (Sensitivity)", "Positive_F1", "Paper_AUC_ROC_Ref"]].to_string(index=False))

    print("\n--- MISCLASSIFICATION SUMMARY TABLE (vs Paper Table 3) ---")
    print(misc_df[["Model", "False_Positives (FP)", "False_Negatives (FN)", "Paper_FP_Ref", "Paper_FN_Ref"]].to_string(index=False))

    # --------------------------------------------------------------------------
    # Generate Plots
    # --------------------------------------------------------------------------
    _plot_confusion_matrices(model_names_map, predictions_dict, y_test, target_cm_dir, title_suffix)
    _plot_roc_curves(model_names_map, predictions_dict, y_test, target_plots_dir, title_suffix)
    _plot_precision_recall_curves(model_names_map, predictions_dict, y_test, target_plots_dir, title_suffix)

    return {
        "overall_df": overall_df,
        "class_df": class_df,
        "misc_df": misc_df,
        "predictions_dict": predictions_dict,
    }


def _plot_confusion_matrices(models_names_map, predictions_dict, y_test, cm_dir, title_suffix=""):
    """Generates and saves individual and combined confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (key, display_name) in enumerate(models_names_map.items()):
        preds = predictions_dict[key]["preds"]
        cm = confusion_matrix(y_test, preds)

        # Plot individual CM
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Readmitted (0)", "Readmitted <30d (1)"],
            yticklabels=["Not Readmitted (0)", "Readmitted <30d (1)"],
        )
        plt.title(f"Confusion Matrix: {display_name} {title_suffix}".strip(), fontsize=12, pad=12)
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.savefig(cm_dir / f"{key}_cm.png", dpi=300)
        plt.close()

        # Plot on combined subplot
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"],
            cbar=False,
        )
        axes[idx].set_title(display_name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    fig.suptitle(f"Confusion Matrices: 30-Day Hospital Readmission Models (N=20,354) {title_suffix}".strip(), fontsize=15, y=0.98)
    plt.tight_layout()
    fig.savefig(cm_dir / "all_models_cm.png", dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrices to: {cm_dir}")


def _plot_roc_curves(models_names_map, predictions_dict, y_test, plots_dir, title_suffix=""):
    """Generates and saves combined publication-grade ROC curves plot."""
    plt.figure(figsize=(9, 7))
    colors = {"logistic_regression": "#1f77b4", "random_forest": "#ff7f0e", "xgboost": "#2ca02c", "dnn": "#d62728"}

    for key, display_name in models_names_map.items():
        probs = predictions_dict[key]["probs"]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{display_name} (AUC = {auc:.3f})", color=colors[key], linewidth=2.0)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", linewidth=1.2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(f"ROC Curves for All Four Machine Learning Models {title_suffix}".strip(), fontsize=14, pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = plots_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to: {out_path}")


def _plot_precision_recall_curves(models_names_map, predictions_dict, y_test, plots_dir, title_suffix=""):
    """Generates and saves Precision-Recall curves plot highlighting minority class detection."""
    plt.figure(figsize=(9, 7))
    colors = {"logistic_regression": "#1f77b4", "random_forest": "#ff7f0e", "xgboost": "#2ca02c", "dnn": "#d62728"}

    baseline_pr = (y_test == 1).mean()
    plt.axhline(baseline_pr, color="k", linestyle="--", label=f"Baseline Prevalence ({baseline_pr*100:.1f}%)", linewidth=1.2)

    for key, display_name in models_names_map.items():
        probs = predictions_dict[key]["probs"]
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        plt.plot(rec, prec, label=f"{display_name} (PR-AUC = {pr_auc:.3f})", color=colors[key], linewidth=2.0)

    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall / Sensitivity (Positive Class)", fontsize=12)
    plt.ylabel("Precision / Positive Predictive Value", fontsize=12)
    plt.title(f"Precision-Recall Curves for 30-Day Readmission Models {title_suffix}".strip(), fontsize=14, pad=12)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = plots_dir / "precision_recall_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall curves to: {out_path}")


if __name__ == "__main__":
    evaluate_all_models()
