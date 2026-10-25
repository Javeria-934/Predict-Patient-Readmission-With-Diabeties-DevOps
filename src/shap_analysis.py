"""
SHAP Interpretability Analysis Module for XGBoost Readmission Model.
Computes TreeExplainer SHAP values on the held-out test data,
generates SHAP summary beeswarm plot, global feature importance ranking bar chart,
local individual prediction explanation, and exports numerical importance CSV.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SHAP_DIR,
    SAVED_MODELS_DIR,
    RANDOM_STATE,
)
from src.preprocessing import split_and_preprocess_data


def run_shap_analysis(xgb_model=None, data_dict=None, max_display=20, n_samples=2000):
    """
    Executes SHAP TreeExplainer analysis on XGBoost.
    Uses a representative sample of test encounters (n=2,000) for fast,
    accurate computation and clean, interpretable visualizations.
    """
    print("=" * 80)
    print("STEP 5: SHAP INTERPRETABILITY ANALYSIS ON XGBOOST MODEL")
    print("=" * 80)

    if data_dict is None:
        data_dict = split_and_preprocess_data()

    if xgb_model is None:
        xgb_path = SAVED_MODELS_DIR / "xgboost.joblib"
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            print(f"Loaded trained XGBoost model from: {xgb_path}")
        else:
            raise FileNotFoundError(f"Trained XGBoost model not found at {xgb_path}. Run training first.")

    X_test_trans = data_dict["X_test_trans"]
    y_test = data_dict["y_test"]
    feature_names = data_dict["feature_names"]

    # Sample a representative subset of test data for SHAP computation
    np.random.seed(RANDOM_STATE)
    if X_test_trans.shape[0] > n_samples:
        sample_indices = np.random.choice(X_test_trans.shape[0], size=n_samples, replace=False)
    else:
        sample_indices = np.arange(X_test_trans.shape[0])

    X_sample_sparse = X_test_trans[sample_indices]
    X_sample_dense = X_sample_sparse.toarray()
    y_sample = y_test[sample_indices]

    print(f"Computing SHAP values for {len(sample_indices):,d} test encounters using TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample_dense)

    # In binary XGBoost, shap_values is a 2D array (n_samples, n_features) representing log-odds impact
    if isinstance(shap_values, list):
        shap_values_matrix = shap_values[1]  # positive class
    else:
        shap_values_matrix = shap_values

    print(f"SHAP values matrix shape: {shap_values_matrix.shape}")

    # 1. Global Feature Importance Calculation: Mean Absolute SHAP Value
    mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    importance_csv_path = SHAP_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"Saved numerical feature importance ranking to: {importance_csv_path}")

    print("\n--- TOP 15 INFLUENTIAL READMISSION PREDICTORS (SHAP) ---")
    print(importance_df.head(15).to_string(index=True))

    # 2. SHAP Summary Beeswarm Plot
    print("\nGenerating SHAP Summary Beeswarm Plot...")
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values_matrix,
        X_sample_dense,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary Plot: Top {max_display} Predictors for 30-Day Readmission (XGBoost)", fontsize=13, pad=12)
    plt.xlabel("SHAP value (impact on model log-odds of readmission)", fontsize=11)
    plt.tight_layout()
    summary_plot_path = SHAP_DIR / "shap_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot to: {summary_plot_path}")

    # 3. Global Importance Bar Chart
    print("Generating SHAP Global Importance Bar Chart...")
    plt.figure(figsize=(10, 7))
    top_n = importance_df.head(max_display).iloc[::-1]  # Reverse for horizontal bar chart
    plt.barh(range(len(top_n)), top_n["Mean_Abs_SHAP"], color="#2ca02c", align="center")
    plt.yticks(range(len(top_n)), top_n["Feature"], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Prediction Magnitude)", fontsize=11)
    plt.title(f"Top {max_display} Features by Mean Absolute SHAP Value", fontsize=13, pad=12)
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    bar_plot_path = SHAP_DIR / "shap_bar_importance.png"
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"Saved SHAP bar importance to: {bar_plot_path}")

    # 4. Local Individual Explanation (Waterfall plot for a high-risk readmitted patient)
    print("Generating Local Explanation for an Individual Encounter...")
    # Find a patient who was readmitted (y_sample == 1) and had high predicted risk
    sample_preds = xgb_model.predict_proba(X_sample_sparse)[:, 1]
    high_risk_candidates = np.where((y_sample == 1) & (sample_preds > 0.6))[0]
    local_idx = high_risk_candidates[0] if len(high_risk_candidates) > 0 else 0

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]

    explanation = shap.Explanation(
        values=shap_values_matrix[local_idx],
        base_values=base_val,
        data=X_sample_dense[local_idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(11, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"Local SHAP Explanation: Patient Encounter #{local_idx} (Predicted Readmission Risk: {sample_preds[local_idx]:.2f})", fontsize=12, pad=12)
    plt.tight_layout()
    waterfall_path = SHAP_DIR / "shap_local_waterfall.png"
    plt.savefig(waterfall_path, dpi=300)
    plt.close()
    print(f"Saved individual local explanation waterfall to: {waterfall_path}")

    return {
        "importance_df": importance_df,
        "shap_values": shap_values_matrix,
        "summary_plot_path": summary_plot_path,
        "bar_plot_path": bar_plot_path,
        "waterfall_path": waterfall_path,
    }


if __name__ == "__main__":
    run_shap_analysis()
