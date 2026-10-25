"""
Model Training and Hyperparameter Tuning Module.
Performs 5-fold Stratified Cross-Validation with Grid Search on training data,
selects optimal hyperparameters based on ROC-AUC, refits models on the full
training set, records training duration, and saves model artifacts.
"""

import sys
import os
import time
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_STATE,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.preprocessing import build_preprocessor, split_and_preprocess_data
from src.models import get_model, PyTorchDNNClassifier


def train_and_tune_models(data_dict=None, weighted=False, save_dir=None, retrain=False):
    """
    Executes 5-fold cross-validation grid search for all four models
    on the training data only, refits on full training set, and saves results.
    weighted=False: Primary Reproduction Experiment (matches paper Table 1-3).
    weighted=True: Secondary Sensitivity Experiment (cost-sensitive clinical detection).
    """
    mode_str = "WEIGHTED (Cost-Sensitive)" if weighted else "UNWEIGHTED (Paper-Matching / Standard 0.5 Threshold)"
    print("=" * 80, flush=True)
    print(f"STEP 3: 5-FOLD STRATIFIED CROSS-VALIDATION & HYPERPARAMETER TUNING [{mode_str}]", flush=True)
    print("=" * 80, flush=True)

    if save_dir is None:
        target_results_dir = RESULTS_DIR if not weighted else (RESULTS_DIR / "secondary_weighted")
    else:
        target_results_dir = Path(save_dir)
    target_models_dir = target_results_dir / "saved_models"
    target_models_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "logistic_regression": target_models_dir / "logistic_regression.joblib",
        "random_forest": target_models_dir / "random_forest.joblib",
        "xgboost": target_models_dir / "xgboost.joblib",
        "dnn": target_models_dir / "dnn.joblib",
    }
    if not retrain and all(p.exists() for p in model_files.values()):
        print(f"Loading pre-trained models from {target_models_dir} (use retrain=True to force re-search)...", flush=True)
        trained_models = {k: joblib.load(p) for k, p in model_files.items()}
        cv_path = target_results_dir / "cv_results.json"
        cv_scores = {}
        training_times = {}
        if cv_path.exists():
            with open(cv_path) as f:
                cv_data = json.load(f)
                cv_scores = cv_data.get("best_cv_scores", {})
                training_times = cv_data.get("training_times_seconds", {})
        params_path = target_results_dir / "best_parameters.json"
        best_params = {}
        if params_path.exists():
            with open(params_path) as f:
                best_params = json.load(f)
        return {
            "trained_models": trained_models,
            "best_params": best_params,
            "cv_scores": cv_scores,
            "training_times": training_times,
        }

    if data_dict is None:
        data_dict = split_and_preprocess_data()

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_train_trans = data_dict["X_train_trans"]
    preprocessor = data_dict["preprocessor"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_params_all = {}
    cv_scores_all = {}
    training_times = {}
    trained_models = {}

    # --------------------------------------------------------------------------
    # Model 1: Logistic Regression
    # --------------------------------------------------------------------------
    print("\n>>> Tuning Model 1: Logistic Regression (5-Fold CV)...", flush=True)
    lr_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
    }
    t0 = time.time()
    lr_base = get_model("logistic_regression", weighted=weighted)
    lr_search = GridSearchCV(
        lr_base,
        lr_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    lr_search.fit(X_train_trans, y_train)
    lr_time = time.time() - t0

    print(f"  Best Logistic Regression params: {lr_search.best_params_}", flush=True)
    print(f"  Best 5-Fold CV ROC-AUC: {lr_search.best_score_:.4f}", flush=True)
    print(f"  Tuning & Refit Time: {lr_time:.2f}s", flush=True)

    best_params_all["logistic_regression"] = lr_search.best_params_
    cv_scores_all["logistic_regression"] = float(lr_search.best_score_)
    training_times["logistic_regression"] = round(lr_time, 2)
    trained_models["logistic_regression"] = lr_search.best_estimator_

    # --------------------------------------------------------------------------
    # Model 2: Random Forest
    # --------------------------------------------------------------------------
    print("\n>>> Tuning Model 2: Random Forest (5-Fold CV)...", flush=True)
    rf_grid = {
        "max_depth": [10, 15],
        "min_samples_split": [5, 10],
    }
    t0 = time.time()
    rf_base = get_model("random_forest", weighted=weighted, n_estimators=100)
    rf_search = GridSearchCV(
        rf_base,
        rf_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    rf_search.fit(X_train_trans, y_train)
    rf_time = time.time() - t0

    print(f"  Best Random Forest params: {rf_search.best_params_}", flush=True)
    print(f"  Best 5-Fold CV ROC-AUC: {rf_search.best_score_:.4f}", flush=True)
    print(f"  Tuning & Refit Time: {rf_time:.2f}s", flush=True)

    best_params_all["random_forest"] = rf_search.best_params_
    cv_scores_all["random_forest"] = float(rf_search.best_score_)
    training_times["random_forest"] = round(rf_time, 2)
    trained_models["random_forest"] = rf_search.best_estimator_

    # --------------------------------------------------------------------------
    # Model 3: XGBoost
    # --------------------------------------------------------------------------
    print("\n>>> Tuning Model 3: XGBoost (5-Fold CV)...", flush=True)
    spw = float((y_train == 0).sum() / (y_train == 1).sum()) if weighted else 1.0
    print(f"  XGBoost scale_pos_weight: {spw:.4f}", flush=True)
    xgb_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    t0 = time.time()
    xgb_base = get_model("xgboost", y_train=y_train, weighted=weighted, n_estimators=100)
    xgb_search = GridSearchCV(
        xgb_base,
        xgb_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    xgb_search.fit(X_train_trans, y_train)
    xgb_time = time.time() - t0

    print(f"  Best XGBoost params: {xgb_search.best_params_}", flush=True)
    print(f"  Best 5-Fold CV ROC-AUC: {xgb_search.best_score_:.4f}", flush=True)
    print(f"  Tuning & Refit Time: {xgb_time:.2f}s", flush=True)

    best_params_all["xgboost"] = {**xgb_search.best_params_, "scale_pos_weight": round(spw, 4)}
    cv_scores_all["xgboost"] = float(xgb_search.best_score_)
    training_times["xgboost"] = round(xgb_time, 2)
    trained_models["xgboost"] = xgb_search.best_estimator_

    # --------------------------------------------------------------------------
    # Model 4: Deep Neural Network (DNN)
    # --------------------------------------------------------------------------
    print("\n>>> Tuning Model 4: Deep Neural Network (DNN)...", flush=True)
    pw = float((y_train == 0).sum() / (y_train == 1).sum()) if weighted else None
    dnn_configs = [
        {"hidden_layer_sizes": (128, 64, 32), "lr": 0.001, "dropout_rate": 0.2, "epochs": 8, "pos_weight": pw},
        {"hidden_layer_sizes": (64, 32), "lr": 0.001, "dropout_rate": 0.1, "epochs": 8, "pos_weight": pw},
    ]

    t0 = time.time()
    best_dnn_score = -1.0
    best_dnn_cfg = None

    cv_dnn = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for cfg_idx, cfg in enumerate(dnn_configs):
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_dnn.split(X_train_trans, y_train)):
            fold_dnn = PyTorchDNNClassifier(**cfg)
            fold_dnn.fit(X_train_trans[train_idx], y_train[train_idx])
            val_probs = fold_dnn.predict_proba(X_train_trans[val_idx])[:, 1]
            fold_auc = roc_auc_score(y_train[val_idx], val_probs)
            fold_scores.append(fold_auc)

        mean_cv = np.mean(fold_scores)
        print(f"  DNN Config {cfg_idx+1} {cfg['hidden_layer_sizes']} Mean CV ROC-AUC: {mean_cv:.4f}", flush=True)
        if mean_cv > best_dnn_score:
            best_dnn_score = mean_cv
            best_dnn_cfg = cfg

    print(f"  Refitting best DNN configuration {best_dnn_cfg['hidden_layer_sizes']} on full training set...", flush=True)
    best_dnn = PyTorchDNNClassifier(**best_dnn_cfg)
    best_dnn.fit(X_train_trans, y_train)
    dnn_time = time.time() - t0

    print(f"  Best DNN CV ROC-AUC: {best_dnn_score:.4f}", flush=True)
    print(f"  Tuning & Refit Time: {dnn_time:.2f}s", flush=True)

    best_dnn_json = dict(best_dnn_cfg)
    best_dnn_json["hidden_layer_sizes"] = list(best_dnn_json["hidden_layer_sizes"])
    best_params_all["dnn"] = best_dnn_json
    cv_scores_all["dnn"] = float(best_dnn_score)
    training_times["dnn"] = round(dnn_time, 2)
    trained_models["dnn"] = best_dnn

    # --------------------------------------------------------------------------
    # Save Metadata and Artifacts
    # --------------------------------------------------------------------------
    with open(target_results_dir / "best_parameters.json", "w") as f:
        json.dump(best_params_all, f, indent=4)

    cv_results_payload = {
        "scoring_metric": "roc_auc",
        "cv_folds": 5,
        "dnn_cv_folds": 3,
        "best_cv_scores": cv_scores_all,
        "training_times_seconds": training_times,
    }
    with open(target_results_dir / "cv_results.json", "w") as f:
        json.dump(cv_results_payload, f, indent=4)

    joblib.dump(preprocessor, target_models_dir / "preprocessor.joblib")
    joblib.dump(trained_models["logistic_regression"], target_models_dir / "logistic_regression.joblib")
    joblib.dump(trained_models["random_forest"], target_models_dir / "random_forest.joblib")
    joblib.dump(trained_models["xgboost"], target_models_dir / "xgboost.joblib")
    joblib.dump(trained_models["dnn"], target_models_dir / "dnn.joblib")

    print(f"\nSaved best parameters to: {target_results_dir / 'best_parameters.json'}", flush=True)
    print(f"Saved CV results to: {target_results_dir / 'cv_results.json'}", flush=True)
    print(f"Saved trained models to: {target_models_dir}", flush=True)

    return {
        "trained_models": trained_models,
        "best_params": best_params_all,
        "cv_scores": cv_scores_all,
        "training_times": training_times,
    }


if __name__ == "__main__":
    train_and_tune_models()
