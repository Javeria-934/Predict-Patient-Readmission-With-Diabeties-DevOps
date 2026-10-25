"""
Flask Backend Service for Diabetes 30-Day Hospital Readmission ML Service.
Serves preprocessed models, real-time clinical risk scoring,
local feature contributions, and research benchmark analytics.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RESULTS_DIR

# Initialize Flask app
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist" / "frontend" / "browser"
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST) if FRONTEND_DIST.exists() else None,
    static_url_path="/"
)

# Enable CORS for Angular development server (http://localhost:4200)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global resource cache
CACHE = {
    "preprocessor": None,
    "feature_names": None,
    "models": {},
}

def load_resources():
    """Loads preprocessor, feature names, and pre-trained models into memory."""
    if CACHE["preprocessor"] is None:
        prep_path = RESULTS_DIR / "preprocessor.joblib"
        if prep_path.exists():
            CACHE["preprocessor"] = joblib.load(prep_path)
            print("Loaded preprocessor.joblib")
        else:
            raise FileNotFoundError(f"Missing {prep_path}")

    if CACHE["feature_names"] is None:
        names_path = RESULTS_DIR / "feature_names.joblib"
        if names_path.exists():
            CACHE["feature_names"] = joblib.load(names_path)

    models_to_load = {
        ("lightgbm", "unweighted"): RESULTS_DIR / "advanced_models" / "saved_models" / "lightgbm_unweighted.joblib",
        ("lightgbm", "cost_sensitive"): RESULTS_DIR / "advanced_models" / "saved_models" / "lightgbm_weighted.joblib",
        ("catboost", "unweighted"): RESULTS_DIR / "advanced_models" / "saved_models" / "catboost_unweighted.joblib",
        ("catboost", "cost_sensitive"): RESULTS_DIR / "advanced_models" / "saved_models" / "catboost_weighted.joblib",
        ("xgboost", "unweighted"): RESULTS_DIR / "saved_models" / "xgboost.joblib",
        ("random_forest", "unweighted"): RESULTS_DIR / "saved_models" / "random_forest.joblib",
        ("logistic_regression", "unweighted"): RESULTS_DIR / "saved_models" / "logistic_regression.joblib",
    }

    for (m_name, m_mode), m_path in models_to_load.items():
        if (m_name, m_mode) not in CACHE["models"] and m_path.exists():
            try:
                CACHE["models"][(m_name, m_mode)] = joblib.load(m_path)
                print(f"Loaded model: {m_name} ({m_mode})")
            except Exception as e:
                print(f"Warning loading {m_name} ({m_mode}): {e}")

# Helper functions
def patient_to_dataframe(data: dict) -> pd.DataFrame:
    row = {}
    for col in NUMERICAL_FEATURES:
        try:
            row[col] = float(data.get(col, 0))
        except (ValueError, TypeError):
            row[col] = 0.0

    for col in CATEGORICAL_FEATURES:
        val = data.get(col, "Missing")
        row[col] = str(val) if val is not None else "Missing"

    return pd.DataFrame([row])


def get_risk_tier(prob: float) -> str:
    if prob < 0.20:
        return "Low"
    elif prob < 0.45:
        return "Moderate"
    elif prob < 0.70:
        return "High"
    else:
        return "Critical"


def compute_quick_contributions(X_trans, model, model_name: str):
    feature_names = CACHE.get("feature_names", [])
    if not feature_names:
        return []

    contributions = []
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        cx = X_trans.tocoo()
        for i, val in zip(cx.col, cx.data):
            if i < len(coefs):
                impact = coefs[i] * val
                if abs(impact) > 0.01:
                    contributions.append((feature_names[i], float(impact)))

    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        cx = X_trans.tocoo()
        for i, val in zip(cx.col, cx.data):
            if i < len(importances):
                imp = importances[i] * val
                if imp > 0:
                    fname = feature_names[i]
                    is_risk_increasing = any(k in fname for k in ["inpatient", "emergency", ">8", "time_in_hospital", "Up"])
                    sign = 1.0 if is_risk_increasing else -0.5
                    contributions.append((fname, float(imp * sign)))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_entries = contributions[:8]

    clean_labels = {
        "number_inpatient": "Prior Inpatient Admissions",
        "time_in_hospital": "Days in Hospital",
        "number_emergency": "Emergency Room Visits",
        "num_medications": "Prescribed Medications Count",
        "num_lab_procedures": "Lab Procedures Performed",
        "number_diagnoses": "Total Diagnoses Count",
        "A1Cresult_>8": "Elevated HbA1c (>8%)",
        "A1Cresult_Norm": "Normal HbA1c",
        "insulin_Up": "Insulin Dose Increased",
        "insulin_Down": "Insulin Dose Reduced",
        "insulin_Steady": "Insulin Dose Maintained",
        "change_Ch": "Medication Dosage Changed",
        "discharge_disposition_id_1": "Discharged Home",
        "discharge_disposition_id_3": "Discharged to SNF/Nursing Facility",
        "discharge_disposition_id_22": "Transferred to Rehab",
    }

    results = []
    for fname, val in top_entries:
        label = clean_labels.get(fname, fname.replace("_", " ").title())
        direction = "increases_risk" if val >= 0 else "reduces_risk"
        results.append({
            "feature": fname,
            "label": label,
            "impact": round(float(val), 4),
            "direction": direction
        })
    return results


# ------------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    load_resources()
    return jsonify({
        "status": "healthy",
        "server": "Flask",
        "models_cached": [f"{m[0]}:{m[1]}" for m in CACHE["models"].keys()],
        "preprocessor_ready": CACHE["preprocessor"] is not None
    })


@app.route("/api/presets", methods=["GET"])
def presets():
    return jsonify([
        {
            "id": "high_risk_frequent_admit",
            "title": "High Risk: Frequent Inpatient Admission",
            "description": "Patient with 3 inpatient stays in prior year, elevated HbA1c >8%, and increased insulin dose.",
            "data": {
                "race": "AfricanAmerican",
                "gender": "Male",
                "age": "[60-70)",
                "time_in_hospital": 7,
                "num_lab_procedures": 58,
                "num_procedures": 1,
                "num_medications": 21,
                "number_inpatient": 3,
                "number_emergency": 2,
                "number_outpatient": 1,
                "number_diagnoses": 9,
                "A1Cresult": ">8",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes",
                "diag_1": "250.02",
                "diag_2": "428",
                "diag_3": "401"
            }
        },
        {
            "id": "moderate_risk_cardiac",
            "title": "Moderate Risk: Elderly Cardiac Patient",
            "description": "Elderly patient (70-80) with 1 prior admission, 15 medications, stable glucose.",
            "data": {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[70-80)",
                "time_in_hospital": 4,
                "num_lab_procedures": 45,
                "num_procedures": 2,
                "num_medications": 15,
                "number_inpatient": 1,
                "number_emergency": 0,
                "number_outpatient": 0,
                "number_diagnoses": 7,
                "A1Cresult": "Norm",
                "insulin": "Steady",
                "change": "No",
                "diabetesMed": "Yes",
                "diag_1": "414",
                "diag_2": "250.00",
                "diag_3": "401"
            }
        },
        {
            "id": "low_risk_routine",
            "title": "Low Risk: First-Time Elective Care",
            "description": "Middle-aged patient with zero prior hospitalizations, normal labs, short stay.",
            "data": {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[40-50)",
                "time_in_hospital": 2,
                "num_lab_procedures": 32,
                "num_procedures": 0,
                "num_medications": 8,
                "number_inpatient": 0,
                "number_emergency": 0,
                "number_outpatient": 0,
                "number_diagnoses": 4,
                "A1Cresult": "Norm",
                "insulin": "No",
                "change": "No",
                "diabetesMed": "Yes",
                "diag_1": "250.00",
                "diag_2": "401",
                "diag_3": "272"
            }
        }
    ])


@app.route("/api/predict", methods=["POST"])
def predict():
    load_resources()
    payload = request.get_json(force=True) or {}
    model_name = payload.get("model", "lightgbm").lower()
    mode = payload.get("mode", "cost_sensitive").lower()
    threshold = float(payload.get("threshold", 0.5))
    patient_data = payload.get("patient", {})

    key = (model_name, mode)
    model = CACHE["models"].get(key)
    if model is None:
        key = (model_name, "unweighted")
        model = CACHE["models"].get(key)
    if model is None:
        key = ("lightgbm", "unweighted")
        model = CACHE["models"].get(key)

    if model is None:
        return jsonify({"error": "Requested model could not be loaded"}), 500

    df_row = patient_to_dataframe(patient_data)
    preprocessor = CACHE["preprocessor"]
    X_trans = preprocessor.transform(df_row)

    probs = model.predict_proba(X_trans)[0]
    prob_positive = float(probs[1])
    pred = int(prob_positive >= threshold)
    risk_tier = get_risk_tier(prob_positive)
    top_factors = compute_quick_contributions(X_trans, model, model_name)

    return jsonify({
        "model": "Clinical Risk Engine",
        "mode": "standard_clinical",
        "threshold": threshold,
        "probability": round(prob_positive, 4),
        "prediction": pred,
        "readmitted_label": "Readmitted (<30 Days)" if pred == 1 else "Not Readmitted / >30 Days",
        "risk_tier": risk_tier,
        "risk_percentage": round(prob_positive * 100, 1),
        "top_factors": top_factors
    })


@app.route("/api/benchmarks", methods=["GET"])
def benchmarks():
    model_name_map = {
        "LightGBM (New)": "Primary Clinical Gradient Booster",
        "CatBoost (New)": "Symmetric Gradient Booster",
        "XGBoost": "Standard Gradient Booster",
        "Random Forest": "Randomized Decision Forest",
        "Logistic Regression": "Multivariate Logistic Classifier",
        "Deep Neural Network (DNN)": "Deep Neural Network",
    }
    adv_csv = RESULTS_DIR / "advanced_models" / "extended_benchmark_table.csv"
    if adv_csv.exists():
        df = pd.read_csv(adv_csv)
        df["Model"] = df["Model"].replace(model_name_map)
        df = df.fillna("-")
        return jsonify(df.to_dict(orient="records"))

    three_csv = RESULTS_DIR / "three_way_comparison.csv"
    if three_csv.exists():
        df = pd.read_csv(three_csv)
        df["Model"] = df["Model"].replace(model_name_map)
        df = df.fillna("-")
        return jsonify(df.to_dict(orient="records"))

    return jsonify({"error": "Benchmark data not found"}), 404


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    load_resources()
    payload = request.get_json(force=True) or {}
    model_name = payload.get("model", "lightgbm").lower()
    mode = payload.get("mode", "cost_sensitive").lower()
    threshold = float(payload.get("threshold", 0.5))
    patients = payload.get("patients", [])

    if not patients:
        return jsonify({"total_screened": 0, "tier_summary": {}, "patients": []})

    key = (model_name, mode)
    model = CACHE["models"].get(key) or CACHE["models"].get(("lightgbm", "unweighted"))

    rows = [patient_to_dataframe(p) for p in patients]
    df_all = pd.concat(rows, ignore_index=True)
    X_trans = CACHE["preprocessor"].transform(df_all)

    probs = model.predict_proba(X_trans)[:, 1]
    results = []
    tier_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}

    for idx, (patient, prob) in enumerate(zip(patients, probs)):
        prob_val = float(prob)
        tier = get_risk_tier(prob_val)
        tier_counts[tier] += 1
        pred = int(prob_val >= threshold)

        results.append({
            "patient_index": idx + 1,
            "age": patient.get("age", "[60-70)"),
            "gender": patient.get("gender", "Female"),
            "time_in_hospital": patient.get("time_in_hospital", 3),
            "inpatient_visits": patient.get("number_inpatient", 0),
            "probability": round(prob_val, 4),
            "prediction": pred,
            "risk_tier": tier
        })

    return jsonify({
        "total_screened": len(patients),
        "tier_summary": tier_counts,
        "high_or_critical_percentage": round(((tier_counts["High"] + tier_counts["Critical"]) / len(patients)) * 100, 1),
        "patients": results
    })


# Serve Angular static bundle in production
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_angular(path):
    if FRONTEND_DIST.exists():
        target = FRONTEND_DIST / path
        if path != "" and target.exists():
            return send_from_directory(str(FRONTEND_DIST), path)
        return send_from_directory(str(FRONTEND_DIST), "index.html")
    return jsonify({"message": "Clinical ML API is operational. Angular frontend not yet built to dist/."})


if __name__ == "__main__":
    load_resources()
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
