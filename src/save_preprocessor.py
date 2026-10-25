"""
Utility to fit and serialize the preprocessor on the training data split.
Saves preprocessor.joblib and feature metadata for low-latency API inference.
"""

import sys
import joblib
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR
from src.preprocessing import split_and_preprocess_data

def save_fitted_preprocessor():
    print("Fitting and serializing preprocessor...")
    data_dict = split_and_preprocess_data()
    preprocessor = data_dict["preprocessor"]
    feature_names = data_dict["feature_names"]
    
    out_path = RESULTS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, out_path)
    print(f"Saved fitted ColumnTransformer to: {out_path}")
    
    # Save feature names list
    names_path = RESULTS_DIR / "feature_names.joblib"
    joblib.dump(feature_names, names_path)
    print(f"Saved feature names list ({len(feature_names)} features) to: {names_path}")

if __name__ == "__main__":
    save_fitted_preprocessor()
