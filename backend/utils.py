import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any
import sys
import __main__
from ml.preprocess import Preprocessor

__main__.Preprocessor = Preprocessor

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "churn_model.joblib"))
BUNDLE = None

def get_bundle():
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = joblib.load(MODEL_PATH)
    return BUNDLE

def predict_single(data: Dict[str, Any]) -> Dict[str, Any]:
    bundle = get_bundle()
    expected_order = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
        "MonthlyCharges", "TotalCharges"
    ]
    
    # Create DataFrame and REORDER columns immediately
    df = pd.DataFrame([data])[expected_order]
    
    preprocessed = bundle["preprocessor"].transform(df)
    print("NaNs after transform:", preprocessed.isna().sum().sum())
    if preprocessed.isna().any().any():
        print(preprocessed.columns[preprocessed.isna().any()].tolist())
    features = bundle["features"]
    X = preprocessed[features].copy()
    
    cat_cols = bundle["preprocessor"].all_cat_cols
    te = bundle["te"]
    
    # LGB / XGB path (target-encoded)
    X_te = X.copy()
    X_te[cat_cols] = te.transform(X_te[cat_cols])
    X_num = X_te[bundle["cols"]]
    
    # CatBoost path
    X_cat = X.copy()
    for c in cat_cols:
        if c in X_cat.columns:
            X_cat[c] = X_cat[c].astype(str)
    
    # Ensemble averages
    lgb_p = np.mean([m.predict_proba(X_num)[:, 1] for m in bundle["lgb_models"]], axis=0)
    xgb_p = np.mean([m.predict_proba(X_num)[:, 1] for m in bundle["xgb_models"]], axis=0)
    cat_p = np.mean([m.predict_proba(X_cat)[:, 1] for m in bundle["cat_models"]], axis=0)
    
    base = np.column_stack([lgb_p, xgb_p, cat_p])
    prob = bundle["meta_model"].predict_proba(base)[:, 1][0]
    
    return {
        "churn_probability": round(float(prob), 4),
        "prediction": "Churn" if prob > 0.5 else "No Churn",
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    }

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    bundle = get_bundle()
    preprocessed = bundle["preprocessor"].transform(df)
    features = bundle["features"]
    X = preprocessed[features].copy()
    
    cat_cols = bundle["preprocessor"].all_cat_cols
    te = bundle["te"]
    
    X_te = X.copy()
    X_te[cat_cols] = te.transform(X_te[cat_cols])
    X_num = X_te[bundle["cols"]]
    
    X_cat = X.copy()
    for c in cat_cols:
        if c in X_cat.columns:
            X_cat[c] = X_cat[c].astype(str)
    
    lgb_p = np.mean([m.predict_proba(X_num)[:, 1] for m in bundle["lgb_models"]], axis=0)
    xgb_p = np.mean([m.predict_proba(X_num)[:, 1] for m in bundle["xgb_models"]], axis=0)
    cat_p = np.mean([m.predict_proba(X_cat)[:, 1] for m in bundle["cat_models"]], axis=0)
    
    base = np.column_stack([lgb_p, xgb_p, cat_p])
    probs = bundle["meta_model"].predict_proba(base)[:, 1]
    
    result = df.copy()
    result["churn_probability"] = np.round(probs, 4)
    result["prediction"] = ["Churn" if p > 0.5 else "No Churn" for p in probs]
    result["risk_level"] = ["High" if p > 0.7 else "Medium" if p > 0.4 else "Low" for p in probs]
    return result