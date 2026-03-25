import sys
import json
import joblib
import pandas as pd
import numpy as np
import os
from preprocess import Preprocessor

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'churn_model.joblib')

def predict():
    try:
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input data provided"}))
            return

        data = json.loads(input_data)

        if not os.path.exists(MODEL_PATH):
            print(json.dumps({"error": "Model bundle not found. Please run ml/train.py first."}))
            return

        bundle = joblib.load(MODEL_PATH)
        lgb_models = bundle.get('lgb_models', [])
        xgb_models = bundle.get('xgb_models', [])
        cat_models = bundle.get('cat_models', [])
        preprocessor = bundle['preprocessor']
        te = bundle['te']
        features = bundle['features']
        has_xgb = bundle.get('has_xgb', False)
        has_cat = bundle.get('has_cat', False)
        cols = bundle['cols']                     # numeric columns for LGBM/XGB

        input_df = pd.DataFrame([data])

        # Feature engineering
        processed_df = preprocessor.transform(input_df.copy())

        # Target Encoding (only for LGBM & XGB)
        cat_cols = preprocessor.all_cat_cols
        processed_df_te = processed_df.copy()
        if cat_cols:
            processed_df_te[cat_cols] = te.transform(processed_df_te[cat_cols])

        # Numeric matrix for LGBM/XGB
        X = processed_df_te.select_dtypes(include=[np.number])
        for col in cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[cols]

        # === Predictions ===
        # LGBM
        lgb_probs = [model.predict_proba(X)[:, 1][0] for model in lgb_models]
        lgb_prob = np.mean(lgb_probs)

        # XGB
        xgb_prob = 0.0
        if has_xgb and xgb_models:
            xgb_probs = [model.predict_proba(X)[:, 1][0] for model in xgb_models]
            xgb_prob = np.mean(xgb_probs)

        # CatBoost (native cats)
        cat_prob = 0.0
        if has_cat and cat_models:
            X_cat = processed_df[features].copy()
            for c in cat_cols:
                if c in X_cat.columns:
                    X_cat[c] = X_cat[c].astype(str)
            cat_probs = [model.predict_proba(X_cat)[:, 1][0] for model in cat_models]
            cat_prob = np.mean(cat_probs)

        # Final blended probability
        probs = [p for p in [lgb_prob, xgb_prob, cat_prob] if p > 0]
        final_prob = np.mean(probs)

        result = {
            "churnProbability": int(final_prob * 100),
            "status": "success",
            "model_type": f"LGBM{' + XGB' if has_xgb else ''}{' + CatBoost' if has_cat else ''} Blend"
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    predict()