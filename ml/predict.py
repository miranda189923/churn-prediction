import sys
import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'ml/churn_model.joblib'

def predict():
    try:
        input_data = sys.stdin.read()
        if not input_data: return
        data = json.loads(input_data)
        
        bundle = joblib.load(MODEL_PATH)
        preprocessor = bundle['preprocessor']
        te = bundle['te']
        meta_model = bundle['meta_model']
        cols = bundle['cols']
        features = bundle['features']
        
        input_df = pd.DataFrame([data])
        proc_df = preprocessor.transform(input_df.copy())
        cat_cols = preprocessor.all_cat_cols
        
        # 1. Base Predictions for LGB/XGB
        proc_df_te = proc_df.copy()
        proc_df_te[cat_cols] = te.transform(proc_df_te[cat_cols])
        X_num_vals = proc_df_te[cols].values # Convert to numpy to avoid warnings
        
        lgb_p = np.mean([m.predict_proba(X_num_vals)[:, 1][0] for m in bundle['lgb_models']])
        xgb_p = np.mean([m.predict_proba(X_num_vals)[:, 1][0] for m in bundle['xgb_models']])
        
        # 2. Base Prediction for CatBoost
        X_cat = proc_df[features].astype(str)
        cat_p = np.mean([m.predict_proba(X_cat)[:, 1][0] for m in bundle['cat_models']])
        
        # 3. Meta-model final blend
        base_preds = np.array([[lgb_p, xgb_p, cat_p]])
        final_prob = meta_model.predict_proba(base_preds)[:, 1][0]
        
        print(json.dumps({
            "churnProbability": int(final_prob * 100),
            "status": "success",
            "model_type": "Stacked Ensemble"
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    predict()