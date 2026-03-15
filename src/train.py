# src/train.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.preprocess import load_data, TenureBucketTransformer, build_preprocessor
from src.model_utils import save_pipeline, export_pipeline_to_onnx

import shap

# Paths and constants
DATA_PATH = "data/telco_customer_churn.csv"
MODEL_PATH = "models/pipeline.pkl"
ONNX_PATH = "models/pipeline.onnx"
RANDOM_STATE = 42

def main():
    # Load data
    df = load_data(DATA_PATH)

    # Convert TotalCharges to numeric if present
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Ensure tenure_bucket exists for preprocessing (TenureBucketTransformer will also be included in pipeline)
    df = TenureBucketTransformer().transform(df)

    # Build preprocessor (returns ColumnTransformer, feature list, and target series)
    preprocessor, feature_cols, _ = build_preprocessor(df)

    # Prepare X and y
    X = df.drop(columns=['Churn']) if 'Churn' in df.columns else df.copy()
    y = df['Churn'].map({'Yes':1,'No':0}) if 'Churn' in df.columns else None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # XGBoost classifier (compatible with recent xgboost versions)
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        verbosity=0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )

    # Full pipeline: include TenureBucketTransformer so the saved pipeline can accept raw inputs
    pipeline = Pipeline(steps=[
        ('tenure_bucket', TenureBucketTransformer()),
        ('preprocessor', preprocessor),
        ('clf', xgb)
    ])

    # Hyperparameter search space
    param_dist = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__colsample_bytree': [0.6, 0.8, 1.0],
        'clf__reg_alpha': [0, 0.1, 1],
        'clf__reg_lambda': [1, 5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=30,
        scoring='roc_auc', n_jobs=-1, cv=cv, verbose=1, random_state=RANDOM_STATE
    )

    print("Starting hyperparameter search")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred_proba = best.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print("Best params:", search.best_params_)
    print("Test AUC:", auc)

    # Save pipeline
    save_pipeline(best, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

    # Optional: export to ONNX
    try:
        # Need a numeric sample for ONNX conversion: transform a small batch using the preprocessor step
        pre = best.named_steps['preprocessor']
        X_sample = pre.transform(X_test.iloc[:5])
        export_pipeline_to_onnx(best, X_sample, ONNX_PATH)
        print("ONNX exported to", ONNX_PATH)
    except Exception as e:
        print("ONNX export failed:", e)

    # SHAP summary plot (optional)
    try:
        clf = best.named_steps['clf']
        pre = best.named_steps['preprocessor']
        X_train_trans = pre.transform(X_train)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train_trans)
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_train_trans, show=False)
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/shap_summary.png', bbox_inches='tight')
        plt.close()
        print("Saved SHAP summary to models/shap_summary.png")
    except Exception as e:
        print("SHAP generation failed:", e)

if __name__ == "__main__":
    main()
