import numpy as np
import pandas as pd
import time
import joblib
import os
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
from preprocess import Preprocessor

# Mute warnings for a cleaner console
warnings.filterwarnings("ignore", category=UserWarning)

class CFG:
    TARGET = 'Churn'
    N_FOLDS = 5
    REPEATS = 2
    OPTUNA_TRIALS = 60
    RANDOM_SEED = 42
    DATA_PATH = "data/telco_customer_churn.csv"

LGB_BASE = {
    'random_state': CFG.RANDOM_SEED,
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'n_jobs': -1,
}

XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'random_state': CFG.RANDOM_SEED,
    'eval_metric': 'auc',
    'tree_method': 'hist', # CPU-optimized
}

CAT_BASE = {
    'random_seed': CFG.RANDOM_SEED,
    'verbose': False,
    'early_stopping_rounds': 50,
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',
    'task_type': 'CPU', 
}

def train_model():
    print("Loading dataset...")
    df = pd.read_csv(CFG.DATA_PATH)
    
    if CFG.TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[CFG.TARGET]):
        df[CFG.TARGET] = df[CFG.TARGET].astype(str).str.strip().str.capitalize().map({'No': 0, 'Yes': 1}).fillna(0).astype(int)

    churn_rate = df[CFG.TARGET].mean()
    pos_weight = (1 - churn_rate) / churn_rate
    print(f"Churn rate: {churn_rate:.3f} → scale_pos_weight = {pos_weight:.2f}")

    train, test = train_test_split(df, test_size=0.2, random_state=CFG.RANDOM_SEED, stratify=df[CFG.TARGET])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    preprocessor = Preprocessor(target_col=CFG.TARGET)
    train, test, features = preprocessor.fit_transform(train, test, df.copy())

    X_full = train[features].copy()
    y_full = train[CFG.TARGET].values
    cat_cols = preprocessor.all_cat_cols

    # --- 1. Tuning LightGBM ---
    def objective_lgb(trial):
        params = {
            **LGB_BASE,
            'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'num_leaves': trial.suggest_int('num_leaves', 31, 128),
            'scale_pos_weight': pos_weight,
        }
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG.RANDOM_SEED)
        aucs = []
        for tr_idx, va_idx in inner_skf.split(X_full, y_full):
            te_i = TargetEncoder(cv=3, random_state=CFG.RANDOM_SEED)
            X_tr = te_i.fit_transform(X_full.iloc[tr_idx][cat_cols], y_full[tr_idx])
            X_va = te_i.transform(X_full.iloc[va_idx][cat_cols])
            model = LGBMClassifier(**params)
            # Use .values to avoid feature name warnings
            model.fit(X_tr.values, y_full[tr_idx], eval_set=[(X_va.values, y_full[va_idx])], callbacks=[early_stopping(50, verbose=False)])
            aucs.append(roc_auc_score(y_full[va_idx], model.predict_proba(X_va.values)[:, 1]))
        return np.mean(aucs)

    print("Tuning LightGBM...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(objective_lgb, n_trials=CFG.OPTUNA_TRIALS)
    
    # --- 2. Tuning CatBoost ---
    def objective_cat(trial):
        params = {
            **CAT_BASE,
            'iterations': trial.suggest_int('iterations', 800, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
        }
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG.RANDOM_SEED)
        aucs = []
        for tr_idx, va_idx in inner_skf.split(X_full, y_full):
            X_tr_i, X_va_i = X_full.iloc[tr_idx].copy(), X_full.iloc[va_idx].copy()
            for d in (X_tr_i, X_va_i):
                for c in cat_cols: d[c] = d[c].astype(str)
            model = CatBoostClassifier(**params)
            model.fit(X_tr_i, y_full[tr_idx], eval_set=(X_va_i, y_full[va_idx]), cat_features=cat_cols, verbose=False)
            aucs.append(roc_auc_score(y_full[va_idx], model.predict_proba(X_va_i)[:, 1]))
        return np.mean(aucs)

    print("Tuning CatBoost...")
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(objective_cat, n_trials=CFG.OPTUNA_TRIALS)

    LGB_PARAMS = {**LGB_BASE, **study_lgb.best_params, 'scale_pos_weight': pos_weight}
    CAT_PARAMS = {**CAT_BASE, **study_cat.best_params}

    # --- 3. Final Stacking Training ---
    print("\nTraining Final Stacked Ensemble...")
    skf_outer = RepeatedStratifiedKFold(n_splits=CFG.N_FOLDS, n_repeats=CFG.REPEATS, random_state=CFG.RANDOM_SEED)
    oof_base_preds = np.zeros((len(train), 3))
    test_base_preds = np.zeros((len(test), 3))
    lgb_models, xgb_models, cat_models = [], [], []

    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        X_tr, y_tr = train.iloc[train_idx][features], train.iloc[train_idx][CFG.TARGET].values
        X_val, y_val = train.iloc[val_idx][features], train.iloc[val_idx][CFG.TARGET].values
        
        te = TargetEncoder(cv=3, random_state=CFG.RANDOM_SEED)
        X_tr_te = X_tr.copy(); X_val_te = X_val.copy(); X_te_te = test[features].copy()
        X_tr_te[cat_cols] = te.fit_transform(X_tr_te[cat_cols], y_tr)
        X_val_te[cat_cols] = te.transform(X_val_te[cat_cols])
        X_te_te[cat_cols] = te.transform(X_te_te[cat_cols])
        
        X_tr_n = X_tr_te.select_dtypes(include=[np.number])
        X_val_n = X_val_te.select_dtypes(include=[np.number])
        X_te_n = X_te_te.select_dtypes(include=[np.number])

        # Train Base Models
        m_lgb = LGBMClassifier(**LGB_PARAMS).fit(X_tr_n.values, y_tr, eval_set=[(X_val_n.values, y_val)], callbacks=[early_stopping(50, verbose=False)])
        m_xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_weight).fit(X_tr_n.values, y_tr, eval_set=[(X_val_n.values, y_val)], verbose=False)
        
        X_tr_c = X_tr.copy(); X_val_c = X_val.copy(); X_te_c = test[features].copy()
        for dfc in (X_tr_c, X_val_c, X_te_c):
            for c in cat_cols: dfc[c] = dfc[c].astype(str)
        m_cat = CatBoostClassifier(**CAT_PARAMS).fit(X_tr_c, y_tr, eval_set=(X_val_c, y_val), cat_features=cat_cols, verbose=False)

        # Store OOF
        oof_base_preds[val_idx, 0] += m_lgb.predict_proba(X_val_n.values)[:, 1] / CFG.REPEATS
        oof_base_preds[val_idx, 1] += m_xgb.predict_proba(X_val_n.values)[:, 1] / CFG.REPEATS
        oof_base_preds[val_idx, 2] += m_cat.predict_proba(X_val_c)[:, 1] / CFG.REPEATS
        
        # Store Models
        lgb_models.append(m_lgb); xgb_models.append(m_xgb); cat_models.append(m_cat)
        print(f"   Fold {i+1} completed.")

    # Train Meta-model
    print("\nTraining Meta-model (Logistic Regression)...")
    meta_model = LogisticRegression().fit(oof_base_preds, train[CFG.TARGET])
    
    # Save Bundle
    te_final = TargetEncoder(random_state=CFG.RANDOM_SEED).fit(train[cat_cols], train[CFG.TARGET])
    bundle = {
        'lgb_models': lgb_models, 'xgb_models': xgb_models, 'cat_models': cat_models,
        'meta_model': meta_model, 'preprocessor': preprocessor, 'te': te_final, 
        'features': features, 'cols': X_tr_n.columns.tolist()
    }
    joblib.dump(bundle, 'ml/churn_model.joblib')
    
    final_p = meta_model.predict_proba(oof_base_preds)[:, 1]
    print("\n" + "="*40 + "\nTRAINING COMPLETE")
    print(f"Stacked OOF AUC: {roc_auc_score(train[CFG.TARGET], final_p):.5f}")
    print(f"Accuracy:        {accuracy_score(train[CFG.TARGET], (final_p > 0.5)):.5f}\n" + "="*40)

if __name__ == "__main__":
    train_model()