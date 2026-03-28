import numpy as np
import pandas as pd
import time
import joblib
import os
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
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
    'tree_method': 'hist',   # CPU-optimized
}

CAT_BASE = {
    'random_seed': CFG.RANDOM_SEED,
    'verbose': False,
    'early_stopping_rounds': 50,
    'eval_metric': 'AUC',
    'auto_class_weights': 'Balanced',
    'task_type': 'CPU',      # <-- CPU version
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
    orig = df.copy()

    preprocessor = Preprocessor(target_col=CFG.TARGET)
    train, test, features = preprocessor.fit_transform(train, test, orig)

    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER TUNING + STACKING META-MODEL")
    print("="*70)

    X_full = train[features].copy()
    y_full = train[CFG.TARGET].values
    cat_cols = preprocessor.all_cat_cols

    # --- 1. Tuning LightGBM (now matches notebook exactly) ---
    def objective_lgb(trial):
        params = {
            **LGB_BASE,
            'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.008, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'num_leaves': trial.suggest_int('num_leaves', 31, 128),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'scale_pos_weight': pos_weight,
        }
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG.RANDOM_SEED + trial.number)
        aucs = []
        for tr_idx, va_idx in inner_skf.split(X_full, y_full):
            X_tr_i = X_full.iloc[tr_idx].copy()
            X_va_i = X_full.iloc[va_idx].copy()
            y_tr_i = y_full[tr_idx]
            y_va_i = y_full[va_idx]
            te_i = TargetEncoder(cv=3, random_state=CFG.RANDOM_SEED)
            X_tr_i[cat_cols] = te_i.fit_transform(X_tr_i[cat_cols], y_tr_i)
            X_va_i[cat_cols] = te_i.transform(X_va_i[cat_cols])
            X_tr_num = X_tr_i.select_dtypes(include=[np.number])
            X_va_num = X_va_i.select_dtypes(include=[np.number])
            model = LGBMClassifier(**params)
            model.fit(X_tr_num, y_tr_i, eval_set=[(X_va_num, y_va_i)], callbacks=[early_stopping(50, verbose=False)])
            pred = model.predict_proba(X_va_num)[:, 1]
            aucs.append(roc_auc_score(y_va_i, pred))
        return np.mean(aucs)

    print("Tuning LightGBM...")
    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.RANDOM_SEED))
    study_lgb.optimize(objective_lgb, n_trials=CFG.OPTUNA_TRIALS)
    print(f"   LGBM best AUC: {study_lgb.best_value:.5f}")

    # --- 2. Tuning CatBoost (now matches notebook exactly) ---
    def objective_cat(trial):
        params = {
            **CAT_BASE,
            'iterations': trial.suggest_int('iterations', 800, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        }
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG.RANDOM_SEED + trial.number)
        aucs = []
        for tr_idx, va_idx in inner_skf.split(X_full, y_full):
            X_tr_i = X_full.iloc[tr_idx].copy()
            X_va_i = X_full.iloc[va_idx].copy()
            y_tr_i = y_full[tr_idx]
            y_va_i = y_full[va_idx]
            cat_features = [col for col in cat_cols if col in X_tr_i.columns]
            for df_ in (X_tr_i, X_va_i):
                for c in cat_features:
                    df_[c] = df_[c].astype(str)
            model = CatBoostClassifier(**params)
            model.fit(X_tr_i, y_tr_i, eval_set=(X_va_i, y_va_i), cat_features=cat_features, verbose=False)
            pred = model.predict_proba(X_va_i)[:, 1]
            aucs.append(roc_auc_score(y_va_i, pred))
        return np.mean(aucs)

    print("Tuning CatBoost...")
    study_cat = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.RANDOM_SEED))
    study_cat.optimize(objective_cat, n_trials=CFG.OPTUNA_TRIALS)
    print(f"   CatBoost best AUC: {study_cat.best_value:.5f}")

    LGB_PARAMS = {**LGB_BASE, **study_lgb.best_params, 'scale_pos_weight': pos_weight}
    CAT_PARAMS = {**CAT_BASE, **study_cat.best_params}

    # --- 3. Final Stacked Ensemble (exact same as notebook) ---
    print("\n" + "="*70)
    print(f"TRAINING FINAL STACKED ENSEMBLE ({CFG.N_FOLDS}×{CFG.REPEATS})")
    print("="*70)

    skf_outer = RepeatedStratifiedKFold(n_splits=CFG.N_FOLDS, n_repeats=CFG.REPEATS, random_state=CFG.RANDOM_SEED)
    n_outer = CFG.N_FOLDS * CFG.REPEATS

    oof_base_preds = np.zeros((len(train), 3))
    test_base_preds = np.zeros((len(test), 3))

    lgb_models, xgb_models, cat_models = [], [], []

    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\nOuter fold {i+1}/{n_outer}")
        X_tr = train.iloc[train_idx][features].copy()
        y_tr = train.iloc[train_idx][CFG.TARGET].values
        X_val = train.iloc[val_idx][features].copy()
        y_val = train.iloc[val_idx][CFG.TARGET].values
        X_te = test[features].copy()

        # Target Encoding
        te = TargetEncoder(cv=3, random_state=CFG.RANDOM_SEED)
        X_tr_te = X_tr.copy()
        X_val_te = X_val.copy()
        X_te_te = X_te.copy()
        X_tr_te[cat_cols] = te.fit_transform(X_tr_te[cat_cols], y_tr)
        X_val_te[cat_cols] = te.transform(X_val_te[cat_cols])
        X_te_te[cat_cols] = te.transform(X_te_te[cat_cols])

        X_tr_num = X_tr_te.select_dtypes(include=[np.number])
        X_val_num = X_val_te.select_dtypes(include=[np.number])
        X_te_num = X_te_te.select_dtypes(include=[np.number])

        # LGBM
        lgb_model = LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(X_tr_num, y_tr, eval_set=[(X_val_num, y_val)], callbacks=[early_stopping(50, verbose=False)])
        lgb_val_p = lgb_model.predict_proba(X_val_num)[:, 1]
        lgb_test_p = lgb_model.predict_proba(X_te_num)[:, 1]
        lgb_models.append(lgb_model)

        # XGB
        xgb_model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_weight)
        xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_val_num, y_val)], verbose=False)
        xgb_val_p = xgb_model.predict_proba(X_val_num)[:, 1]
        xgb_test_p = xgb_model.predict_proba(X_te_num)[:, 1]
        xgb_models.append(xgb_model)

        # CatBoost
        X_tr_cat = X_tr.copy()
        X_val_cat = X_val.copy()
        X_te_cat = X_te.copy()
        cat_features = [col for col in cat_cols if col in X_tr_cat.columns]
        for dfc in (X_tr_cat, X_val_cat, X_te_cat):
            for c in cat_features:
                dfc[c] = dfc[c].astype(str)
        cat_model = CatBoostClassifier(**CAT_PARAMS)
        cat_model.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), cat_features=cat_features, verbose=False)
        cat_val_p = cat_model.predict_proba(X_val_cat)[:, 1]
        cat_test_p = cat_model.predict_proba(X_te_cat)[:, 1]
        cat_models.append(cat_model)

        # Store OOF + test predictions
        oof_base_preds[val_idx, 0] += lgb_val_p / CFG.REPEATS
        oof_base_preds[val_idx, 1] += xgb_val_p / CFG.REPEATS
        oof_base_preds[val_idx, 2] += cat_val_p / CFG.REPEATS

        test_base_preds[:, 0] += lgb_test_p / n_outer
        test_base_preds[:, 1] += xgb_test_p / n_outer
        test_base_preds[:, 2] += cat_test_p / n_outer

        fold_auc = roc_auc_score(y_val, np.mean([lgb_val_p, xgb_val_p, cat_val_p], axis=0))
        print(f"   Fold {i+1} Mean AUC: {fold_auc:.5f}")

    # Train Meta-model
    print("\nTraining Meta-model (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=CFG.RANDOM_SEED)
    meta_model.fit(oof_base_preds, train[CFG.TARGET])

    final_oof_preds = meta_model.predict_proba(oof_base_preds)[:, 1]

    te_final = TargetEncoder(random_state=CFG.RANDOM_SEED).fit(train[cat_cols], train[CFG.TARGET])

    bundle = {
        'lgb_models': lgb_models,
        'xgb_models': xgb_models,
        'cat_models': cat_models,
        'meta_model': meta_model,
        'preprocessor': preprocessor,
        'te': te_final,
        'features': features,
        'cols': X_tr_num.columns.tolist()
    }
    joblib.dump(bundle, 'ml/churn_model.joblib')

    y_true = train[CFG.TARGET].values
    y_pred_binary = (final_oof_preds > 0.5).astype(int)

    print("\n" + "="*70 + "\nTRAINING COMPLETE")
    print(f"Stacked OOF AUC: {roc_auc_score(y_true, final_oof_preds):.5f}")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred_binary):.5f}\n" + "="*70)

if __name__ == "__main__":
    train_model()