# app/streamlit_app.py
import sys
import pathlib

# Ensure project root is on sys.path so "import src" works when Streamlit runs
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import textwrap

from src.preprocess import load_data, TenureBucketTransformer

MODEL_PATH = "models/pipeline.pkl"
DATA_SAMPLE_PATH = "data/telco_customer_churn.csv"


@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)


@st.cache_data
def load_sample(path=DATA_SAMPLE_PATH):
    try:
        df = load_data(path)
        return df
    except Exception:
        return pd.DataFrame()


def _ensure_expected_columns(X_raw: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """
    Ensure X_raw contains all columns expected by the ColumnTransformer preprocessor.
    Adds missing columns filled with NaN and reorders to match preprocessor.feature_names_in_ if available.
    """
    X = X_raw.copy()
    expected = getattr(preprocessor, "feature_names_in_", None)
    if expected is None:
        # Best-effort fallback: collect column lists from transformers_
        cols = []
        try:
            for name, trans, cols_spec in preprocessor.transformers_:
                if cols_spec in ("drop", "remainder"):
                    continue
                if isinstance(cols_spec, (list, tuple)):
                    cols.extend(cols_spec)
        except Exception:
            cols = list(X.columns)
        expected = np.array(cols)

    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = np.nan
    # Reorder to expected order if possible
    try:
        X = X[expected.tolist()]
    except Exception:
        pass
    return X


def _wrap_feature_names(names, width=30):
    return [textwrap.fill(str(n), width=width) for n in names]


def _get_feature_names_from_preprocessor(preprocessor, X_aligned):
    """
    Try to obtain feature names after preprocessing step.
    Prefer ColumnTransformer.get_feature_names_out if available; otherwise fall back to X_aligned.columns.
    """
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            return preprocessor.get_feature_names_out(X_aligned.columns).tolist()
    except Exception:
        pass

    try:
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if cols in ("drop", "remainder"):
                continue
            if isinstance(cols, (list, tuple)):
                try:
                    if hasattr(trans, "get_feature_names_out"):
                        out = trans.get_feature_names_out(cols)
                        names.extend(out.tolist())
                    else:
                        names.extend(cols)
                except Exception:
                    names.extend(cols)
        if names:
            return names
    except Exception:
        pass

    return X_aligned.columns.tolist()


def single_customer_form(sample_df: pd.DataFrame) -> pd.DataFrame:
    st.header("Single Customer Prediction")
    input_vals = {}
    if sample_df.empty:
        st.write("No sample data available. Upload a CSV in Batch scoring or place dataset in data/")
        return pd.DataFrame()
    # Use first row as template
    template = sample_df.iloc[0]
    for col in sample_df.columns:
        if col in ['Churn', 'customerID']:
            continue
        val = template[col]
        if pd.api.types.is_numeric_dtype(sample_df[col]) or isinstance(val, (int, float, np.number)):
            default = float(val) if not pd.isna(val) else 0.0
            input_vals[col] = st.number_input(col, value=default)
        else:
            options = sample_df[col].dropna().unique().tolist()
            if len(options) == 0:
                input_vals[col] = st.text_input(col, value="")
            else:
                input_vals[col] = st.selectbox(col, options=options, index=0)
    return pd.DataFrame([input_vals])


def batch_upload():
    st.header("Batch scoring")
    uploaded = st.file_uploader("Upload CSV for batch scoring", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview", df.head())
        if st.button("Run batch prediction"):
            model = load_model()
            try:
                # Apply tenure bucketing to match training pipeline expectations
                df_proc = TenureBucketTransformer().transform(df.copy())
                preprocessor = model.named_steps.get('preprocessor')
                if preprocessor is not None:
                    df_aligned = _ensure_expected_columns(df_proc, preprocessor)
                else:
                    df_aligned = df_proc
                probs = model.predict_proba(df_aligned)[:, 1]
                df['churn_probability'] = probs
                st.download_button("Download results CSV", df.to_csv(index=False).encode('utf-8'),
                                   file_name='batch_predictions.csv')
                st.write(df.head())
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


def _plot_shap_bar_mean_abs(shap_values, feature_names, top_n=20):
    """
    Create a simple bar chart of mean(|SHAP|) per feature into a Matplotlib figure and return it.
    This is compact and avoids overlapping text issues.
    """
    # If shap_values is a list (multi-output), pick the first output
    sv = shap_values[0] if isinstance(shap_values, (list, tuple)) else shap_values
    # sv shape: (n_samples, n_features)
    try:
        mean_abs = np.mean(np.abs(sv), axis=0)
    except Exception:
        # fallback if sv is already aggregated
        mean_abs = np.array(sv)

    # Build DataFrame for sorting
    fnames = feature_names if feature_names is not None else [f"f{i}" for i in range(len(mean_abs))]
    df_imp = pd.DataFrame({"feature": fnames, "mean_abs_shap": mean_abs})
    df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).head(top_n).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(df_imp))), dpi=120)
    ax.barh(df_imp['feature'], df_imp['mean_abs_shap'], color='C0')
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature importance (mean absolute SHAP)")
    plt.tight_layout()
    return fig


def explain_prediction(model, X_raw: pd.DataFrame, proba: float):
    """
    Simplified SHAP explanation: show a single, robust bar chart of mean absolute SHAP values.
    Waterfall/local plots removed to avoid rendering issues and overlapping text.
    """
    st.subheader("Explanation (SHAP)")
    try:
        # 1) Apply tenure bucketing to the raw input (same logic used in training)
        X_proc = TenureBucketTransformer().transform(X_raw.copy())

        # 2) Align columns expected by the preprocessor
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor is not None:
            X_aligned = _ensure_expected_columns(X_proc, preprocessor)
            X_trans = preprocessor.transform(X_aligned)
            feature_names = _get_feature_names_from_preprocessor(preprocessor, X_aligned)
        else:
            X_aligned = X_proc
            X_trans = X_proc.values
            feature_names = X_proc.columns.tolist()

        clf = model.named_steps.get('clf')
        if clf is None:
            st.write("Model classifier not found in pipeline.")
            return

        # Compute SHAP values (TreeExplainer for tree models)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)

        st.write(f"Predicted churn probability: **{proba:.3f}**")

        # Prepare feature names for plotting (wrap long names)
        wrapped_names = _wrap_feature_names(feature_names, width=30) if feature_names is not None else None

        # Create and show a compact bar chart of mean absolute SHAP values
        n_features = X_trans.shape[1] if hasattr(X_trans, 'shape') else len(wrapped_names or [])
        top_n = min(25, n_features)
        fig = _plot_shap_bar_mean_abs(shap_values, wrapped_names, top_n)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        # If SHAP fails for any reason, show a friendly message and skip plots
        st.error(f"SHAP explanation failed: {e}")
        st.info("SHAP plots have been disabled for this input to avoid rendering issues.")


def main():
    st.title("Telco Customer Churn Predictor")
    st.markdown("Use the form for single-customer prediction or upload a CSV for batch scoring.")
    model = load_model()
    sample = load_sample()
    mode = st.radio("Mode", ["Single prediction", "Batch scoring"])
    if mode == "Single prediction":
        df_input = single_customer_form(sample)
        if not df_input.empty and st.button("Predict"):
            try:
                # Apply tenure bucketing before prediction so pipeline/preprocessor sees expected columns
                df_proc = TenureBucketTransformer().transform(df_input.copy())
                preprocessor = model.named_steps.get('preprocessor')
                if preprocessor is not None:
                    df_aligned = _ensure_expected_columns(df_proc, preprocessor)
                else:
                    df_aligned = df_proc
                proba = model.predict_proba(df_aligned)[:, 1][0]
                st.success(f"Churn probability: {proba:.3f}")
                explain_prediction(model, df_input, proba)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        batch_upload()


if __name__ == "__main__":
    main()