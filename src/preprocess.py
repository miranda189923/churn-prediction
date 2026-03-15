# src/preprocess.py
import pandas as pd
import numpy as np
import sklearn
from packaging import version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from typing import Tuple, List, Optional

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)

class TenureBucketTransformer(BaseEstimator, TransformerMixin):
    """Create tenure buckets from a 'tenure' column."""
    def __init__(self, bins: Optional[List[float]] = None, labels: Optional[List[str]] = None):
        self.bins = bins or [0, 12, 24, 48, 60, np.inf]
        self.labels = labels or ['0-12','12-24','24-48','48-60','60+']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'tenure' in X.columns:
            X['tenure_bucket'] = pd.cut(X['tenure'], bins=self.bins, labels=self.labels)
        return X

def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], Optional[pd.Series]]:
    """
    Build a ColumnTransformer preprocessor for the dataset.
    Returns (preprocessor, feature_columns, target_series_or_None).
    """
    df = df.copy()
    # Drop obvious ID columns
    drop_cols = ['customerID']
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    target = 'Churn'
    if target in df.columns:
        y = df[target].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

    # Heuristic: detect object columns that are numeric-like and convert them
    obj_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = obj_cols.copy()

    for col in obj_cols:
        # try to coerce to numeric without modifying original df
        try:
            pd.to_numeric(X[col].dropna().iloc[:10])
            # if conversion of sample succeeds, treat as numeric
            numeric_cols.append(col)
            categorical_cols.remove(col)
        except Exception:
            pass

    # Ensure columns exist
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # OneHotEncoder kwarg compatibility across sklearn versions
    ohe_kwargs = {}
    try:
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            ohe_kwargs['sparse_output'] = False
        else:
            ohe_kwargs['sparse'] = False
    except Exception:
        # Fallback: try older kwarg
        ohe_kwargs['sparse'] = False

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', **ohe_kwargs))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    return preprocessor, X.columns.tolist(), y
