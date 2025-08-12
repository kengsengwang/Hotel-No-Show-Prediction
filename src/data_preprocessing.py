from __future__ import annotations
from typing import Dict, Any, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocess(preprocess_inputs: Dict[str, Any]) -> ColumnTransformer:
    """Build a ColumnTransformer using the provided numeric / categorical lists."""
    num_cols: List[str] = list(preprocess_inputs.get("num_cols", []))
    cat_cols: List[str] = list(preprocess_inputs.get("cat_cols", []))

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
