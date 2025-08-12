# src/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

TARGET_NAME: str = "no_show"


def load_data(path: Path | str) -> DataFrame:
    """Load CSV into a DataFrame with basic sanity checks."""
    csv_path: Path = Path(path)
    df: DataFrame = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Loaded empty DataFrame from: {csv_path}")
    return df


def _infer_feature_types(df: DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical feature columns (excluding the target)."""
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. Columns: {list(df.columns)}"
        )

    X_cols: List[str] = [c for c in df.columns if c != target_col]
    num_cols: List[str] = df[X_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols: List[str] = [c for c in X_cols if c not in num_cols]
    return num_cols, cat_cols


def train_test_split_stratified(
    df: DataFrame,
    *,
    target_col: str = TARGET_NAME,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[DataFrame, DataFrame, Series, Series, Dict[str, Any]]:
    """
    Split features/target with stratification on the target.
    Returns: X_train, X_test, y_train, y_test, preprocess_inputs
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    y_raw: Series = df[target_col]

    # Coerce to 0/1 int if needed
    if y_raw.dtype == "bool":
        y: Series = y_raw.astype(int)
    elif y_raw.dtype.kind in ("i", "u"):
        y = y_raw.astype(int)
    else:
        mapping: Dict[Any, Any] = {
            "no": 0, "No": 0, "NO": 0, "N": 0, "false": 0, "False": 0,
            "yes": 1, "Yes": 1, "YES": 1, "Y": 1, "true": 1, "True": 1,
        }
        y = y_raw.map(mapping).fillna(y_raw)
        try:
            y = y.astype(int)
        except Exception as exc:  # pragma: no cover
            raise ValueError(
                f"Target '{target_col}' must be binary (0/1 or bool)."
            ) from exc

    X: DataFrame = df.drop(columns=[target_col])

    num_cols, cat_cols = _infer_feature_types(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocess_inputs: Dict[str, Any] = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return X_train, X_test, y_train, y_test, preprocess_inputs
