# src/data_loader.py
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pandas as pd


def ensure_binary_no_show(df: pd.DataFrame, col: str = "no_show") -> pd.DataFrame:
    """
    Normalise df[col] to strict 0/1 ints.
    Accepts 'yes/no', 'true/false', '0/1', mixed types, and NaN.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe")

    s = df[col].astype("string").str.strip().str.lower()
    s = s.replace({"yes": "1", "true": "1", "no": "0", "false": "0"})
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")  # keep NA-safe first
    df[col] = s.astype("int64")
    return df


def safe_cast_ints(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Cast listed columns to int safely (coerce bad values to NaN → fill 0 → int64)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
