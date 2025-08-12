from __future__ import annotations
import pandas as pd
from pandas import DataFrame

def clean_data(df: DataFrame) -> DataFrame:
    """Basic cleaning; extend as needed."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df
