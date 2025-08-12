# src/model_evaluation.py
from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_and_save(
    models: Dict[str, Pipeline],
    X_test: DataFrame,
    y_test: Series,
    results_dir: Path,
) -> DataFrame:
    """
    Evaluate each fitted model on X_test/y_test and save a metrics table.
    Returns a DataFrame indexed by model name with accuracy, f1, roc_auc.
    """
    rows: list[dict[str, float | str]] = []

    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        rows.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            }
        )

    metrics_df: DataFrame = (
        pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(results_dir / "metrics.csv")
    return metrics_df
