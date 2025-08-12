# src/model_train.py
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
from __future__ import annotations
from typing import Dict, Union, Any, cast
import time
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb  # type: ignore
    has_xgb: bool = True
except Exception:
    xgb = None  # type: ignore
    has_xgb = False

Preprocess = Union[ColumnTransformer, Pipeline]


def _tick(msg: str) -> float:
    print(msg, flush=True)
    return time.perf_counter()


def _tock(t0: float, name: str) -> None:
    print(f"[train] {name} done in {time.perf_counter() - t0:.1f}s", flush=True)


def train_models(
    preprocess: Preprocess,
    X_train: DataFrame,
    y_train: NDArray[np.int64],
    models_to_run: tuple[str, ...] = ("logreg", "rf", "xgb"),
) -> Dict[str, Pipeline]:
    """
    Train models with faster defaults.
    Returns a dict of fitted Pipelines keyed by model name.
    """
    models: Dict[str, Pipeline] = {}

    # ---- Logistic Regression (fast, good baseline)
    if "logreg" in models_to_run:
        logreg = Pipeline(steps=[
            ("preprocess", preprocess),
            ("scale", StandardScaler(with_mean=False)),  # safe with one-hot; helps convergence
            ("clf", LogisticRegression(
                solver="saga",
                penalty="l2",
                C=1.0,
                max_iter=3000,  # lower from 5000
                random_state=42,
            )),
        ])
        t0 = _tick("[train] fitting logreg…")
        logreg.fit(cast(Any, X_train), cast(Any, y_train))
        _tock(t0, "logreg")
        models["logreg"] = logreg

    # ---- Random Forest (parallel; fewer trees)
    if "rf" in models_to_run:
        rf = Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", RandomForestClassifier(
                n_estimators=200,       # 300 -> 200
                max_depth=12,          # cap depth to speed up
                min_samples_leaf=2,
                n_jobs=-1,             # use all cores
                random_state=42,
            )),
        ])
        t0 = _tick("[train] fitting random forest…")
        rf.fit(cast(Any, X_train), cast(Any, y_train))
        _tock(t0, "rf")
        models["rf"] = rf

    # ---- XGBoost (parallel; hist; fewer trees)
    if has_xgb and xgb is not None and "xgb" in models_to_run:
        xgb_pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", xgb.XGBClassifier(
                n_estimators=350,       # 500 -> 350
                learning_rate=0.10,     # slightly larger for fewer trees
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,              # use all cores
                tree_method="hist",
                eval_metric="logloss",
                verbosity=0,
            )),
        ])
        t0 = _tick("[train] fitting xgboost…")
        xgb_pipe.fit(cast(Any, X_train), cast(Any, y_train))
        _tock(t0, "xgb")
        models["xgb"] = xgb_pipe

    return models
