# src/main.py
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

from src.data_loader import load_data, train_test_split_stratified
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.data_preprocessing import build_preprocess
from src.model_train import train_models
from src.model_evaluation import evaluate_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hotel No-Show Prediction: end-to-end pipeline"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "hotel_no_show_cleaned.csv",
        help="Path to the input CSV (default: data/hotel_no_show_cleaned.csv)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Directory to save metrics/reports (default: results/)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logreg,rf,xgb",
        help="Comma-separated list of models to run: logreg,rf,xgb (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path: Path = args.data
    results_dir: Path = args.results
    models_to_run = tuple(m.strip() for m in args.models.split(",") if m.strip())

    start = time.time()
    print("=== Hotel No-Show Prediction Pipeline ===")

    # --- Safety checks
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Load
    print(f"[1/6] Loading data from: {data_path}")
    df = load_data(data_path)
    print(f"       → Loaded {len(df):,} rows, {df.shape[1]} columns")

    # --- 2) Clean
    print("[2/6] Cleaning data…")
    df = clean_data(df)

    # --- 3) Feature engineering
    print("[3/6] Engineering features…")
    df = engineer_features(df)

    # --- 4) Split
    print("[4/6] Train/Test split…")
    X_train, X_test, y_train, y_test, preprocess_inputs = train_test_split_stratified(df)
    print(f"       → X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- 5) Preprocess + Train
    print(f"[5/6] Building preprocessors and training models: {models_to_run}")
    preprocess = build_preprocess(preprocess_inputs)

    # Make Pylance happy: Series -> NumPy int64 array
    y_train_np = y_train.astype("int64").to_numpy()

    models = train_models(preprocess, X_train, y_train_np, models_to_run=models_to_run)

    # --- 6) Evaluate
    print("[6/6] Evaluating and saving metrics…")
    metrics_df = evaluate_and_save(models, X_test, y_test, results_dir)

    print("\n=== Metrics ===")
    print(metrics_df)

    elapsed = time.time() - start
    print(f"\nSaved reports and metrics to {results_dir}")
    print(f"Done ✅  (elapsed: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
