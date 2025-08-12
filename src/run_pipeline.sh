#!/usr/bin/env bash
set -e

DATA_PATH_BASH="/c/Users/DELL/Documents/Hotel-No-Show-Prediction/data/hotel_no_show_cleaned.csv"
RESULTS_DIR="results"

echo "=== Checking dataset path ==="
[ -f "$DATA_PATH_BASH" ] || { echo "[ERROR] $DATA_PATH_BASH not found"; exit 1; }

if command -v cygpath >/dev/null 2>&1; then
  DATA_PATH_WIN="$(cygpath -w "$DATA_PATH_BASH")"
else
  DATA_PATH_WIN="C:/Users/DELL/Documents/Hotel-No-Show-Prediction/data/hotel_no_show_cleaned.csv"
fi

echo "=== Activating py311env ==="
source ~/miniforge3/Scripts/activate py311env

echo "=== Running pipeline ==="
python -m src.main --data "$DATA_PATH_WIN" --results "$RESULTS_DIR"

echo "=== Metrics ==="
[ -f "$RESULTS_DIR/metrics.csv" ] && cat "$RESULTS_DIR/metrics.csv" || echo "No metrics.csv"
