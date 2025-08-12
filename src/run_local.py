#!/usr/bin/env python3
"""
Run the Hotel No-Show pipeline with your CURRENT Python interpreter.
Usage examples:
  python scripts/run_local.py
  python scripts/run_local.py --data "C:/.../data/hotel_no_show_cleaned.csv" --results results --models logreg,rf
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "hotel_no_show_cleaned.csv")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--models", type=str, default="logreg,rf,xgb",
                        help="Comma-separated: logreg,rf,xgb")
    args = parser.parse_args()

    # Rebuild argv for src.main (so its own parser sees the same args)
    argv = ["src.main", "--data", str(args.data), "--results", str(args.results), "--models", args.models]

    # Import and call your existing entrypoint
    from src.main import main as pipeline_main
    old_argv = sys.argv
    try:
        sys.argv = argv
        pipeline_main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    main()
