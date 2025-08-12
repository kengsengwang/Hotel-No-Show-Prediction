#!/usr/bin/env python3
"""
Start script for the Hotel No-Show Prediction ML pipeline.
Run this in VS Code or terminal to execute the full process.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from src.main import main  # noqa: E402


if __name__ == "__main__":
    print(">>> Starting Hotel No-Show Prediction Pipeline <<<\n")
    main()
    print("\n>>> Pipeline finished successfully! <<<")
