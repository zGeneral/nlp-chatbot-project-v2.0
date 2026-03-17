"""
run.py — Launcher for the data pipeline and model training.

Usage:
    python run.py phase1   # Run data pipeline (once)
    python run.py train    # Train both models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase1 (data pipeline) or train (model training).")
    parser.add_argument("phase", choices=["phase1", "train"], help="Which script to run.")
    args = parser.parse_args()

    if args.phase == "phase1":
        from phase1 import PHASE1_CONFIG, main as phase1_main
        phase1_main(cfg=dict(PHASE1_CONFIG), script_name="phase1")
    else:
        from train import main as train_main
        train_main(script_name="train")


if __name__ == "__main__":
    main()
