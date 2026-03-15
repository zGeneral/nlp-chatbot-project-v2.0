"""
run.py — Unified launcher for the data pipeline and model training.

Usage
-----
    # Data pipeline (run once to generate artifacts):
    python run.py phase1

    # Smoke test — verify full code path end-to-end (2 epochs, 10K pairs):
    python run.py train --mode sample

    # Full training run:
    python run.py train --mode full
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))


def run_phase1() -> None:
    from phase1 import PHASE1_CONFIG, main as phase1_main
    phase1_main(cfg=dict(PHASE1_CONFIG), script_name="phase1")


def run_train(mode: str) -> None:
    from config import CONFIG
    from train import main as train_main

    cfg = dict(CONFIG)
    if mode == "sample":
        cfg.update({
            "max_train_samples": 10_000,
            "num_epochs":        2,
            "batch_size":        256,
            "grad_accum_steps":  1,
            "num_workers":       2,
        })

    sys.argv = [sys.argv[0], "--mode", mode]
    train_main(cfg=cfg, script_name="train")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Launcher for phase1 (data pipeline) and train (seq2seq training).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "phase",
        choices=["phase1", "train"],
        help="Which script to run.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sample"],
        default="full",
        help="Train mode: full=standard training; sample=smoke test (10K pairs, 2 epochs).",
    )
    args = parser.parse_args()

    if args.phase == "phase1":
        run_phase1()
    elif args.phase == "train":
        run_train(args.mode)


if __name__ == "__main__":
    main()
