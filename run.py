"""
run.py — Unified launcher for phase1 (data pipeline) and train (model training).

Accepts a --drive-dir argument that redirects ALL artifact, checkpoint, log,
and TensorBoard output to a Google Drive folder (or any other path). When
omitted, paths default to the local new/ directory (same as running the
scripts directly).

Usage
-----
# Smoke test — verify full code path works (2 epochs, 10K pairs):
    python run.py train --mode sample

# Full A100 training run:
    python run.py train --mode full

# Data pipeline only:
    python run.py phase1

# With Google Drive (the typical Colab workflow):
    python run.py phase1 --drive-dir /content/drive/MyDrive/nlp-chatbot-v2
    python run.py train  --drive-dir /content/drive/MyDrive/nlp-chatbot-v2 --mode sample
    python run.py train  --drive-dir /content/drive/MyDrive/nlp-chatbot-v2 --mode full

Design notes
------------
- Drive is mounted by the Colab setup script (setup_colab.sh) BEFORE this
  script runs. This script does NOT mount Drive itself.
- Config overrides are injected in-process via dict.update(), so the same
  CONFIG / PHASE1_CONFIG objects used inside phase1.py / train.py pick them up.
- The corpus CSV (dialogueText_301.csv) stays in data/Ubuntu-dialogue-corpus/
  inside the cloned repo (symlinked from Drive if preferred). Only outputs
  are redirected to Drive.
- All Drive output folders are created automatically.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Ensure repo root is on sys.path regardless of cwd ─────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))


def _apply_drive_overrides(cfg: dict, drive_dir: str) -> dict:
    """Redirect all output paths in cfg to a Google Drive folder.

    Only output directories are redirected. Input paths (corpus_dir,
    spm_model_path when used as input) are left pointing at the cloned
    repo unless they have already been generated (in which case they are
    also in drive_dir).
    """
    d = Path(drive_dir)
    overrides = {
        "artifact_dir":          str(d / "artifacts"),
        "checkpoint_dir":        str(d / "checkpoints"),
        "tensorboard_dir":       str(d / "tb_logs"),
        "log_dir":               str(d / "logs"),
        # These are outputs of phase1 AND inputs of train — both resolve
        # to the same Drive path so the chain works end-to-end.
        "spm_model_path":        str(d / "artifacts" / "stage5_spm.model"),
        "embedding_matrix_path": str(d / "artifacts" / "stage8_embedding_matrix.npy"),
    }
    cfg.update(overrides)

    # Create all output directories up-front so downstream code never fails
    # on a missing directory even if an early stage is skipped.
    for key in ("artifact_dir", "checkpoint_dir", "tensorboard_dir", "log_dir"):
        os.makedirs(cfg[key], exist_ok=True)

    return cfg


def _print_path_summary(cfg: dict, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label} — effective output paths")
    print(f"{'─'*60}")
    for key in ("artifact_dir", "checkpoint_dir", "tensorboard_dir", "log_dir"):
        if key in cfg:
            print(f"  {key:<26}: {cfg[key]}")
    print(f"{'─'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# phase1
# ──────────────────────────────────────────────────────────────────────────────

def run_phase1(drive_dir: str | None) -> None:
    from phase1 import PHASE1_CONFIG, main as phase1_main

    cfg = dict(PHASE1_CONFIG)

    if drive_dir:
        cfg = _apply_drive_overrides(cfg, drive_dir)

    _print_path_summary(cfg, "phase1")
    phase1_main(cfg=cfg, script_name="phase1")


# ──────────────────────────────────────────────────────────────────────────────
# train
# ──────────────────────────────────────────────────────────────────────────────

def run_train(drive_dir: str | None, mode: str) -> None:
    from config import CONFIG, get_a100_overrides
    from train import main as train_main

    cfg = dict(CONFIG)

    # Apply hardware / mode overrides first, then Drive paths on top,
    # so Drive paths always win regardless of what overrides do.
    if mode == "full":
        cfg.update(get_a100_overrides())
    elif mode == "sample":
        cfg.update({
            "max_train_samples": 10_000,
            "num_epochs":        2,
            "batch_size":        256,
            "grad_accum_steps":  1,
            "num_workers":       2,
        })

    if drive_dir:
        cfg = _apply_drive_overrides(cfg, drive_dir)

    _print_path_summary(cfg, f"train --mode {mode}")

    # train.main() uses argparse internally for the banner, but we've already
    # resolved the config here. Patch sys.argv so argparse sees --mode without
    # re-parsing our own args.
    sys.argv = [sys.argv[0], "--mode", mode]
    train_main(cfg=cfg, script_name="train")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

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
        "--drive-dir",
        default=None,
        metavar="PATH",
        help=(
            "Root folder on Google Drive (or any path) to store ALL outputs. "
            "E.g. /content/drive/MyDrive/nlp-chatbot-v2  "
            "Omit to use local paths inside the repo."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sample"],
        default="full",
        help=(
            "Train mode only. "
            "full=A100 optimised; sample=smoke test (10K pairs, 2 epochs)."
        ),
    )
    args = parser.parse_args()

    if args.phase == "phase1":
        run_phase1(args.drive_dir)
    elif args.phase == "train":
        run_train(args.drive_dir, args.mode)


if __name__ == "__main__":
    main()
