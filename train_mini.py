"""
train_mini.py — Fast training run on mini artifacts for quick iteration.

Runs exactly the same training loop as train.py, but with:
  - artifacts from artifacts_mini/  (10% Ubuntu corpus, ~75k train pairs)
  - max_train_samples = 15,000      (further subset for speed)
  - num_epochs = 10                 (vs 20 for full run)
  - early stopping patience = 3
  - checkpoints → checkpoints_mini/
  - TensorBoard → tb_logs_mini/
  - After training, auto-runs evaluate_mini.py for full model analysis

Purpose: catch training pathologies (mode collapse, no learning, overfitting)
in ~10-30 minutes before committing to a full multi-hour training run.

Usage:
    cd new/
    python train_mini.py
"""

import copy
import sys
from pathlib import Path

# ── Ensure new/ is on the path regardless of working directory ──────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import CONFIG
import train as _train


def main() -> None:
    cfg = copy.deepcopy(CONFIG)

    # ── Mini overrides ───────────────────────────────────────────────────────
    cfg["artifact_dir"]        = str(_HERE / "artifacts_mini")
    cfg["spm_model_path"]      = str(_HERE / "artifacts_mini" / "stage5_spm.model")
    cfg["embedding_matrix_path"] = str(_HERE / "artifacts_mini" / "stage8_embedding_matrix.npy")
    cfg["checkpoint_dir"]      = str(_HERE / "checkpoints_mini")
    cfg["tensorboard_dir"]     = str(_HERE / "tb_logs_mini")
    cfg["log_dir"]             = str(_HERE / "logs")
    cfg["max_train_samples"]   = 50_000   # enough signal for 44M-param model without long runtime
    cfg["num_epochs"]          = 20       # early stopping will cut this short if loss plateaus
    cfg["patience"]            = 5        # allow 5 non-improving epochs before stopping
    cfg["grad_accum_steps"]    = 1        # disable accumulation — no benefit at mini scale
    cfg["batch_size"]          = 64       # smaller for CPU comfort; raise on GPU
    cfg["num_workers"]         = 0        # macOS / Windows safe

    print("=" * 60)
    print(f"  TRAIN MINI — quick sanity check on 50,000 pairs / up to 20 epochs")
    print("=" * 60)
    print(f"  artifact_dir   : {cfg['artifact_dir']}")
    print(f"  checkpoint_dir : {cfg['checkpoint_dir']}")
    print(f"  max_train_samples: {cfg['max_train_samples']:,}")
    print(f"  num_epochs     : {cfg['num_epochs']}")
    print()

    _train.main(cfg=cfg, script_name="train_mini")

    # ── Auto-run evaluate_mini after training ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete — starting evaluate_mini")
    print("=" * 60)
    import evaluate_mini as _eval_mini
    _eval_mini.main(cfg=cfg)


if __name__ == "__main__":
    main()
