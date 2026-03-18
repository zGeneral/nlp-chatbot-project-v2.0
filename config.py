"""
config.py — Single source of truth for all hyperparameters and paths.

Every other file imports from here. No magic numbers anywhere else.

Implementation notes:
- CONFIG is a plain dict (not a dataclass) for easy JSON serialisation
  and per-script override via CONFIG.update({...}).
- All paths are computed relative to this file's location so the
  code works correctly regardless of which directory you run from.
"""

import json
import os
from pathlib import Path

# ── Tokenization ──────────────────────────────────────────────────────────────
_TOKENIZATION = {
    "vocab_size":            32000,   # SentencePiece BPE vocabulary size
    "spm_vocab_size":        32000,   # alias used by SPM training scripts
    "embed_dim":             300,     # token embedding dimensionality
    # Special token IDs — must match SentencePieceTrainer pad_id/bos_id/eos_id
    "pad_idx":               0,       # <pad>
    "unk_idx":               1,       # <unk>
    "sos_idx":               2,       # <sos> / bos
    "eos_idx":               3,       # <eos>
}

# ── Architecture ──────────────────────────────────────────────────────────────
_ARCHITECTURE = {
    "enc_hidden_dim":        512,     # per-direction; effective = 1024 (bidirectional)
    "dec_hidden_dim":        1024,    # matches bidirectional encoder output
    "projection_dim":        512,     # encoder → decoder state projection size
    "attn_dim":              256,     # Bahdanau attention hidden size
    "num_layers":            2,       # stacked LSTM layers in encoder and decoder
    "shared_embeddings":     True,    # encoder and decoder share the embedding matrix
    # Dropout — kept together so a single config change tunes all three.
    "dropout_embed":         0.3,     # embedding dropout (light: preserve FastText signal)
    "dropout_lstm":          0.5,     # inter-layer LSTM dropout (aggressive regularisation)
    "dropout_out":           0.4,     # output/projection dropout
}

# ── Training ──────────────────────────────────────────────────────────────────
_TRAINING = {
    "learning_rate":         3e-4,    # peak LR (reached after warmup; cosine decays from here)
    "weight_decay":          1e-5,    # L2 regularisation
    "max_grad_norm":         1.0,     # gradient clipping
    "batch_size":            256,     # per-step batch size
    "grad_accum_steps":      2,       # effective batch = batch_size × grad_accum_steps = 512
    "num_epochs":            20,      # total training epochs
    "amp_dtype":             "bfloat16",  # automatic mixed precision dtype
    "patience":              4,       # early stopping patience (0 = disabled); monitoring
                                      # begins only after Phase 1 ends (see get_tf_ratio)
}

# ── Teacher-Forcing Schedule ───────────────────────────────────────────────────
# 3-phase schedule: full TF → linear annealing → floor.
_TF_SCHEDULE = {
    "tf_schedule": {
        "phase1_end":       5,    # last epoch (inclusive) at TF = phase1_tf
        "phase1_tf":        1.0,  # TF ratio for epochs 1 → phase1_end
        "phase2_end":       12,   # last epoch (inclusive) of linear annealing
        "phase2_start_tf":  0.9,  # TF at the first epoch of phase 2 (phase1_end + 1)
        "phase2_end_tf":    0.5,  # TF at the last epoch of phase 2 (phase2_end)
        "phase3_tf":        0.5,  # floor TF held for all remaining epochs
    },
}

# ── LR Scheduler ──────────────────────────────────────────────────────────────
# ReduceLROnPlateau: halves LR when val loss stalls; scheduler.step(val_loss) once per epoch.
_LR_SCHEDULER = {
    "lr_scheduler_patience": 3,    # ReduceLROnPlateau patience (epochs)
    "lr_scheduler_factor":   0.5,  # LR multiplicative decay factor on plateau
}

# ── Loss ──────────────────────────────────────────────────────────────────────
_LOSS = {
    "label_smoothing":       0.0,
}

# ── Data ──────────────────────────────────────────────────────────────────────
_DATA = {
    "max_ctx_tokens":        256,  # max tokens per context (encoder input, no <sos>)
    "max_ctx_turns":         8,    # max dialogue turns retained in context
    "max_resp_tokens":       50,   # max response tokens (excl. <sos>/<eos>);
                                   # padded length = 52 (includes <sos> + <eos>)
    "num_workers":           7,    # DataLoader workers — static: 7 of 8 CPUs, 1 reserved for OS
    "fasttext_workers":      7,    # FastText training workers (phase1 only)
}

# ── Path root (resolves to this file's directory, wherever you run from) ──────
_NEW_DIR = Path(__file__).resolve().parent   # .../nlp-chatbot-project-v2.0/

# ── Paths ─────────────────────────────────────────────────────────────────────
# Absolute paths computed from this file's location — work from any cwd.
_PATHS = {
    "artifact_dir":           str(_NEW_DIR / "artifacts"),
    "checkpoint_dir":         str(_NEW_DIR / "checkpoints"),
    "log_dir":                str(_NEW_DIR / "logs"),
    "spm_model_path":         str(_NEW_DIR / "artifacts" / "stage5_spm.model"),
    "embedding_matrix_path":  str(_NEW_DIR / "artifacts" / "stage8_embedding_matrix.npy"),
}

# ── Reproducibility ───────────────────────────────────────────────────────────
_REPRODUCIBILITY = {
    "seed": 42,   # global random seed for torch, numpy, python random, DataLoader
}

# ── Master CONFIG ─────────────────────────────────────────────────────────────
CONFIG: dict = {
    **_TOKENIZATION,
    **_ARCHITECTURE,
    **_TRAINING,
    **_TF_SCHEDULE,
    **_LR_SCHEDULER,
    **_LOSS,
    **_DATA,
    **_PATHS,
    **_REPRODUCIBILITY,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds and configure hardware-specific acceleration.

    Must be called at the top of train.py main() and evaluate.py main()
    to ensure results are reproducible across runs.

    CUDA hardware settings (applied when CUDA is available):
      - cudnn.benchmark = True:  auto-tunes cuDNN kernel selection on the first
            batch; gives a persistent speedup since our padded input shapes are
            fixed epoch-to-epoch.
      - cudnn.deterministic = False:  allows faster non-deterministic cuDNN
            algorithms; results remain statistically reproducible via the seed.
      - allow_tf32 = True:  enables TF32 acceleration for float32 matmul and cuDNN ops.

    On non-CUDA hardware (MPS / CPU), full determinism is preserved.
    """
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    if _torch.cuda.is_available():
        # Speed over bit-exact reproducibility.
        _torch.backends.cudnn.deterministic = False
        _torch.backends.cudnn.benchmark = True
        # TF32 acceleration.
        _torch.backends.cuda.matmul.allow_tf32 = True
        _torch.backends.cudnn.allow_tf32 = True
    else:
        # Local / CPU / MPS: full determinism for reproducible debugging.
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False


def get_tf_ratio(epoch: int, config: dict) -> float:
    """Return the teacher-forcing ratio for a given 1-indexed epoch.

    Three phases:
      Phase 1 (epochs 1 → phase1_end):         TF = phase1_tf  (constant, full TF)
      Phase 2 (epochs phase1_end+1 → phase2_end): TF decays linearly from
                                                  phase2_start_tf → phase2_end_tf
      Phase 3 (epochs phase2_end+1 → end):     TF = phase3_tf  (constant floor)

    The phase-2 decay uses the formula from the training spec:
        tf = phase2_start_tf
             - (epoch - phase2_start_epoch)
             * (phase2_start_tf - phase2_end_tf) / (phase2_end - phase2_start_epoch)

    Examples::

        >>> get_tf_ratio(1,  CONFIG)   # → 1.0   (phase 1)
        >>> get_tf_ratio(5,  CONFIG)   # → 1.0   (phase 1, last epoch)
        >>> get_tf_ratio(6,  CONFIG)   # → 0.9   (phase 2, first epoch)
        >>> get_tf_ratio(9,  CONFIG)   # → 0.7   (phase 2, midpoint)
        >>> get_tf_ratio(12, CONFIG)   # → 0.5   (phase 2, last epoch)
        >>> get_tf_ratio(13, CONFIG)   # → 0.5   (phase 3)
        >>> get_tf_ratio(20, CONFIG)   # → 0.5   (phase 3)

    Args:
        epoch:  Current training epoch, **1-indexed**.
        config: Config dict containing a ``"tf_schedule"`` sub-dict with keys
                ``phase1_end``, ``phase1_tf``, ``phase2_end``,
                ``phase2_start_tf``, ``phase2_end_tf``, and ``phase3_tf``.

    Returns:
        Teacher-forcing ratio as a float in [0, 1].
    """
    schedule = config["tf_schedule"]
    if epoch <= schedule["phase1_end"]:
        return schedule["phase1_tf"]
    if epoch <= schedule["phase2_end"]:
        # Linear interpolation: phase2_start_tf → phase2_end_tf over
        # the epochs in [phase1_end+1, phase2_end].
        phase2_start_epoch = schedule["phase1_end"] + 1
        tf_range = schedule["phase2_start_tf"] - schedule["phase2_end_tf"]
        step_range = schedule["phase2_end"] - phase2_start_epoch   # denominator
        tf = schedule["phase2_start_tf"] - (epoch - phase2_start_epoch) * (
            tf_range / step_range
        )
        return tf
    return schedule["phase3_tf"]


if __name__ == "__main__":
    print(json.dumps(CONFIG, indent=2))
