"""
config.py — Single source of truth for all hyperparameters and paths.

Every other file in new/ imports from here. No magic numbers anywhere else.

Implementation notes:
- CONFIG is a plain dict (not a dataclass) for easy JSON serialisation
  and Colab cell override via CONFIG.update({...}).
- All paths are computed relative to this file's location (new/) so the
  code works correctly regardless of which directory you run from —
  project root, new/, or any other working directory.
- Notebooks may still override paths with absolute Drive paths, but it
  is no longer required.
"""

import json
import os
from pathlib import Path

# ── Tokenization ──────────────────────────────────────────────────────────────
_TOKENIZATION = {
    "vocab_size":            16000,   # SentencePiece BPE vocabulary size
    "spm_vocab_size":        16000,   # alias used by SPM training scripts
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
    "learning_rate":         3e-4,    # flat LR, no warmup
    "weight_decay":          1e-5,    # L2 regularisation
    "max_grad_norm":         1.0,     # gradient clipping threshold (gentler than 0.5)
    "batch_size":            256,     # per-step batch size
    "grad_accum_steps":      2,       # effective batch = batch_size × grad_accum_steps = 512
    "num_epochs":            20,      # total training epochs
    "amp_dtype":             "bfloat16",  # automatic mixed precision dtype
}

# ── Teacher-Forcing Schedule ───────────────────────────────────────────────────
# Pure epoch-lookup table — no decay formula.
# Epochs 1–15:  TF = 1.0  (full teacher forcing)
# Epochs 16–18: TF = 0.8  (slight exposure bias correction)
# Epochs 19–20: TF = 0.5  (free-running warm-up before inference)
_TF_SCHEDULE = {
    "tf_schedule": {
        "phase1_end":  15,   # last epoch (inclusive) at TF = phase1_tf
        "phase1_tf":   1.0,  # TF ratio for epochs 1 through phase1_end
        "phase2_end":  18,   # last epoch (inclusive) at TF = phase2_tf
        "phase2_tf":   0.8,  # TF ratio for epochs phase1_end+1 through phase2_end
        "phase3_tf":   0.5,  # TF ratio for all remaining epochs
    },
}

# ── LR Scheduler ──────────────────────────────────────────────────────────────
_LR_SCHEDULER = {
    "lr_scheduler_patience": 3,    # ReduceLROnPlateau patience (epochs)
    "lr_scheduler_factor":   0.5,  # LR multiplicative decay factor on plateau
}

# ── Loss ──────────────────────────────────────────────────────────────────────
_LOSS = {
    "label_smoothing":       0.0,  # OFF — smoothing fights the model at TF=1.0
}

# ── Data ──────────────────────────────────────────────────────────────────────
_DATA = {
    "max_ctx_tokens":        100,  # max tokens per context (encoder input, no <sos>)
    "max_ctx_turns":         8,    # max dialogue turns retained in context
    "max_resp_tokens":       40,   # max response tokens (excl. <sos>/<eos>);
                                   # padded length = 42 (includes <sos> + <eos>)
    "num_workers":           0,    # DataLoader workers; Colab can override to 4
}

# ── Path root (always resolves to the new/ directory, wherever you run from) ──
_NEW_DIR = Path(__file__).resolve().parent   # .../NLP_Final_Project_v2/new/

# ── Paths ─────────────────────────────────────────────────────────────────────
# Absolute paths computed from this file's location — work from any cwd.
# Notebooks may still override with Drive paths via CONFIG.update({...}).
_PATHS = {
    "artifact_dir":           str(_NEW_DIR / "artifacts"),
    "checkpoint_dir":         str(_NEW_DIR / "checkpoints"),
    "tensorboard_dir":        str(_NEW_DIR / "tb_logs"),
    "log_dir":                str(_NEW_DIR / "logs"),
    "spm_model_path":         str(_NEW_DIR / "artifacts" / "stage5_spm.model"),
    "embedding_matrix_path":  str(_NEW_DIR / "artifacts" / "stage8_embedding_matrix.npy"),
}

# ── Chat Inference ────────────────────────────────────────────────────────────
_CHAT = {
    "chat_max_history_turns": 10,   # turns kept in sliding context window
    "top_p":                  0.9,  # nucleus sampling cumulative probability
    "temperature":            0.8,  # softmax temperature for sampling
    "ngram_block":            3,    # block repeated n-grams during decoding
    "max_decode_len":         40,   # maximum tokens generated per response
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
    **_CHAT,
    **_REPRODUCIBILITY,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility (AC2-C1).

    Must be called at the top of train.py main() and evaluate.py main()
    to ensure results are reproducible across runs.

    Note: DataLoader worker seeds are controlled separately via
    ``worker_init_fn`` when ``num_workers > 0``.
    """
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

def get_tf_ratio(epoch: int, config: dict) -> float:
    """Return the teacher-forcing ratio for a given 1-indexed epoch.

    The ratio is determined by a pure epoch-lookup table stored in
    ``config["tf_schedule"]``; no decay formula is used.

    Examples::

        >>> get_tf_ratio(1,  CONFIG)   # → 1.0  (phase 1)
        >>> get_tf_ratio(16, CONFIG)   # → 0.8  (phase 2)
        >>> get_tf_ratio(19, CONFIG)   # → 0.5  (phase 3)

    Args:
        epoch:  Current training epoch, **1-indexed**.
        config: Config dict containing a ``"tf_schedule"`` sub-dict with keys
                ``phase1_end``, ``phase1_tf``, ``phase2_end``, ``phase2_tf``,
                and ``phase3_tf``.

    Returns:
        Teacher-forcing ratio as a float in [0, 1].
    """
    schedule = config["tf_schedule"]
    if epoch <= schedule["phase1_end"]:
        return schedule["phase1_tf"]
    if epoch <= schedule["phase2_end"]:
        return schedule["phase2_tf"]
    return schedule["phase3_tf"]


if __name__ == "__main__":
    print(json.dumps(CONFIG, indent=2))
