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
import math
import os
from pathlib import Path


def _container_cpu_count() -> int:
    """Return the number of CPUs actually assigned to this process.

    Handles three environments correctly:
      - Linux container (cgroup v2, e.g. OpenShift): reads /sys/fs/cgroup/cpu.max
        so the host CPU count (e.g. 256) is never used — only the quota (e.g. 8).
      - Linux container (cgroup v1): reads cpu.cfs_quota_us / cpu.cfs_period_us.
      - Bare-metal Linux / Windows / macOS: falls back to os.cpu_count().

    Returns at least 1.
    """
    # cgroup v2  (OpenShift / modern Docker)
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if cpu_max != "max":
            quota, period = cpu_max.split()
            return max(1, int(quota) // int(period))
    except (FileNotFoundError, ValueError):
        pass

    # cgroup v1
    try:
        quota  = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (FileNotFoundError, ValueError):
        pass

    # Bare-metal / Windows / macOS
    return max(1, os.cpu_count() or 1)


# Number of CPUs assigned to this process (container-aware).
_CPU_COUNT = _container_cpu_count()

# Workers = all assigned CPUs minus 1 reserved for the OS scheduler.
# Windows uses multiprocessing 'spawn' so DataLoader worker startup is
# expensive; beyond ~4 workers the overhead exceeds the throughput gain.
# On Linux / containers the cgroup quota is respected and all CPUs are used.
_WORKERS = max(1, _CPU_COUNT - 1)
if os.name == "nt":                    # Windows
    _WORKERS = min(4, _WORKERS)

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
    "batch_size":            1024,    # A100-80GB: 2× throughput; Adam's adaptive v_t absorbs variance reduction
    "grad_accum_steps":      1,       # no accumulation needed
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
    "lr_scheduler_patience": 4,    # ReduceLROnPlateau patience (epochs)
                                    # 4 (not 3) — val set is only ~187 batches so
                                    # val loss is noisier; extra epoch avoids
                                    # premature LR halving on a single noisy reading
    "lr_scheduler_factor":   0.5,  # LR multiplicative decay factor on plateau
}

# ── Loss ──────────────────────────────────────────────────────────────────────
_LOSS = {
    "label_smoothing":       0.1,   # prevents overconfidence on short IRC responses
                                      # (median resp=14 tok); standard seq2seq practice
}

# ── Data ──────────────────────────────────────────────────────────────────────
_DATA = {
    "max_ctx_tokens":        256,  # max tokens per context (encoder input, no <sos>)
    "max_ctx_turns":         8,    # max dialogue turns retained in context
    "max_resp_tokens":       50,   # max response tokens (excl. <sos>/<eos>);
                                   # padded length = 52 (includes <sos> + <eos>)
    "num_workers":           _WORKERS,  # DataLoader workers (cpu_count - 1, cgroup-aware)
    "fasttext_workers":      _WORKERS,  # FastText training workers (phase1 only)
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
