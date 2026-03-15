"""
dataset.py — PyTorch Dataset and DataLoader for BPE-tokenised Ubuntu pairs.

Reads the JSONL files produced by phase1.py Stage 6.
Each line: {"ctx": [int, ...], "resp": [int, ...]}

Key design decisions:
  - All splits loaded fully into memory on __init__ (1.5M pairs ≈ 400 MB total,
    well within typical RAM budgets; avoids per-epoch file I/O overhead)
  - collate_fn returns a plain dict (not a tuple) for readability in train.py
  - num_workers=4 on Windows RTX3080-12GB (set in config.py); pin_memory=True
    for maximum CPU→GPU throughput
  - persistent_workers=True (when num_workers > 0) avoids re-spawning worker
    processes every epoch — important for the full 20-epoch training run
"""

import json
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class BucketBatchSampler:
    """
    Batch sampler that groups sequences by context length to minimise padding.

    The Ubuntu corpus has high length variance (contexts range from 5 to 100
    tokens). With a random sampler, a batch of 256 sequences is padded to the
    length of the longest context in that batch — often near the 100-token cap.
    Most sequences are much shorter, so the majority of LSTM steps process <pad>.

    Algorithm (per epoch):
      1. Shuffle all sample indices randomly (seed + epoch for reproducibility).
      2. Split into non-overlapping buckets of `batch_size * bucket_size_factor`
         samples each.  Within each bucket, sort by ctx length — samples of
         similar length will end up in the same batch.
      3. Form batches of `batch_size` from the sorted buckets.
      4. Shuffle batch order (within-bucket sort is invisible to the model —
         only the order of batches within an epoch varies).

    Result: LSTM unroll depth per batch ≈ mean(ctx lengths in bucket) instead
    of max(ctx lengths in dataset). Expected speedup: 20–40% per epoch.

    Use ``set_epoch(epoch)`` each epoch so the shuffle varies while remaining
    reproducible across restarts.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        bucket_size_factor: int = 100,  # bucket = batch_size × factor samples
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size = batch_size * bucket_size_factor
        self.epoch = 0

        # Precompute ctx lengths once — used for sorting within each bucket.
        # Handle both full UbuntuPairDataset and torch.utils.data.Subset.
        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            base = dataset.dataset
            self._lengths = [len(base.pairs[i]["ctx"]) for i in dataset.indices]
        else:
            self._lengths = [len(p["ctx"]) for p in dataset.pairs]

        self._n = len(self._lengths)

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch before iterating to vary the bucket shuffle."""
        self.epoch = epoch

    def __iter__(self):
        import random as _rng
        rng = _rng.Random(self.seed + self.epoch)

        # Step 1 — global shuffle preserves stochasticity across buckets.
        indices = list(range(self._n))
        rng.shuffle(indices)

        # Step 2 — sort within each bucket by ctx length.
        buckets = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start: start + self.bucket_size]
            bucket.sort(key=lambda i: self._lengths[i])
            buckets.append(bucket)

        # Step 3 — form fixed-size batches from the sorted stream.
        flat = [i for bucket in buckets for i in bucket]
        batches = [flat[i: i + self.batch_size]
                   for i in range(0, len(flat), self.batch_size)]

        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()

        # Step 4 — shuffle batch order so the model doesn't see length-ordered
        # batches epoch after epoch (would bias gradient statistics).
        rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return self._n // self.batch_size
        return (self._n + self.batch_size - 1) // self.batch_size


class UbuntuPairDataset(Dataset):
    """
    Loads BPE-tokenised Ubuntu dialogue pairs from a JSONL file produced by
    phase1.py Stage 6.  All pairs are held in memory; at ~50 IDs per pair and
    8 bytes per int, 1.5 M pairs occupy roughly 400 MB — comfortably feasible
    on any modern training machine.
    """

    def __init__(self, jsonl_path: str, max_ctx_len: int, max_resp_len: int) -> None:
        """
        Args:
            jsonl_path:   Path to stage6_{split}_ids.jsonl.
            max_ctx_len:  Hard truncation length for context sequences.
            max_resp_len: Hard truncation length for response sequences
                          (should include <sos> and <eos>, i.e. max_resp_tokens + 2).
        """
        self.pairs: List[Dict[str, List[int]]] = []
        self.skipped_count: int = 0

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Keep the LAST max_ctx_len tokens (most recent dialogue turns),
                    # matching the right-end truncation applied in phase1 stage6.
                    raw_ctx = record["ctx"]
                    ctx  = raw_ctx[-max_ctx_len:] if len(raw_ctx) > max_ctx_len else raw_ctx
                    resp = record["resp"][:max_resp_len]
                except (json.JSONDecodeError, KeyError):
                    self.skipped_count += 1
                    warnings.warn(f"Malformed line {lineno} in {jsonl_path} — skipping.")
                    continue

                if not ctx or not resp:
                    self.skipped_count += 1
                    warnings.warn(
                        f"Empty ctx or resp after truncation at line {lineno} "
                        f"in {jsonl_path} — skipping."
                    )
                    continue

                self.pairs.append({"ctx": ctx, "resp": resp})

        print(f"Loaded {len(self.pairs)} pairs from {jsonl_path} (skipped {self.skipped_count})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Return {"ctx": List[int], "resp": List[int]}."""
        return self.pairs[idx]


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently for reproducibility.

    Called once per worker process at startup. Without this, all workers
    share the same numpy random state (forked from the main process),
    which produces correlated behaviour if any per-sample randomness is
    added in future. NumPy seed = global seed (42) + worker_id ensures
    each worker is deterministic but distinct.
    """
    import numpy as _np
    _np.random.seed(42 + worker_id)


def collate_fn(batch: List[Dict[str, List[int]]], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of variable-length pairs to the longest sequence in that batch.

    No sort needed — Encoder uses enforce_sorted=False.

    Returns:
        src         — LongTensor [batch, max_ctx_len_in_batch]   padded context IDs
        src_lengths — LongTensor [batch]                         actual ctx lengths
        trg         — LongTensor [batch, max_resp_len_in_batch]  padded response IDs
        trg_lengths — LongTensor [batch]                         actual resp lengths
    """
    ctxs = [torch.tensor(item["ctx"], dtype=torch.long) for item in batch]
    resps = [torch.tensor(item["resp"], dtype=torch.long) for item in batch]

    src_lengths = torch.tensor([len(c) for c in ctxs], dtype=torch.long)
    trg_lengths = torch.tensor([len(r) for r in resps], dtype=torch.long)

    src = pad_sequence(ctxs, batch_first=True, padding_value=pad_idx)
    trg = pad_sequence(resps, batch_first=True, padding_value=pad_idx)

    return {
        "src":         src,
        "src_lengths": src_lengths,
        "trg":         trg,
        "trg_lengths": trg_lengths,
    }


def build_dataloaders(
    artifact_dir: str,
    batch_size: int = 256,
    num_workers: int = 0,
    max_ctx_len: int = 100,
    max_resp_len: int = 42,    # 40 tokens + <sos> + <eos>
    pad_idx: int = 0,
    max_train_samples: int = 0,  # 0 = use all; set >0 to subsample train split
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from Phase 1 Stage 6 artifacts.

    All three splits are loaded fully into memory via UbuntuPairDataset.
    Files expected in artifact_dir:
        stage6_train_ids.jsonl, stage6_val_ids.jsonl, stage6_test_ids.jsonl

    Args:
        artifact_dir:       Directory containing stage6_*_ids.jsonl files.
        batch_size:         Samples per batch.
        num_workers:        DataLoader worker processes (0 = main process; use 4 on Windows RTX).
        max_ctx_len:        Hard truncation for context sequences passed to UbuntuPairDataset.
        max_resp_len:       Hard truncation for response sequences (incl. <sos>/<eos>).
        pad_idx:            Padding token ID (must match config pad_idx, typically 0).
        max_train_samples:  If > 0, randomly subsample train split to this many pairs.
                            Val and test are always loaded in full. Used by train_mini.py.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    import random as _random
    artifact_dir = Path(artifact_dir)
    _collate = partial(collate_fn, pad_idx=pad_idx)

    train_ds = UbuntuPairDataset(
        str(artifact_dir / "stage6_train_ids.jsonl"), max_ctx_len, max_resp_len
    )

    # Subsample train if requested (mini runs for fast iteration).
    if max_train_samples > 0 and len(train_ds) > max_train_samples:
        indices = _random.Random(42).sample(range(len(train_ds)), max_train_samples)
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, indices)
        print(f"  [mini] Train subsampled: {max_train_samples:,} of {len(train_ds.dataset):,} pairs (seed=42)")

    val_ds = UbuntuPairDataset(
        str(artifact_dir / "stage6_val_ids.jsonl"), max_ctx_len, max_resp_len
    )
    test_ds = UbuntuPairDataset(
        str(artifact_dir / "stage6_test_ids.jsonl"), max_ctx_len, max_resp_len
    )

    pin = torch.cuda.is_available()

    # Common kwargs for all three loaders.
    # prefetch_factor keeps the GPU fed between LSTM forward passes; only valid
    # when num_workers > 0 (DataLoader raises if set with num_workers=0).
    _common = dict(
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 4} if num_workers > 0 else {}),
    )

    # Train loader uses BucketBatchSampler to minimise padding waste.
    # batch_sampler is incompatible with batch_size/shuffle/drop_last kwargs.
    _bucket = BucketBatchSampler(
        train_ds,
        batch_size=batch_size,
        drop_last=True,
        seed=42,
    )
    train_loader = DataLoader(train_ds, batch_sampler=_bucket, **_common)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_common,
    )

    return train_loader, val_loader, test_loader
