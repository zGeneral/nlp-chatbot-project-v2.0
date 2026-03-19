"""
train.py — Training loop for clean-from-scratch Seq2Seq chatbot.

Trains both "baseline" (no attention) and "attention" (Bahdanau) models.
Both models are trained with identical hyperparameters — the only difference
is the attention mechanism — providing a controlled apples-to-apples ablation.

  TEACHER FORCING STRATEGY (3-phase schedule):
    Epochs  1– 5:  TF = 1.0        — Foundation: burn in basic token representations.
                                      Loss drops from ~5.5 → ~4.25 here.
    Epochs  6–12:  TF 0.9 → 0.5    — Annealing: linear decay; decoder begins
                                      practicing self-feeding while representations
                                      are still being refined.
    Epochs 13–20:  TF = 0.5        — Maturation: hold at floor; both models
                                      fully adapt to semi-autoregressive generation.

  TF floor = 0.5 for both models. The baseline decoder has no attention to
  recover from compounding errors; TF < 0.5 causes collapse. Floor is kept
  identical for a fair comparison.

  LR STRATEGY: ReduceLROnPlateau.
    factor=0.5, patience=3, min_lr=1e-5.
    scheduler.step(val_loss) is called once PER EPOCH after validation.
    The Phase 2 reset (best_val_loss=inf at epoch 6) prevents plateau from
    prematurely halving LR during the TF=1.0 exposure-bias phase.

  EARLY STOPPING:
    Patience = 4 epochs, monitored only from Phase 2 onward (epoch > 5).

  GRADIENT CLIPPING:
    clip = 1.0. Allows valid large gradients during TF=1.0 phase.
"""

import os
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sacrebleu as _sacrebleu
import sentencepiece as _spm

from config import CONFIG, get_tf_ratio, set_seed
from dataset import build_dataloaders
from models import build_model


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: dict,
    device: torch.device,
    epoch: int,
    global_step: int,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None,
) -> Tuple[float, float, int]:
    """
    One training epoch with bf16 AMP and gradient accumulation.

    Returns:
        (avg_train_loss, avg_grad_norm, updated_global_step)
    """
    model.train()

    vocab_size: int = config["vocab_size"]
    grad_accum_steps: int = config["grad_accum_steps"]
    max_grad_norm: float = config["max_grad_norm"]
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))

    tf_ratio: float = get_tf_ratio(epoch, config)

    total_loss = 0.0
    total_grad_norm = 0.0
    _accum_loss = 0.0   # accumulates every micro-batch loss for correct window average
    n_updates = 0
    num_batches = len(loader)

    pbar = tqdm(
        enumerate(loader),
        total=num_batches,
        desc=f"  Epoch {epoch:3d} train",
        dynamic_ncols=True,
        unit="batch",
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in pbar:
        src: torch.Tensor = batch["src"].to(device, non_blocking=True)                   # [B, src_len]
        src_lengths: torch.Tensor = batch["src_lengths"].to(device, non_blocking=True)   # [B]
        trg: torch.Tensor = batch["trg"].to(device, non_blocking=True)                   # [B, trg_len]

        # ── Forward pass under bf16 AMP ────────────────────────────────────
        # device.type is used dynamically so CPU debug runs don't warn/fail.
        _device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        with torch.amp.autocast(device_type=_device_type, dtype=_amp_dtype,
                                enabled=_device_type == "cuda"):
            # output: [B, trg_len-1, vocab_size]
            output = model(src, src_lengths, trg, teacher_forcing_ratio=tf_ratio)

            # trg[:, 1:] excludes <sos>; output already aligns with trg[1:] steps.
            loss = criterion(
                output.reshape(-1, vocab_size),
                trg[:, 1:].reshape(-1),
            )
            # Scale loss for gradient accumulation so each micro-step contributes
            # an equal 1/grad_accum_steps share to the final gradient.
            scaled_loss = loss / grad_accum_steps

        scaled_loss.backward()

        # Accumulate every micro-batch loss BEFORE the should_step gate so the
        # reported train loss is the true average across the full accum window,
        # not just the last micro-batch (which was the previous bug).
        _accum_loss += loss.item()

        is_last_batch = (batch_idx + 1) == num_batches
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch

        if should_step:
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            ).item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Divide by the number of micro-batches in this window.
            # The last window may be smaller if num_batches % grad_accum_steps != 0.
            _window = (batch_idx % grad_accum_steps) + 1
            total_loss += _accum_loss / _window
            _accum_loss = 0.0

            total_grad_norm += grad_norm
            n_updates += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm:.3f}",
            )

    avg_train_loss = total_loss / max(n_updates, 1)
    avg_grad_norm = total_grad_norm / max(n_updates, 1)
    return avg_train_loss, avg_grad_norm, global_step


def evaluate_loss(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """
    Primary validation pass — teacher forcing FULLY ENABLED (ratio=1.0).

    Using TF=1.0 here gives a loss that is:
      - Directly comparable to train loss (same conditioning)
      - Stable across all epochs (unaffected by the model's own generation quality)
      - The correct signal for ReduceLROnPlateau and best-checkpoint saving

    The previous design used TF=0.0 for val loss, which caused a spurious spike
    at epoch 2 when the model learned EOS: post-EOS tokens became garbage and
    inflated val loss even as the model was genuinely improving.

    Returns:
        (val_loss, val_ppl)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    _device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

    with torch.no_grad():
        for batch in tqdm(loader, desc="  val", unit="batch", dynamic_ncols=True, leave=False):
            src: torch.Tensor = batch["src"].to(device, non_blocking=True)
            src_lengths: torch.Tensor = batch["src_lengths"].to(device, non_blocking=True)
            trg: torch.Tensor = batch["trg"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=_device_type, dtype=amp_dtype,
                                    enabled=_device_type == "cuda"):
                output = model(src, src_lengths, trg, teacher_forcing_ratio=1.0)

            vocab_size: int = output.size(-1)
            loss = criterion(
                output.reshape(-1, vocab_size),
                trg[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


def evaluate_generation(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    eos_idx: int = 3,
    pad_idx: int = 0,
    sp_model=None,
    n_gen_samples: int = 1024,
) -> Tuple[float, float, float, float, float]:
    """
    Generation-quality pass — teacher forcing DISABLED (ratio=0.0).

    Runs on the first n_gen_samples validation items (may span multiple
    batches depending on val_loader batch size) to keep per-epoch overhead low.

    Five metrics are computed:

    gen_loss:
        Cross-entropy over *active* positions only — positions up to and
        including the first EOS token.  Post-EOS tokens are masked out so
        the loss is not inflated by garbage generated after the model's
        intended stopping point.  Formula:
            active = (cumsum(preds == eos_idx) <= 1) & (trg != pad_idx)
            gen_loss = sum(CE[active]) / n_active

    avg_pred_len:
        Mean number of tokens generated up to and including the first EOS.
        Collapse detector: a sudden drop to 2–3 signals the decoder has
        learned to always output EOS immediately (mode collapse).

    bleu:
        Corpus BLEU-4 (sacrebleu) on the decoded n_gen_samples pairs.
        Hypotheses and references are decoded from token IDs using the
        SentencePiece model.  Returns 0.0 if sp_model is None.

    Returns:
        (gen_loss, avg_pred_len, bleu, token_f1, avg_active_tokens)
    """
    model.eval()
    total_ce_sum = 0.0        # sum of per-token CE over all active positions (for true global avg)
    total_pred_len = 0.0
    total_active_tokens = 0   # total active (pre-EOS) token count across all batches
    n_batches = 0
    n_seen = 0
    hypotheses: List[str] = []
    references: List[str] = []
    _device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    _special = {0, 1, 2, eos_idx, pad_idx}  # unk/bos/eos/pad — strip from BLEU decode

    with torch.no_grad():
        for batch in tqdm(loader, desc="  gen", unit="batch", dynamic_ncols=True, leave=False):
            if n_seen >= n_gen_samples:
                break

            src: torch.Tensor = batch["src"].to(device, non_blocking=True)
            src_lengths: torch.Tensor = batch["src_lengths"].to(device, non_blocking=True)
            trg: torch.Tensor = batch["trg"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=_device_type, dtype=amp_dtype,
                                    enabled=_device_type == "cuda"):
                output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)

            vocab_size: int = output.size(-1)
            preds = output.argmax(-1)  # [B, T]

            # ── Post-EOS mask ─────────────────────────────────────────────────
            # cumsum of EOS hits across time: 0 before first EOS, ≥1 at and after.
            # Keep positions where cumsum ≤ 1 (up to and including first EOS).
            eos_cumsum = preds.eq(eos_idx).long().cumsum(dim=1)          # [B, T]
            active = (eos_cumsum <= 1) & trg[:, 1:].ne(pad_idx)         # [B, T]

            # Intentionally uses raw F.cross_entropy (no label_smoothing) so
            # gen_loss measures sharp per-token CE — what the model actually
            # predicts, not the smoothed training objective.  This makes it a
            # cleaner proxy for generation quality than the smoothed criterion.
            per_tok = F.cross_entropy(
                output.reshape(-1, vocab_size),
                trg[:, 1:].reshape(-1),
                ignore_index=pad_idx,
                reduction="none",
            )  # [B*T]
            n_active = active.reshape(-1).sum().clamp(min=1)
            # Accumulate raw CE sum (not per-batch average) so the final
            # avg_gen_loss is a true global average over all active tokens,
            # not an average-of-batch-averages that over-weights small batches.
            total_ce_sum += per_tok[active.reshape(-1)].sum().item()
            total_active_tokens += n_active.item()

            # ── PredLen (collapse detector) ───────────────────────────────────
            has_eos   = preds.eq(eos_idx).any(dim=1)
            first_eos = preds.eq(eos_idx).long().argmax(dim=1).add(1)
            full_len  = torch.full_like(first_eos, preds.size(1))
            lengths   = torch.where(has_eos, first_eos, full_len)
            total_pred_len += lengths.float().mean().item()

            # ── BLEU decode ───────────────────────────────────────────────────
            if sp_model is not None:
                for i in range(preds.size(0)):
                    hyp_ids = preds[i].cpu().tolist()
                    ref_ids = trg[i, 1:].cpu().tolist()
                    try:
                        hyp_ids = hyp_ids[:hyp_ids.index(eos_idx)]
                    except ValueError:
                        pass
                    try:
                        ref_ids = ref_ids[:ref_ids.index(eos_idx)]
                    except ValueError:
                        pass
                    hyp_ids = [t for t in hyp_ids if t not in _special]
                    ref_ids = [t for t in ref_ids  if t not in _special]
                    hypotheses.append(sp_model.decode(hyp_ids))
                    references.append(sp_model.decode(ref_ids))

            n_batches += 1
            n_seen += preds.size(0)

    # True global average: total CE sum / total active tokens (not avg-of-avgs)
    avg_gen_loss = total_ce_sum / max(total_active_tokens, 1)
    avg_pred_len = total_pred_len / max(n_batches, 1)
    # Store total_active_tokens (not per-batch avg) so gen_loss is reconstructable from history
    avg_active_tokens = total_active_tokens / max(n_batches, 1)  # per-batch avg for display
    # force=True suppresses sacrebleu's "tokenized period" warning — SPM decode()
    # produces spaces around punctuation that sacrebleu misidentifies as tokenized
    # MT output.  Our text is correctly decoded; the spacing is a SPM artifact.
    bleu = _sacrebleu.corpus_bleu(hypotheses, [references], force=True).score if hypotheses else 0.0

    # ── Corpus Token F1 (unigram word overlap) ────────────────────────────────
    # More robust than BLEU-4 for technical support: many valid phrasings of
    # the same answer score near-zero BLEU but high token overlap.
    # Computed at corpus level: accumulate TP/hyp_len/ref_len across samples,
    # then compute precision/recall/F1 once (avoids averaging of averages).
    tp_total = hyp_total = ref_total = 0
    for hyp_str, ref_str in zip(hypotheses, references):
        hyp_toks = hyp_str.split()
        ref_toks = ref_str.split()
        common   = sum((Counter(hyp_toks) & Counter(ref_toks)).values())
        tp_total  += common
        hyp_total += len(hyp_toks)
        ref_total += len(ref_toks)
    p = tp_total / max(hyp_total, 1)
    r = tp_total / max(ref_total, 1)
    token_f1 = (2 * p * r / (p + r) * 100) if (p + r) > 0 else 0.0

    return avg_gen_loss, avg_pred_len, bleu, token_f1, avg_active_tokens


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Build AdamW optimizer and ReduceLROnPlateau scheduler.

    IMPORTANT: scheduler.step(val_loss) must be called ONCE PER EPOCH after
    validation — NOT once per optimizer step.

    Combined with the Phase 2 reset (best_val_loss=inf at epoch phase1_end+1),
    the plateau scheduler won't prematurely reduce LR during TF=1.0 because:
      - Epochs 1–5 (Phase 1): early stopping is disabled → even if plateau fires,
        LR reduction just lets the model fine-tune (no early exit).
      - At epoch 6 (Phase 2 start): best_val_loss resets → plateau's internal
        `best` is also reset (via a fresh scheduler), so the first val_loss of
        Phase 2 becomes the new reference. LR will only reduce if Phase 2 val
        loss fails to improve for lr_patience=3 consecutive epochs.

    Args:
        model:   The model whose parameters to optimise.
        config:  CONFIG dict — must contain ``learning_rate``, ``weight_decay``,
                 ``lr_scheduler_patience``, ``lr_scheduler_factor``, ``lr_min``.

    Returns:
        (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("lr_scheduler_factor", 0.5),
        patience=config.get("lr_scheduler_patience", 3),
        min_lr=config.get("lr_min", 1e-5),
    )
    return optimizer, scheduler


def train_model(model_type: str, config: dict, device: torch.device) -> Dict[str, List]:
    """
    Full training run for one model_type ("baseline" or "attention").

    Saves:
      {checkpoint_dir}/{model_type}_best.pt      — best val-loss checkpoint (atomic)
      {checkpoint_dir}/{model_type}_last.pt      — last-epoch checkpoint (atomic, for resume)
      {checkpoint_dir}/{model_type}_history.json — training history

    Returns:
        Dict with per-epoch train_loss, val_loss, tf_ratios, and lrs lists.
    """
    # Stash model_type in config so train_epoch can name periodic checkpoints.
    config = dict(config)
    config["_model_type"] = model_type

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        artifact_dir=config["artifact_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_ctx_len=config["max_ctx_tokens"],
        max_resp_len=config["max_resp_tokens"] + 2,   # +2 for <sos> and <eos>
        pad_idx=config["pad_idx"],
    )

    # ── 1b. SentencePiece model for BLEU decoding ─────────────────────────────
    _sp = None
    _spm_path = config.get("spm_model_path", "")
    if _spm_path and Path(_spm_path).exists():
        _sp = _spm.SentencePieceProcessor()
        _sp.load(_spm_path)
        print(f"[{model_type}] SP model loaded for BLEU: {_spm_path}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model = build_model(model_type, config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{model_type}] Trainable parameters: {num_params:,}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        print(f"  GPU memory after model load: {allocated:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved")

    # ── 3. Optimizer + scheduler ──────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    # ── 4. Loss ───────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=config["pad_idx"],
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))

    # ── 5. Resume — prefer last-epoch checkpoint to avoid re-running epochs ──
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_type}_best.pt")
    last_ckpt_path = os.path.join(checkpoint_dir, f"{model_type}_last.pt")

    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0
    _patience: int = config.get("patience", 0)      # 0 = disabled; set >0 in train_mini
    _no_improve: int = 0
    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss":   [],   # TF=1.0 — primary signal for scheduler + checkpoint
        "gen_loss":          [],   # TF=0.0, post-EOS masked — generation quality proxy
        "avg_active_tokens": [],   # avg active (pre-EOS) tokens per batch — gen_loss denominator
        "bleu":              [],   # corpus BLEU-4 on n_gen_samples val pairs
        "token_f1":          [],   # corpus token F1 (unigram word overlap, %) — robust for technical support
        "avg_pred_len": [], # collapse detector
        "tf_ratios":  [],
        "lrs":        [],
    }

    # Prefer last checkpoint (has most recent epoch) over best checkpoint
    resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else (
        best_ckpt_path if os.path.exists(best_ckpt_path) else None
    )
    if resume_path:
        print(f"[{model_type}] Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        if "history" in ckpt:
            history = ckpt["history"]

        # Issue #4 fix: derive best_val_loss from history minimum, not from the
        # checkpoint's val_loss field.  Resuming from last.pt at epoch N gives
        # val_loss=epoch_N_loss, which may be worse than the true best (saved in
        # best.pt at an earlier epoch).
        #
        # Phase 2 resume caveat: if we crash and resume during Phase 2, using
        # min(all history) would restore the Phase 1 best (e.g. 3.9) as the
        # baseline, defeating the Phase 2 reset (which set best=inf so Phase 2
        # epochs only compete against each other).  Fix: when resuming mid-Phase
        # 2, slice history to Phase 2 entries only.
        if history.get("val_loss"):
            _phase1_end = config["tf_schedule"]["phase1_end"]
            if start_epoch > _phase1_end + 1:
                # Resuming inside Phase 2 — only Phase 2 val_losses count.
                _phase2_losses = history["val_loss"][_phase1_end:]
                best_val_loss = min(_phase2_losses) if _phase2_losses else float("inf")
            else:
                best_val_loss = min(history["val_loss"])
        else:
            best_val_loss = ckpt.get("val_loss", float("inf"))

        # Issue #6 fix: restore _no_improve from checkpoint so patience budget
        # survives a crash/resume cycle instead of silently resetting to 0.
        _no_improve = ckpt.get("no_improve", 0)

        # ReduceLROnPlateau: load optimizer state to restore the exact LR at
        # the time the checkpoint was saved. The scheduler is NOT loaded from
        # state dict — it starts fresh each resume, which is safe because
        # ReduceLROnPlateau reads the current LR from optimizer.param_groups.
        # This avoids scheduler-type mismatches between runs (e.g., cosine
        # checkpoint loaded into a plateau scheduler).
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{model_type}] Scheduler starts fresh at LR = {lr_now:.3e}")

        print(f"[{model_type}] Resumed at epoch {start_epoch}, step {global_step}, "
              f"best_val={best_val_loss:.4f}")

    num_epochs: int = config["num_epochs"]
    print(f"[{model_type}] Training epochs {start_epoch}–{num_epochs}")
    print(f"  Val(TF1)  : TF=1.0, full val set  — drives scheduler + checkpoint (epoch-comparable)")
    print(f"  GenLoss   : TF=0.0, post-EOS mask  — note: scale shifts as Len shrinks (not epoch-comparable)")
    print(f"  BLEU/F1   : corpus BLEU-4 + token F1 on {config.get('n_gen_samples', 1024)} val samples")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # flush async GPU ops so wall time is accurate

        # Advance bucket sampler seed so each epoch shuffles differently.
        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(epoch)

        # 7a. Teacher forcing ratio for this epoch (constant within epoch).
        tf_ratio = get_tf_ratio(epoch, config)

        # 7b. Train.
        train_loss, avg_gnorm, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            epoch=epoch,
            global_step=global_step,
            scheduler=scheduler,
        )

        # 7c. Validate — two passes.
        #     evaluate_loss : TF=1.0, full val set  → scheduler + checkpoint
        #     evaluate_generation : TF=0.0, first 1024 items → gen metrics + BLEU
        val_loss, val_ppl = evaluate_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=_amp_dtype,
        )
        gen_loss, avg_pred_len, bleu, token_f1, avg_active_tokens = evaluate_generation(
            model=model,
            loader=val_loader,
            device=device,
            amp_dtype=_amp_dtype,
            eos_idx=config.get("eos_idx", 3),
            pad_idx=config.get("pad_idx", 0),
            sp_model=_sp,
            n_gen_samples=config.get("n_gen_samples", 1024),
        )

        # 7d. Step ReduceLROnPlateau with TF=1.0 val loss (stable primary signal).
        scheduler.step(val_loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # ensure all GPU work is done before measuring
        elapsed = time.time() - epoch_start
        lr: float = optimizer.param_groups[0]["lr"]

        # 7f. Save best checkpoint (atomic: write .tmp then os.replace).
        # Reset best_val_loss at the Phase 2 boundary.  When TF drops at
        # epoch phase1_end+1, the decoder switches from ground-truth to its
        # own predictions, causing a discontinuous jump in validation loss.
        # Comparing against the TF=1.0 best is unfair — the model needs
        # several epochs to recover and potentially surpass it.  Resetting
        # here gives it a clean baseline for the annealing phase.
        _phase1_end = config["tf_schedule"]["phase1_end"]
        if epoch == _phase1_end + 1:
            best_val_loss = float("inf")
            _no_improve = 0
            # Reset ReduceLROnPlateau's internal state so Phase 1 bad-epoch
            # counts don't consume Phase 2 patience budget.  Without this the
            # scheduler's own `best` and `num_bad_epochs` carry over, and LR
            # can halve on epoch 7 before the model has adapted to lower TF.
            scheduler.best = float("inf")
            scheduler.num_bad_epochs = 0
            print(f"[{model_type}] Phase 2 begins (epoch {epoch}) — "
                  f"resetting best_val_loss, early-stopping counter, and scheduler state")

        # Capture improvement BEFORE updating best_val_loss.
        _improved = val_loss < best_val_loss
        if _improved:
            best_val_loss = val_loss
            ckpt_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "train_loss": train_loss,
                "tf_ratio": tf_ratio,
                "no_improve": _no_improve,
                "config": dict(config),
                "history": history,
            }
            tmp_path = best_ckpt_path + ".tmp"
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, best_ckpt_path)

        # 7g. Update history.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["gen_loss"].append(gen_loss)
        history["avg_active_tokens"].append(avg_active_tokens)
        history["bleu"].append(bleu)
        history["token_f1"].append(token_f1)
        history["avg_pred_len"].append(avg_pred_len)
        history["tf_ratios"].append(tf_ratio)
        history["lrs"].append(lr)

        # Early stopping active from Phase 2 onward.
        if _patience > 0 and epoch > config["tf_schedule"]["phase1_end"]:
            if _improved:
                _no_improve = 0
            else:
                _no_improve += 1
                if _no_improve >= _patience:
                    print(f"[{model_type}] Early stopping at epoch {epoch} "
                          f"(no improvement for {_patience} epochs)")
                    # Save last checkpoint before breaking.
                    last_ckpt_data = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_type": model_type,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "no_improve": _no_improve,
                        "config": dict(config),
                        "history": history,
                    }
                    tmp_last = last_ckpt_path + ".tmp"
                    torch.save(last_ckpt_data, tmp_last)
                    os.replace(tmp_last, last_ckpt_path)
                    break

        # 7g2. Save last-epoch checkpoint (resume-friendly — prefers this over best).
        last_ckpt_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "no_improve": _no_improve,
            "config": dict(config),
            "history": history,
        }
        tmp_last = last_ckpt_path + ".tmp"
        torch.save(last_ckpt_data, tmp_last)
        os.replace(tmp_last, last_ckpt_path)

        # 7h. Epoch summary.
        _improved_marker = " ✓" if _improved else ""
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val(TF1): {val_loss:.4f} | "
            f"PPL: {val_ppl:.2f} | "
            f"GenLoss: {gen_loss:.4f} | "
            f"BLEU: {bleu:.2f} | "
            f"F1: {token_f1:.2f} | "
            f"Len: {avg_pred_len:.1f} | "
            f"LR: {lr:.2e} | "
            f"TF: {tf_ratio:.2f} | "
            f"Grad: {avg_gnorm:.3f} | "
            f"{elapsed:.1f}s"
            f"{_improved_marker}"
        )

    # ── 8. Save history JSON (atomic write) ─────────────────────────────────
    history_path = os.path.join(checkpoint_dir, f"{model_type}_history.json")
    _tmp_hist = history_path + ".tmp"
    with open(_tmp_hist, "w") as fh:
        json.dump(history, fh, indent=2)
    os.replace(_tmp_hist, history_path)
    print(f"[{model_type}] History saved to {history_path}")

    return history


def main(cfg: dict = None, script_name: str = "train") -> None:
    """Train baseline model, then attention model.

    Args:
        cfg:         Config dict overrides. Defaults to CONFIG from config.py.
        script_name: Used for the run log filename (e.g. "train").
    """
    # ── Build active config — start from provided cfg or global CONFIG ────────
    active_cfg = dict(cfg if cfg is not None else CONFIG)

    set_seed(active_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Startup banner ────────────────────────────────────────────────────────
    from config import _GPU_PROFILES, _gpu_profile
    eff_batch = active_cfg["batch_size"] * active_cfg["grad_accum_steps"]
    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    _matched  = next((k for k in _GPU_PROFILES if k in gpu_name), None)
    _profile_label = f"profile={_matched!r}" if _matched else "profile=default (no match)"
    print("=" * 70)
    print(f"  Device  : {device}  ({gpu_name})")
    print(f"  Profile : {_profile_label}")
    print(f"  Batch   : {active_cfg['batch_size']} × {active_cfg['grad_accum_steps']} accum "
          f"= {eff_batch} effective")
    print(f"  Epochs  : {active_cfg['num_epochs']}  |  LR : {active_cfg['learning_rate']:.1e}")
    print("=" * 70)

    os.makedirs(active_cfg["checkpoint_dir"], exist_ok=True)

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING: baseline (no attention)")
    print("=" * 70)
    baseline_history = train_model("baseline", active_cfg, device)

    # Clear GPU cache between models.  After baseline training the CUDA
    # allocator retains cached VRAM blocks.  The attention model needs extra
    # per-step memory (attention weight tensors over the full source sequence)
    # and can OOM due to fragmentation even when total capacity is sufficient.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("  GPU cache cleared between models.")

    # ── Separator ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING: attention (Bahdanau)")
    print("=" * 70)

    # ── Attention ─────────────────────────────────────────────────────────────
    attention_history = train_model("attention", active_cfg, device)

    print("\nTraining complete.")
    print(f"  Baseline  best val loss : {min(baseline_history['val_loss']):.4f}")
    print(f"  Attention best val loss : {min(attention_history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
