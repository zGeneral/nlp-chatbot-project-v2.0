"""
finetune.py — Fine-tune a trained run-3 checkpoint with a lower TF floor.

Loads a baseline or attention checkpoint produced by train.py and continues
training with a reduced teacher-forcing floor (default: 0.30) to address the
exposure-bias / repetition-loop problem identified in run-3 inference.

Design rationale
────────────────
Run 3 training converged well (Val=5.1305, PPL=169.1) but inference shows
repetition loops ("apt-get install vim sudo apt-get install vim") because:
  1. TF floor = 0.50 means the model was never trained to recover from its own
     multi-step errors; it is out-of-distribution at inference time.
  2. The peaked distribution on high-frequency tokens (▁i, ▁you) amplifies
     any initial error in a greedy / beam chain.

Fine-tuning from the converged checkpoint with TF floor = 0.30 (and
optionally 0.0 for run 5) gives the decoder a harder curriculum without
re-learning vocabulary and grammar from scratch.

Key differences from train.py
──────────────────────────────
  - Loads weights + optimizer + scheduler from checkpoint (warm start).
  - No Phase 1 (TF=1.0) — weights are already burned in.
  - Flat LR from the start (lower than original: default 1e-4).
  - Phase 3 TF floor overridden to ft_tf_floor (default 0.30).
  - TF decays linearly from ft_tf_start (default 0.50) → ft_tf_floor over
    ft_anneal_epochs (default 5), then holds at floor for remaining epochs.
  - Early stopping and LR scheduler remain active throughout.
  - Saves to {model_type}_ft_best.pt / {model_type}_ft_last.pt so run-3
    checkpoints are never overwritten.

Usage
─────
    python finetune.py --model baseline
    python finetune.py --model attention
    python finetune.py --model baseline --tf-floor 0.0 --epochs 15
    python finetune.py --model baseline --tf-floor 0.3 --lr 5e-5 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import sentencepiece as _spm
from tqdm import tqdm

from config import CONFIG, set_seed
from dataset import build_dataloaders
from models import build_model

# Re-use heavy-lifting functions from train.py directly.
from train import (
    train_epoch,
    evaluate_loss,
    evaluate_generation,
    log_decoded_samples,
    log_probe_responses,
    compute_attention_entropy,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _ft_tf_ratio(epoch: int, ft_tf_start: float, ft_tf_floor: float,
                 ft_anneal_epochs: int) -> float:
    """Linear decay from ft_tf_start → ft_tf_floor over ft_anneal_epochs, then hold."""
    if ft_anneal_epochs <= 0 or ft_tf_start <= ft_tf_floor:
        return ft_tf_floor
    if epoch <= ft_anneal_epochs:
        frac = (epoch - 1) / max(ft_anneal_epochs - 1, 1)
        return ft_tf_start - frac * (ft_tf_start - ft_tf_floor)
    return ft_tf_floor


# ── Main fine-tune loop ───────────────────────────────────────────────────────

def finetune(
    model_type: str,
    ft_tf_floor: float,
    ft_tf_start: float,
    ft_anneal_epochs: int,
    ft_lr: float,
    ft_epochs: int,
    config: dict,
    device: torch.device,
    lr_schedule: str = "cosine",
    patience: int = 0,
) -> Dict[str, List]:
    """
    Fine-tune model_type from its best run-3 checkpoint.

    Args:
        model_type:        "baseline" or "attention"
        ft_tf_floor:       TF floor to anneal to (e.g. 0.0 or 0.30)
        ft_tf_start:       Starting TF (should match run-3 floor, default 0.50)
        ft_anneal_epochs:  Epochs over which to linearly decay TF to floor
        ft_lr:             Initial learning rate (lower than run-3, e.g. 1e-4)
        ft_epochs:         Total fine-tune epochs
        config:            Base CONFIG dict (will be shallow-copied + patched)
        device:            torch.device
        lr_schedule:       "cosine" | "constant" | "plateau"
                           cosine: full LR during anneal, decays over full run (recommended)
                           constant: fixed LR throughout
                           plateau: ReduceLROnPlateau on gen_loss (not val_loss)
        patience:          Early stopping patience on gen_loss (0 = disabled)
                           Default 0 because val_loss WILL rise during TF→0 fine-tuning;
                           using val_loss for early stopping terminates runs prematurely.

    Returns:
        history dict with per-epoch metrics
    """
    config = dict(config)
    config["_model_type"] = model_type

    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    src_ckpt = os.path.join(checkpoint_dir, f"{model_type}_best.pt")
    ft_best  = os.path.join(checkpoint_dir, f"{model_type}_ft_best.pt")
    ft_last  = os.path.join(checkpoint_dir, f"{model_type}_ft_last.pt")

    if not os.path.exists(src_ckpt):
        raise FileNotFoundError(
            f"Source checkpoint not found: {src_ckpt}\n"
            "Run train.py (run 3) first to produce the base checkpoint."
        )

    print("=" * 70)
    print(f"  FINE-TUNE: {model_type.upper()}")
    print("=" * 70)
    print(f"  Source checkpoint : {src_ckpt}")
    print(f"  TF schedule       : {ft_tf_start:.2f} → {ft_tf_floor:.2f} "
          f"over {ft_anneal_epochs} epochs, then hold")
    print(f"  Learning rate     : {ft_lr:.2e}")
    print(f"  Epochs            : {ft_epochs}")
    print(f"  Label smoothing   : {config.get('label_smoothing', 0.0)}")
    print(f"  Output best       : {ft_best}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        artifact_dir=config["artifact_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_ctx_len=config["max_ctx_tokens"],
        max_resp_len=config["max_resp_tokens"] + 2,
        pad_idx=config["pad_idx"],
    )

    # ── SPM for BLEU ──────────────────────────────────────────────────────────
    _sp = None
    _spm_path = config.get("spm_model_path", "")
    if _spm_path and Path(_spm_path).exists():
        _sp = _spm.SentencePieceProcessor()
        _sp.load(_spm_path)

    # ── Model — build architecture then load weights ──────────────────────────
    model = build_model(model_type, config, device)

    ckpt = torch.load(src_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    src_epoch = ckpt.get("epoch", "?")
    src_val   = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded weights from epoch {src_epoch}  val_loss={src_val:.4f}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}\n")

    # ── Optimizer — fresh, lower LR ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ft_lr,
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # ── LR scheduler — cosine by default (plateau is wrong for TF→0 fine-tune)
    # ReduceLROnPlateau halves LR when val_loss stops improving. But val_loss
    # WILL rise as TF drops — it would reduce LR exactly when the model needs
    # full learning capacity to adapt to autoregressive generation.
    # Cosine: full LR during annealing, smooth decay during stabilisation.
    # Plateau (on gen_loss): available for comparison; tracks the right metric.
    lr_schedule = lr_schedule.lower()
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ft_epochs, eta_min=config.get("lr_min", 1e-5),
        )
        _scheduler_on_gen = False
    elif lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=ft_epochs)
        _scheduler_on_gen = False
    else:  # "plateau" — steps on gen_loss, not val_loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.get("lr_scheduler_factor", 0.5),
            patience=config.get("lr_scheduler_patience", 3),
            min_lr=config.get("lr_min", 1e-5),
        )
        _scheduler_on_gen = True

    print(f"  LR schedule       : {lr_schedule}  (early-stop patience={patience}, "
          f"0=disabled)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=config["pad_idx"],
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    # (amp_dtype handled internally by train_epoch/evaluate_* via config)

    # ── History ───────────────────────────────────────────────────────────────
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [], "gen_loss": [],
        "bleu": [], "token_f1": [], "avg_pred_len": [],
        "avg_active_tokens": [], "attn_entropy": [],
        "tf_ratios": [], "lrs": [],
    }

    best_val_loss = float("inf")
    best_gen_loss = float("inf")   # primary criterion for ft checkpoint selection
    _patience     = patience       # 0 = disabled (default for fine-tuning)
    _no_improve   = 0              # tracks gen_loss stagnation (not val_loss)
    global_step   = 0

    # Build a ft_config that overrides the TF schedule so train_epoch picks up
    # the right tf_ratio via get_tf_ratio(epoch, config).  We monkey-patch by
    # storing the per-epoch ratio directly into the schedule each iteration.
    ft_config = dict(config)

    print(f"  {'Ep':>3}  {'TF':>5}  {'Val(TF1)':>9}  {'PPL':>8}  "
          f"{'BLEU':>6}  {'F1':>6}  {'Len':>5}  {'LR':>9}  {'Ent':>6}  Time")
    print("  " + "─" * 80)

    for epoch in range(1, ft_epochs + 1):
        t0 = time.time()

        tf_ratio = _ft_tf_ratio(epoch, ft_tf_start, ft_tf_floor, ft_anneal_epochs)

        # Inject tf_ratio into ft_config so train_epoch reads it correctly.
        # We override the schedule to a constant for this epoch: set phase1_end
        # to a value that puts every epoch in Phase 3, and set phase3_tf to our
        # computed ratio.
        ft_config["tf_schedule"] = dict(config.get("tf_schedule", {}))
        ft_config["tf_schedule"]["phase1_end"]   = 0   # never Phase 1
        ft_config["tf_schedule"]["phase2_end"]   = 0   # never Phase 2
        ft_config["tf_schedule"]["phase3_tf"]    = tf_ratio

        # ── Train ────────────────────────────────────────────────────────────
        train_loss, avg_gnorm, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=ft_config,
            device=device,
            epoch=epoch,
            global_step=global_step,
        )

        # ── Validation (TF=1.0) ───────────────────────────────────────────────
        val_loss, val_ppl = evaluate_loss(
            model=model, loader=val_loader, criterion=criterion,
            device=device,
        )

        # ── Generation metrics (TF=0.0, 1024 samples) ─────────────────────────
        n_gen = config.get("n_gen_samples", 1024)
        gen_loss, avg_pred_len, bleu, token_f1, avg_active_tokens = evaluate_generation(
            model=model, loader=val_loader,
            device=device, sp_model=_sp, n_gen_samples=n_gen,
        )

        # ── Attention entropy ─────────────────────────────────────────────────
        attn_entropy: Optional[float] = None
        if model_type == "attention":
            attn_entropy = compute_attention_entropy(
                model=model, loader=val_loader,
                device=device, n_samples=256,
            )

        # ── Decoded samples (val-set items, random each epoch) ───────────────
        log_decoded_samples(
            model=model, loader=val_loader, device=device,
            sp_model=_sp, n_samples=10,
            model_type=model_type, epoch=epoch,
        )

        # ── Fixed probe questions (same 16 Ubuntu IRC prompts every epoch) ────
        log_probe_responses(
            model=model, device=device, sp_model=_sp,
            model_type=model_type, epoch=epoch, config=ft_config,
        )

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ── Scheduler step ────────────────────────────────────────────────────
        # cosine / constant: step() takes no metric argument
        # plateau: step on gen_loss, NOT val_loss — val_loss WILL rise during TF→0
        if _scheduler_on_gen:
            scheduler.step(gen_loss)
        else:
            scheduler.step()

        # ── Build current checkpoint data (always, not just on improvement) ──
        # Defined here so both best-val and best-gen checkpoints can use it,
        # and so the secondary F1 branch never references a stale/undefined var.
        _current_ckpt = {
            "epoch":            epoch,
            "ft_epoch":         epoch,
            "source_epoch":     src_epoch,
            "global_step":      global_step,
            "model_type":       model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "val_loss":         val_loss,
            "val_ppl":          val_ppl,
            "gen_loss":         gen_loss,
            "token_f1":         token_f1,
            "train_loss":       train_loss,
            "tf_ratio":         tf_ratio,
            "ft_tf_floor":      ft_tf_floor,
            "config":           config,
            "history":          history,
        }

        # ── Best val-loss checkpoint (epoch-comparable, primary signal) ───────
        _improved = val_loss < best_val_loss
        if _improved:
            best_val_loss = val_loss
            tmp = ft_best + ".tmp"
            torch.save(_current_ckpt, tmp)
            os.replace(tmp, ft_best)

        # ── Best gen-loss checkpoint (free-running quality — key for TF→0) ───
        # Reviewer point 3: val_loss (TF=1.0) and gen_loss (TF=0.0) can diverge
        # during low-TF fine-tuning. Save a separate gen checkpoint so we can
        # pick whichever performs better at inference time.
        ft_best_gen = ft_best.replace("_ft_best.pt", "_ft_best_gen.pt")
        _gen_improved = gen_loss < best_gen_loss
        if _gen_improved:
            best_gen_loss = gen_loss
            tmp_gen = ft_best_gen + ".tmp"
            torch.save(_current_ckpt, tmp_gen)
            os.replace(tmp_gen, ft_best_gen)

        # ── History ───────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["gen_loss"].append(gen_loss)
        history["avg_active_tokens"].append(avg_active_tokens)
        history["bleu"].append(bleu)
        history["token_f1"].append(token_f1)
        history["avg_pred_len"].append(avg_pred_len)
        history["attn_entropy"].append(attn_entropy)
        history["tf_ratios"].append(tf_ratio)
        history["lrs"].append(lr)

        # ── Secondary F1 checkpoint ───────────────────────────────────────────
        _f1_hist = history["token_f1"]
        if (epoch > ft_anneal_epochs + 2
                and len(_f1_hist) >= 4
                and not _improved):
            _f1_smooth_now  = (_f1_hist[-1] + _f1_hist[-2]) / 2.0
            _f1_smooth_prev = (_f1_hist[-3] + _f1_hist[-4]) / 2.0
            if _f1_smooth_now - _f1_smooth_prev >= 1.0:
                _f1_ckpt = ft_best.replace("_ft_best.pt", "_ft_best_f1.pt")
                _f1_data = {**_current_ckpt, "f1_smooth": _f1_smooth_now}
                _f1_tmp = _f1_ckpt + ".tmp"
                torch.save(_f1_data, _f1_tmp)
                os.replace(_f1_tmp, _f1_ckpt)
                print(f"  [ft] F1 checkpoint saved "
                      f"(smoothed F1 {_f1_smooth_prev:.2f}→{_f1_smooth_now:.2f})")

        # ── Early stopping on gen_loss (disabled by default: patience=0) ───────
        # We do NOT stop on val_loss — it will rise as TF→0 by design.
        # If patience>0, stop only when free-running gen_loss stops improving.
        if _patience > 0:
            if _gen_improved:
                _no_improve = 0
            else:
                _no_improve += 1
                if _no_improve >= _patience:
                    print(f"\n  [{model_type}] Early stopping at ft-epoch {epoch} "
                          f"(gen_loss no improvement for {_patience} epochs)")
                    _save_last(model, optimizer, epoch, global_step, model_type,
                               val_loss, _no_improve, config, history, ft_last)
                    break

        # ── Last checkpoint ───────────────────────────────────────────────────
        _save_last(model, optimizer, epoch, global_step, model_type,
                   val_loss, _no_improve, config, history, ft_last)

        # ── Epoch line ────────────────────────────────────────────────────────
        _mark   = " ✓" if _gen_improved else ""   # ✓ marks gen_loss improvement
        _ent    = f"{attn_entropy:.3f}" if attn_entropy is not None else "  n/a"
        print(
            f"  Ep {epoch:2d}/{ft_epochs}  "
            f"TF:{tf_ratio:.2f}  "
            f"Val:{val_loss:.4f}  "
            f"PPL:{val_ppl:7.2f}  "
            f"BLEU:{bleu:.2f}  "
            f"F1:{token_f1:.2f}  "
            f"Len:{avg_pred_len:.1f}  "
            f"LR:{lr:.2e}  "
            f"Ent:{_ent}  "
            f"{elapsed:.1f}s{_mark}"
        )

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(checkpoint_dir, f"{model_type}_ft_history.json")
    tmp_hist  = hist_path + ".tmp"
    with open(tmp_hist, "w") as fh:
        json.dump(history, fh, indent=2)
    os.replace(tmp_hist, hist_path)
    print(f"\n  [{model_type}] Fine-tune history saved → {hist_path}")
    print(f"  [{model_type}] Best ft val_loss = {best_val_loss:.4f}  → {ft_best}")
    print(f"  [{model_type}] Best ft gen_loss = {best_gen_loss:.4f}  → {ft_best_gen}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history


def _save_last(model, optimizer, epoch, global_step, model_type,
               val_loss, no_improve, config, history, path):
    data = {
        "epoch":            epoch,
        "global_step":      global_step,
        "model_type":       model_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "val_loss":         val_loss,
        "no_improve":       no_improve,
        "config":           config,
        "history":          history,
    }
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune run-3 checkpoint with lower TF floor")
    parser.add_argument("--model",          type=str,   default="baseline",
                        choices=["baseline", "attention", "both"],
                        help="Which model to fine-tune (default: baseline)")
    parser.add_argument("--tf-floor",       type=float, default=0.30,
                        help="TF floor to anneal to (default: 0.30)")
    parser.add_argument("--tf-start",       type=float, default=0.50,
                        help="TF ratio at start of fine-tune (default: 0.50, matching run-3 floor)")
    parser.add_argument("--anneal-epochs",  type=int,   default=5,
                        help="Epochs to linearly decay TF to floor (default: 5)")
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Starting learning rate (default: 1e-4, lower than run-3)")
    parser.add_argument("--epochs",         type=int,   default=12,
                        help="Total fine-tune epochs (default: 12)")
    parser.add_argument("--lr-schedule",    type=str,   default="cosine",
                        choices=["cosine", "constant", "plateau"],
                        help="LR schedule: cosine (default), constant, or plateau on gen_loss")
    parser.add_argument("--patience",       type=int,   default=0,
                        help="Early-stop patience on gen_loss; 0=disabled (default)")
    parser.add_argument("--seed",           type=int,   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _seed = args.seed if args.seed is not None else CONFIG.get("seed", 42)
    set_seed(_seed)

    models_to_run = (["baseline", "attention"] if args.model == "both"
                     else [args.model])

    for model_type in models_to_run:
        finetune(
            model_type=model_type,
            ft_tf_floor=args.tf_floor,
            ft_tf_start=args.tf_start,
            ft_anneal_epochs=args.anneal_epochs,
            ft_lr=args.lr,
            ft_epochs=args.epochs,
            config=CONFIG,
            device=device,
            lr_schedule=args.lr_schedule,
            patience=args.patience,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  GPU cache cleared.\n")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
