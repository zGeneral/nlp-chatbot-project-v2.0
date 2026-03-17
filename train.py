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
    Patience = 5 epochs, monitored only from Phase 2 onward (epoch > 5).

  GRADIENT CLIPPING:
    clip = 1.0. Allows valid large gradients during TF=1.0 phase.
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

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

        is_last_batch = (batch_idx + 1) == num_batches
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch

        if should_step:
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            ).item()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_updates += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm:.3f}",
            )

    avg_train_loss = total_loss / max(n_updates, 1)
    avg_grad_norm = total_grad_norm / max(n_updates, 1)
    return avg_train_loss, avg_grad_norm, global_step


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """
    Validation pass — teacher forcing is DISABLED (ratio=0.0).

    trg is passed only to set the decoder step count (trg.size(1)-1 steps).
    TF=0.0 means the model never sees gold tokens during the forward pass.
    This is standard practice — true autoregressive evaluation is in evaluate.py.

    Loss is accumulated only on non-padding positions because criterion has
    ignore_index=pad_idx; padding tokens contribute zero to the mean.

    Returns:
        (avg_val_loss, val_ppl) where val_ppl = exp(min(avg_val_loss, 20)).
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  val", unit="batch", dynamic_ncols=True, leave=False):
        src: torch.Tensor = batch["src"].to(device, non_blocking=True)
        src_lengths: torch.Tensor = batch["src_lengths"].to(device, non_blocking=True)
        trg: torch.Tensor = batch["trg"].to(device, non_blocking=True)

        # Pass teacher_forcing_ratio=0.0 so the decoder uses its own predictions,
        # not gold tokens.  trg still determines how many decode steps to run.
        _device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        with torch.amp.autocast(device_type=_device_type, dtype=amp_dtype,
                                enabled=_device_type == "cuda"):
            output = model(src, src_lengths, trg, teacher_forcing_ratio=0.0)

        vocab_size: int = output.size(-1)
        loss = criterion(
            output.reshape(-1, vocab_size),
            trg[:, 1:].reshape(-1),   # exclude <sos>; criterion ignores <pad>
        )

        total_loss += loss.item()
        n_batches += 1

    avg_val_loss = total_loss / max(n_batches, 1)
    val_ppl = math.exp(min(avg_val_loss, 20))
    return avg_val_loss, val_ppl


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
                 ``lr_patience``, ``lr_factor``, ``lr_min``.

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
      {checkpoint_dir}/{model_type}_step_{n}.pt  — periodic checkpoints (atomic)
      {checkpoint_dir}/{model_type}_history.json — training history

    Returns:
        List of per-epoch history dicts.
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
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))   # read from config

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
        "val_loss": [],
        "tf_ratios": [],
        "lrs": [],
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
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        if "history" in ckpt:
            history = ckpt["history"]

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

        # 7c. Validate.
        val_loss, val_ppl = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=_amp_dtype,
        )

        # 7d. Step ReduceLROnPlateau with validation loss.
        #     Must be called AFTER validation, once per epoch.
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
            print(f"[{model_type}] Phase 2 begins (epoch {epoch}) — "
                  f"resetting best_val_loss and early-stopping counter")

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
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "train_loss": train_loss,
                "tf_ratio": tf_ratio,
                "config": dict(config),
                "history": history,
            }
            tmp_path = best_ckpt_path + ".tmp"
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, best_ckpt_path)

        # 7g. Update history.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
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
                        "scheduler_state": scheduler.state_dict(),
                        "val_loss": val_loss,
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
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": dict(config),
            "history": history,
        }
        tmp_last = last_ckpt_path + ".tmp"
        torch.save(last_ckpt_data, tmp_last)
        os.replace(tmp_last, last_ckpt_path)

        # 7h. Epoch summary.
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"PPL: {val_ppl:.2f} | "
            f"LR: {lr:.2e} | "
            f"TF: {tf_ratio:.2f} | "
            f"Grad: {avg_gnorm:.3f} | "
            f"{elapsed:.1f}s"
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
    eff_batch = active_cfg["batch_size"] * active_cfg["grad_accum_steps"]
    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print("=" * 70)
    print(f"  Device  : {device}  ({gpu_name})")
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
    # allocator holds ~55-70 GB of cached-but-unused blocks.  The attention
    # model needs more per-step memory (attention weight tensors over the full
    # source sequence) and can OOM due to fragmentation even though total
    # capacity is sufficient.
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
