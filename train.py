"""
train.py — Training loop for clean-from-scratch Seq2Seq chatbot.

Trains both "baseline" (no attention) and "attention" (Bahdanau) models.
The training regime is the single most important departure from the prior codebase:

  TEACHER FORCING STRATEGY (key decision):
    Epochs  1–15:  TF = 1.0  — model always sees correct tokens; burns in sharp patterns
    Epochs 16–18:  TF = 0.8  — gentle awareness of own output errors
    Epochs 19–20:  TF = 0.5  — minimal adaptation; patterns already locked in

  Rationale:
    The prior codebase decayed TF to 0.25, which caused the perplexity-quality
    inversion: better val_loss = more generic output. TF=1.0 for most of training
    is empirically validated by osamadev/seq2seq-chatbot (BLEU-4=0.1386) which
    uses TF=1.0 for all 10 epochs.

  LR STRATEGY:
    Flat LR=3e-4, AdamW, no warmup.
    Prior codebase: LR=3e-3 + complex warmup + ReduceLROnPlateau → Sisyphus trap.
    Conservative flat LR avoids second-moment calibration issues at TF transitions.

  GRADIENT CLIPPING:
    clip=1.0 (prior: 0.5). Allows valid large gradients during TF=1.0 phase.
"""

import os
import json
import math
import time
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG, get_tf_ratio, set_seed
from dataset import build_dataloaders
from models import build_model

# bf16 does not underflow like fp16 — GradScaler is not needed.
# torch.amp.autocast with dtype=torch.bfloat16 is sufficient.


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: dict,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    scheduler,
) -> Tuple[float, float, int]:
    """
    One training epoch with bf16 AMP and gradient accumulation.

    Returns:
        (avg_train_loss, avg_grad_norm, updated_global_step)

    Notes:
        - global_step = optimizer steps, NOT forward passes.
          It increments only when an optimizer step is taken (after accumulation).
        - bf16 does not underflow like fp16 — GradScaler is not needed.
        - NaN loss: batch is skipped (nan_count incremented, no backward).
        - NaN grad norm: optimizer.step() is skipped (nan_count incremented).
        - Periodic checkpoint written every 2000 optimizer steps (atomic).
    """
    model.train()

    vocab_size: int = config["vocab_size"]
    grad_accum_steps: int = config["grad_accum_steps"]
    max_grad_norm: float = config["max_grad_norm"]
    checkpoint_dir: str = config["checkpoint_dir"]
    periodic_ckpt_steps: int = 2000   # save a step-level checkpoint every N optimizer steps
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))   # torch.bfloat16

    # Compute tf_ratio once — it is constant within an epoch.
    tf_ratio: float = get_tf_ratio(epoch, config)

    total_loss = 0.0
    total_grad_norm = 0.0
    n_updates = 0      # number of successful optimizer steps this epoch
    nan_count = 0      # batches skipped due to NaN loss or NaN grad norm
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
        src: torch.Tensor = batch["src"].to(device)                   # [B, src_len]
        src_lengths: torch.Tensor = batch["src_lengths"].to(device)   # [B]
        trg: torch.Tensor = batch["trg"].to(device)                   # [B, trg_len]

        # ── Forward pass under bf16 autocast ───────────────────────────────
        # bf16 does not underflow like fp16 — GradScaler is not needed.
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

        # ── NaN loss guard ──────────────────────────────────────────────────
        if not torch.isfinite(loss):
            nan_count += 1
            # Zero accumulated gradients so this bad batch has no residual effect.
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss="NaN", nan_skip=nan_count)
            continue

        scaled_loss.backward()

        # ── Optimizer step guard (gradient accumulation) ────────────────────
        is_last_batch = (batch_idx + 1) == num_batches
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or is_last_batch

        if should_step:
            # Compute grad norm before clipping for logging.
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            ).item()

            # ── NaN grad norm guard ─────────────────────────────────────────
            if not math.isfinite(grad_norm):
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss=f"{loss.item():.4f}", grad_norm="NaN", nan_skip=nan_count)
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # global_step = optimizer steps, NOT forward passes.
            global_step += 1

            total_loss += loss.item()
            total_grad_norm += grad_norm
            n_updates += 1

            # ── TensorBoard step-level logging ──────────────────────────────
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/grad_norm_step", grad_norm, global_step)

            # ── Periodic checkpoint (atomic: write .tmp then os.replace) ────
            if global_step % periodic_ckpt_steps == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{config.get('_model_type', 'model')}_step_{global_step}.pt",
                )
                tmp_path = ckpt_path + ".tmp"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "train_loss_so_far": total_loss / max(n_updates, 1),
                        "tf_ratio": tf_ratio,
                    },
                    tmp_path,
                )
                os.replace(tmp_path, ckpt_path)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm:.3f}",
                nan_skip=nan_count,
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
        src: torch.Tensor = batch["src"].to(device)
        src_lengths: torch.Tensor = batch["src_lengths"].to(device)
        trg: torch.Tensor = batch["trg"].to(device)

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

        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1

    avg_val_loss = total_loss / max(n_batches, 1)
    # Cap inside exp() to avoid overflow on very early / diverged runs.
    val_ppl = math.exp(min(avg_val_loss, 20))
    return avg_val_loss, val_ppl


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Build AdamW optimizer and ReduceLROnPlateau scheduler.

    No warmup: LR=3e-4 is already conservative; warmup would waste epochs.
    ReduceLROnPlateau activates from epoch 1 (not post-warmup).

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
        patience=config["lr_scheduler_patience"],
        factor=config["lr_scheduler_factor"],
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
    # FIX M6 — exact call as specified in INTERFACES.md §4
    train_loader, val_loader, _ = build_dataloaders(
        artifact_dir=config["artifact_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_ctx_len=config["max_ctx_tokens"],
        max_resp_len=config["max_resp_tokens"] + 2,   # +2 for <sos> and <eos>
        pad_idx=config["pad_idx"],
        max_train_samples=config.get("max_train_samples", 0),
    )

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model = build_model(model_type, config, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{model_type}] Trainable parameters: {num_params:,}")

    # ── 3. Optimizer + scheduler ──────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    # ── 4. Loss ───────────────────────────────────────────────────────────────
    # label_smoothing=0.0 during TF=1.0 phase — smoothing fights sharp token
    # predictions that the TF schedule is designed to burn in.
    criterion = nn.CrossEntropyLoss(
        ignore_index=config["pad_idx"],
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    _amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))   # read from config

    # ── 5. TensorBoard ────────────────────────────────────────────────────────
    tb_dir = os.path.join(config["tensorboard_dir"], model_type)
    writer = SummaryWriter(log_dir=tb_dir)

    # ── 6. Resume — prefer last-epoch checkpoint to avoid re-running epochs ──
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
    # to avoid silently discarding epochs on Colab disconnect (QA2-M1).
    resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else (
        best_ckpt_path if os.path.exists(best_ckpt_path) else None
    )
    if resume_path:
        print(f"[{model_type}] Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"[{model_type}] Resumed at epoch {start_epoch}, step {global_step}, "
              f"best_val={best_val_loss:.4f}")

    num_epochs: int = config["num_epochs"]
    print(f"[{model_type}] Training epochs {start_epoch}–{num_epochs}")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

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
            writer=writer,
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

        # 7d. LR scheduler step on validation loss.
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start
        lr: float = optimizer.param_groups[0]["lr"]

        # 7e. TensorBoard epoch-level logging.
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/ppl", val_ppl, epoch)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("learning_rate", lr, epoch)
        writer.add_scalar("tf_ratio", tf_ratio, epoch)

        # 7f. Save best checkpoint (atomic: write .tmp then os.replace).
        # Capture improvement BEFORE updating best_val_loss so early stopping
        # can use the same flag (avoids off-by-one: after the update,
        # val_loss == best_val_loss which would look like "no improvement").
        _improved = val_loss < best_val_loss
        if _improved:
            best_val_loss = val_loss
            # AC2-C2: include run provenance in every best checkpoint.
            try:
                _git = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                _git = "unknown"
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
                "git_hash": _git,
                "torch_version": torch.__version__,
                "run_timestamp": time.strftime("%Y%m%d_%H%M%S"),
            }
            tmp_path = best_ckpt_path + ".tmp"
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, best_ckpt_path)

        # 7g. Update history.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["tf_ratios"].append(tf_ratio)
        history["lrs"].append(lr)

        # Early stopping (used by train_mini; disabled when patience=0).
        if _patience > 0:
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

        # 7g2. Save last-epoch checkpoint (for Colab resume — prefers this over best).
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

    writer.close()

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
        script_name: Used for the run log filename (e.g. "train_mini").
    """
    from logging_utils import setup_run_logging
    active_cfg = cfg if cfg is not None else CONFIG
    setup_run_logging(script_name, log_dir=active_cfg.get("log_dir", "new/logs"))

    # AC2-C1: set all random seeds for full reproducibility.
    set_seed(active_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(active_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(active_cfg["tensorboard_dir"], exist_ok=True)

    # AC2-C2: log environment metadata (git hash, versions, timestamp).
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    run_info = {
        "git_hash":       git_hash,
        "python_version": sys.version,
        "torch_version":  torch.__version__,
        "cuda_version":   torch.version.cuda or "cpu",
        "run_timestamp":  time.strftime("%Y%m%d_%H%M%S"),
        "seed":           active_cfg.get("seed", 42),
        "device":         str(device),
    }
    run_info_path = os.path.join(active_cfg["checkpoint_dir"], "run_info.json")
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2)
    print(f"Run info saved → {run_info_path}")
    print(f"  git: {git_hash}  |  torch: {torch.__version__}  |  seed: {run_info['seed']}")

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING: baseline (no attention)")
    print("=" * 70)
    baseline_history = train_model("baseline", active_cfg, device)

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
