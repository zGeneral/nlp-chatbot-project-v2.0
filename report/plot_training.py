"""
report/plot_training.py — Training Performance Visualizations
==============================================================
Reads training history from checkpoint files and generates publication-ready
figures for the project report.

Usage:
    python report/plot_training.py
    python report/plot_training.py --checkpoint-dir checkpoints/
    python report/plot_training.py --demo          # synthetic data (no checkpoints needed)

Output (saved to report/figures/):
    fig1_loss_curves.png/.pdf       Train & val loss — both models
    fig2_perplexity.png/.pdf        Validation perplexity — both models
    fig3_lr_tf_schedule.png/.pdf    LR decay + TF ratio across epochs
    fig4_overfit_gap.png/.pdf       Train–val loss gap (overfitting diagnostic)
    fig5_model_comparison.png/.pdf  Side-by-side best val-loss per epoch

All figures exported at 300 DPI (PNG) and as vector PDF.
"""

import argparse
import math
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Try SciencePlots (publication-quality style) ─────────────────────────────
try:
    import scienceplots  # noqa: F401
    _SCIENCE_STYLE = ["science", "grid"]
except ImportError:
    _SCIENCE_STYLE = None

# ── Try seaborn for extra polish ──────────────────────────────────────────────
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

# ── Try torch for checkpoint loading ─────────────────────────────────────────
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Phase boundary epochs (match config.py defaults)
PHASE1_END = 5    # TF=1.0 up to and including this epoch
PHASE2_END = 12   # linear TF annealing ends here; TF=0.5 floor after

# Colour palette — distinct, colour-blind friendly, looks great in print
PALETTE = {
    "baseline":  "#2C7BB6",   # steel blue
    "attention": "#D7191C",   # crimson
    "train":     "#7BAFD4",   # light blue (dashed train lines)
    "train_att": "#E87070",   # light red  (dashed train lines, attention)
    "phase":     "#AAAAAA",   # phase boundary lines
    "fill":      0.12,        # alpha for shaded regions
}


# ─────────────────────────────────────────────────────────────────────────────
# Style setup
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    if _SCIENCE_STYLE:
        plt.style.use(_SCIENCE_STYLE)
        # SciencePlots enables LaTeX by default — disable if not available
        try:
            import shutil
            if not shutil.which("latex"):
                plt.rcParams["text.usetex"] = False
        except Exception:
            plt.rcParams["text.usetex"] = False
    elif _HAS_SNS:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
        "figure.constrained_layout.use": True,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_history(checkpoint_dir: str, model_type: str) -> dict | None:
    """Load history dict from best or last checkpoint for a model."""
    if not _HAS_TORCH:
        return None
    for suffix in ("best", "last"):
        path = os.path.join(checkpoint_dir, f"{model_type}_{suffix}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            h = ckpt.get("history", {})
            if h and h.get("val_loss"):
                print(f"  Loaded {model_type} from {path} "
                      f"({len(h['val_loss'])} epochs)")
                return h
    return None


def _demo_history(seed: int, offset: float = 0.0, epochs: int = 20) -> dict:
    """Generate synthetic training history for demo/testing purposes."""
    rng = np.random.default_rng(seed)
    ep = np.arange(1, epochs + 1)

    # Simulate train loss: fast drop then slow convergence
    train_loss = 8.0 * np.exp(-0.22 * ep) + 4.2 + offset + rng.normal(0, 0.04, epochs)

    # Validate loss: higher than train, improvement stalls slightly
    val_loss = train_loss + 0.35 + offset + np.linspace(0.0, 0.3, epochs) + rng.normal(0, 0.06, epochs)

    # LR: starts at 3e-4, halves twice (epochs ~8 and ~14)
    lrs = np.full(epochs, 3e-4)
    lrs[8:] = 1.5e-4
    lrs[14:] = 7.5e-5
    lrs += rng.normal(0, 1e-6, epochs)

    # TF ratio: phase 1 flat, phase 2 linear, phase 3 floor
    tf = np.ones(epochs)
    for i, e in enumerate(ep):
        if e <= PHASE1_END:
            tf[i] = 1.0
        elif e <= PHASE2_END:
            tf[i] = 1.0 - 0.5 * (e - PHASE1_END) / (PHASE2_END - PHASE1_END)
        else:
            tf[i] = 0.5

    return {
        "train_loss": train_loss.tolist(),
        "val_loss":   val_loss.tolist(),
        "lrs":        lrs.tolist(),
        "tf_ratios":  tf.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _epochs(h: dict) -> np.ndarray:
    return np.arange(1, len(h["val_loss"]) + 1)


def _add_phase_bands(ax, n_epochs: int) -> None:
    """Shade training phases and add labels."""
    p1 = min(PHASE1_END, n_epochs)
    p2 = min(PHASE2_END, n_epochs)

    ymin, ymax = ax.get_ylim()

    ax.axvspan(0.5, p1 + 0.5, alpha=0.06, color="#4CAF50", zorder=0)
    ax.axvspan(p1 + 0.5, p2 + 0.5, alpha=0.06, color="#FFC107", zorder=0)
    if n_epochs > p2:
        ax.axvspan(p2 + 0.5, n_epochs + 0.5, alpha=0.06, color="#FF5722", zorder=0)

    for boundary in (p1 + 0.5, p2 + 0.5):
        if boundary < n_epochs + 0.5:
            ax.axvline(boundary, color=PALETTE["phase"], lw=0.8, ls="--", zorder=1)

    # Phase labels — place near top of axes
    label_y = ymin + 0.94 * (ymax - ymin)
    phase_mids = [(0.5 + p1 + 0.5) / 2, (p1 + 0.5 + p2 + 0.5) / 2]
    phase_labels = ["Phase 1\n(TF=1.0)", "Phase 2\n(anneal)"]
    if n_epochs > p2:
        phase_mids.append((p2 + 0.5 + n_epochs + 0.5) / 2)
        phase_labels.append("Phase 3\n(TF=0.5)")

    for xm, lbl in zip(phase_mids, phase_labels):
        if 0.5 < xm < n_epochs + 0.5:
            ax.text(xm, label_y, lbl, ha="center", va="top",
                    fontsize=7, color="#555555", style="italic")


def _integer_xticks(ax, n: int) -> None:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=min(n, 10)))


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {name}.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Train & Validation Loss
# ─────────────────────────────────────────────────────────────────────────────

def fig1_loss_curves(h_base: dict, h_att: dict) -> None:
    ep_b = _epochs(h_base)
    ep_a = _epochs(h_att)
    n = max(len(ep_b), len(ep_a))

    fig, ax = plt.subplots(figsize=(7, 4))

    # Validation loss (solid, prominent)
    ax.plot(ep_b, h_base["val_loss"],   color=PALETTE["baseline"],  ls="-",  marker="o",
            ms=4, label="Baseline — val loss")
    ax.plot(ep_a, h_att["val_loss"],    color=PALETTE["attention"], ls="-",  marker="s",
            ms=4, label="Attention — val loss")

    # Train loss (dashed, secondary)
    if h_base.get("train_loss"):
        ax.plot(ep_b, h_base["train_loss"], color=PALETTE["train"],    ls="--", lw=1.2,
                label="Baseline — train loss")
    if h_att.get("train_loss"):
        ax.plot(ep_a, h_att["train_loss"],  color=PALETTE["train_att"], ls="--", lw=1.2,
                label="Attention — train loss")

    _add_phase_bands(ax, n)
    _integer_xticks(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss (nats)")
    ax.set_title("Training and Validation Loss")
    ax.legend(loc="upper right", framealpha=0.9)
    _save(fig, "fig1_loss_curves")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Validation Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def fig2_perplexity(h_base: dict, h_att: dict) -> None:
    ep_b = _epochs(h_base)
    ep_a = _epochs(h_att)
    n = max(len(ep_b), len(ep_a))

    ppl_b = [math.exp(min(v, 20)) for v in h_base["val_loss"]]
    ppl_a = [math.exp(min(v, 20)) for v in h_att["val_loss"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep_b, ppl_b, color=PALETTE["baseline"],  ls="-", marker="o", ms=4,
            label="Baseline")
    ax.plot(ep_a, ppl_a, color=PALETTE["attention"], ls="-", marker="s", ms=4,
            label="Attention")
    ax.fill_between(ep_b, ppl_b, alpha=PALETTE["fill"], color=PALETTE["baseline"])
    ax.fill_between(ep_a, ppl_a, alpha=PALETTE["fill"], color=PALETTE["attention"])

    _add_phase_bands(ax, n)
    _integer_xticks(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Validation Perplexity (lower is better)")
    ax.legend(loc="upper right", framealpha=0.9)
    _save(fig, "fig2_perplexity")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — LR Decay + TF Schedule
# ─────────────────────────────────────────────────────────────────────────────

def fig3_lr_tf_schedule(h_base: dict, h_att: dict) -> None:
    ep_b = _epochs(h_base)
    ep_a = _epochs(h_att)
    n = max(len(ep_b), len(ep_a))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # Top: LR
    if h_base.get("lrs"):
        ax1.step(ep_b, h_base["lrs"], color=PALETTE["baseline"],  where="post",
                 label="Baseline LR")
    if h_att.get("lrs"):
        ax1.step(ep_a, h_att["lrs"],  color=PALETTE["attention"], where="post",
                 ls="--", label="Attention LR")
    ax1.set_ylabel("Learning Rate")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax1.set_title("Learning Rate (ReduceLROnPlateau) & Teacher Forcing Schedule")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Bottom: TF ratio
    if h_base.get("tf_ratios"):
        ax2.plot(ep_b, h_base["tf_ratios"], color=PALETTE["baseline"],  ls="-",
                 marker="o", ms=4, label="Baseline TF ratio")
    if h_att.get("tf_ratios"):
        ax2.plot(ep_a, h_att["tf_ratios"],  color=PALETTE["attention"], ls="-",
                 marker="s", ms=4, label="Attention TF ratio")
    ax2.set_ylim(-0.05, 1.10)
    ax2.axhline(1.0, color=PALETTE["phase"], lw=0.7, ls=":")
    ax2.axhline(0.5, color=PALETTE["phase"], lw=0.7, ls=":")
    ax2.set_ylabel("Teacher Forcing Ratio")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right", framealpha=0.9)

    for ax in (ax1, ax2):
        _add_phase_bands(ax, n)
        _integer_xticks(ax, n)

    _save(fig, "fig3_lr_tf_schedule")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Generalisation Gap (Train–Val Loss)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_overfit_gap(h_base: dict, h_att: dict) -> None:
    if not h_base.get("train_loss") or not h_att.get("train_loss"):
        print("  Skipping fig4 — train_loss not available in history")
        return

    ep_b = _epochs(h_base)
    ep_a = _epochs(h_att)
    n = max(len(ep_b), len(ep_a))

    gap_b = np.array(h_base["val_loss"]) - np.array(h_base["train_loss"])
    gap_a = np.array(h_att["val_loss"])  - np.array(h_att["train_loss"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep_b, gap_b, color=PALETTE["baseline"],  ls="-", marker="o", ms=4,
            label="Baseline")
    ax.plot(ep_a, gap_a, color=PALETTE["attention"], ls="-", marker="s", ms=4,
            label="Attention")
    ax.fill_between(ep_b, 0, gap_b, alpha=PALETTE["fill"], color=PALETTE["baseline"])
    ax.fill_between(ep_a, 0, gap_a, alpha=PALETTE["fill"], color=PALETTE["attention"])
    ax.axhline(0, color="#333333", lw=0.8, ls="--")

    _add_phase_bands(ax, n)
    _integer_xticks(ax, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss − Train Loss (nats)")
    ax.set_title("Generalisation Gap (lower = less overfitting)")
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, "fig4_overfit_gap")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Model Comparison: Best Val Loss per Epoch
# ─────────────────────────────────────────────────────────────────────────────

def fig5_model_comparison(h_base: dict, h_att: dict) -> None:
    ep_b = _epochs(h_base)
    ep_a = _epochs(h_att)
    n = max(len(ep_b), len(ep_a))

    # Running best (cumulative minimum) — shows when each model peaked
    best_b = np.minimum.accumulate(h_base["val_loss"])
    best_a = np.minimum.accumulate(h_att["val_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: raw val loss side-by-side
    ax1.plot(ep_b, h_base["val_loss"], color=PALETTE["baseline"],  ls="-", marker="o",
             ms=4, label="Baseline")
    ax1.plot(ep_a, h_att["val_loss"],  color=PALETTE["attention"], ls="-", marker="s",
             ms=4, label="Attention")
    _add_phase_bands(ax1, n)
    _integer_xticks(ax1, n)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss (nats)")
    ax1.set_title("Validation Loss per Epoch")
    ax1.legend(framealpha=0.9)

    # Right: running best
    ax2.plot(ep_b, best_b, color=PALETTE["baseline"],  ls="-", marker="o",
             ms=4, label="Baseline best")
    ax2.plot(ep_a, best_a, color=PALETTE["attention"], ls="-", marker="s",
             ms=4, label="Attention best")
    ax2.fill_between(ep_b, best_b, alpha=PALETTE["fill"], color=PALETTE["baseline"])
    ax2.fill_between(ep_a, best_a, alpha=PALETTE["fill"], color=PALETTE["attention"])

    # Annotate final best values
    ax2.annotate(f"{best_b[-1]:.3f}",
                 xy=(ep_b[-1], best_b[-1]),
                 xytext=(ep_b[-1] - 1.5, best_b[-1] + 0.08),
                 fontsize=8, color=PALETTE["baseline"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["baseline"], lw=0.8))
    ax2.annotate(f"{best_a[-1]:.3f}",
                 xy=(ep_a[-1], best_a[-1]),
                 xytext=(ep_a[-1] - 1.5, best_a[-1] - 0.15),
                 fontsize=8, color=PALETTE["attention"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["attention"], lw=0.8))

    _add_phase_bands(ax2, n)
    _integer_xticks(ax2, n)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Best Val Loss So Far (nats)")
    ax2.set_title("Cumulative Best Validation Loss")
    ax2.legend(framealpha=0.9)

    _save(fig, "fig5_model_comparison")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Summary Table (bonus: key metrics as a styled table figure)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_summary_table(h_base: dict, h_att: dict) -> None:
    best_val_b   = min(h_base["val_loss"])
    best_val_a   = min(h_att["val_loss"])
    best_ppl_b   = math.exp(min(best_val_b, 20))
    best_ppl_a   = math.exp(min(best_val_a, 20))
    best_epoch_b = int(np.argmin(h_base["val_loss"])) + 1
    best_epoch_a = int(np.argmin(h_att["val_loss"])) + 1
    final_lr_b   = h_base["lrs"][-1] if h_base.get("lrs") else "—"
    final_lr_a   = h_att["lrs"][-1]  if h_att.get("lrs")  else "—"

    rows = [
        ["Metric",                "Baseline",                          "Attention"],
        ["Total epochs",          str(len(h_base["val_loss"])),        str(len(h_att["val_loss"]))],
        ["Best val loss (nats)",  f"{best_val_b:.4f}",                 f"{best_val_a:.4f}"],
        ["Best val perplexity",   f"{best_ppl_b:.1f}",                 f"{best_ppl_a:.1f}"],
        ["Best epoch",            str(best_epoch_b),                   str(best_epoch_a)],
        ["Final LR",
         f"{final_lr_b:.2e}" if isinstance(final_lr_b, float) else final_lr_b,
         f"{final_lr_a:.2e}" if isinstance(final_lr_a, float) else final_lr_a],
    ]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")

    col_widths = [0.38, 0.31, 0.31]
    table = ax.table(
        cellText=[r[1:] for r in rows[1:]],
        colLabels=rows[0][1:],
        rowLabels=[r[0] for r in rows[1:]],
        cellLoc="center",
        loc="center",
        colWidths=col_widths[1:],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header and best-column cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("#CCCCCC")

    # Highlight better model in val_loss and ppl rows
    better = 1 if best_val_b <= best_val_a else 2
    for metric_row in (1, 2):
        cell = table[metric_row, better - 1]
        cell.set_facecolor("#D4EDDA")

    ax.set_title("Training Summary — Baseline vs. Attention", fontsize=12, pad=12)
    _save(fig, "fig6_summary_table")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training performance figures")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(Path(os.path.dirname(__file__)).parent / "checkpoints"),
        help="Directory containing *_best.pt / *_last.pt files (default: <project_root>/checkpoints/)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic data — no checkpoints required (useful for testing)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of epochs for demo data (default: 20)",
    )
    args = parser.parse_args()

    _apply_style()

    print("=" * 60)
    print("  Training Performance Plots")
    print("=" * 60)

    if args.demo:
        print("  Mode: DEMO (synthetic data)")
        h_base = _demo_history(seed=42,  offset=0.0,  epochs=args.epochs)
        h_att  = _demo_history(seed=137, offset=-0.1, epochs=args.epochs)
    else:
        print(f"  Loading checkpoints from: {args.checkpoint_dir}")
        h_base = _load_history(args.checkpoint_dir, "baseline")
        h_att  = _load_history(args.checkpoint_dir, "attention")

        if h_base is None or h_att is None:
            missing = []
            if h_base is None: missing.append("baseline")
            if h_att  is None: missing.append("attention")
            print(f"\n  WARNING: Missing checkpoints for: {', '.join(missing)}")
            print("  Falling back to demo data for missing model(s).")
            if h_base is None:
                h_base = _demo_history(seed=42,  offset=0.0,  epochs=args.epochs)
            if h_att is None:
                h_att  = _demo_history(seed=137, offset=-0.1, epochs=args.epochs)

    print(f"\n  Baseline : {len(h_base['val_loss'])} epochs  "
          f"| best val loss = {min(h_base['val_loss']):.4f}")
    print(f"  Attention: {len(h_att['val_loss'])} epochs  "
          f"| best val loss = {min(h_att['val_loss']):.4f}")
    print(f"\n  Saving figures to: {FIGURES_DIR}/\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("  [1/6] Loss curves …")
        fig1_loss_curves(h_base, h_att)

        print("  [2/6] Perplexity …")
        fig2_perplexity(h_base, h_att)

        print("  [3/6] LR + TF schedule …")
        fig3_lr_tf_schedule(h_base, h_att)

        print("  [4/6] Generalisation gap …")
        fig4_overfit_gap(h_base, h_att)

        print("  [5/6] Model comparison …")
        fig5_model_comparison(h_base, h_att)

        print("  [6/6] Summary table …")
        fig6_summary_table(h_base, h_att)

    print(f"\n  Done — {len(os.listdir(FIGURES_DIR))} files in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
