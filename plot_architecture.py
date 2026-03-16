"""
plot_architecture.py — Publication-quality figures for the Architecture report.

Figures produced (report/figures/, PNG + PDF at 300 DPI):
  fig_a1  — Full encoder-decoder architecture overview (both models)
  fig_a2  — Baseline decoder step (fixed context, single timestep)
  fig_a3  — Attention decoder step (dynamic context, Bahdanau)
  fig_a4  — Parameter breakdown: stacked bar baseline vs attention
  fig_a5  — Bahdanau vs Luong attention comparison table / flow

Usage:
    python plot_architecture.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette (consistent with other report figures) ────────────────────────────
BLUE    = "#2196F3"
LBLUE   = "#BBDEFB"
RED     = "#F44336"
LRED    = "#FFCDD2"
GREEN   = "#4CAF50"
LGREEN  = "#C8E6C9"
ORANGE  = "#FF9800"
LORANGE = "#FFE0B2"
PURPLE  = "#9C27B0"
LPURPLE = "#E1BEE7"
TEAL    = "#009688"
LTEAL   = "#B2DFDB"
GREY    = "#9E9E9E"
LGREY   = "#F5F5F5"
DARK    = "#263238"
WHITE   = "#FFFFFF"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved → {name}.png / .pdf")


def _box(ax, x, y, w, h, label, sublabel=None,
         fc=LBLUE, ec=BLUE, fontsize=9, bold=False,
         radius=0.02, lc=DARK):
    """Draw a rounded-rectangle box with centred label (and optional sublabel)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(box)
    fw = "bold" if bold else "normal"
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, color=lc, zorder=4)
    if sublabel:
        ax.text(x, y - 0.028, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=GREY, zorder=4, style="italic")


def _arrow(ax, x0, y0, x1, y1, color=DARK, lw=1.4, label=None,
           label_side="right", shrink=4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, shrinkA=shrink, shrinkB=shrink))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx = 0.025 if label_side == "right" else -0.025
        ax.text(mx + dx, my, label, fontsize=7.5, color=color,
                ha="left" if label_side == "right" else "right", va="center",
                style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a1 — Full architecture overview (encoder → bridge → decoder, both models)
# ─────────────────────────────────────────────────────────────────────────────
def plot_overview():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))

    for col, (ax, title, is_attn) in enumerate(zip(
            axes,
            ["Baseline Model  (no attention)", "Attention Model  (Bahdanau)"],
            [False, True])):

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8,
                     color=RED if is_attn else BLUE)

        W = 0.30   # box width
        Hs = 0.07  # box height small
        Hm = 0.09  # box height medium

        # ── Token IDs (input) ────────────────────────────────────────────────
        _box(ax, 0.5, 0.93, W, Hs, "src_ids  [B, T_src]",
             fc=LGREY, ec=GREY, fontsize=8.5)

        # ── Embedding ────────────────────────────────────────────────────────
        _box(ax, 0.5, 0.82, W, Hs, "Shared Embedding",
             sublabel="16 000 × 300  |  drop=0.30",
             fc=LPURPLE, ec=PURPLE)
        _arrow(ax, 0.5, 0.895, 0.5, 0.855)

        # ── BiLSTM encoder ───────────────────────────────────────────────────
        _box(ax, 0.5, 0.68, W, Hm + 0.02,
             "Bi-LSTM Encoder  ×2 layers",
             sublabel="hidden=512/dir  |  drop=0.50  |  14.4M params",
             fc=LBLUE, ec=BLUE, bold=True)
        _arrow(ax, 0.5, 0.857, 0.5, 0.723)

        # outputs label
        ax.text(0.5, 0.622, "encoder_outputs [B, T_src, 1024]",
                ha="center", fontsize=7.8, color=BLUE, style="italic")
        ax.text(0.5, 0.600, "h_n, c_n [4, B, 512]",
                ha="center", fontsize=7.8, color=BLUE, style="italic")

        # ── Bridge ───────────────────────────────────────────────────────────
        _box(ax, 0.5, 0.525, W, Hs, "Bridge",
             sublabel="Linear(1024→1024) × 2  |  2.1M params",
             fc=LTEAL, ec=TEAL)
        _arrow(ax, 0.5, 0.595, 0.5, 0.560)

        ax.text(0.5, 0.475, "h_0, c_0  [2, B, 1024]",
                ha="center", fontsize=7.8, color=TEAL, style="italic")

        # ── Decoder label ────────────────────────────────────────────────────
        decoder_col = RED if is_attn else ORANGE

        # Decoder input (teacher-forced token)
        _box(ax, 0.18, 0.38, 0.22, Hs, "y_{t-1}  (prev token)",
             fc=LGREY, ec=GREY, fontsize=8)

        _box(ax, 0.18, 0.27, 0.22, Hs, "Embedding + Dropout",
             fc=LPURPLE, ec=PURPLE, fontsize=8)
        _arrow(ax, 0.18, 0.355, 0.18, 0.305)

        if is_attn:
            # Attention context path
            # encoder_outputs → Attention block
            _box(ax, 0.79, 0.44, 0.26, Hm,
                 "Bahdanau Attention",
                 sublabel="W_enc + W_dec → 256-d  |  0.5M params",
                 fc=LRED, ec=RED, bold=True, fontsize=8.5)

            # Arrow from encoder outputs to attention (top-left of attn box)
            ax.annotate("", xy=(0.79, 0.481), xytext=(0.65, 0.637),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                        connectionstyle="arc3,rad=-0.25",
                                        shrinkA=4, shrinkB=4))
            ax.text(0.72, 0.585, "encoder_outputs\n[B, T_src, 1024]",
                    fontsize=7, color=RED, ha="center", style="italic")

            # α weights label
            ax.text(0.79, 0.386, "α_t, c_t  [B, 1024]",
                    ha="center", fontsize=7.5, color=RED, style="italic")

            # arrow from attention to LSTM input merger
            _arrow(ax, 0.79, 0.383, 0.545, 0.305, color=RED)
            # cat label
            ax.text(0.60, 0.32, "c_t", fontsize=7.5, color=RED, style="italic")

            lstm_sublabel = "hidden=1024  |  drop=0.50  |  input=[embed;c_t] 1324-d"
        else:
            # Fixed context path
            ax.annotate("", xy=(0.545, 0.27), xytext=(0.65, 0.625),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                        connectionstyle="arc3,rad=-0.2",
                                        shrinkA=4, shrinkB=4))
            ax.text(0.67, 0.44, "ctx_fixed\n(last timestep)\n[B, 1024]",
                    fontsize=7.5, color=ORANGE, ha="center", style="italic",
                    bbox=dict(facecolor=LORANGE, edgecolor=ORANGE,
                              boxstyle="round,pad=0.2", alpha=0.8))
            lstm_sublabel = "hidden=1024  |  drop=0.50  |  input=[embed;ctx] 1324-d"

        # ── Decoder LSTM ──────────────────────────────────────────────────────
        _box(ax, 0.38, 0.27, 0.32, Hm,
             "Decoder LSTM  ×2 layers",
             sublabel=lstm_sublabel,
             fc=LORANGE if not is_attn else LRED,
             ec=ORANGE if not is_attn else RED,
             bold=True, fontsize=8)
        _arrow(ax, 0.18, 0.247, 0.22, 0.27, color=GREY)

        # Bridge → decoder init
        _arrow(ax, 0.5, 0.472, 0.44, 0.31, color=TEAL)
        ax.text(0.435, 0.40, "h_0, c_0", fontsize=7.5, color=TEAL,
                ha="right", style="italic")

        # ── Bottleneck projection ─────────────────────────────────────────────
        _box(ax, 0.38, 0.155, W, Hs,
             "Linear(2048→512) + tanh + Dropout",
             sublabel="cat([s_t ; c_t])  →  512-d",
             fc=LGREEN, ec=GREEN, fontsize=8)
        _arrow(ax, 0.38, 0.247, 0.38, 0.19)

        # ── Output head ───────────────────────────────────────────────────────
        _box(ax, 0.38, 0.055, W, Hs,
             "Linear(512 → 16 000)  →  logits",
             sublabel="8.2M params  |  vocab softmax",
             fc=LGREEN, ec=GREEN, fontsize=8)
        _arrow(ax, 0.38, 0.120, 0.38, 0.090)

        # ── Param count badge ─────────────────────────────────────────────────
        total = "44 337 024" if is_attn else "43 812 480"
        ax.text(0.5, 0.005, f"Total parameters: {total}",
                ha="center", fontsize=9, fontweight="bold",
                color=RED if is_attn else BLUE)

    fig.suptitle("Seq2Seq LSTM Architecture — Baseline vs Attention",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(pad=1.5)
    _save(fig, "fig_a1_architecture_overview")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a2 — Baseline decoder: single timestep (fixed context)
# ─────────────────────────────────────────────────────────────────────────────
def plot_baseline_decoder():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Baseline Decoder — Single Timestep  (fixed context)",
                 fontsize=11, fontweight="bold", pad=8)

    W, Hs, Hm = 0.26, 0.07, 0.09

    # Previous token
    _box(ax, 0.18, 0.88, W, Hs, "y_{t-1}  (previous token)",
         fc=LGREY, ec=GREY, fontsize=8.5)

    # Embedding
    _box(ax, 0.18, 0.74, W, Hs, "Embedding(300) + Dropout(0.30)",
         fc=LPURPLE, ec=PURPLE, fontsize=8.5)
    _arrow(ax, 0.18, 0.845, 0.18, 0.775)
    ax.text(0.19, 0.808, "[B, 300]", fontsize=7.5, color=PURPLE, style="italic")

    # Fixed context (from encoder last timestep)
    _box(ax, 0.72, 0.74, 0.30, Hm,
         "ctx_fixed",
         sublabel="encoder_outputs[:, −1, :]  →  [B, 1024]",
         fc=LORANGE, ec=ORANGE, fontsize=8.5)
    ax.text(0.72, 0.66, "Fixed — same at every decoder step", ha="center",
            fontsize=7.5, color=ORANGE, style="italic")

    # Cat operation
    _box(ax, 0.40, 0.60, 0.22, 0.06, "cat([embed ; ctx])",
         sublabel="→  [B, 1 324]",
         fc=LGREY, ec=GREY, fontsize=8)
    _arrow(ax, 0.18, 0.705, 0.295, 0.60, color=PURPLE)
    _arrow(ax, 0.57, 0.74, 0.515, 0.63, color=ORANGE)

    # LSTM
    _box(ax, 0.40, 0.46, 0.28, Hm, "2-layer LSTM",
         sublabel="hidden=1024  |  drop=0.50\noutputs s_t  [B, 1024]",
         fc=LORANGE, ec=ORANGE, bold=True, fontsize=8.5)
    _arrow(ax, 0.40, 0.570, 0.40, 0.508)
    # Recurrent arrow
    ax.annotate("", xy=(0.56, 0.505), xytext=(0.56, 0.415),
                arrowprops=dict(arrowstyle="<|-", color=ORANGE, lw=1.2,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(0.64, 0.46, "h_{t-1}, c_{t-1}", fontsize=7.5, color=ORANGE,
            ha="left", style="italic")

    # Second cat with ctx_fixed
    _box(ax, 0.40, 0.325, 0.26, 0.06, "cat([s_t ; ctx_fixed])",
         sublabel="→  [B, 2 048]",
         fc=LGREY, ec=GREY, fontsize=8)
    _arrow(ax, 0.40, 0.415, 0.40, 0.358)
    ax.annotate("", xy=(0.535, 0.328), xytext=(0.72, 0.695),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1,
                                connectionstyle="arc3,rad=0.35",
                                shrinkA=4, shrinkB=4))

    # Bottleneck
    _box(ax, 0.40, 0.205, W, Hs, "Linear(2048→512) + tanh + Dropout(0.40)",
         fc=LGREEN, ec=GREEN, fontsize=8)
    _arrow(ax, 0.40, 0.295, 0.40, 0.242)
    ax.text(0.41, 0.268, "[B, 512]", fontsize=7.5, color=GREEN, style="italic")

    # Output head
    _box(ax, 0.40, 0.090, W, Hs, "Linear(512 → 16 000)  →  logits",
         fc=LGREEN, ec=GREEN, fontsize=8.5)
    _arrow(ax, 0.40, 0.170, 0.40, 0.128)
    ax.text(0.41, 0.150, "[B, 16 000]", fontsize=7.5, color=GREEN, style="italic")

    # Key limitation annotation
    ax.text(0.5, 0.005,
            "⚠  Context is fixed — decoder cannot focus on different source positions at different steps",
            ha="center", fontsize=8.5, color=ORANGE,
            bbox=dict(facecolor=LORANGE, edgecolor=ORANGE, alpha=0.8,
                      boxstyle="round,pad=0.3"))

    fig.tight_layout()
    _save(fig, "fig_a2_baseline_decoder")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a3 — Attention decoder: single timestep (dynamic Bahdanau context)
# ─────────────────────────────────────────────────────────────────────────────
def plot_attention_decoder():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Attention Decoder — Single Timestep  (Bahdanau, dynamic context)",
                 fontsize=11, fontweight="bold", pad=8)

    W, Hs, Hm = 0.23, 0.065, 0.085

    # Previous token
    _box(ax, 0.16, 0.90, W, Hs, "y_{t-1}  (previous token)",
         fc=LGREY, ec=GREY, fontsize=8.5)

    # Embedding
    _box(ax, 0.16, 0.775, W, Hs, "Embedding(300) + Dropout",
         fc=LPURPLE, ec=PURPLE, fontsize=8.5)
    _arrow(ax, 0.16, 0.868, 0.16, 0.808)
    ax.text(0.175, 0.838, "[B, 300]", fontsize=7.5, color=PURPLE, style="italic")

    # c_{t-1} previous context
    _box(ax, 0.47, 0.90, W, Hs, "c_{t-1}  (prev context vector)",
         sublabel="[B, 1024]  |  zeros at t=0",
         fc=LRED, ec=RED, fontsize=8)

    # Cat embed + c_{t-1}
    _box(ax, 0.30, 0.660, 0.22, 0.06, "cat([embed ; c_{t-1}])",
         sublabel="→  [B, 1 324]",
         fc=LGREY, ec=GREY, fontsize=8)
    _arrow(ax, 0.16, 0.743, 0.245, 0.660, color=PURPLE)
    _arrow(ax, 0.47, 0.868, 0.365, 0.690, color=RED)

    # LSTM
    _box(ax, 0.30, 0.530, 0.26, Hm, "2-layer LSTM",
         sublabel="hidden=1024  |  drop=0.50\noutputs s_t  [B, 1024]",
         fc=LRED, ec=RED, bold=True, fontsize=8.5)
    _arrow(ax, 0.30, 0.630, 0.30, 0.573)
    ax.annotate("", xy=(0.445, 0.572), xytext=(0.445, 0.488),
                arrowprops=dict(arrowstyle="<|-", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(0.51, 0.530, "h_{t-1}, c_{t-1}", fontsize=7.5, color=RED,
            ha="left", style="italic")

    # ── Attention block ───────────────────────────────────────────────────────
    # Encoder outputs
    _box(ax, 0.82, 0.780, 0.26, Hm,
         "encoder_outputs",
         sublabel="[B, T_src, 1024]\nprecomputed keys W_enc·h_i → [B, T, 256]",
         fc=LBLUE, ec=BLUE, fontsize=8)

    # W_dec(s_t) path
    _box(ax, 0.65, 0.620, 0.20, Hs, "W_dec(s_t)",
         sublabel="→  [B, 256]",
         fc=LRED, ec=RED, fontsize=8)
    _arrow(ax, 0.42, 0.530, 0.555, 0.620, color=RED)
    ax.text(0.48, 0.59, "s_t", fontsize=7.5, color=RED, style="italic")

    # Energy computation
    _box(ax, 0.75, 0.490, 0.26, Hm,
         "Energy  e_{t,i}",
         sublabel="v · tanh(W_enc·h_i + W_dec·s_t)\n→  [B, T_src, 1]",
         fc=LPURPLE, ec=PURPLE, bold=True, fontsize=8)
    _arrow(ax, 0.65, 0.588, 0.715, 0.530, color=RED)
    _arrow(ax, 0.82, 0.737, 0.82, 0.535, color=BLUE)
    ax.text(0.835, 0.635, "keys\n[B, T, 256]", fontsize=7.5, color=BLUE, style="italic")

    # Softmax → α
    _box(ax, 0.75, 0.375, 0.24, Hs, "softmax (pad masked)",
         sublabel="α_t  →  [B, T_src, 1]",
         fc=LPURPLE, ec=PURPLE, fontsize=8)
    _arrow(ax, 0.75, 0.448, 0.75, 0.411)

    # Context vector
    _box(ax, 0.75, 0.265, 0.26, Hm,
         "Context  c_t",
         sublabel="Σ α_{t,i} · h_i\n→  [B, 1024]",
         fc=LRED, ec=RED, bold=True, fontsize=8.5)
    _arrow(ax, 0.75, 0.343, 0.75, 0.309)
    # encoder_outputs → context
    ax.annotate("", xy=(0.88, 0.266), xytext=(0.82, 0.737),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1,
                                connectionstyle="arc3,rad=0.3",
                                shrinkA=4, shrinkB=4))
    ax.text(0.90, 0.50, "encoder_outputs\n[B, T, 1024]",
            fontsize=7, color=BLUE, ha="left", style="italic")

    # c_t → cat + c_{t-1} feedback
    _arrow(ax, 0.75, 0.265, 0.47, 0.895, color=RED)
    ax.text(0.54, 0.73, "c_t becomes\nc_{t-1} next step",
            fontsize=7, color=RED, ha="center", style="italic",
            bbox=dict(facecolor="white", edgecolor=RED, alpha=0.75,
                      boxstyle="round,pad=0.2"))

    # cat(s_t, c_t)
    _box(ax, 0.30, 0.265, 0.24, Hs, "cat([s_t ; c_t])",
         sublabel="→  [B, 2 048]",
         fc=LGREY, ec=GREY, fontsize=8)
    _arrow(ax, 0.30, 0.488, 0.30, 0.300)
    _arrow(ax, 0.62, 0.265, 0.425, 0.265, color=RED)

    # Bottleneck
    _box(ax, 0.30, 0.160, W, Hs, "Linear(2048→512) + tanh + Dropout(0.40)",
         fc=LGREEN, ec=GREEN, fontsize=8)
    _arrow(ax, 0.30, 0.233, 0.30, 0.193)
    ax.text(0.315, 0.212, "[B, 512]", fontsize=7.5, color=GREEN, style="italic")

    # Output head
    _box(ax, 0.30, 0.060, W, Hs, "Linear(512 → 16 000)  →  logits",
         fc=LGREEN, ec=GREEN, fontsize=8.5)
    _arrow(ax, 0.30, 0.128, 0.30, 0.095)

    # Key advantage
    ax.text(0.5, 0.005,
            "✓  c_t is recomputed every step — decoder attends to different source positions as it generates each token",
            ha="center", fontsize=8.5, color=RED,
            bbox=dict(facecolor=LRED, edgecolor=RED, alpha=0.8,
                      boxstyle="round,pad=0.3"))

    fig.tight_layout()
    _save(fig, "fig_a3_attention_decoder")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a4 — Parameter breakdown: stacked bar baseline vs attention
# ─────────────────────────────────────────────────────────────────────────────
def plot_parameter_breakdown():
    components = ["Embedding\n(shared)", "Encoder\nBiLSTM", "Bridge",
                  "Decoder\nLSTM", "Bottleneck\n+ Output head", "Attention\nweights"]

    baseline = [4_800_000, 9_633_792, 2_099_200, 18_022_400, 9_257_088, 0]
    attention = [4_800_000, 9_633_792, 2_099_200, 18_022_400, 9_257_088, 524_544]

    colors = [PURPLE, BLUE, TEAL, ORANGE, GREEN, RED]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                             gridspec_kw={"width_ratios": [1.8, 1]})

    # ── Left: stacked bars ────────────────────────────────────────────────────
    ax = axes[0]
    models   = ["Baseline\n43,812,480", "Attention\n44,337,024"]
    datasets = [baseline, attention]

    bottoms = [0, 0]
    bars_collection = []
    for i, (comp, bl, at, col) in enumerate(zip(components, baseline, attention, colors)):
        vals = [bl, at]
        b = ax.bar(models, vals, bottom=bottoms, color=col, alpha=0.85,
                   label=comp, edgecolor="white", linewidth=1.2)
        bars_collection.append((b, vals, bottoms[:]))
        for j in range(2):
            mid = bottoms[j] + vals[j] / 2
            if vals[j] > 500_000:
                ax.text(j, mid, f"{vals[j]/1e6:.2f}M",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Breakdown by Component", pad=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_ylim(0, 50_000_000)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)

    # ── Right: attention overhead donut ────────────────────────────────────────
    ax2 = axes[1]
    sizes  = [43_812_480, 524_544]
    cols   = [BLUE, RED]
    labels = [f"Shared with baseline\n43,812,480 params", f"Attention only\n524,544 params"]
    wedges, _ = ax2.pie(sizes, colors=cols, startangle=90,
                         wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
                         explode=[0, 0.08])
    ax2.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               fontsize=8.5, framealpha=0.9)
    ax2.text(0, 0, "+1.2%\noverhead", ha="center", va="center",
             fontsize=10, fontweight="bold", color=RED)
    ax2.set_title("Attention\nParameter Overhead", pad=8)

    fig.suptitle("Model Parameter Breakdown — Baseline vs Attention",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(pad=2.0)
    _save(fig, "fig_a4_parameter_breakdown")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a5 — Bahdanau vs Luong comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_bahdanau_vs_luong():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Bahdanau vs Luong Attention — Design Comparison",
                 fontsize=12, fontweight="bold", pad=10)

    # Column headers
    cols_x  = [0.07, 0.38, 0.72]
    headers = ["Property", "Bahdanau (additive)  ✓ chosen", "Luong (multiplicative)"]
    hcols   = [DARK, GREEN, GREY]
    for x, h, c in zip(cols_x, headers, hcols):
        ax.text(x, 0.935, h, fontsize=10, fontweight="bold", color=c, va="top")

    # Divider line under header
    ax.axhline(0.900, xmin=0.0, xmax=1.0, color=GREY, lw=1.2)

    rows = [
        ("Query timing",
         "s_{t−1}  (BEFORE LSTM step)\n→ context informs current generation",
         "s_t  (AFTER LSTM step)\n→ context only rescores output"),
        ("Score function",
         "v · tanh(W_enc·h_i + W_dec·s)\nAdditive MLP, non-linear",
         "s_t · h_i  (dot)\nor s_t · W · h_i  (general)"),
        ("Projection",
         "Both projected to 256-d\n→ controls energy scale, avoids saturation",
         "No projection (dot product)\n→ large inner products at 1024-d"),
        ("Softmax stability",
         "✓ Stable — 256-d projection keeps energies bounded",
         "⚠ Risk of saturation at 1024-d without √d scaling"),
        ("Expressiveness",
         "✓ Non-linear alignment function\n→ better for noisy IRC multi-turn context",
         "Linear dot product\n→ sufficient for clean parallel text (NMT)"),
    ]

    row_colors_bahdanau = [LGREEN, LGREEN, LGREEN, LGREEN, LGREEN]
    row_colors_luong    = [LGREY,  LGREY,  LGREY,  "#FFF9C4", LGREY]

    y_start, row_h = 0.865, 0.155

    for i, (prop, bah, luong) in enumerate(rows):
        y_top = y_start - i * row_h
        y_mid = y_top - row_h / 2

        # Alternating row background
        if i % 2 == 0:
            ax.axhspan(y_top - row_h, y_top, color="#F8F8F8", zorder=0)

        # Property column
        ax.text(cols_x[0], y_mid + 0.01, prop,
                fontsize=8.5, color=DARK, fontweight="bold", va="center")

        # Bahdanau column
        bg_b = FancyBboxPatch((cols_x[1] - 0.01, y_top - row_h + 0.008),
                               0.32, row_h - 0.016,
                               boxstyle="round,pad=0.005",
                               facecolor=LGREEN, edgecolor=GREEN,
                               linewidth=0.7, alpha=0.5, zorder=1)
        ax.add_patch(bg_b)
        ax.text(cols_x[1], y_mid, bah, fontsize=8, color=DARK, va="center",
                linespacing=1.4, zorder=2)

        # Luong column
        luong_col = "#FFF176" if "⚠" in luong else LGREY
        bg_l = FancyBboxPatch((cols_x[2] - 0.01, y_top - row_h + 0.008),
                               0.32, row_h - 0.016,
                               boxstyle="round,pad=0.005",
                               facecolor=luong_col, edgecolor=GREY,
                               linewidth=0.7, alpha=0.6, zorder=1)
        ax.add_patch(bg_l)
        ax.text(cols_x[2], y_mid, luong, fontsize=8, color=DARK, va="center",
                linespacing=1.4, zorder=2)

    # Bottom divider
    ax.axhline(y_start - len(rows) * row_h, color=GREY, lw=1.2)

    # Footer note
    ax.text(0.5, 0.015,
            "Chosen: Bahdanau — stable energy scale, non-linear alignment, "
            "context fed into the LSTM step rather than rescoring after it",
            ha="center", fontsize=8.5, color=GREEN, style="italic",
            bbox=dict(facecolor=LGREEN, edgecolor=GREEN,
                      boxstyle="round,pad=0.3", alpha=0.8))

    fig.tight_layout()
    _save(fig, "fig_a5_bahdanau_vs_luong")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Architecture report figures...")
    plot_overview()
    plot_baseline_decoder()
    plot_attention_decoder()
    plot_parameter_breakdown()
    plot_bahdanau_vs_luong()
    print(f"\nAll figures saved to {FIG_DIR}")
