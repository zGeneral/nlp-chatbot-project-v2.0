"""
plot_architecture.py — Publication-quality figures for the Architecture report.
Uses Graphviz (dot layout) for flowchart diagrams, matplotlib for charts.

Figures produced (report/figures/, PNG + PDF at 300 DPI):
  fig_a1  — Full encoder-decoder architecture overview (both models side by side)
  fig_a2  — Baseline decoder step (fixed context, single timestep)
  fig_a3  — Attention decoder step (dynamic context, Bahdanau)
  fig_a4  — Parameter breakdown: stacked bar + overhead donut
  fig_a5  — Bahdanau vs Luong attention comparison table

Usage:
    python report/plot_architecture.py
"""

import os
import sys
from pathlib import Path

# Ensure the Graphviz binaries are on PATH (winget installs here)
os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"

import graphviz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT    = Path(__file__).parent        # report/
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_EMBED  = "#CE93D8"   # purple  — embedding
C_ENC    = "#90CAF9"   # blue    — encoder
C_BRIDGE = "#80CBC4"   # teal    — bridge
C_DEC    = "#FFCC80"   # orange  — decoder (baseline)
C_ATTN   = "#EF9A9A"   # red     — attention
C_PROJ   = "#A5D6A7"   # green   — projection / output
C_IO     = "#ECEFF1"   # grey    — input/output tokens
C_OP     = "#FFF9C4"   # yellow  — cat / operation nodes
DARK     = "#263238"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save_gv(src: graphviz.Source, name: str, debug: bool = False):
    """Render graphviz source to PNG and PDF."""
    if debug:
        (FIG_DIR / f"{name}.dot").write_text(src.source, encoding="utf-8")
    out = str(FIG_DIR / name)
    src.render(out, format="png", cleanup=True)
    src.render(out, format="pdf", cleanup=True)
    print(f"  Saved → {name}.png / .pdf")


def _save_mpl(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {name}.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Graphviz style helpers
# ─────────────────────────────────────────────────────────────────────────────
_GRAPH_ATTRS = {
    "bgcolor": "white",
    "fontname": "Helvetica",
    "fontsize": "11",
    "nodesep": "0.40",
    "ranksep": "0.55",
    "splines": "spline",
    "pad": "0.55",
    "dpi": "300",
}

_EDGE_ATTRS = {
    "fontname": "Helvetica",
    "fontsize": "9",
    "color": "#546E7A",
    "arrowsize": "0.7",
}

def _node(g, name, label, fillcolor=C_IO, shape="box",
          fontsize="10", style="filled,rounded", width="2.2", height="0.45"):
    g.node(name, label=label, fillcolor=fillcolor, style=style,
           shape=shape, fontname="Helvetica", fontsize=fontsize,
           width=width, height=height, margin="0.15,0.08")


def _edge(g, a, b, label="", color="#546E7A", style="solid"):
    g.edge(a, b, label=label, fontname="Helvetica", fontsize="9",
           color=color, fontcolor=color, style=style,
           arrowsize="0.75", penwidth="1.3")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a1 — Side-by-side architecture overview (two sub-graphs in one image)
# ─────────────────────────────────────────────────────────────────────────────
def plot_overview():
    g = graphviz.Digraph("architecture_overview",
                          graph_attr={**_GRAPH_ATTRS,
                                      "rankdir": "TB",
                                      "label": "Seq2Seq LSTM Architecture — Baseline vs Attention",
                                      "labelloc": "t",
                                      "fontsize": "14",
                                      "fontname": "Helvetica-Bold",
                                      "nodesep": "0.4",
                                      "ranksep": "0.55"})

    for model in ("baseline", "attention"):
        pfx = "b_" if model == "baseline" else "a_"
        is_attn = (model == "attention")
        title = "Baseline  (no attention)" if not is_attn else "Attention  (Bahdanau)"
        title_color = "#1565C0" if not is_attn else "#B71C1C"

        with g.subgraph(name=f"cluster_{model}") as sg:
            sg.attr(label=title, fontsize="12", fontname="Helvetica-Bold",
                    fontcolor=title_color, style="rounded",
                    color=("#90CAF9" if not is_attn else "#EF9A9A"),
                    penwidth="2", bgcolor="#FAFAFA")

            # Input
            _node(sg, pfx+"src", "src_ids  [B, T_src]",
                  fillcolor=C_IO, shape="parallelogram", width="2.4")

            # Embedding
            _node(sg, pfx+"emb", "Shared Embedding\n16 000 × 300  |  drop=0.30",
                  fillcolor=C_EMBED, width="2.8")

            # BiLSTM Encoder
            _node(sg, pfx+"enc", "BiLSTM Encoder  ×2 layers\nhidden=512/dir  |  drop=0.50\n14.4M params",
                  fillcolor=C_ENC, width="3.0", height="0.65", fontsize="9")

            # Bridge
            _node(sg, pfx+"bri", "Bridge\nLinear(1024→1024) ×2  |  tanh\n2.1M params",
                  fillcolor=C_BRIDGE, width="2.8", height="0.58")

            # Decoder LSTM
            dec_label = (
                "Decoder LSTM  ×2 layers\nhidden=1024  |  drop=0.50\ninput: [embed ; c_t]  1324-d"
                if is_attn else
                "Decoder LSTM  ×2 layers\nhidden=1024  |  drop=0.50\ninput: [embed ; ctx_fixed]  1324-d"
            )
            dec_color = C_ATTN if is_attn else C_DEC
            _node(sg, pfx+"dec", dec_label,
                  fillcolor=dec_color, width="3.2", height="0.65", fontsize="9")

            # Context node
            if is_attn:
                _node(sg, pfx+"attn",
                      "Bahdanau Attention\nW_enc·h_i + W_dec·s_t → 256-d\nα_t → c_t  [B, 1024]",
                      fillcolor=C_ATTN, shape="diamond", width="3.2", height="0.75",
                      fontsize="9", style="filled")
            else:
                _node(sg, pfx+"ctx",
                      "ctx_fixed\nencoder_outputs[:, −1, :]\n[B, 1024]  — constant",
                      fillcolor=C_DEC, shape="note", width="2.8", height="0.65",
                      fontsize="9")

            # Bottleneck
            _node(sg, pfx+"proj",
                  "Linear(2048→512) + tanh + Dropout\n→ [B, 512]",
                  fillcolor=C_PROJ, width="3.0")

            # Output head
            _node(sg, pfx+"out",
                  "Linear(512 → 16 000)  →  logits",
                  fillcolor=C_PROJ, width="3.0")

            # Output token
            _node(sg, pfx+"trg", "logits  [B, T_trg, 16 000]",
                  fillcolor=C_IO, shape="parallelogram", width="2.8")

            # Edges
            _edge(sg, pfx+"src",  pfx+"emb")
            _edge(sg, pfx+"emb",  pfx+"enc")
            _edge(sg, pfx+"enc",  pfx+"bri",
                  label=" states [4,B,512]")
            _edge(sg, pfx+"bri",  pfx+"dec",
                  label=" h₀,c₀ [2,B,1024]")

            if is_attn:
                _edge(sg, pfx+"enc",  pfx+"attn",
                      color="#C62828", style="dashed")
                _edge(sg, pfx+"dec",  pfx+"attn",
                      label=" s_t", color="#C62828", style="dashed")
                _edge(sg, pfx+"attn", pfx+"dec",
                      label=" c_t [B,1024]", color="#C62828")
                _edge(sg, pfx+"dec",  pfx+"proj",
                      label=" cat[s_t;c_t] 2048-d")
            else:
                _edge(sg, pfx+"enc",  pfx+"ctx",
                      label=" last timestep", color="#E65100", style="dashed")
                _edge(sg, pfx+"ctx",  pfx+"dec",
                      label=" concat each step", color="#E65100")
                _edge(sg, pfx+"dec",  pfx+"proj",
                      label=" cat[s_t;ctx] 2048-d")

            _edge(sg, pfx+"proj", pfx+"out")
            _edge(sg, pfx+"out",  pfx+"trg")

            # Total param badge
            total = "44,337,024" if is_attn else "43,812,480"
            sg.node(pfx+"total",
                    label=f"Total: {total} params",
                    shape="plaintext", fontsize="10", fontname="Helvetica-Bold",
                    fontcolor=title_color)
            _edge(sg, pfx+"trg", pfx+"total", style="invis")

    _save_gv(g, "fig_a1_architecture_overview")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a2 — Baseline decoder: single timestep
# ─────────────────────────────────────────────────────────────────────────────
def plot_baseline_decoder():
    g = graphviz.Digraph("baseline_decoder",
                          graph_attr={**_GRAPH_ATTRS,
                                      "rankdir": "TB",
                                      "label": "Baseline Decoder — Single Timestep (fixed context)",
                                      "labelloc": "t", "fontsize": "13",
                                      "fontname": "Helvetica-Bold",
                                      "ranksep": "0.55", "nodesep": "0.5"})

    _node(g, "prev",   "y_{t-1}  (previous token)",
          fillcolor=C_IO, shape="parallelogram", width="2.6")
    _node(g, "emb",    "Embedding(300) + Dropout(0.30)\n→ [B, 300]",
          fillcolor=C_EMBED, width="2.8")
    _node(g, "ctx",    "ctx_fixed  =  encoder_outputs[:, −1, :]\n[B, 1024]  — same at every decoder step",
          fillcolor=C_DEC, shape="note", width="3.8", height="0.55", fontsize="9")
    _node(g, "cat1",   "cat([embed ; ctx_fixed])\n→ [B, 1 324]",
          fillcolor=C_OP, width="2.8")
    _node(g, "lstm",   "2-layer LSTM\nhidden=1024  |  drop=0.50\n→ s_t  [B, 1024]",
          fillcolor=C_DEC, width="2.8", height="0.62", fontsize="9")
    _node(g, "cat2",   "cat([s_t ; ctx_fixed])\n→ [B, 2 048]",
          fillcolor=C_OP, width="2.8")
    _node(g, "proj",   "Linear(2048→512) + tanh + Dropout(0.40)\n→ [B, 512]",
          fillcolor=C_PROJ, width="3.4")
    _node(g, "out",    "Linear(512 → 16 000)  →  logits [B, 16 000]",
          fillcolor=C_PROJ, width="3.4")

    _node(g, "warn",
          "⚠  ctx_fixed never changes — decoder cannot\n"
          "focus on different source positions each step",
          fillcolor="#FFF3E0", shape="note", width="3.8",
          fontsize="9", style="filled")

    _edge(g, "prev",  "emb")
    _edge(g, "emb",   "cat1",  label="[B,300]", color="#7B1FA2")
    _edge(g, "ctx",   "cat1",  label="[B,1024]", color="#E65100", style="dashed")
    _edge(g, "cat1",  "lstm",  label="[B,1324]")
    _edge(g, "lstm",  "cat2",  label="s_t [B,1024]")
    _edge(g, "ctx",   "cat2",  label="[B,1024]", color="#E65100", style="dashed")
    _edge(g, "cat2",  "proj")
    _edge(g, "proj",  "out")
    _edge(g, "out",   "warn",  style="invis")

    # Recurrent state
    g.node("rec", label="h_{t-1}, c_{t-1}\n(recurrent state)",
           fillcolor="#E0F2F1", style="filled,rounded", shape="box",
           fontname="Helvetica", fontsize="9", width="1.9", height="0.5")
    _edge(g, "rec",  "lstm",  label="prev state", color="#00796B", style="dashed")
    _edge(g, "lstm", "rec",   label="next state", color="#00796B", style="dashed")

    _save_gv(g, "fig_a2_baseline_decoder")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a3 — Attention decoder: single timestep
# ─────────────────────────────────────────────────────────────────────────────
def plot_attention_decoder():
    g = graphviz.Digraph("attention_decoder",
                          graph_attr={**_GRAPH_ATTRS,
                                      "rankdir": "TB",
                                      "label": "Attention Decoder — Single Timestep (Bahdanau)",
                                      "labelloc": "t", "fontsize": "13",
                                      "fontname": "Helvetica-Bold",
                                      "ranksep": "0.60", "nodesep": "0.5"})

    # Main decode path
    _node(g, "prev",   "y_{t-1}  (previous token)",
          fillcolor=C_IO, shape="parallelogram", width="2.6")
    _node(g, "cprev",  "c_{t-1}  (prev context vector)\n[B, 1024]  |  zeros at t = 0",
          fillcolor=C_ATTN, width="2.8", height="0.5", fontsize="9")
    _node(g, "emb",    "Embedding(300) + Dropout(0.30)\n→ [B, 300]",
          fillcolor=C_EMBED, width="2.8")
    _node(g, "cat1",   "cat([embed ; c_{t-1}])\n→ [B, 1 324]",
          fillcolor=C_OP, width="2.6")
    _node(g, "lstm",   "2-layer LSTM\nhidden=1024  |  drop=0.50\n→ s_t  [B, 1024]",
          fillcolor=C_ATTN, width="2.8", height="0.62", fontsize="9")

    # Attention sub-graph
    with g.subgraph(name="cluster_attn") as sa:
        sa.attr(label="Bahdanau Attention Mechanism",
                style="rounded,dashed", color="#C62828",
                fontname="Helvetica-Bold", fontsize="10", fontcolor="#C62828",
                bgcolor="#FFF5F5", penwidth="1.8")

        _node(sa, "enc_out", "encoder_outputs\n[B, T_src, 1024]",
              fillcolor=C_ENC, width="2.8", height="0.55", fontsize="9")
        _node(sa, "wdec",    "W_dec(s_t)\n→ [B, 256]",
              fillcolor=C_ATTN, width="2.4")
        _node(sa, "energy",  "e_{t,i} = v · tanh(W_enc·h_i + W_dec·s_t)\n→ [B, T_src, 1]  (pad masked to −∞)",
              fillcolor="#F8BBD0", width="3.8", height="0.55", fontsize="9")
        _node(sa, "softmax", "softmax(e_{t,i})  →  α_t\n[B, T_src, 1]",
              fillcolor="#F8BBD0", width="3.0")
        _node(sa, "ctx",     "c_t = Σ α_{t,i} · h_i\n→ [B, 1024]",
              fillcolor=C_ATTN, shape="diamond", width="3.0", height="0.65",
              fontsize="10", style="filled")

        _edge(sa, "wdec",    "energy",  label="query [B,256]",  color="#C62828")
        _edge(sa, "enc_out", "energy",  label="keys [B,T,256]", color="#C62828")
        _edge(sa, "energy",  "softmax")
        _edge(sa, "softmax", "ctx",     label="α_t weights", color="#C62828")
        _edge(sa, "enc_out", "ctx",     color="#C62828", style="dashed")

    # Continue decode path
    _node(g, "cat2",   "cat([s_t ; c_t])\n→ [B, 2 048]",
          fillcolor=C_OP, width="2.6")
    _node(g, "proj",   "Linear(2048→512) + tanh + Dropout(0.40)\n→ [B, 512]",
          fillcolor=C_PROJ, width="3.4")
    _node(g, "out",    "Linear(512 → 16 000)  →  logits [B, 16 000]",
          fillcolor=C_PROJ, width="3.4")

    _node(g, "note",
          "✓  c_t recomputed every step — decoder attends\n"
          "to different source positions as it generates",
          fillcolor="#E8F5E9", shape="note", width="3.8",
          fontsize="9", style="filled")

    # Recurrent state
    g.node("rec", label="h_{t-1}, c_{t-1}\n(LSTM state)",
           fillcolor="#E0F2F1", style="filled,rounded", shape="box",
           fontname="Helvetica", fontsize="9", width="1.8", height="0.5")

    # Edges
    _edge(g, "prev",  "emb")
    _edge(g, "cprev", "cat1",   label="c_{t-1} [B,1024]", color="#C62828", style="dashed")
    _edge(g, "emb",   "cat1",   label="[B,300]", color="#7B1FA2")
    _edge(g, "cat1",  "lstm")
    _edge(g, "rec",   "lstm",   label="prev state", color="#00796B", style="dashed")
    _edge(g, "lstm",  "rec",    label="next state", color="#00796B", style="dashed")
    _edge(g, "lstm",  "wdec",   label="s_t [B,1024]", color="#C62828")
    _edge(g, "lstm",  "cat2",   label="s_t [B,1024]")
    _edge(g, "ctx",   "cat2",   label="c_t [B,1024]", color="#C62828")
    _edge(g, "ctx",   "cprev",  label="→ c_{t-1}\n(next step)",
          color="#C62828", style="dashed")
    _edge(g, "cat2",  "proj")
    _edge(g, "proj",  "out")
    _edge(g, "out",   "note",   style="invis")

    _save_gv(g, "fig_a3_attention_decoder")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a4 — Parameter breakdown (matplotlib — chart, not flowchart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_parameter_breakdown():
    import matplotlib.ticker as mticker

    components = ["Embedding\n(shared)", "Encoder\nBiLSTM", "Bridge",
                  "Decoder\nLSTM", "Bottleneck +\nOutput head", "Attention\nweights"]
    baseline   = [4_800_000, 9_633_792, 2_099_200, 18_022_400, 9_257_088, 0]
    attention  = [4_800_000, 9_633_792, 2_099_200, 18_022_400, 9_257_088, 524_544]
    colors     = ["#CE93D8", "#90CAF9", "#80CBC4", "#FFCC80", "#A5D6A7", "#EF9A9A"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.8, 1]})

    models   = ["Baseline\n43,812,480", "Attention\n44,337,024"]
    datasets = [baseline, attention]
    bottoms  = [0, 0]

    for comp, bl, at, col in zip(components, baseline, attention, colors):
        vals = [bl, at]
        ax.bar(models, vals, bottom=bottoms, color=col, alpha=0.9,
               label=comp, edgecolor="white", linewidth=1.2)
        for j in range(2):
            mid = bottoms[j] + vals[j] / 2
            if vals[j] > 500_000:
                ax.text(j, mid, f"{vals[j]/1e6:.2f}M",
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Breakdown by Component", pad=8, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_ylim(0, 50_000_000)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Donut
    sizes  = [43_812_480, 524_544]
    cols2  = ["#90CAF9", "#EF9A9A"]
    wedges, _ = ax2.pie(sizes, colors=cols2, startangle=90,
                         wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
                         explode=[0, 0.08])
    ax2.legend(wedges,
               [f"Shared with baseline\n43,812,480",
                f"Attention only\n524,544  (+1.2%)"],
               loc="lower center", bbox_to_anchor=(0.5, -0.14),
               fontsize=8.5, framealpha=0.9)
    ax2.text(0, 0, "+1.2%\noverhead", ha="center", va="center",
             fontsize=11, fontweight="bold", color="#C62828")
    ax2.set_title("Attention\nParameter Overhead", pad=8, fontweight="bold")

    fig.suptitle("Model Parameter Breakdown — Baseline vs Attention",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    _save_mpl(fig, "fig_a4_parameter_breakdown")


# ─────────────────────────────────────────────────────────────────────────────
# fig_a5 — Bahdanau vs Luong comparison (matplotlib table — chart, not flowchart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_bahdanau_vs_luong():
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Bahdanau vs Luong Attention — Design Comparison",
                 fontsize=13, fontweight="bold", pad=12)

    cols_x = [0.07, 0.38, 0.72]

    # Column headers
    for x, h, c in zip(cols_x,
                        ["Property", "Bahdanau (additive)  ✓ chosen", "Luong (multiplicative)"],
                        [DARK, "#2E7D32", "#546E7A"]):
        ax.text(x, 0.935, h, fontsize=10.5, fontweight="bold", color=c, va="top")
    ax.axhline(0.900, color="#B0BEC5", lw=1.4)

    rows = [
        ("Query timing",
         "s_{t−1}  (BEFORE LSTM step)\nContext informs what to generate next",
         "s_t  (AFTER LSTM step)\nContext only rescores existing output"),
        ("Score function",
         "v · tanh(W_enc·h_i + W_dec·s)\nNon-linear additive MLP",
         "s_t · h_i  (dot product)\nor  s_t · W · h_i  (general)"),
        ("Projection",
         "Both projected to 256-d\nControls energy scale, prevents saturation",
         "No projection\nRaw 1024-d dot product"),
        ("Softmax stability",
         "✓  Stable — 256-d energies stay bounded",
         "⚠  Risk of saturation at 1024-d\n(Transformers add √d scaling to fix this)"),
        ("Expressiveness",
         "✓  Non-linear — learns complex alignment\nBetter for noisy multi-turn IRC context",
         "Linear compatibility function\nSufficient for clean parallel text (NMT)"),
    ]

    y_start, row_h = 0.865, 0.155

    for i, (prop, bah, luong) in enumerate(rows):
        y_top = y_start - i * row_h
        y_mid = y_top - row_h / 2

        if i % 2 == 0:
            ax.axhspan(y_top - row_h, y_top, color="#FAFAFA", zorder=0)

        ax.text(cols_x[0], y_mid + 0.01, prop,
                fontsize=9, color=DARK, fontweight="bold", va="center")

        bg_b = FancyBboxPatch((cols_x[1] - 0.01, y_top - row_h + 0.01),
                               0.31, row_h - 0.02,
                               boxstyle="round,pad=0.005",
                               facecolor="#E8F5E9", edgecolor="#A5D6A7",
                               linewidth=0.8, alpha=0.6, zorder=1)
        ax.add_patch(bg_b)
        ax.text(cols_x[1], y_mid, bah, fontsize=8.5, color=DARK, va="center",
                linespacing=1.5, zorder=2)

        luong_fc = "#FFF9C4" if "⚠" in luong else "#F5F5F5"
        luong_ec = "#F9A825" if "⚠" in luong else "#B0BEC5"
        bg_l = FancyBboxPatch((cols_x[2] - 0.01, y_top - row_h + 0.01),
                               0.31, row_h - 0.02,
                               boxstyle="round,pad=0.005",
                               facecolor=luong_fc, edgecolor=luong_ec,
                               linewidth=0.8, alpha=0.7, zorder=1)
        ax.add_patch(bg_l)
        ax.text(cols_x[2], y_mid, luong, fontsize=8.5, color=DARK, va="center",
                linespacing=1.5, zorder=2)

    ax.axhline(y_start - len(rows) * row_h, color="#B0BEC5", lw=1.4)
    ax.text(0.5, 0.012,
            "Selected: Bahdanau — stable energy scale, non-linear alignment, "
            "context fed INTO the LSTM step rather than rescoring after it",
            ha="center", fontsize=9, color="#2E7D32", style="italic",
            bbox=dict(facecolor="#E8F5E9", edgecolor="#A5D6A7",
                      boxstyle="round,pad=0.35", alpha=0.9))

    fig.tight_layout()
    _save_mpl(fig, "fig_a5_bahdanau_vs_luong")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Architecture report figures (Graphviz + matplotlib)...")
    plot_overview()
    plot_baseline_decoder()
    plot_attention_decoder()
    plot_parameter_breakdown()
    plot_bahdanau_vs_luong()
    print(f"\nAll figures saved to {FIG_DIR}")

