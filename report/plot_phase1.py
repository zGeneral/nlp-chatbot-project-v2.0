"""
plot_phase1.py — Generate publication-quality figures for the Phase 1 pipeline report.

Figures produced (saved to report/figures/ as PNG + PDF at 300 DPI):
  fig_p1  — Full data-reduction funnel (waterfall chart)
  fig_p2  — Stage 2 dialogue filter breakdown (horizontal bar)
  fig_p3  — Stage 4 pair filter breakdown (horizontal bar)
  fig_p4  — Stage 4.5 domain filter composition (stacked bar + set sizes)
  fig_p5  — Token length distributions: context and response (two-panel)
  fig_p6  — Temporal train/val/test split timeline

Usage:
    python report/plot_phase1.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent      # report/
ROOT        = _SCRIPT_DIR.parent         # project root
ARTIFACTS  = ROOT / "artifacts"
FIG_DIR    = _SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style (matches evaluation/training figures) ──────────────────────
BLUE   = "#2196F3"
RED    = "#F44336"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GREY   = "#9E9E9E"
DARK   = "#263238"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


def _save(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved → {name}.png / .pdf")


def _fmt(n: int) -> str:
    """Format large integer with comma thousands separator."""
    return f"{n:,}"


# ─────────────────────────────────────────────────────────────────────────────
# fig_p1 — Full pipeline funnel (waterfall / step chart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_funnel():
    # Verified numbers from stats JSONs + pipeline report
    # Each row: (label, count, unit, colour, right-side annotation)
    rows = [
        ("Stage 1 — Load",             1_852_868, "dialogues", BLUE,   None),
        ("Stage 2 — Clean & Filter",   1_315_108, "dialogues", BLUE,   "−537,760  (29.0%)"),
        ("Stage 3 — Split (train)",    1_259_711, "dialogues", BLUE,   "−55,397  (train subset)"),
        ("Stage 4 — Pair Generation",  1_500_000, "pairs",     ORANGE, "×1.19 expansion"),
        ("Stage 4.5 — Domain Filter",  1_103_539, "pairs",     ORANGE, "−396,461  (26.4%)"),
        ("Stage 6 — Encoded (final)",  1_103_539, "pairs",     GREEN,  "GPU-ready JSONL ✓"),
    ]

    labels  = [r[0] for r in rows]
    counts  = [r[1] for r in rows]
    units   = [r[2] for r in rows]
    colors  = [r[3] for r in rows]
    annots  = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    bar_h = 0.52
    y_pos = list(range(len(rows)))[::-1]
    max_val = max(counts)

    bars = ax.barh(y_pos, counts, height=bar_h, color=colors, alpha=0.85, zorder=2)

    # Count labels (right of bar)
    for yp, cnt, unit in zip(y_pos, counts, units):
        ax.text(cnt + max_val * 0.008, yp, f"{_fmt(cnt)} {unit}",
                va="center", ha="left", fontsize=9, color=DARK, fontweight="bold")

    # Discard / annotation labels (inside bar, left-aligned)
    for yp, ann in zip(y_pos, annots):
        if ann:
            ax.text(max_val * 0.01, yp, ann,
                    va="center", ha="left", fontsize=8.5,
                    color="white", fontweight="bold", alpha=0.92)

    # Connector dashes between stages
    for i in range(len(rows) - 1):
        x_val = min(counts[i], counts[i + 1])
        ax.plot([x_val, x_val],
                [y_pos[i + 1] + bar_h / 2, y_pos[i] - bar_h / 2],
                color=GREY, lw=1.2, ls="--", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Record count", fontsize=10)
    ax.set_title("Phase 1 — Full Data Reduction Funnel", pad=10)
    ax.set_xlim(0, max_val * 1.32)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"))
    ax.grid(axis="x", ls=":", alpha=0.4, zorder=0)

    patches = [
        mpatches.Patch(color=BLUE,   alpha=0.85, label="Dialogue records"),
        mpatches.Patch(color=ORANGE, alpha=0.85, label="Context-response pairs"),
        mpatches.Patch(color=GREEN,  alpha=0.85, label="Final GPU-ready dataset"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "fig_p1_funnel")


# ─────────────────────────────────────────────────────────────────────────────
# fig_p2 — Stage 2 dialogue filter breakdown
# ─────────────────────────────────────────────────────────────────────────────
def plot_stage2_filters():
    filters = {
        "Speaker dominance\n(one speaker > 80% of turns)":  234_864,
        "Temporal hard ceiling\n(single gap > 3,600 s)":    175_962,
        "Low alternation\n(ratio < 0.15)":                   82_045,
        "Non-dyadic\n(≠ 2 speakers)":                        26_402,
        "Temporal gap ratio\n(> 30% large-gap turns)":       16_658,
        "Too few turns\n(< 2 turns)":                         1_749,
        "Too few valid dates":                                    80,
    }

    labels = list(filters.keys())[::-1]
    values = list(filters.values())[::-1]
    total  = sum(filters.values())

    palette = [BLUE, BLUE, BLUE, ORANGE, ORANGE, RED, RED][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=palette, alpha=0.85, height=0.6)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{_fmt(val)}  ({pct:.1f}%)", va="center", fontsize=9, color=DARK)

    ax.set_xlabel("Dialogues discarded")
    ax.set_title(f"Stage 2 — Dialogue Filter Breakdown  (total discarded: {_fmt(total)})", pad=10)
    ax.set_xlim(0, max(values) * 1.42)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1e3)}k"))
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_p2_stage2_filters")


# ─────────────────────────────────────────────────────────────────────────────
# fig_p3 — Stage 4 pair filter breakdown
# ─────────────────────────────────────────────────────────────────────────────
def plot_stage4_filters():
    filters = {
        "Response too short\n(< 5 word-level tokens)":       971_906,
        "Incoherent pair\n(< 5 shared content words)":       748_826,
        "Response too long\n(> 40 tokens)":                  255_579,
        "Context too short\n(< 3 word-level tokens)":         33_012,
        "Bot response\n(scripted / moderation)":               8_942,
        "Echo pair\n(response verbatim in context)":           4_298,
        "Diversity cap\n(same response > 500×)":               1_730,
    }

    labels = list(filters.keys())[::-1]
    values = list(filters.values())[::-1]
    total  = sum(filters.values())

    # Colour by severity
    palette = [RED, RED, ORANGE, ORANGE, BLUE, BLUE, GREY][::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, values, color=palette, alpha=0.85, height=0.6)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_width() + total * 0.003, bar.get_y() + bar.get_height() / 2,
                f"{_fmt(val)}  ({pct:.1f}%)", va="center", fontsize=9, color=DARK)

    ax.set_xlabel("Pairs discarded (train split)")
    ax.set_title(f"Stage 4 — Pair Filter Breakdown  (total discarded: {_fmt(total)})", pad=10)
    ax.set_xlim(0, max(values) * 1.52)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune="both"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"))
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig_p3_stage4_filters")


# ─────────────────────────────────────────────────────────────────────────────
# fig_p4 — Stage 4.5 domain filter composition (stacked bar + overlap breakdown)
# ─────────────────────────────────────────────────────────────────────────────
def plot_domain_filter():
    # From stage4_5_filter_stats.json (train split)
    n_cmd_only  = 878_067 - 283_535   # cmd signal only = cmd total − both
    n_q_only    = 509_007 - 283_535   # question only   = question total − both
    n_both      = 283_535             # matched both signals
    n_filtered  = 396_461             # neither signal

    total = n_cmd_only + n_q_only + n_both + n_filtered

    # Two-panel layout: left = stacked bar, right = breakdown donut
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.4]})

    # ── Left: horizontal stacked bar ──────────────────────────────────────────
    categories = ["Kept\n(1,103,539)", "Filtered out\n(396,461)"]
    kept_parts = [n_cmd_only, n_both, n_q_only]
    kept_colors = [BLUE, PURPLE, GREEN]
    kept_labels = [f"Command signal only\n({_fmt(n_cmd_only)})",
                   f"Both signals\n({_fmt(n_both)})",
                   f"Question signal only\n({_fmt(n_q_only)})"]

    left = 0
    for val, col, lbl in zip(kept_parts, kept_colors, kept_labels):
        ax1.barh(0, val, left=left, color=col, alpha=0.85, height=0.5, label=lbl)
        if val > 30_000:
            ax1.text(left + val / 2, 0, f"{val/1e3:.0f}k",
                     ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        left += val

    ax1.barh(1, n_filtered, color=RED, alpha=0.75, height=0.5, label=f"No domain signal\n({_fmt(n_filtered)})")
    ax1.text(n_filtered / 2, 1, f"{n_filtered/1e3:.0f}k",
             ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(categories, fontsize=9)
    ax1.set_xlabel("Pairs (train split)")
    ax1.set_title("Stage 4.5 — Domain Filter Outcome", pad=8)
    ax1.set_xlim(0, 1_300_000)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1e3)}k"))
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.grid(axis="x", ls=":", alpha=0.4)

    # ── Right: signal overlap breakdown (donut) ────────────────────────────────
    sizes  = [n_cmd_only, n_both, n_q_only, n_filtered]
    colors = [BLUE, PURPLE, GREEN, RED]
    explode = [0.02, 0.06, 0.02, 0.02]
    wedge_labels = [
        f"Command only\n{n_cmd_only/total*100:.1f}%",
        f"Both signals\n{n_both/total*100:.1f}%",
        f"Question only\n{n_q_only/total*100:.1f}%",
        f"Filtered out\n{n_filtered/total*100:.1f}%",
    ]

    wedges, texts = ax2.pie(sizes, colors=colors, explode=explode,
                             startangle=90, counterclock=False,
                             wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5})

    # Legend instead of wedge labels (cleaner)
    ax2.legend(wedges, wedge_labels, loc="center left", bbox_to_anchor=(0.85, 0.5),
               fontsize=8.5, framealpha=0.9)

    # Centre annotation
    ax2.text(0, 0, f"1,500,000\npairs\nin", ha="center", va="center",
             fontsize=9, color=DARK, fontweight="bold")

    ax2.set_title("Signal overlap breakdown\n(train split)", pad=8)

    fig.tight_layout(pad=2.0)
    _save(fig, "fig_p4_domain_filter")


# ─────────────────────────────────────────────────────────────────────────────
# fig_p5 — Token length distributions (context + response, two panels)
# ─────────────────────────────────────────────────────────────────────────────
def plot_length_distributions():
    # Context percentile data (from Appendix G / dataset_stats.py)
    ctx_percentiles = [0, 25, 50, 75, 90, 95, 99, 100]  # approximate %ile labels
    ctx_train = [2,   38,  63,  82,  95,  98, 100, 100]
    ctx_val   = [2,   42,  66,  84,  96,  99, 100, 100]
    ctx_test  = [2,   46,  67,  88,  98, 100, 100, 100]

    # Response percentile data
    resp_percentiles = [0, 10, 25, 50, 75, 90, 99, 100]
    resp_train = [2,   6,   9,  14,  21,  29,  40,  42]
    resp_val   = [2,   6,  10,  15,  22,  30,  40,  42]
    resp_test  = [2,   6,  10,  15,  23,  31,  40,  42]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Context lengths ────────────────────────────────────────────────────────
    pct_labels = ["min", "p25", "p50\n(median)", "p75", "p90", "p95", "p99", "max (cap)"]
    x = np.arange(len(pct_labels))
    w = 0.26

    ax1.bar(x - w,   ctx_train, width=w, color=BLUE,   alpha=0.85, label="Train")
    ax1.bar(x,       ctx_val,   width=w, color=ORANGE, alpha=0.85, label="Val")
    ax1.bar(x + w,   ctx_test,  width=w, color=RED,    alpha=0.85, label="Test")

    ax1.axhline(100, color=GREY, lw=1.5, ls="--", label="Cap (100 tokens)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pct_labels, fontsize=8.5)
    ax1.set_ylabel("Tokens")
    ax1.set_title("Context Token Length Distribution", pad=8)
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=8.5)
    ax1.grid(axis="y", ls=":", alpha=0.4)

    # Annotation: means
    ax1.annotate("mean train = 58.6\nmean val = 59.4\nmean test = 67.9",
                 xy=(0.67, 0.22), xycoords="axes fraction",
                 fontsize=8, color=DARK,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=GREY))

    # ── Response lengths ───────────────────────────────────────────────────────
    resp_labels = ["min", "p10", "p25", "p50\n(median)", "p75", "p90", "p99", "max (cap)"]
    x2 = np.arange(len(resp_labels))

    ax2.bar(x2 - w,   resp_train, width=w, color=BLUE,   alpha=0.85, label="Train")
    ax2.bar(x2,       resp_val,   width=w, color=ORANGE, alpha=0.85, label="Val")
    ax2.bar(x2 + w,   resp_test,  width=w, color=RED,    alpha=0.85, label="Test")

    ax2.axhline(40, color=GREY, lw=1.5, ls="--", label="Cap (40 tokens)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(resp_labels, fontsize=8.5)
    ax2.set_ylabel("Tokens")
    ax2.set_title("Response Token Length Distribution", pad=8)
    ax2.set_ylim(0, 50)
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    ax2.annotate("mean train = 16.2  |  mean val = 17.4  |  mean test = 17.8\n"
                 "Only ~2.5% of responses reach the 40-token cap",
                 xy=(0.02, 0.88), xycoords="axes fraction",
                 fontsize=8, color=DARK,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=GREY))

    fig.suptitle("Phase 1 — Stage 6 Token Length Distributions", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig_p5_token_lengths")


# ─────────────────────────────────────────────────────────────────────────────
# fig_p6 — Temporal split timeline
# ─────────────────────────────────────────────────────────────────────────────
def plot_temporal_split():
    fig, ax = plt.subplots(figsize=(11, 3.6))

    # Use VISUAL widths (not proportional) so val/test are readable.
    # Actual proportions are shown in the text labels.
    # train=0.68, val=0.16, test=0.16  (enlarged for legibility; marked "not to scale")
    splits = [
        ("TRAIN",  0.00, 0.68, BLUE,   "1,259,711 dialogues\n95.8%  |  2004 – Apr 2012"),
        ("VAL",    0.68, 0.84, ORANGE, "27,550 dialogues\n2.1%  |  Apr – Aug 2012"),
        ("TEST",   0.84, 1.00, RED,    "27,847 dialogues\n2.1%  |  Aug 2012+"),
    ]

    bar_y, bar_h = 0.45, 0.42

    for name, x0, x1, col, detail in splits:
        ax.barh(bar_y, x1 - x0, left=x0, height=bar_h,
                color=col, alpha=0.85, edgecolor="white", linewidth=2.5)
        cx = (x0 + x1) / 2
        ax.text(cx, bar_y + 0.05, name,
                ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(cx, bar_y - 0.22, detail,
                ha="center", va="top", fontsize=8.5, color=DARK, linespacing=1.5)

    # Divider lines at exact split boundaries
    for xd, label in [(0.68, "2012-04-27\n(val start)"), (0.84, "2012-08-07\n(test start)")]:
        ax.axvline(xd, color=DARK, lw=1.6, ls="--", alpha=0.65, ymin=0.25, ymax=0.85)
        ax.text(xd, bar_y + bar_h / 2 + 0.10, label,
                ha="center", va="bottom", fontsize=8, color=DARK, style="italic")

    # Corpus start label
    ax.text(0.00, bar_y + bar_h / 2 + 0.10, "Corpus start\n2004",
            ha="left", va="bottom", fontsize=8, color=DARK, style="italic")

    # Zero-overlap badge
    ax.text(0.5, 0.07,
            "✓  Zero thread-level overlap between splits  (verified by stage3_stats.json)",
            ha="center", va="bottom", fontsize=9, color=GREEN,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9",
                      edgecolor=GREEN, linewidth=1.2))

    # Not-to-scale note
    ax.text(0.99, 0.97, "* Val/Test bands enlarged for legibility (not to scale)",
            ha="right", va="top", fontsize=7.5, color=GREY, style="italic",
            transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Stage 3 — Temporal Train / Val / Test Split", pad=10,
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_p6_temporal_split")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Phase 1 report figures...")
    plot_funnel()
    plot_stage2_filters()
    plot_stage4_filters()
    plot_domain_filter()
    plot_length_distributions()
    plot_temporal_split()
    print(f"\nAll figures saved to {FIG_DIR}")
