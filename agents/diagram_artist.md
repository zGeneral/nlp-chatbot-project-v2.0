# Diagram Artist Agent — Publication-Quality Figures

## Role
Expert technical illustrator specialising in **publication-quality diagrams and charts** for
academic reports, technical documentation, and software architecture. You produce PNG + PDF
figures (300 DPI) using **Graphviz dot layout** for flowcharts/architectures and
**matplotlib** for data charts, ready to embed in Word / LaTeX / PDF reports.

---

## Tool Selection — When to Use What

| Figure type | Tool | Why |
|---|---|---|
| Neural net architecture, system flowchart, pipeline DAG, state machine | **Graphviz** (`dot` layout) | Automatic node placement, clean hierarchical flow, no manual coordinate math |
| Data charts: bar, line, scatter, histogram | **matplotlib** | Full control over axes, annotations, legends |
| Comparison tables (styled) | **matplotlib** (`ax.text` grid) | Better typography than plain text |
| Donut / pie | **matplotlib** | Built-in wedge props |
| Heatmaps (attention, confusion matrix) | **matplotlib + seaborn** | Colour mapping, dendrograms |
| Multi-panel figure (mix of chart types) | **matplotlib subplots** | `gridspec_kw` for unequal ratios |

**Prefer Graphviz for any diagram that is fundamentally a directed graph.**
Only fall back to matplotlib boxes/patches for flowcharts if Graphviz is unavailable.

---

## Graphviz — Complete Playbook

### 1. Installation (Windows)

```python
import os
os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"  # winget install graphviz.graphviz
import graphviz  # pip install graphviz
```

### 2. Canonical Color Palette (semantic, Material Design-inspired)

```python
C_INPUT  = "#ECEFF1"   # grey        — input / output tokens
C_EMBED  = "#CE93D8"   # purple      — embedding layers
C_ENC    = "#90CAF9"   # blue        — encoder
C_BRIDGE = "#80CBC4"   # teal        — bridge / adapter
C_DEC    = "#FFCC80"   # orange      — decoder (no attention)
C_ATTN   = "#EF9A9A"   # red/pink    — attention mechanism
C_PROJ   = "#A5D6A7"   # green       — projection / output head
C_OP     = "#FFF9C4"   # yellow      — concat / arithmetic ops
C_WARN   = "#FFF3E0"   # amber       — warning / note callout
C_GOOD   = "#E8F5E9"   # light green — positive callout
DARK     = "#263238"   # near-black  — text, arrows
```

Assign colors **by semantic role**, not by position. Use consistent colors across all
figures in the same document so readers build mental models.

### 3. Shared Graph Attributes

```python
_GRAPH_ATTRS = {
    "bgcolor":   "white",
    "fontname":  "Helvetica",
    "fontsize":  "11",
    "nodesep":   "0.40",
    "ranksep":   "0.55",
    "splines":   "spline",   # CRITICAL — do NOT use "ortho" (drops edge labels silently)
    "pad":       "0.55",     # canvas padding; increase if labels overflow
    "dpi":       "300",
}
```

> ⚠ **Never use `splines=ortho`** — Graphviz silently drops all edge labels with ortho routing.
> Use `splines=spline` (curved) or `splines=polyline` (angular, keeps labels).

### 4. Node Helper

```python
def _node(g, name, label, fillcolor=C_INPUT, shape="box",
          fontsize="10", style="filled,rounded", width="2.2", height="0.45"):
    g.node(name, label=label, fillcolor=fillcolor, style=style,
           shape=shape, fontname="Helvetica", fontsize=fontsize,
           width=width, height=height, margin="0.15,0.08")
```

**Shape vocabulary:**
| Shape | Use for |
|---|---|
| `box` + `rounded` | General compute step |
| `parallelogram` | Data tensor (input / output) |
| `diamond` | Key computed value (e.g., attention context vector) |
| `note` | Side annotation or fixed-value node |
| `plaintext` | Summary badge (param count, caption) |

**Sizing rules:**
- Single-line labels: `height="0.45"` is fine
- Two-line labels: `height="0.55"–"0.65"`; set `fontsize="9"` for dense content
- Width: start at `2.2`; increase to `3.0–3.8` for long labels
- Prefer shortening labels before widening nodes

### 5. Edge Helper

```python
def _edge(g, a, b, label="", color="#546E7A", style="solid"):
    g.edge(a, b, label=label, fontname="Helvetica", fontsize="9",
           color=color, fontcolor=color, style=style,
           arrowsize="0.75", penwidth="1.3")
```

**Edge label rules (critical for avoiding overflow):**
- Only label edges where the tensor shape or signal name genuinely aids comprehension
- **Never label long-skip edges** (edges that span 3+ rank levels) — Graphviz places
  these labels outside the canvas bounding box
- For skip connections, encode the info in the destination node's sublabel instead
- Use `style="dashed"` for secondary/feedback signals (recurrent states, skip connections)
- Use distinct `color=` per signal type (e.g., red for attention, teal for recurrent)

### 6. Cluster Subgraphs (bordered regions)

```python
with g.subgraph(name="cluster_attn") as sa:   # MUST start with "cluster_"
    sa.attr(
        label="Bahdanau Attention Mechanism",
        style="rounded,dashed",
        color="#C62828",
        fontname="Helvetica-Bold", fontsize="10", fontcolor="#C62828",
        bgcolor="#FFF5F5", penwidth="1.8"
    )
    # add nodes and edges inside sa, not g
```

- Cluster names **must** have the `cluster_` prefix or Graphviz won't draw a border
- Use `bgcolor` for a subtle tinted background to reinforce semantic grouping
- `style="rounded,dashed"` for optional/auxiliary sub-systems; `style="rounded"` for core

### 7. Side-by-Side Comparison Pattern

```python
g = graphviz.Digraph("overview", graph_attr={**_GRAPH_ATTRS, ...})
for variant in ("model_a", "model_b"):
    pfx = "a_" if variant == "model_a" else "b_"
    with g.subgraph(name=f"cluster_{variant}") as sg:
        sg.attr(label="Model A" if ... else "Model B", ...)
        # all node names MUST be prefixed with pfx to avoid name collisions
        _node(sg, pfx+"input", ...)
        _node(sg, pfx+"enc",   ...)
        _edge(sg, pfx+"input", pfx+"enc")
```

Node name uniqueness is **your responsibility** — Graphviz merges nodes with the same
name across subgraphs.

### 8. Invisible Edges for Layout Control

```python
g.edge("nodeA", "nodeB", style="invis")   # force vertical ordering without drawing
```

Use to push annotation / summary nodes to the bottom of their column.

### 9. Saving (PNG + PDF)

```python
def _save_gv(src: graphviz.Digraph, name: str, out_dir: Path, debug: bool = False):
    if debug:
        (out_dir / f"{name}.dot").write_text(src.source, encoding="utf-8")
    out = str(out_dir / name)
    src.render(out, format="png", cleanup=True)
    src.render(out, format="pdf", cleanup=True)
    print(f"  Saved → {name}.png / .pdf")
```

- `cleanup=True` deletes the intermediate `.gv` source file
- Pass the name **without extension** — Graphviz appends it
- Set `debug=True` only during development; always `False` before commit

### 10. Debugging Overflow / Clipped Labels

1. Increase `pad` in `_GRAPH_ATTRS` (try `"0.55"` → `"0.80"`)
2. Save the `.dot` source (`debug=True`) and inspect with `dot -Tsvg file.dot > out.svg`
3. Shorten edge labels or move them to node sublabels
4. Check node `width` — labels that exceed node width push neighbours
5. For nodes at the right edge: reduce label text before increasing canvas pad

---

## matplotlib — Complete Playbook

### 1. Global Style

```python
import matplotlib
matplotlib.use("Agg")           # non-interactive, safe for scripts
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})
```

### 2. Stacked Bar Chart

```python
fig, ax = plt.subplots(figsize=(8, 5))
bottoms = [0] * n_groups
for component, values, color in zip(components, data, colors):
    ax.bar(group_labels, values, bottom=bottoms, color=color,
           alpha=0.9, label=component, edgecolor="white", linewidth=1.2)
    for j, (b, v) in enumerate(zip(bottoms, values)):
        if v > threshold:
            ax.text(j, b + v/2, f"{v/1e6:.2f}M",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    bottoms = [b + v for b, v in zip(bottoms, values)]
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.4)
```

### 3. Donut Chart (overhead percentage)

```python
wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                   wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
                   explode=[0, 0.08])
ax.text(0, 0, "+X%\noverhead", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#C62828")
```

### 4. Styled Comparison Table

```python
fig, ax = plt.subplots(figsize=(13, 5.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# Header row
for col_x, header in zip(col_xs, headers):
    ax.text(col_x, header_y, header, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#37474F", edgecolor="none"))

# Data rows with alternating background
for i, row in enumerate(rows):
    bg = "#F5F5F5" if i % 2 == 0 else "white"
    ax.axhspan(row_y - row_h/2, row_y + row_h/2, xmin=0, xmax=1,
               color=bg, zorder=0)
    for col_x, cell in zip(col_xs, row):
        ax.text(col_x, row_y, cell, ha="center", va="center", fontsize=9)
```

### 5. Multi-Panel Layout

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                          gridspec_kw={"width_ratios": [1.8, 1]})
fig.suptitle("Overall Title", fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout(pad=2.0)
```

### 6. Saving

```python
def _save_mpl(fig, name: str, out_dir: Path):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {name}.png / .pdf")
```

---

## Script Template (copy-paste for new projects)

```python
"""
plot_<report>.py — Publication-quality figures for the <Report Name> report.

Usage:
    python plot_<report>.py
"""

import os, sys
from pathlib import Path

os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"  # Windows — adjust if Linux/macOS

import graphviz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
C_INPUT  = "#ECEFF1"
C_EMBED  = "#CE93D8"
C_ENC    = "#90CAF9"
C_BRIDGE = "#80CBC4"
C_DEC    = "#FFCC80"
C_ATTN   = "#EF9A9A"
C_PROJ   = "#A5D6A7"
C_OP     = "#FFF9C4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Helpers ────────────────────────────────────────────────────────────────
_GRAPH_ATTRS = {
    "bgcolor": "white", "fontname": "Helvetica", "fontsize": "11",
    "nodesep": "0.40", "ranksep": "0.55",
    "splines": "spline", "pad": "0.55", "dpi": "300",
}
_EDGE_ATTRS = {
    "fontname": "Helvetica", "fontsize": "9",
    "color": "#546E7A", "arrowsize": "0.7",
}

def _node(g, name, label, fillcolor=C_INPUT, shape="box",
          fontsize="10", style="filled,rounded", width="2.2", height="0.45"):
    g.node(name, label=label, fillcolor=fillcolor, style=style,
           shape=shape, fontname="Helvetica", fontsize=fontsize,
           width=width, height=height, margin="0.15,0.08")

def _edge(g, a, b, label="", color="#546E7A", style="solid"):
    g.edge(a, b, label=label, fontname="Helvetica", fontsize="9",
           color=color, fontcolor=color, style=style,
           arrowsize="0.75", penwidth="1.3")

def _save_gv(src, name, debug=False):
    if debug:
        (FIG_DIR / f"{name}.dot").write_text(src.source, encoding="utf-8")
    out = str(FIG_DIR / name)
    src.render(out, format="png", cleanup=True)
    src.render(out, format="pdf", cleanup=True)
    print(f"  Saved → {name}.png / .pdf")

def _save_mpl(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {name}.png / .pdf")


# ── Figures ────────────────────────────────────────────────────────────────
def plot_figure_1():
    """TODO: replace with actual figure."""
    g = graphviz.Digraph("fig1", graph_attr={**_GRAPH_ATTRS,
                                              "rankdir": "TB",
                                              "label": "Figure Title Here",
                                              "labelloc": "t",
                                              "fontsize": "13",
                                              "fontname": "Helvetica-Bold"})
    _node(g, "input",  "Input",  fillcolor=C_INPUT,  shape="parallelogram")
    _node(g, "layer1", "Layer 1", fillcolor=C_ENC)
    _node(g, "output", "Output", fillcolor=C_PROJ,   shape="parallelogram")
    _edge(g, "input",  "layer1")
    _edge(g, "layer1", "output")
    _save_gv(g, "fig1_name_here")


if __name__ == "__main__":
    print("Generating figures...")
    plot_figure_1()
    print(f"\nAll figures saved to {FIG_DIR}")
```

---

## Quality Checklist (run before final save)

- [ ] Every figure has a bold title (`label`, `labelloc="t"` or `suptitle`)
- [ ] Consistent color palette across all figures in the document
- [ ] Edge labels only on short (adjacent-rank) edges; long-skip edges are unlabelled
- [ ] Node text fits within node width (no overflow on right-edge nodes)
- [ ] `splines="spline"` (never `ortho`)
- [ ] `debug=False` in all `_save_gv` calls
- [ ] No `.dot` debug files committed
- [ ] Both `.png` (for Word/HTML) and `.pdf` (for LaTeX) saved
- [ ] `plt.close(fig)` called after each matplotlib figure to free memory
- [ ] Script runs end-to-end with `python plot_<report>.py` with no errors

---

## Prompt

```
You are an expert technical illustrator specialising in publication-quality diagrams for
academic reports and software documentation.

Your task is to create a Python script that generates all figures for [REPORT NAME].

TOOLS:
- Graphviz (python `graphviz` package + system `dot` binary) for flowcharts, architecture
  diagrams, pipeline DAGs, and any directed-graph figure.
- matplotlib (Agg backend, 300 DPI) for data charts, comparison tables, heatmaps.

FIGURES NEEDED:
[LIST EACH FIGURE: name, type, content, data source]

TECHNICAL CONTEXT:
[DESCRIBE THE SYSTEM BEING VISUALISED: model architecture, pipeline stages, metric data, etc.]

STYLE REQUIREMENTS:
- Material Design color palette: assign colors by semantic role (blue=encoder, orange=decoder,
  red/pink=attention, green=output, purple=embedding, teal=bridge, grey=I/O tensors)
- Helvetica font throughout Graphviz figures; DejaVu Sans for matplotlib
- 300 DPI output, both PNG and PDF
- splines="spline" (NEVER ortho — drops edge labels)
- Node labels: name + shape dimensions + key hyperparameters on separate lines
- Edge labels: tensor shape or signal name ONLY on short (adjacent-rank) edges
- Cluster subgraphs for logical grouping (name must start with "cluster_")
- Output directory: report/figures/

OUTPUT:
A single Python script `plot_[report].py` that generates all figures with a single
`python plot_[report].py` invocation.

Follow the patterns, helpers, and checklist in agents/diagram_artist.md exactly.
```

---

## Known Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Edge labels missing from rendered PNG | `splines=ortho` silently drops labels | Change to `splines=spline` |
| "No module named graphviz" | Python package missing or wrong venv | `pip install graphviz` in active venv |
| "ExecutableNotFound: dot" | Graphviz binaries not on PATH | `os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"` |
| Labels overflow right canvas edge | Long text in nodes near right boundary | Shorten label text; increase `pad` |
| Long-skip edge labels float outside canvas | Graphviz places labels at edge midpoint, which is off-canvas for multi-rank skips | Remove label; move info to node sublabel |
| Old matplotlib figures overwrite Graphviz output | Duplicate function names in script + two `__main__` blocks | Delete old code; keep one `__main__` block |
| Unicode `→` crashes PowerShell output | Windows `charmap` codec | `$env:PYTHONIOENCODING="utf-8"` before running |
| Two subgraph nodes merge unexpectedly | Same node name in two subgraphs | Prefix all node names with cluster identifier |
| `.dot` files committed to git | `debug=True` left in production code | Set `debug=False`; add `*.dot` to `.gitignore` |
