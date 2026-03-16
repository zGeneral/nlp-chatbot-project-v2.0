"""
render_to_png.py
Renders .excalidraw files to PNG using matplotlib.
Handles: rectangles, text, arrows (including multi-point).
Usage: python report/excalidraw/render_to_png.py
"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import withStroke
import matplotlib.patheffects as pe
import numpy as np

_THIS_DIR = Path(__file__).parent                  # report/excalidraw/
OUT_DIR   = _THIS_DIR.parent / "figures"           # report/figures/
SRC_DIR   = _THIS_DIR                              # report/excalidraw/
DPI       = 150
PAD       = 40          # canvas padding in excalidraw px

FILES = [
    "fig_tm1_training_loop.excalidraw",
    "fig_tm2_tf_phases.excalidraw",
    "fig_tm3_grad_accum.excalidraw",
    "fig_tm4_early_stopping.excalidraw",
]


def hex_to_rgb(h, opacity=100):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
    return (r, g, b, opacity/100)


def blend_on_white(color_rgba):
    """Pre-multiply alpha over white background."""
    r, g, b, a = color_rgba
    return (r*a + 1*(1-a), g*a + 1*(1-a), b*a + 1*(1-a), 1.0)


def render(src_path: Path, out_path: Path):
    data = json.loads(src_path.read_text(encoding="utf-8"))
    els  = data["elements"]

    # --- bounding box of all elements ---
    xs, ys, xe, ye = [], [], [], []
    for e in els:
        if "x" in e and "y" in e:
            xs.append(e["x"]); ys.append(e["y"])
            xe.append(e["x"] + e.get("width",  0))
            ye.append(e["y"] + e.get("height", 0))
        if e["type"] == "arrow":
            for p in e.get("points", []):
                xs.append(e["x"] + p[0]); ys.append(e["y"] + p[1])

    x_min, y_min = min(xs)-PAD, min(ys)-PAD
    x_max, y_max = max(xe)+PAD, max(ye)+PAD
    W = x_max - x_min
    H = y_max - y_min

    fig_w = W / DPI * 1.0
    fig_h = H / DPI * 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)   # invert y so 0 is top
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Sort: rectangles first (z-order by array index)
    def z_order(e):
        t = e["type"]
        return 0 if t == "rectangle" else (1 if t == "text" else 2)

    for e in sorted(els, key=z_order):
        if e.get("isDeleted"):
            continue
        etype = e["type"]
        op    = e.get("opacity", 100)

        # ── RECTANGLE ─────────────────────────────────────────────────────
        if etype == "rectangle":
            x, y, w, h  = e["x"], e["y"], e["width"], e["height"]
            fill_hex     = e.get("backgroundColor", "#FFFFFF")
            stroke_hex   = e.get("strokeColor",     "#07182D")
            sw           = e.get("strokeWidth", 1) * 0.6
            dashed       = e.get("strokeStyle") == "dashed"

            if fill_hex in ("transparent", "none"):
                face = (1,1,1,0)
            else:
                face = blend_on_white(hex_to_rgb(fill_hex, op))

            edge = hex_to_rgb(stroke_hex, 100)[:3]

            ls = (0, (4, 3)) if dashed else "solid"
            roundness = 8 if e.get("roundness") else 0
            style = f"round,pad=0,rounding_size={roundness}"

            rect = FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={roundness}",
                facecolor=face, edgecolor=edge,
                linewidth=sw, linestyle=ls, zorder=1
            )
            ax.add_patch(rect)

        # ── TEXT ──────────────────────────────────────────────────────────
        elif etype == "text":
            tx, ty  = e["x"], e["y"]
            text    = e.get("text", "")
            fs      = e.get("fontSize", 14)
            color   = e.get("strokeColor", "#07182D")
            ff      = e.get("fontFamily", 2)

            # fontFamily 2 = sans-serif, 3 = monospace
            family  = "monospace" if ff == 3 else "DejaVu Sans"
            # Convert excalidraw px fontSize to matplotlib points (approx)
            pt_size = fs * 0.72

            rgb = hex_to_rgb(color, 100)[:3]
            ax.text(
                tx, ty, text,
                fontsize=pt_size, color=rgb,
                family=family, va="top", ha="left",
                zorder=3, wrap=False,
                linespacing=1.25,
            )

        # ── ARROW ─────────────────────────────────────────────────────────
        elif etype == "arrow":
            pts      = e.get("points", [[0,0],[0,10]])
            ox, oy   = e["x"], e["y"]
            stroke   = e.get("strokeColor", "#07182D")
            sw       = e.get("strokeWidth", 1) * 0.7
            dashed   = e.get("strokeStyle") == "dashed"
            end_head = e.get("endArrowhead", "arrow")

            abs_pts = [(ox + p[0], oy + p[1]) for p in pts]
            color   = hex_to_rgb(stroke, 100)[:3]
            ls      = "--" if dashed else "-"

            if len(abs_pts) == 2:
                (x0, y0), (x1, y1) = abs_pts
                dx, dy = x1-x0, y1-y0
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->" if end_head else "-",
                        color=color, lw=sw,
                        linestyle=ls,
                        mutation_scale=10,
                    ), zorder=2
                )
            else:
                # multi-segment polyline + arrowhead at end
                xs2 = [p[0] for p in abs_pts]
                ys2 = [p[1] for p in abs_pts]
                ax.plot(xs2, ys2, color=color, lw=sw, ls=ls, zorder=2)
                # draw arrowhead at last segment
                if end_head:
                    x0r, y0r = abs_pts[-2]
                    x1r, y1r = abs_pts[-1]
                    ax.annotate(
                        "", xy=(x1r, y1r), xytext=(x0r, y0r),
                        arrowprops=dict(
                            arrowstyle="->", color=color, lw=sw,
                            mutation_scale=10,
                        ), zorder=2
                    )

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> {out_path.name}  ({int(W)}x{int(H)} px canvas)")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else FILES
    for fname in targets:
        src = SRC_DIR / fname
        out = OUT_DIR / (Path(fname).stem + ".png")
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        print(f"Rendering {fname} ...")
        render(src, out)

print("\nDone.")
