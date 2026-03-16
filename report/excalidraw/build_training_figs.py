"""
build_training_figs.py
Generates 4 Excalidraw training methodology diagrams for the NLP chatbot report.
"""
import json
import math
from pathlib import Path

OUT_DIR = Path(__file__).parent   # report/excalidraw/

# ─────────────────────── Cisco Brand Palette ────────────────────────
C_ENC    = "#02C8FF"
C_DEC    = "#0A60FF"
C_RED    = "#EB4651"
C_BRIDGE = "#B4B9C0"
C_WHITE  = "#FFFFFF"
C_LGREY  = "#F0F0F0"
C_MGREY  = "#E8E8E8"

S_ENC    = "#02C8FF"
S_DEC    = "#0A60FF"
S_RED    = "#EB4651"
S_BRIDGE = "#6B6B6B"
S_DARK   = "#07182D"
S_GREY   = "#6B6B6B"

FONT = 2

# ─────────────────────── ID / seed generator ──────────────────────────
_counter = [1000]

def nxt():
    _counter[0] += 1
    return _counter[0]

def _eid():
    _counter[0] += 1
    return f"el{_counter[0]}"

# ─────────────────────── Size helpers ─────────────────────────────────
def tw(text, fs):
    return math.ceil(max(len(l) for l in text.split('\n')) * 0.55 * fs)

def th(text, fs):
    return math.ceil(len(text.split('\n')) * fs * 1.25)

# ─────────────────────── Element builders ─────────────────────────────
def txt(eid, x, y, text, fs, color="#07182D"):
    return {
        "type": "text", "id": eid, "x": x, "y": y,
        "width": tw(text, fs), "height": th(text, fs),
        "angle": 0, "seed": nxt(), "version": 1, "versionNonce": nxt(),
        "isDeleted": False, "groupIds": [], "frameId": None, "boundElements": [],
        "updated": 1700000000000, "link": None, "locked": False,
        "strokeColor": color, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "text": text, "fontSize": fs, "fontFamily": 2,
        "textAlign": "left", "verticalAlign": "top",
        "containerId": None, "originalText": text, "autoResize": True, "lineHeight": 1.25,
    }

def rect(eid, x, y, w, h, fill="#FFFFFF", stroke="#07182D", sw=1, opacity=100):
    return {
        "type": "rectangle", "id": eid, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "seed": nxt(), "version": 1, "versionNonce": nxt(),
        "isDeleted": False, "groupIds": [], "frameId": None, "boundElements": [],
        "updated": 1700000000000, "link": None, "locked": False,
        "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid",
        "strokeWidth": sw, "strokeStyle": "solid",
        "roughness": 0, "opacity": opacity, "roundness": {"type": 3},
    }

def arrow(eid, x, y, pts, stroke="#07182D", sw=1, dashed=False):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    w = max(abs(max(xs) - min(xs)), 1); h = max(abs(max(ys) - min(ys)), 1)
    return {
        "type": "arrow", "id": eid, "x": x, "y": y, "width": w, "height": h,
        "points": pts, "angle": 0, "seed": nxt(), "version": 1, "versionNonce": nxt(),
        "isDeleted": False, "groupIds": [], "frameId": None, "boundElements": [],
        "updated": 1700000000000, "link": None, "locked": False,
        "strokeColor": stroke, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": sw, "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 0, "opacity": 100,
        "endArrowhead": "arrow", "startArrowhead": None, "elbowed": False,
        "startBinding": None, "endBinding": None,
    }

def save(name, elements):
    xs = [e["x"] for e in elements if "x" in e]
    ys = [e["y"] for e in elements if "y" in e]
    sx = -(min(xs) - 40) if xs else 0
    sy = -(min(ys) - 40) if ys else 0
    doc = {
        "type": "excalidraw", "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff", "gridSize": None,
            "zoom": {"value": 1.0}, "scrollX": sx, "scrollY": sy,
        },
        "files": {},
    }
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"Saved {path} ({len(elements)} elements)")


# ══════════════════════════════════════════════════════════════════════
# Diagram 1 — fig_tm1_training_loop.excalidraw
# ══════════════════════════════════════════════════════════════════════
def build_tm1():
    els = []

    # Title
    els.append(txt(_eid(), 120, 20, "Training Loop — Epoch 1 → 20", 20))

    # EPOCH ZONE
    epoch_zone_h = 900
    els.append(rect(_eid(), 40, 65, 580, epoch_zone_h, fill=C_ENC, stroke=S_ENC, sw=1, opacity=20))
    els.append(txt(_eid(), 55, 75, "EPOCH LOOP  (1 → 20)", 14, color=S_ENC))

    # Pre-batch boxes
    pre_boxes = [
        (105, "tf_ratio ← get_tf_ratio(epoch)"),
        (163, "optimizer.zero_grad()"),
    ]
    for y, label in pre_boxes:
        els.append(rect(_eid(), 80, y, 500, 48, fill=C_WHITE, stroke=S_DARK))
        els.append(txt(_eid(), 90, y + 12, label, 13))

    # Arrows between pre-batch boxes
    els.append(arrow(_eid(), 330, 153, [[0, 0], [0, 10]], stroke=S_DARK))

    # BATCH ZONE
    els.append(rect(_eid(), 60, 230, 540, 390, fill=C_DEC, stroke=S_DEC, sw=1, opacity=20))
    els.append(txt(_eid(), 75, 240, "BATCH LOOP", 14, color=S_DEC))

    # Batch step boxes
    batch_boxes = [
        (268, "with autocast(cuda, bf16):  logits = model(src, src_len, trg, tf_ratio)", C_WHITE, S_DARK, 1, 100),
        (326, "loss = CrossEntropyLoss(logits, trg[:,1:])   ← pad masked", C_WHITE, S_DARK, 1, 100),
        (384, "if loss is NaN → zero_grad, skip batch", C_RED, S_RED, 1, 35),
        (442, "(loss / accum_steps).backward()   ← scale gradient", C_WHITE, S_DARK, 1, 100),
        (500, "if step%2==0: clip_grad_norm_ + optimizer.step() + zero_grad", C_WHITE, S_DARK, 1, 100),
        (558, "if global_step % 2000 == 0 → save step checkpoint", C_WHITE, S_DARK, 1, 100),
    ]
    for y, label, fill, stroke, sw, opacity in batch_boxes:
        els.append(rect(_eid(), 100, y, 460, 48, fill=fill, stroke=stroke, sw=sw, opacity=opacity))
        els.append(txt(_eid(), 110, y + 12, label, 12))

    # Arrows within batch zone
    for y_from in [316, 374, 432, 490, 548]:
        els.append(arrow(_eid(), 330, y_from, [[0, 0], [0, 10]], stroke=S_DARK))

    # Arrow from pre-batch to batch zone
    els.append(arrow(_eid(), 330, 211, [[0, 0], [0, 19]], stroke=S_DARK))

    # Post-batch boxes
    post_boxes = [
        (640, "val_loss ← evaluate(model, val_loader, tf=0.0)   ← no TF in val", C_WHITE, S_DARK, 1, 100),
        (698, "scheduler.step(val_loss)   ← may halve LR", C_WHITE, S_DARK, 1, 100),
    ]
    for y, label, fill, stroke, sw, opacity in post_boxes:
        els.append(rect(_eid(), 80, y, 500, 48, fill=fill, stroke=stroke, sw=sw, opacity=opacity))
        els.append(txt(_eid(), 90, y + 12, label, 12))

    # Arrow from batch zone to post-batch
    els.append(arrow(_eid(), 330, 620, [[0, 0], [0, 20]], stroke=S_DARK))
    els.append(arrow(_eid(), 330, 688, [[0, 0], [0, 10]], stroke=S_DARK))

    # Phase-reset box
    els.append(rect(_eid(), 80, 756, 500, 48, fill=C_DEC, stroke=S_DEC, sw=1, opacity=35))
    els.append(txt(_eid(), 90, 768, "if epoch == 6: reset best_val_loss, no_improve   ← Phase 2 reset", 12))
    els.append(arrow(_eid(), 330, 746, [[0, 0], [0, 10]], stroke=S_DARK))

    # Final best/early-stop box
    els.append(rect(_eid(), 80, 814, 500, 48, fill=C_WHITE, stroke=S_DARK))
    els.append(txt(_eid(), 90, 826, "if val_loss < best → save best.pt   else: no_improve+1 → early stop", 12))
    els.append(arrow(_eid(), 330, 804, [[0, 0], [0, 10]], stroke=S_DARK))

    # Last epoch box
    els.append(rect(_eid(), 80, 872, 500, 48, fill=C_LGREY, stroke=S_GREY))
    els.append(txt(_eid(), 90, 884, "save last.pt   (always)", 12))
    els.append(arrow(_eid(), 330, 862, [[0, 0], [0, 10]], stroke=S_DARK))

    save("fig_tm1_training_loop.excalidraw", els)


# ══════════════════════════════════════════════════════════════════════
# Diagram 2 — fig_tm2_tf_phases.excalidraw
# ══════════════════════════════════════════════════════════════════════
def build_tm2():
    els = []

    # Title
    els.append(txt(_eid(), 100, 20, "Teacher Forcing — Three-Phase Schedule", 20))

    # Phase 1 zone
    els.append(rect(_eid(), 80, 70, 200, 300, fill=C_ENC, stroke=S_ENC, opacity=25))
    els.append(txt(_eid(), 110, 110, "PHASE 1\nEpochs 1–5\nTF = 1.00\n(fixed)", 14, color=S_ENC))

    # Phase 2 zone
    els.append(rect(_eid(), 280, 70, 280, 300, fill=C_DEC, stroke=S_DEC, opacity=25))
    els.append(txt(_eid(), 310, 110, "PHASE 2\nEpochs 6–12\nTF 0.90 → 0.50\n(linear decay)", 14, color=S_DEC))

    # Phase 3 zone
    els.append(rect(_eid(), 560, 70, 240, 300, fill=C_LGREY, stroke=S_GREY, opacity=100))
    els.append(txt(_eid(), 590, 110, "PHASE 3\nEpochs 13–20\nTF = 0.50\n(floor)", 14, color=S_GREY))

    # Epoch axis line
    els.append(arrow(_eid(), 80, 380, [[0, 0], [730, 0]], stroke=S_DARK))
    els.append(txt(_eid(), 790, 372, "Epoch", 14))

    # Epoch tick labels
    tick_labels = [
        (155, "1"), (255, "5"), (285, "6"), (545, "12"), (565, "13"), (785, "20"),
    ]
    for x, label in tick_labels:
        els.append(txt(_eid(), x, 385, label, 12))

    # TF y-axis labels
    els.append(txt(_eid(), 40, 75, "1.00", 13))
    els.append(txt(_eid(), 40, 155, "0.90", 13))
    els.append(txt(_eid(), 40, 355, "0.50", 13))

    # Decision note box
    els.append(rect(_eid(), 80, 430, 720, 52, fill=C_RED, stroke=S_RED, opacity=35))
    els.append(txt(_eid(), 90, 443,
        "Phase 2 reset at epoch 6: best_val_loss and no_improve counter reset to allow independent convergence window",
        13, color=S_DARK))

    save("fig_tm2_tf_phases.excalidraw", els)


# ══════════════════════════════════════════════════════════════════════
# Diagram 3 — fig_tm3_grad_accum.excalidraw
# ══════════════════════════════════════════════════════════════════════
def build_tm3():
    els = []

    # Title
    els.append(txt(_eid(), 80, 20, "Gradient Accumulation — Effective Batch = 512", 20))

    # Batch 0 row
    els.append(rect(_eid(), 80, 80, 120, 60, fill=C_ENC, stroke=S_ENC, opacity=35))
    els.append(txt(_eid(), 88, 96, "Batch 0\n[B=256]", 13, color=S_DARK))

    els.append(rect(_eid(), 250, 80, 200, 60, fill=C_WHITE, stroke=S_DARK))
    els.append(txt(_eid(), 258, 96, "Forward Pass\nlogits = model(...)", 13))

    els.append(rect(_eid(), 500, 80, 240, 60, fill=C_WHITE, stroke=S_DARK))
    els.append(txt(_eid(), 508, 96, "(loss / 2).backward()\ngradients accumulate", 13))

    # Arrows for Batch 0 row
    els.append(arrow(_eid(), 200, 110, [[0, 0], [50, 0]], stroke=S_DARK))
    els.append(arrow(_eid(), 450, 110, [[0, 0], [50, 0]], stroke=S_DARK))

    # Batch 1 row
    els.append(rect(_eid(), 80, 180, 120, 60, fill=C_DEC, stroke=S_DEC, opacity=35))
    els.append(txt(_eid(), 88, 196, "Batch 1\n[B=256]", 13, color=S_DARK))

    els.append(rect(_eid(), 250, 180, 200, 60, fill=C_WHITE, stroke=S_DARK))
    els.append(txt(_eid(), 258, 196, "Forward Pass\nlogits = model(...)", 13))

    els.append(rect(_eid(), 500, 180, 240, 60, fill=C_WHITE, stroke=S_DARK))
    els.append(txt(_eid(), 508, 196, "(loss / 2).backward()\ngradients accumulate", 13))

    # Arrows for Batch 1 row
    els.append(arrow(_eid(), 200, 210, [[0, 0], [50, 0]], stroke=S_DARK))
    els.append(arrow(_eid(), 450, 210, [[0, 0], [50, 0]], stroke=S_DARK))

    # Merge arrows from accumulate boxes to optimizer
    els.append(arrow(_eid(), 620, 140, [[0, 0], [0, 160]], stroke=S_DARK))
    els.append(arrow(_eid(), 620, 240, [[0, 0], [0, 60]], stroke=S_DARK))

    # Optimizer step box
    els.append(rect(_eid(), 400, 310, 360, 60, fill=C_RED, stroke=S_RED, sw=2, opacity=35))
    els.append(txt(_eid(), 410, 326, "clip_grad_norm_(1.0)  →  optimizer.step()  →  zero_grad()", 14, color=S_DARK))

    # Result annotation box
    els.append(rect(_eid(), 80, 400, 680, 50, fill=C_ENC, stroke=S_ENC, opacity=25))
    els.append(txt(_eid(), 90, 413,
        "Effective batch size = 256 × 2 = 512 samples   |   loss divided by accum_steps before backward",
        14, color=S_DARK))

    save("fig_tm3_grad_accum.excalidraw", els)


# ══════════════════════════════════════════════════════════════════════
# Diagram 4 — fig_tm4_early_stopping.excalidraw
# ══════════════════════════════════════════════════════════════════════
def build_tm4():
    els = []

    # Title
    els.append(txt(_eid(), 80, 20, "Early Stopping — Phase-Gated Counter", 20))

    # Phase 1 zone (disabled)
    els.append(rect(_eid(), 40, 70, 320, 200, fill=C_LGREY, stroke=S_GREY, opacity=100))
    els.append(txt(_eid(), 60, 90, "PHASE 1\nEpochs 1–5\nEarly stopping: DISABLED", 14, color=S_GREY))

    # Phase 2+ zone (active)
    els.append(rect(_eid(), 400, 70, 480, 360, fill=C_DEC, stroke=S_DEC, opacity=20))
    els.append(txt(_eid(), 420, 82, "PHASE 2+\nEpochs 6–20\nEarly stopping: ACTIVE", 14, color=S_DEC))

    # "val_loss improved" box (Cisco Blue)
    els.append(rect(_eid(), 420, 140, 200, 60, fill=C_ENC, stroke=S_ENC, opacity=35))
    els.append(txt(_eid(), 428, 156, "val_loss improved\n→ no_improve = 0", 13, color=S_DARK))

    # "val_loss worse" box (neutral)
    els.append(rect(_eid(), 660, 140, 200, 60, fill=C_WHITE, stroke=S_DARK))
    els.append(txt(_eid(), 668, 156, "val_loss worse\n→ no_improve += 1", 13, color=S_DARK))

    # "no_improve >= 4" box (Cisco Red — halt)
    els.append(rect(_eid(), 540, 280, 240, 60, fill=C_RED, stroke=S_RED, sw=2, opacity=35))
    els.append(txt(_eid(), 548, 296, "no_improve >= 4\n→ HALT  (best.pt saved)", 13, color=S_DARK))

    # Arrow: Phase 1 → Phase 2 zone
    els.append(arrow(_eid(), 360, 170, [[0, 0], [40, 0]], stroke=S_DARK))

    # Arrow: "val_loss worse" down-left to halt
    els.append(arrow(_eid(), 760, 200, [[0, 0], [0, 80], [-100, 80]], stroke=S_DARK))

    # Arrow: "val_loss improved" → annotation below
    els.append(arrow(_eid(), 520, 200, [[0, 0], [0, 30]], stroke=S_ENC))
    els.append(txt(_eid(), 430, 235, "→ save best.pt", 12, color=S_ENC))

    # Reset box (between zones)
    els.append(rect(_eid(), 200, 310, 200, 50, fill=C_DEC, stroke=S_DEC, opacity=25))
    els.append(txt(_eid(), 210, 323, "epoch==6: reset\nbest_val_loss, counter", 13, color=S_DARK))

    save("fig_tm4_early_stopping.excalidraw", els)


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_tm1()
    build_tm2()
    build_tm3()
    build_tm4()
    print("\nAll 4 training methodology diagrams generated.")
