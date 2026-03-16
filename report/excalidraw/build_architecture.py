"""
build_architecture.py
Generates 5 Excalidraw architecture diagrams for the Seq2Seq LSTM chatbot report.
"""
import json
import math
from pathlib import Path

OUT_DIR = Path(__file__).parent   # report/excalidraw/

# ─────────────────────── Cisco Brand Palette ────────────────────────
# Fills used at opacity 35 for component boxes, 20 for zone backgrounds
C_ENC    = "#02C8FF"   # Cisco Blue   — Encoder domain (BiLSTM, embeddings)
C_EMBED  = "#02C8FF"   # Cisco Blue   — Shared embedding (encoder domain)
C_DEC    = "#0A60FF"   # Medium Blue  — Decoder domain (LSTM)
C_ATTN   = "#0A60FF"   # Medium Blue  — Attention mechanism (decoder sub-domain)
C_BRIDGE = "#B4B9C0"   # 30% Midnight — Bridge / projection (neutral)
C_PROJ   = "#EB4651"   # Cisco Red    — Output / logits critical path
C_IO     = "#FFFFFF"   # White        — Input/output neutral boxes
C_OP     = "#FFFFFF"   # White        — Concat / operation nodes
C_STATE  = "#F0F0F0"   # Light grey   — Recurrent state side boxes
C_NOTE   = "#F0F0F0"   # Light grey   — Note / annotation boxes

Z_ENC    = "#02C8FF"   # Cisco Blue   — Encoder zone background (opacity 20)
Z_DEC    = "#0A60FF"   # Medium Blue  — Decoder zone background (opacity 20)
Z_ATTN   = "#0A60FF"   # Medium Blue  — Attention zone background (opacity 20)
Z_BRIDGE = "#B4B9C0"   # Neutral      — Bridge zone background

S_ENC    = "#02C8FF"   # Cisco Blue stroke
S_BRIDGE = "#6B6B6B"   # Medium Grey stroke
S_DEC    = "#0A60FF"   # Medium Blue stroke
S_ATTN   = "#0A60FF"   # Medium Blue stroke
S_EMBED  = "#02C8FF"   # Cisco Blue stroke (shared embedding = encoder domain)
S_PROJ   = "#EB4651"   # Cisco Red stroke (output / critical path)
S_OP     = "#6B6B6B"   # Medium Grey stroke (concat ops)
S_STATE  = "#6B6B6B"   # Medium Grey stroke (state side boxes)
S_IO     = "#6B6B6B"   # Medium Grey stroke (input/output boxes)
S_DARK   = "#07182D"   # Midnight Blue (default text and arrows)

# ─────────────────────── Element Helpers ───────────────────────────────
_seed = 2000

def nxt():
    global _seed
    _seed += 1
    return _seed

def tw(text, fs):
    return math.ceil(max(len(l) for l in text.split('\n')) * 0.55 * fs)

def th(text, fs):
    return math.ceil(len(text.split('\n')) * fs * 1.25)

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

def rect(eid, x, y, w, h, fill, stroke="#07182D", sw=1, opacity=100, dashed=False):
    return {
        "type": "rectangle", "id": eid, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "seed": nxt(), "version": 1, "versionNonce": nxt(),
        "isDeleted": False, "groupIds": [], "frameId": None, "boundElements": [],
        "updated": 1700000000000, "link": None, "locked": False,
        "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid",
        "strokeWidth": sw, "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 0, "opacity": opacity, "roundness": {"type": 3},
    }

def arrow(eid, x, y, pts, stroke="#07182D", sw=1, dashed=False):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w  = max(abs(max(xs) - min(xs)), 1)
    h  = max(abs(max(ys) - min(ys)), 1)
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


# ═══════════════════════════════════════════════════════════════════════
# fig_a1_overview — Side-by-side Baseline vs Attention
# ═══════════════════════════════════════════════════════════════════════
def build_fig_a1():
    els = []

    BW     = 300   # box width
    LX     = 60    # left column left-edge
    RX     = 520   # right column left-edge
    LCX    = LX + BW // 2   # 210 — left spine centre-x
    RCX    = RX + BW // 2   # 670 — right spine centre-x

    BH_IO  = 62
    BH     = 72
    BH_CTX = 92

    Y_IN   = 110
    Y_EM   = 235   # gap: 235-110-62 = 63
    Y_ENC  = 378   # gap: 378-235-72 = 71
    Y_BR   = 521   # gap: 521-378-72 = 71
    Y_CTX  = 664   # gap: 664-521-72 = 71
    Y_DEC  = 827   # gap: 827-664-92 = 71
    Y_PROJ = 970   # gap: 970-827-72 = 71
    Y_OUT  = 1113  # gap: 1113-970-72 = 71

    # ── Title ──────────────────────────────────────────────────────────
    els.append(txt("a1_title", LX, 36,
                   "Seq2Seq Architecture \u2014 Baseline vs Attention", 22))

    # ── Zone backgrounds (drawn FIRST for lowest z-order) ──────────────
    Z_TOP = Y_IN - 28
    Z_H   = (Y_OUT + BH_IO + 30) - Z_TOP
    els.append(rect("a1_lzone", LX-22, Z_TOP, BW+44, Z_H, Z_ENC, S_ENC, 1, 20))
    els.append(rect("a1_rzone", RX-22, Z_TOP, BW+44, Z_H, Z_DEC, S_DEC, 1, 20))
    els.append(txt("a1_lbl_bl", LX+6,  Z_TOP+7, "BASELINE",  14, S_ENC))
    els.append(txt("a1_lbl_at", RX+6,  Z_TOP+7, "ATTENTION", 14, S_DEC))

    # ── Helper: draw same component on both spines ──────────────────────
    def both(pfx, y, h, fill, stroke, label, fs=13, opacity=100):
        for px in (LX, RX):
            p = "l" if px == LX else "r"
            els.append(rect(f"a1_{pfx}_{p}_r", px, y, BW, h, fill, stroke, 2, opacity))
            els.append(txt(f"a1_{pfx}_{p}_t", px+9, y+9, label, fs, stroke))

    def arrs(pfx, y_start, y_end, lbl=""):
        gap = y_end - y_start
        for (cx, p) in ((LCX, "l"), (RCX, "r")):
            els.append(arrow(f"a1_ar_{pfx}_{p}", cx, y_start, [[0, 0], [0, gap]]))
            if lbl:
                els.append(txt(f"a1_lbl_{pfx}_{p}", cx+4, y_start+6, lbl, 11, "#6b7280"))

    # ── Spine rows (shared on both sides) ──────────────────────────────
    both("in",  Y_IN,   BH_IO, C_IO,     S_IO,
         "Input Tokens\nsrc_ids  [B, T]")
    arrs("ie",  Y_IN + BH_IO, Y_EM, "[B, T]")

    both("em",  Y_EM,   BH,    C_EMBED,  S_EMBED,
         "Shared Embedding\n16000\u00d7300  |  drop=0.30", opacity=35)
    arrs("ee",  Y_EM + BH, Y_ENC, "[B, T, 300]")

    both("enc", Y_ENC,  BH,    C_ENC,    S_ENC,
         "BiLSTM Encoder \u00d72\nhidden=512/dir  \u2192  [B,T,1024]", opacity=35)
    arrs("eb",  Y_ENC + BH, Y_BR, "states [4,B,512]")

    both("br",  Y_BR,   BH,    C_BRIDGE, S_BRIDGE,
         "Bridge Projection\nLinear(1024\u21921024)\u00d72 + tanh", opacity=35)
    arrs("bc",  Y_BR + BH, Y_CTX, "h\u2080, c\u2080  [2,B,1024]")

    # ── Context boxes — DIFFERENT per side ─────────────────────────────
    # Baseline: fixed context
    els.append(rect("a1_ctx_l_r", LX, Y_CTX, BW, BH_CTX, C_DEC, S_DEC, 2, 35))
    els.append(txt("a1_ctx_l_t", LX+9, Y_CTX+9,
                   "ctx_fixed\nencoder_outputs[:,-1,:]\n[B,1024]  \u2014  constant", 13, S_DEC))

    # Attention: Bahdanau
    els.append(rect("a1_ctx_r_r", RX, Y_CTX, BW, BH_CTX, C_ATTN, S_ATTN, 2, 35))
    els.append(txt("a1_ctx_r_t", RX+9, Y_CTX+9,
                   "Bahdanau Attention\ne=v\u00b7tanh(W_enc\u00b7h_i+W_dec\u00b7s_t)\nc_t [B,1024]  \u2014  dynamic",
                   13, S_ATTN))

    # Arrows context → decoder
    ctx_bot = Y_CTX + BH_CTX
    for (cx, p, sc) in ((LCX, "l", S_DEC), (RCX, "r", S_ATTN)):
        gap = Y_DEC - ctx_bot
        els.append(arrow(f"a1_ar_cd_{p}", cx, ctx_bot, [[0, 0], [0, gap]], sc))
        els.append(txt(f"a1_lbl_cd_{p}", cx+4, ctx_bot+6, "[B,1024] ctx", 11, "#6b7280"))

    both("dec",  Y_DEC,  BH,    C_DEC,  S_DEC,
         "Decoder LSTM \u00d72\nhidden=1024  |  drop=0.50", opacity=35)
    arrs("dp",   Y_DEC + BH, Y_PROJ, "[B,1024] s_t")

    both("proj", Y_PROJ, BH,    C_PROJ, S_PROJ,
         "Bottleneck + Output Head\nLinear(2048\u2192512)+tanh\u219216000", opacity=35)
    arrs("po",   Y_PROJ + BH, Y_OUT, "[B,16000]")

    both("out",  Y_OUT,  BH_IO, C_IO,   S_IO,
         "Output Logits\n[B, V=16000]")

    save("fig_a1_overview.excalidraw", els)


# ═══════════════════════════════════════════════════════════════════════
# fig_a2_baseline_decoder — Baseline decoder single timestep
# ═══════════════════════════════════════════════════════════════════════
def build_fig_a2():
    els = []

    BW = 300; LX = 80; CX = LX + BW // 2   # CX = 230

    Y_PREV = 60;   H_PREV = 62
    Y_EM   = 185;  H_EM   = 72
    Y_CAT1 = 330;  H_CAT1 = 66
    Y_LSTM = 470;  H_LSTM = 92
    Y_CAT2 = 640;  H_CAT2 = 66
    Y_LIN1 = 785;  H_LIN1 = 72
    Y_LIN2 = 935;  H_LIN2 = 72
    Y_OUT  = 1080; H_OUT  = 62

    els.append(txt("a2_title", LX, 22,
                   "Baseline Decoder \u2014 Single Timestep (step t)", 18))

    def sbox(pfx, y, h, fill, stroke, label, fs=13, opacity=100):
        els.append(rect(f"{pfx}_r", LX, y, BW, h, fill, stroke, 2, opacity))
        els.append(txt(f"{pfx}_t", LX+9, y+9, label, fs, stroke))

    def darr(pfx, y_start, y_end, lbl=""):
        els.append(arrow(f"a2_ar_{pfx}", CX, y_start, [[0, 0], [0, y_end - y_start]]))
        if lbl:
            els.append(txt(f"a2_lbl_{pfx}", CX+5, y_start+6, lbl, 11, "#6b7280"))

    # ── Main spine ──────────────────────────────────────────────────────
    sbox("a2_prev", Y_PREV, H_PREV, C_IO,     S_IO,
         "y_{t-1}   (prev output token)\n[B]  token index")
    darr("pe", Y_PREV + H_PREV, Y_EM, "[B]")

    sbox("a2_em",   Y_EM,   H_EM,   C_EMBED,  S_EMBED,
         "Embedding (shared)\n16000\u00d7300  |  drop=0.30", opacity=35)
    darr("ec", Y_EM + H_EM, Y_CAT1, "embed  [B, 300]")

    sbox("a2_cat1", Y_CAT1, H_CAT1, C_OP,     S_OP,
         "cat([embed, ctx_fixed])\n[B, 1324]  (300 + 1024)")
    darr("cl", Y_CAT1 + H_CAT1, Y_LSTM, "[B, 1324]")

    sbox("a2_lstm", Y_LSTM, H_LSTM, C_DEC,    S_DEC,
         "2-layer Decoder LSTM\nhidden=1024  |  drop=0.50\noutput: s_t  [B, 1024]", opacity=35)
    darr("lc", Y_LSTM + H_LSTM, Y_CAT2, "s_t  [B, 1024]")

    sbox("a2_cat2", Y_CAT2, H_CAT2, C_OP,     S_OP,
         "cat([s_t, ctx_fixed])\n[B, 2048]  (1024 + 1024)")
    darr("c2l", Y_CAT2 + H_CAT2, Y_LIN1, "[B, 2048]")

    sbox("a2_lin1", Y_LIN1, H_LIN1, C_PROJ,   S_PROJ,
         "Linear(2048\u2192512) + tanh + Drop\nbottleneck  \u2192  [B, 512]", opacity=35)
    darr("l12", Y_LIN1 + H_LIN1, Y_LIN2, "[B, 512]")

    sbox("a2_lin2", Y_LIN2, H_LIN2, C_PROJ,   S_PROJ,
         "Linear(512 \u2192 16000)\noutput head", opacity=35)
    darr("lo", Y_LIN2 + H_LIN2, Y_OUT, "[B, 16000]")

    sbox("a2_out",  Y_OUT,  H_OUT,  C_IO,     S_IO,
         "Logits   [B, V=16000]")

    # ── ctx_fixed side box ──────────────────────────────────────────────
    CTX_X = 460; CTX_W = 265; CTX_H = 112
    cat1_cy = Y_CAT1 + H_CAT1 // 2   # 363
    cat2_cy = Y_CAT2 + H_CAT2 // 2   # 673
    CTX_Y   = (cat1_cy + cat2_cy) // 2 - CTX_H // 2   # ~462

    els.append(rect("a2_ctx_r", CTX_X, CTX_Y, CTX_W, CTX_H, C_DEC, S_DEC, 2, 35))
    els.append(txt("a2_ctx_t", CTX_X+9, CTX_Y+10,
                   "ctx_fixed\nencoder_outputs[:, -1, :]\n[B, 1024]  \u2014  constant", 13, S_DEC))

    ctx_lx     = CTX_X
    ctx_cy_val = CTX_Y + CTX_H // 2
    cat1_rx    = LX + BW   # right edge of spine = 380
    cat2_rx    = LX + BW

    # Dashed amber arrows: ctx_fixed → cat1 (diagonal)
    dx1 = cat1_rx - ctx_lx           # 380 - 460 = -80
    dy1 = cat1_cy - ctx_cy_val
    els.append(arrow("a2_ar_ctx1", ctx_lx, ctx_cy_val, [[0, 0], [dx1, dy1]], S_DEC, 1, True))
    els.append(txt("a2_lbl_c1",
                   ctx_lx + dx1 // 2 - 15, ctx_cy_val + dy1 // 2 - 14, "ctx", 11, S_DEC))

    # Dashed amber arrows: ctx_fixed → cat2 (diagonal)
    dx2 = cat2_rx - ctx_lx
    dy2 = cat2_cy - ctx_cy_val
    els.append(arrow("a2_ar_ctx2", ctx_lx, ctx_cy_val, [[0, 0], [dx2, dy2]], S_DEC, 1, True))
    els.append(txt("a2_lbl_c2",
                   ctx_lx + dx2 // 2 - 15, ctx_cy_val + dy2 // 2 + 6, "ctx", 11, S_DEC))

    # ── Recurrent state side box ─────────────────────────────────────────
    ST_X = 460; ST_W = 240; ST_H = 76
    ST_Y = CTX_Y + CTX_H + 22

    els.append(rect("a2_state_r", ST_X, ST_Y, ST_W, ST_H, C_STATE, S_STATE, 1, 100, True))
    els.append(txt("a2_state_t", ST_X+9, ST_Y+10,
                   "h_{t-1}, c_{t-1}  \u2192  LSTM\n\u2192  h_t, c_t  [2, B, 1024]", 12, S_STATE))

    # Elbow arrow: LSTM right edge → rightward → state box
    lstm_cy  = Y_LSTM + H_LSTM // 2
    state_cy = ST_Y + ST_H // 2
    dx_st    = ST_X - (LX + BW)          # 460 - 380 = 80
    dy_st    = state_cy - lstm_cy
    els.append(arrow("a2_ar_st", LX + BW, lstm_cy,
                     [[0, 0], [dx_st, 0], [dx_st, dy_st]], S_STATE, 1, True))

    # ── Warning note ──────────────────────────────────────────────────────
    NOTE_Y = Y_OUT + H_OUT + 42
    els.append(rect("a2_note_r", LX, NOTE_Y, 548, 66, C_NOTE, "#94a3b8"))
    els.append(txt("a2_note_t", LX+12, NOTE_Y+13,
                   "\u26a0  ctx_fixed is constant \u2014 decoder cannot focus\n"
                   "   on different source positions at each timestep",
                   13, "#475569"))

    save("fig_a2_baseline_decoder.excalidraw", els)


# ═══════════════════════════════════════════════════════════════════════
# fig_a3_attention_decoder — Attention decoder single timestep
# ═══════════════════════════════════════════════════════════════════════
def build_fig_a3():
    els = []

    BW = 300; LX = 80; CX = LX + BW // 2   # CX = 230

    Y_PREV = 60;   H_PREV = 62
    Y_EM   = 185;  H_EM   = 72
    Y_CAT1 = 330;  H_CAT1 = 66
    Y_LSTM = 470;  H_LSTM = 92
    # Attention runs alongside/below LSTM on the right; CAT2 is after c_t
    Y_CAT2 = 860;  H_CAT2 = 66
    Y_LIN1 = 1005; H_LIN1 = 72
    Y_LIN2 = 1155; H_LIN2 = 72
    Y_OUT  = 1300; H_OUT  = 62

    # ── Attention zone constants (declared early for zone background) ───
    AZ_X = 460; AZ_Y = 420; AZ_W = 380; AZ_H = 450
    A_LX  = AZ_X + 20            # 480
    A_BW  = 260
    A_ACX = A_LX + A_BW // 2    # 610  (centre x of attention nodes)
    A_H   = 54                   # height of each attention node

    A_ENC_Y = 442
    A_KEY_Y = A_ENC_Y + A_H + 18   # 514
    A_SCO_Y = A_KEY_Y + A_H + 18   # 586
    A_ALF_Y = A_SCO_Y + A_H + 18   # 658
    A_CT_Y  = A_ALF_Y + A_H + 18   # 730

    # ── Title ──────────────────────────────────────────────────────────
    els.append(txt("a3_title", LX, 22,
                   "Attention Decoder \u2014 Single Timestep (Bahdanau)", 18))

    # ── Attention zone background — FIRST for lowest z-order ───────────
    els.append(rect("a3_attn_zone", AZ_X, AZ_Y, AZ_W, AZ_H, Z_ATTN, S_ATTN, 2, 20))
    els.append(txt("a3_attn_lbl", AZ_X+10, AZ_Y+7, "Bahdanau Attention", 14, S_ATTN))

    # ── Main spine ──────────────────────────────────────────────────────
    def sbox(pfx, y, h, fill, stroke, label, fs=13, opacity=100):
        els.append(rect(f"{pfx}_r", LX, y, BW, h, fill, stroke, 2, opacity))
        els.append(txt(f"{pfx}_t", LX+9, y+9, label, fs, stroke))

    def darr(pfx, y_start, y_end, lbl=""):
        els.append(arrow(f"a3_ar_{pfx}", CX, y_start, [[0, 0], [0, y_end - y_start]]))
        if lbl:
            els.append(txt(f"a3_lbl_{pfx}", CX+5, y_start+6, lbl, 11, "#6b7280"))

    sbox("a3_prev", Y_PREV, H_PREV, C_IO,    S_IO,
         "y_{t-1}   (prev output token)\n[B]  token index")
    darr("pe", Y_PREV + H_PREV, Y_EM, "[B]")

    sbox("a3_em",   Y_EM,   H_EM,   C_EMBED, S_EMBED,
         "Embedding (shared)\n16000\u00d7300  |  drop=0.30", opacity=35)
    darr("ec", Y_EM + H_EM, Y_CAT1, "embed  [B, 300]")

    sbox("a3_cat1", Y_CAT1, H_CAT1, C_OP,    S_OP,
         "cat([embed, c_{t-1}])\n[B, 1324]  (300 + 1024)")
    darr("cl", Y_CAT1 + H_CAT1, Y_LSTM, "[B, 1324]")

    sbox("a3_lstm", Y_LSTM, H_LSTM, C_ATTN,  S_ATTN,
         "2-layer Decoder LSTM\nhidden=1024  |  drop=0.50\noutput: s_t  [B, 1024]", opacity=35)

    # Dashed stub: s_t descends the main spine while attention runs on right
    gap_stub = Y_CAT2 - (Y_LSTM + H_LSTM)
    els.append(arrow("a3_ar_ls", CX, Y_LSTM + H_LSTM,
                     [[0, 0], [0, gap_stub]], "#94a3b8", 1, True))
    els.append(txt("a3_lbl_ls", CX+5, Y_LSTM + H_LSTM + 12, "s_t  [B,1024]", 11, "#6b7280"))

    sbox("a3_cat2", Y_CAT2, H_CAT2, C_OP,    S_OP,
         "cat([s_t, c_t])\n[B, 2048]  (1024 + 1024)")
    darr("c2l", Y_CAT2 + H_CAT2, Y_LIN1, "[B, 2048]")

    sbox("a3_lin1", Y_LIN1, H_LIN1, C_PROJ,  S_PROJ,
         "Linear(2048\u2192512) + tanh + Drop\nbottleneck  \u2192  [B, 512]", opacity=35)
    darr("l12", Y_LIN1 + H_LIN1, Y_LIN2, "[B, 512]")

    sbox("a3_lin2", Y_LIN2, H_LIN2, C_PROJ,  S_PROJ,
         "Linear(512 \u2192 16000)\noutput head", opacity=35)
    darr("lo", Y_LIN2 + H_LIN2, Y_OUT, "[B, 16000]")

    sbox("a3_out",  Y_OUT,  H_OUT,  C_IO,    S_IO,
         "Logits   [B, V=16000]")

    # ── Attention nodes inside zone ──────────────────────────────────────
    def abox(pfx, y, fill, stroke, label, fs=12, opacity=35):
        els.append(rect(f"{pfx}_r", A_LX, y, A_BW, A_H, fill, stroke, 1, opacity))
        els.append(txt(f"{pfx}_t", A_LX+7, y+7, label, fs, stroke))

    abox("a3_a_enc", A_ENC_Y, C_ENC,  S_ENC,
         "encoder_outputs\n[B, T, 1024]")
    abox("a3_a_key", A_KEY_Y, C_ATTN, S_ATTN,
         "W_enc\u00b7h_i  \u2192  keys  [B,T,256]\n(precomputed once per utterance)")
    abox("a3_a_sco", A_SCO_Y, C_ATTN, S_ATTN,
         "e_{t,i}=v\u00b7tanh(keys+W_dec\u00b7s_t)\n\u2192 scores  [B, T, 1]   (attn_dim=256)")
    abox("a3_a_alf", A_ALF_Y, C_ATTN, S_ATTN,
         "softmax(+pad_mask)  \u2192  \u03b1_t\n[B, T]  attention weights")
    abox("a3_a_ct",  A_CT_Y,  C_ATTN, S_ATTN,
         "c_t = \u03a3 \u03b1_{t,i} \u00b7 h_i\n[B, 1024]  context vector")

    # Arrows inside attention zone (vertical chain)
    for (yf, hf, yt, pfx) in [
        (A_ENC_Y, A_H, A_KEY_Y, "ek"),
        (A_KEY_Y, A_H, A_SCO_Y, "ks"),
        (A_SCO_Y, A_H, A_ALF_Y, "sa"),
        (A_ALF_Y, A_H, A_CT_Y,  "ac"),
    ]:
        els.append(arrow(f"a3_ar_{pfx}", A_ACX, yf + hf,
                         [[0, 0], [0, yt - yf - hf]], S_ATTN, 1))

    # Arrow: LSTM right → score node (s_t feeds as query)
    lstm_cy = Y_LSTM + H_LSTM // 2   # 516
    sco_ly  = A_SCO_Y + A_H // 2    # 613
    dx_lq   = A_LX - (LX + BW)      # 480 - 380 = 100
    dy_lq   = sco_ly - lstm_cy       # 613 - 516 = 97
    els.append(arrow("a3_ar_sq", LX + BW, lstm_cy,
                     [[0, 0], [dx_lq, dy_lq]], S_ATTN, 2, True))
    els.append(txt("a3_lbl_sq", LX + BW + 5, lstm_cy + dy_lq // 2 - 12, "s_t", 12, S_ATTN))

    # Arrow: c_t → CAT2  (elbow: down, then sweep left to spine centre)
    ct_bot  = A_CT_Y + A_H           # 784
    dy_dn   = Y_CAT2 - 22 - ct_bot   # 860-22-784 = 54
    dx_lft  = CX - A_ACX             # 230 - 610 = -380
    dy_tot  = Y_CAT2 - ct_bot        # 860 - 784 = 76
    els.append(arrow("a3_ar_ct2", A_ACX, ct_bot,
                     [[0, 0], [0, dy_dn], [dx_lft, dy_dn], [dx_lft, dy_tot]],
                     S_ATTN, 2))
    els.append(txt("a3_lbl_ct2",
                   A_ACX + dx_lft // 2 - 10, ct_bot + dy_dn - 16,
                   "c_t  [B,1024]", 12, S_ATTN))

    # Arrow: c_t feedback loop → c_{t-1} feeds CAT1 next step (dashed)
    ct_lx   = A_LX                          # 480
    ct_cy   = A_CT_Y + A_H // 2             # 757
    fb_x    = LX - 62                        # 18  — left of main spine
    cat1_rx = LX + BW                        # 380
    cat1_cy = Y_CAT1 + H_CAT1 // 2          # 363
    els.append(arrow("a3_ar_fb", ct_lx, ct_cy,
                     [[0, 0],
                      [fb_x - ct_lx, 0],
                      [fb_x - ct_lx, cat1_cy - ct_cy],
                      [cat1_rx - ct_lx, cat1_cy - ct_cy]],
                     S_ATTN, 1, True))
    els.append(txt("a3_lbl_fb",
                   fb_x - 10, (ct_cy + cat1_cy) // 2 - 10, "c_{t-1}", 12, S_ATTN))

    # ── Recurrent state box (LEFT of main spine) ─────────────────────────
    ST_W = 220; ST_H = 76; ST_X = LX - ST_W - 30   # -170
    ST_Y = Y_LSTM + (H_LSTM - ST_H) // 2
    els.append(rect("a3_state_r", ST_X, ST_Y, ST_W, ST_H, C_STATE, S_STATE, 1, 100, True))
    els.append(txt("a3_state_t", ST_X+9, ST_Y+10,
                   "h_{t-1}, c_{t-1}  \u2192  LSTM\n\u2192  h_t, c_t  [2,B,1024]", 12, S_STATE))
    # Arrow: state right edge → LSTM left edge (horizontal)
    state_cy = ST_Y + ST_H // 2
    els.append(arrow("a3_ar_stl", ST_X + ST_W, state_cy,
                     [[0, 0], [LX - (ST_X + ST_W), 0]], S_STATE, 1, True))

    save("fig_a3_attention_decoder.excalidraw", els)


# ═══════════════════════════════════════════════════════════════════════
# fig_a4_params — Parameter breakdown table
# ═══════════════════════════════════════════════════════════════════════
def build_fig_a4():
    els = []

    TBL_X = 60; TBL_Y = 100
    RH = 52    # row height
    HH = 58    # header height
    CW = [240, 295, 158, 158]   # Component | Calculation | Baseline | Attention
    TW = sum(CW)

    els.append(txt("a4_title", TBL_X, 38,
                   "Parameter Breakdown \u2014 Seq2Seq LSTM Chatbot", 22))

    # ── Header row ──────────────────────────────────────────────────────
    HEADERS = ["Component", "Calculation", "Baseline", "Attention"]
    HBG     = ["#07182D", "#07182D", C_ENC,     C_DEC]
    HTC     = ["#FFFFFF",  "#FFFFFF",  "#07182D", "#07182D"]
    x = TBL_X
    for i, (h, hb, ht, cw) in enumerate(zip(HEADERS, HBG, HTC, CW)):
        els.append(rect(f"a4_hdr_{i}", x, TBL_Y, cw, HH, hb, "#111827", 1))
        els.append(txt(f"a4_hdrt_{i}", x+8, TBL_Y + (HH - 17) // 2, h, 14, ht))
        x += cw

    # ── Data rows ────────────────────────────────────────────────────────
    # (component, calculation, baseline, attention, row_fill, row_stroke)
    ROWS = [
        ("Embedding (shared)",  "16000 \u00d7 300",
         "4,800,000",  "4,800,000",  C_EMBED,  S_EMBED),
        ("Encoder BiLSTM",      "2\u00d7 layers, hidden=512/dir",
         "9,633,792",  "9,633,792",  C_ENC,    S_ENC),
        ("Bridge",              "Linear(1024\u21921024)\u00d72",
         "2,099,200",  "2,099,200",  C_BRIDGE, S_BRIDGE),
        ("Decoder LSTM",        "2\u00d7 layers, hidden=1024",
         "18,022,400", "18,022,400", C_DEC,    S_DEC),
        ("Bottleneck + Output", "2048\u2192512\u219216000",
         "9,257,088",  "9,257,088",  C_PROJ,   S_PROJ),
        ("Attention weights",   "W_enc+W_dec+v, dim=256",
         "\u2014",      "524,544",   C_ATTN,   S_ATTN),
        ("TOTAL",               "",
         "43,812,480", "44,337,024", "#e2e8f0", "#475569"),
    ]

    for r_idx, (comp, calc, base, attn, rf, rs) in enumerate(ROWS):
        y  = TBL_Y + HH + r_idx * RH
        sw = 2 if r_idx == len(ROWS) - 1 else 1
        # Col 0: Component (row colour)
        # Col 1: Calculation (neutral)
        # Col 2: Baseline value (C_DEC tint)
        # Col 3: Attention value (C_ATTN tint)
        COLORED_FILLS = {C_EMBED, C_ENC, C_BRIDGE, C_DEC, C_PROJ, C_ATTN}
        col_fills  = [rf,       "#FFFFFF", C_DEC,    C_ATTN]
        col_strok  = [rs,       "#94a3b8", S_DEC,    S_ATTN]
        if r_idx == len(ROWS) - 1:
            col_fills = [rf, "#F0F0F0", C_ENC, C_DEC]
        x = TBL_X
        for c_idx, (cell, cw, cf, cs) in enumerate(
                zip([comp, calc, base, attn], CW, col_fills, col_strok)):
            cop = 35 if cf in COLORED_FILLS else 100
            els.append(rect(f"a4_r{r_idx}_c{c_idx}", x, y, cw, RH, cf, cs, sw, cop))
            fs  = 14 if r_idx == len(ROWS) - 1 else 13
            tc  = rs if c_idx == 0 else "#07182D"
            els.append(txt(f"a4_rt{r_idx}_c{c_idx}", x+7, y + (RH - fs) // 2, cell, fs, tc))
            x += cw

    # ── Summary box ──────────────────────────────────────────────────────
    SUM_Y = TBL_Y + HH + len(ROWS) * RH + 32
    els.append(rect("a4_sum_r", TBL_X, SUM_Y, TW, 62, C_DEC, S_DEC, 2, 35))
    els.append(txt("a4_sum_t", TBL_X+16, SUM_Y+16,
                   "+524,544 params  (+1.2% overhead for full Bahdanau attention mechanism)",
                   15, "#07182D"))

    save("fig_a4_params.excalidraw", els)


# ═══════════════════════════════════════════════════════════════════════
# fig_a5_comparison — Bahdanau vs Luong comparison table
# ═══════════════════════════════════════════════════════════════════════
def build_fig_a5():
    els = []

    TBL_X = 60; TBL_Y = 100
    RH = 72    # row height (taller for multi-line cells)
    HH = 58
    CW = [222, 322, 282]   # Dimension | Bahdanau | Luong
    TW = sum(CW)

    els.append(txt("a5_title", TBL_X, 38,
                   "Attention Mechanism Comparison \u2014 Bahdanau vs Luong", 22))

    # ── Header ───────────────────────────────────────────────────────────
    HEADERS = ["Dimension", "Bahdanau  (this model)", "Luong"]
    HBGS    = ["#07182D", C_ENC, C_DEC]
    HTCS    = ["#FFFFFF",  "#07182D", "#07182D"]
    x = TBL_X
    for i, (h, hb, ht, cw) in enumerate(zip(HEADERS, HBGS, HTCS, CW)):
        els.append(rect(f"a5_hdr_{i}", x, TBL_Y, cw, HH, hb, "#111827", 1))
        els.append(txt(f"a5_hdrt_{i}", x+8, TBL_Y + (HH - 17) // 2, h, 14, ht))
        x += cw

    # ── Data rows ────────────────────────────────────────────────────────
    BAH_F = C_ENC;     BAH_S = S_ENC
    LUO_F = "#F0F0F0"; LUO_S = "#6B6B6B"
    DIM_F = "#FFFFFF";  DIM_S = "#07182D"

    ROWS = [
        ("Alignment timing",
         "Uses s_{t-1} before LSTM step\n(input-feeding / Luong-style variant)",
         "Uses s_t after LSTM step\n(standard Luong)"),
        ("Scoring function",
         "v\u00b7tanh(W_enc\u00b7h_i + W_dec\u00b7s_t)\nadditive MLP  \u2014  non-linear",
         "s_t \u00b7 h_i  dot-product\nlinear similarity"),
        ("Projection dim",
         "attn_dim = 256\n(stable for additive scoring)",
         "dim = 1024\n(saturation risk at high d)"),
        ("Expressiveness",
         "Non-linear MLP \u2014\ncaptures richer alignment patterns",
         "Linear dot-product \u2014\nlimited expressiveness"),
        ("Recommendation",
         "\u2713  Better for generation tasks\n(IRC dialogue, multi-turn chat)",
         "\u2717  Risk of softmax saturation\nat d=1024 in long utterances"),
    ]

    for r_idx, (dim, bah, luo) in enumerate(ROWS):
        y  = TBL_Y + HH + r_idx * RH
        sw = 2 if r_idx == len(ROWS) - 1 else 1
        x  = TBL_X
        COLORED_FILLS_A5 = {C_ENC, C_DEC, C_ATTN, C_EMBED, C_BRIDGE, C_PROJ}
        for c_idx, (cell, cw, cf, cs) in enumerate(
                zip([dim, bah, luo], CW,
                    [DIM_F, BAH_F, LUO_F],
                    [DIM_S, BAH_S, LUO_S])):
            cop = 35 if cf in COLORED_FILLS_A5 else 100
            els.append(rect(f"a5_r{r_idx}_c{c_idx}", x, y, cw, RH, cf, cs, sw, cop))
            els.append(txt(f"a5_rt{r_idx}_c{c_idx}", x+7, y+8, cell, 12, cs))
            x += cw

    # ── Note below table ─────────────────────────────────────────────────
    NOTE_Y = TBL_Y + HH + len(ROWS) * RH + 28
    els.append(rect("a5_note_r", TBL_X, NOTE_Y, TW, 66, C_ENC, S_ENC, 1, 35))
    els.append(txt("a5_note_t", TBL_X+12, NOTE_Y+14,
                   ">> Bahdanau chosen: avoids softmax saturation at 1024-d, "
                   "more expressive alignment for technical IRC chat",
                   13, "#07182D"))

    save("fig_a5_comparison.excalidraw", els)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_fig_a1()
    build_fig_a2()
    build_fig_a3()
    build_fig_a4()
    build_fig_a5()
    print("\nAll 5 Excalidraw architecture files generated successfully.")
