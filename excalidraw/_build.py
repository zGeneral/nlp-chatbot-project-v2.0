import json, math

def tw(text, fs):
    longest = max(len(l) for l in text.split('\n'))
    return math.ceil(longest * 0.55 * fs)

def th(text, fs):
    lines = len(text.split('\n'))
    return math.ceil(lines * fs * 1.25)

_seed = 1000
def nxt():
    global _seed; _seed += 1; return _seed

def rect(id, x, y, w, h, fill, stroke="#07182D", sw=1, opacity=100):
    return {"type":"rectangle","id":id,"x":x,"y":y,"width":w,"height":h,
            "angle":0,"seed":nxt(),"version":1,"versionNonce":nxt(),
            "isDeleted":False,"groupIds":[],"frameId":None,"boundElements":[],
            "strokeColor":stroke,"backgroundColor":fill,"fillStyle":"solid",
            "strokeWidth":sw,"strokeStyle":"solid","roughness":0,"opacity":opacity,
            "roundness":{"type":3}}

def txt(id, x, y, text, fs, color="#07182D"):
    return {"type":"text","id":id,"x":x,"y":y,
            "width":tw(text,fs),"height":th(text,fs),
            "angle":0,"seed":nxt(),"version":1,"versionNonce":nxt(),
            "isDeleted":False,"groupIds":[],"frameId":None,"boundElements":[],
            "strokeColor":color,"backgroundColor":"transparent","fillStyle":"solid",
            "strokeWidth":1,"strokeStyle":"solid","roughness":0,"opacity":100,
            "text":text,"fontSize":fs,"fontFamily":2,
            "textAlign":"left","verticalAlign":"top",
            "containerId":None,"originalText":text,"autoResize":True,"lineHeight":1.25}

def arrow(id, x, y, pts, stroke="#07182D", sw=1, style="solid"):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    w = max(abs(max(xs)-min(xs)), 1); h = max(abs(max(ys)-min(ys)), 1)
    return {"type":"arrow","id":id,"x":x,"y":y,"width":w,"height":h,
            "points":pts,"angle":0,"seed":nxt(),"version":1,"versionNonce":nxt(),
            "isDeleted":False,"groupIds":[],"frameId":None,"boundElements":[],
            "strokeColor":stroke,"backgroundColor":"transparent","fillStyle":"solid",
            "strokeWidth":sw,"strokeStyle":style,"roughness":0,"opacity":100,
            "endArrowhead":"arrow","startArrowhead":None,
            "startBinding":None,"endBinding":None}

AMBER="#f59e0b"; BLUE="#a5d8ff"; PURP="#d0bfff"; TEAL="#c3fae8"
ORG="#ffd8a8";   GRN="#b2f2bb";  ZBLUE="#dbe4ff"; ZYELL="#fff3bf"

elements = [
    rect("enc_zone",180,58,380,572,ZBLUE,"transparent",0,35),
    txt("enc_lbl",196,66,"ENCODER",12,"#4a9eed"),
    rect("dec_zone",180,646,380,278,ZYELL,"transparent",0,35),
    txt("dec_lbl",196,654,"DECODER",12,AMBER),
    txt("title",235,20,"Baseline  Seq2Seq  Architecture",22),

    rect("input",220,78,300,60,BLUE),
    txt("t_in",230,100,"Input Tokens   src_ids  [B, T]",15),
    arrow("a_ie",370,138,[[0,0],[0,72]]),

    rect("embed",220,210,300,70,PURP),
    txt("t_em1",283,228,"Shared Embedding",14),
    txt("t_em2",261,248,"16 000 \u00d7 300  |  drop=0.30",14),
    arrow("a_ee",370,280,[[0,0],[0,80]]),

    rect("encoder",220,360,300,82,BLUE),
    txt("t_en1",277,377,"BiLSTM Encoder  \u00d72",14),
    txt("t_en2",241,397,"hidden=512/dir  |  14.4M params",14),
    arrow("a_ec",520,401,[[0,0],[120,0]],AMBER,1,"dashed"),
    txt("t_ec_lbl",535,380,"last timestep",12,AMBER),
    arrow("a_eb",370,442,[[0,0],[0,78]]),
    txt("t_eb_lbl",378,470,"states [4, B, 512]",12,"#546e7a"),

    rect("ctx",640,355,260,92,ORG,AMBER),
    txt("t_ctx1",728,368,"ctx_fixed",14),
    txt("t_ctx2",651,387,"enc_outputs[:, -1, :]",13),
    txt("t_ctx3",665,406,"[B, 1024]  \u2014 const",13),
    arrow("a_cd",770,447,[[0,0],[0,263],[-250,263]],AMBER,1,"dashed"),
    txt("t_cd_lbl",782,560,"concat each step",12,AMBER),

    rect("bridge",220,520,300,70,TEAL),
    txt("t_br1",324,535,"Bridge",14),
    txt("t_br2",249,555,"Linear(1024\u21921024)\u00d72  |  tanh",14),
    arrow("a_bd",370,590,[[0,0],[0,80]]),
    txt("t_bd_lbl",378,620,"h\u2080, c\u2080  [2, B, 1024]",12,"#546e7a"),

    rect("decoder",220,670,300,80,ORG),
    txt("t_de1",276,688,"Decoder LSTM  \u00d72",14),
    txt("t_de2",263,708,"hidden=1024  |  drop=0.50",14),
    arrow("a_do",370,750,[[0,0],[0,80]]),

    rect("output",220,830,300,70,GRN,"#07182D",2),
    txt("t_out1",271,848,"Projection + Output Head",14),
    txt("t_out2",262,868,"Linear(2048\u2192512\u219216 000)",14),
]

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "gridSize": None,
        "zoom": {"value": 1.0},
        "scrollX": -140,
        "scrollY": 20
    },
    "files": {}
}

out = "excalidraw/baseline_architecture.excalidraw"
with open(out, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)

print("Text element sizes:")
for el in elements:
    if el["type"] == "text":
        print(f"  {el['id']:12s}  w={el['width']:4d}  h={el['height']:3d}  '{el['text'][:40]}'")
print(f"\nSaved {out}  ({len(json.dumps(doc))} bytes)")
