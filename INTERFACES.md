# Interfaces & Data Contracts
## `new/` — Clean-From-Scratch Implementation

This file is the **binding contract** between all modules in `new/`.
If `phase1.py` writes it, `dataset.py`/`train.py`/`evaluate.py` must read it
using exactly the formats defined here. Never change a format without updating
this file first.

---

## 1. Special Token IDs (reserved — do not change)

These IDs are hardcoded in SentencePiece via `user_defined_symbols` in `PHASE1_CONFIG`.
SentencePiece assigns IDs 0–3 to user-defined symbols in the order they are declared.

| Token   | String  | ID |
|---------|---------|-----|
| Padding | `<pad>` | 0   |
| Unknown | `<unk>` | 1   |
| Start   | `<sos>` | 2   |
| End     | `<eos>` | 3   |

**Rule:** `config.py` must define `pad_idx=0`, `unk_idx=1`, `sos_idx=2`, `eos_idx=3`
as constants. No other file should hard-code these numbers.

---

## 2. Artifact File Formats

### `stage6_vocab.json`
Written by: `phase1.py` Stage 6  
Read by: `dataset.py`, `evaluate.py`, `chat.py`

```json
{
  "<pad>": 0,
  "<unk>": 1,
  "<sos>": 2,
  "<eos>": 3,
  "▁the": 4,
  "▁sudo": 5,
  "..."
}
```

- Keys are SentencePiece piece strings (may start with `▁` = word-initial space)
- Values are integer IDs
- Total entries = `spm_vocab_size` (e.g. 16000)
- To get `idx2word`: `{v: k for k, v in vocab.items()}`

---

### `stage6_{split}_ids.jsonl`
Written by: `phase1.py` Stage 6  
Read by: `dataset.py`  
Files: `stage6_train_ids.jsonl`, `stage6_val_ids.jsonl`, `stage6_test_ids.jsonl`

One JSON object per line:
```json
{"ctx": [2, 145, 67, 890, 3], "resp": [2, 234, 56, 3]}
```

- `ctx`:  List[int] — context token IDs, **including** leading `<sos>` (ID 2)
- `resp`: List[int] — response token IDs, **including** leading `<sos>` and trailing `<eos>`
- Lengths are already truncated to `max_ctx_tokens` and `max_resp_tokens+2` by phase1
- No padding — padding is added by `collate_fn` at batch time
- `resp[0]` is always `<sos>` (2); decoder input is `resp[:-1]`, target is `resp[1:]`

---

### `stage8_embedding_matrix.npy`
Written by: `phase1.py` Stage 8  
Read by: `models.py` → `create_pretrained_embedding()`

- Shape: `[vocab_size, embed_dim]` = e.g. `[16000, 300]`
- dtype: `float32`
- Row 0 (`<pad>`) must be all zeros
- Row ordering matches `stage6_vocab.json` integer IDs exactly

---

### `stage5_spm.model`
Written by: `phase1.py` Stage 5  
Read by: `phase1.py` Stage 6, `evaluate.py`, `chat.py`

Standard SentencePiece `.model` binary file.
Load with: `sp = spm.SentencePieceProcessor(); sp.Load("stage5_spm.model")`

---

### `stage{n}_stats.json`
Written by: each stage in `phase1.py`  
Read by: humans + `phase1.py` (for reporting)

Free-form dict. Each stage should include at minimum:
```json
{
  "stage": 4,
  "elapsed": "1m 25.6s",
  "n_input": 123456,
  "n_output": 98765
}
```

---

## 3. Checkpoint Format

Written by: `train.py`  
Read by: `train.py` (resume), `evaluate.py`, `chat.py`

```python
{
    "epoch":             int,           # 1-indexed epoch that produced this checkpoint
    "global_step":       int,           # total optimizer steps at save time
    "model_type":        str,           # "attention" or "baseline"
    "model_state_dict":  OrderedDict,   # from model.state_dict()
    "optimizer_state":   dict,          # from optimizer.state_dict()
    "scheduler_state":   dict,          # from scheduler.state_dict()
    "val_loss":          float,
    "val_ppl":           float,
    "train_loss":        float,
    "tf_ratio":          float,         # TF ratio used in this epoch
    "config":            dict,          # snapshot of CONFIG at training time
    "history": {
        "train_loss":  [float, ...],    # one per epoch so far
        "val_loss":    [float, ...],
        "tf_ratios":   [float, ...],
        "lrs":         [float, ...],
    }
}
```

**Rules:**
- Always save atomically: write to `.tmp` then `os.replace(tmp, final)`
- Best checkpoint filename: `{model_type}_best.pt`
- Periodic checkpoint: `{model_type}_step_{global_step}.pt`
- Resume logic: load all state, restore epoch counter, restore `lr_stepped_down` flags

---

## 4. Dataset `__getitem__` Return Format

`UbuntuPairDataset.__getitem__()` returns:
```python
{"ctx": List[int], "resp": List[int]}
```

`collate_fn()` returns a **dict** (not a tuple):
```python
{
    "src":         torch.LongTensor,   # [batch, ctx_len_padded]
    "src_lengths": torch.LongTensor,   # [batch]  — actual unpadded ctx lengths
    "trg":         torch.LongTensor,   # [batch, resp_len_padded]
    "trg_lengths": torch.LongTensor,   # [batch]  — actual unpadded resp lengths
}
```

**In `train.py` / `evaluate.py`:** always access by key name, never by index:
```python
src         = batch["src"].to(device)
src_lengths = batch["src_lengths"].to(device)
trg         = batch["trg"].to(device)
```

**Loss computation:**
```python
# model output: [batch, trg_len-1, vocab_size]
output = model(src, src_lengths, trg, teacher_forcing_ratio=tf)
loss = criterion(
    output.reshape(-1, vocab_size),
    trg[:, 1:].reshape(-1),         # shift: exclude <sos>, include <eos>
)
```

---

## 5. Model Forward Signatures

All models must accept this exact signature (enables drop-in swap baseline ↔ attention):

```python
model.forward(
    src:                  torch.LongTensor,   # [batch, src_len]
    src_lengths:          torch.LongTensor,   # [batch]
    trg:                  torch.LongTensor,   # [batch, trg_len]
    teacher_forcing_ratio: float = 1.0,
) -> torch.Tensor                             # [batch, trg_len-1, vocab_size]
```

`build_model(model_type, config, device)` must accept `"attention"` or `"baseline"`
and return a `Seq2Seq` instance with the above signature.

---

## 6. Inference Decode Signatures

Both `greedy_decode` and `top_p_decode` in `evaluate.py`/`chat.py` must accept:

```python
def greedy_decode(
    model,
    src:         torch.Tensor,   # [batch, src_len]  already on device
    src_lengths: torch.Tensor,   # [batch]
    sos_idx:     int,
    eos_idx:     int,
    max_len:     int,
    device:      torch.device,
) -> List[List[int]]:            # batch of token ID lists (no padding, no EOS)
```

Stop condition: emit `<eos>` token OR reach `max_len` steps.
Strip `<sos>` and `<eos>` before returning.

---

## 7. Dependencies (`new/` only)

Add these to the project `requirements.txt` (already configured for CUDA 12.4):

```
sentencepiece>=0.2.0          # BPE tokenization
gensim>=4.3.0                 # FastText training
sacrebleu>=2.3.0              # BLEU corpus evaluation (tokenizer-safe)
rouge-score>=0.1.3            # ROUGE-L evaluation
torch>=2.6.0+cu124            # CUDA 12.4 — RTX 3080 12 GB
tensorboard>=2.14.0
tqdm>=4.66.0
numpy>=1.24.0
```

---

## 8. Directory Layout

```
C:\git\nlp-chatbot-project-v2.0\
├── artifacts\                  ← Phase 1 outputs (stage1–8 files)
│   ├── stage5_spm.model
│   ├── stage6_vocab.json
│   ├── stage6_train_ids.jsonl
│   ├── stage6_val_ids.jsonl
│   ├── stage6_test_ids.jsonl
│   ├── stage7_fasttext.model   (+ .wv.vectors* sidecar files)
│   └── stage8_embedding_matrix.npy
├── checkpoints\                ← Training outputs
│   ├── attention_best.pt
│   ├── baseline_best.pt
│   ├── run_info.json
│   ├── attention_heatmap.png
│   ├── attention_manual_samples.json
│   └── baseline_manual_samples.json
└── tb_logs\                    ← TensorBoard
    ├── attention\
    └── baseline\
```

All paths in `config.py` are relative to the project root and work as-is on Windows.

---

## 9. Manual Evaluation Rubric (G3)

Each entry in `*_manual_samples.json` is rated on four axes (1–5 scale):

| Axis | 1 (poor) | 3 (acceptable) | 5 (excellent) |
|------|----------|----------------|---------------|
| **Fluency** | Ungrammatical / fragmented | Mostly grammatical | Native-sounding, natural |
| **Coherence** | No relation to context | Loosely related | Logically follows context |
| **Relevance** | Off-topic / generic filler | On-topic but vague | Specific, addresses the question |
| **Specificity** | "ok", "yes", "i don't know" | Some detail | Concrete entities, apt vocabulary |

**Procedure:**
1. Evaluator reads `src` (context) and `tgt` (ground-truth reference) — do NOT reveal `hyp` first.
2. Rate `hyp` (model output) on each axis independently.
3. Record mean ± std across 50 samples per model.
4. Flag any responses that are factually false, offensive, or repetitive (these are error categories, not rated on the 1–5 scale).

**Baseline for comparison:** compare baseline vs attention scores on Specificity and Distinct-2. Distinct-2 < 0.05 signals the model is collapsing to generic responses (the quality-inversion failure mode we are solving).

**Inter-rater reliability (AC2-I4 — methodological note):** This rubric is designed for a single annotator. For full academic rigour, a second independent rater should score 20–25 of the 50 samples and compute Cohen's κ per axis. If a second rater is unavailable, acknowledge this as a study limitation in the write-up and cite Liu et al. (2016) who identify single-annotator subjectivity as a general challenge for open-domain dialogue evaluation.

```
config.py       ← no imports from new/
phase1.py       ← no imports from new/ (standalone pipeline)
dataset.py      ← from config import CONFIG
models.py       ← from config import CONFIG
train.py        ← from config import CONFIG, get_tf_ratio
                   from dataset import build_dataloaders
                   from models import build_model
evaluate.py     ← from config import CONFIG
                   from models import build_model
                   from dataset import build_dataloaders
chat.py         ← from config import CONFIG
                   from models import build_model
```

**Rule:** No circular imports. `config.py` is the only shared dependency.
`phase1.py` is a standalone script — it does not import from models or train.
