# NLP Chatbot Project v2.0

Seq2Seq chatbot trained on the **Ubuntu IRC Dialogue Corpus** (~1.1M conversation pairs).  
Compares a **baseline** LSTM encoder-decoder (no attention) against a **Bahdanau attention** model in a controlled ablation study — same architecture, same hyperparameters, attention is the only variable.

---

## Architecture

| Component | Detail |
|---|---|
| Encoder | 2-layer bidirectional LSTM, 512 hidden units per direction (1024 effective) |
| Decoder | 2-layer LSTM, 1024 hidden units |
| Attention | Bahdanau (additive), 256-dim hidden layer |
| Embeddings | 300-dim FastText BPE embeddings, shared encoder/decoder |
| Vocabulary | 16 000 SentencePiece BPE tokens |
| Parameters | ~44M (both models identical) |

## Training schedule

| Phase | Epochs | Teacher Forcing |
|---|---|---|
| Foundation | 1–5 | 1.0 (full TF) |
| Annealing | 6–12 | 0.9 → 0.5 (linear decay) |
| Maturation | 13–20 | 0.5 (floor) |

LR: cosine annealing with 500-step linear warmup (peak 3e-4 → floor 1e-5).  
Label smoothing: 0.1. Early stopping: patience 4, monitored from epoch 6 onward.

---

## Project structure

```
nlp-chatbot-project-v2.0/
├── run.py                  ← unified launcher (phase1 + train, Drive-aware)
├── setup_colab.sh          ← one-shot Colab A100 session setup
├── phase1.py               ← data pipeline (8 stages: load → tokenise → embed)
├── train.py                ← training loop (baseline + attention)
├── models.py               ← encoder, decoder, attention, seq2seq
├── dataset.py              ← PyTorch Dataset + DataLoaders
├── config.py               ← single source of truth for all hyperparameters
├── evaluate.py             ← BLEU / ROUGE-L / BERTScore evaluation
├── chat.py / chatv2.py     ← interactive inference
├── agents/                 ← Copilot agent prompts (architect, developer, QA…)
├── report/                 ← training methodology and analysis notes
├── notebooks/              ← Colab training notebooks (alternative to terminal)
├── data/
│   └── Ubuntu-dialogue-corpus/   ← place dialogueText_301.csv here
├── artifacts/              ← generated: SPM model, embeddings, JSONL splits
├── checkpoints/            ← generated: model checkpoints
├── logs/                   ← generated: run logs
└── tb_logs/                ← generated: TensorBoard event files
```

---

## Uploading pre-built artifacts from Windows

If you already ran phase1 on a Windows machine, use the included PowerShell script to zip and upload the 5 required artifact files directly to Google Drive:

```powershell
# From the repo root on Windows — runs automatically:
.\upload_artifacts_to_drive.ps1
```

It tries (in order):
1. **Google Drive for Desktop** — copies files directly into your synced Drive folder (fastest, no extra tools)
2. **rclone** — if installed ([rclone.org](https://rclone.org)) and configured with a `gdrive` remote
3. **Manual fallback** — saves a zip to your Desktop with upload instructions

The 5 files it uploads:

| File | Description |
|---|---|
| `stage5_spm.model` | SentencePiece BPE tokeniser |
| `stage6_train_ids.jsonl` | ~1.1M training pairs (BPE token IDs) |
| `stage6_val_ids.jsonl` | ~47K validation pairs |
| `stage6_test_ids.jsonl` | ~47K test pairs |
| `stage8_embedding_matrix.npy` | 16 000 × 300 FastText embedding matrix |

After upload, they land at `MyDrive/nlp-chatbot-v2/artifacts/` — exactly where `run.py` and Colab expect them. **Skip phase1 entirely and go straight to training.**

```powershell
# Override source path if your artifacts are in a different folder:
.\upload_artifacts_to_drive.ps1 -ArtifactsDir "D:\my-project\new\artifacts"

# Just create the zip without uploading (upload manually):
.\upload_artifacts_to_drive.ps1 -ZipOnly
```

---



This is the recommended way to run the full training. Everything works from the **terminal** — no notebook cells required except the one-time Drive mount.

### Prerequisites

- A Google account with Google Drive
- A Colab A100 instance (Runtime → Change runtime type → A100 GPU)
- The Ubuntu Dialogue Corpus CSV: `dialogueText_301.csv` (~1.7 GB)  
  Already stored in your Drive at: `MyDrive/nlp-chatbot-v2/data/Ubuntu-dialogue-corpus/dialogueText_301.csv`

---

### Step 1 — Mount Google Drive (notebook cell, once per session)

Open a notebook cell and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

This is the **only** thing that needs a notebook cell. Everything else runs in the terminal.

---

### Step 2 — Run the setup script (terminal, once per session)

Open a Colab terminal (the `>_` icon in the left sidebar) and run:

```bash
cd /content
curl -sO https://raw.githubusercontent.com/zGeneral/nlp-chatbot-project-v2.0/A100/setup_colab.sh
bash setup_colab.sh
```

This script automatically:
- Verifies Drive is mounted
- Clones the `A100` branch into `/content/nlp-chatbot`
- Installs PyTorch and all pip requirements
- Creates output folders on Drive (`artifacts/`, `checkpoints/`, `tb_logs/`, `logs/`)
- Symlinks the Ubuntu corpus CSV from Drive into the repo

---

### Step 3 — Run the data pipeline (once ever, ~30–60 min)

```bash
cd /content/nlp-chatbot

python run.py phase1 --drive-dir /content/drive/MyDrive/nlp-chatbot-v2
```

This runs all 8 stages of `phase1.py`:

| Stage | What it does | Output |
|---|---|---|
| 1 | Load raw CSV → structured dialogues | `stage1_dialogues.pkl` |
| 2 | Clean dialogues (filter noise, IRC artifacts) | `stage2_clean_dialogues.pkl` |
| 3 | Temporal train/val/test split | `stage3_*.pkl` |
| 4 | Generate (context, response) pairs | `stage4_*_pairs.pkl` |
| 4.5 | Domain-focused filtering | in-memory |
| 5 | Train SentencePiece BPE tokeniser | `stage5_spm.model` |
| 6 | Encode all pairs to BPE token IDs | `stage6_*_ids.jsonl` |
| 7 | Train FastText embeddings | `stage7_fasttext.model` |
| 8 | Build 300-dim embedding matrix | `stage8_embedding_matrix.npy` |

**Idempotent**: if a stage's output files already exist in Drive, that stage is skipped. You can safely re-run after a disconnect.

---

### Step 4 — Smoke test (~5 min)

Always run this before committing to the full training. It exercises the complete code path (both models, checkpoints, TensorBoard, validation) on 10K pairs for 2 epochs:

```bash
python run.py train --drive-dir /content/drive/MyDrive/nlp-chatbot-v2 --mode sample
```

If this completes without errors, the full run will work.

---

### Step 5 — Full A100 training (~8–12 hours)

```bash
python run.py train --drive-dir /content/drive/MyDrive/nlp-chatbot-v2 --mode full
```

This trains both models sequentially (baseline first, then attention) with A100-optimised settings:

| Setting | Value |
|---|---|
| Batch size | 1024 (A100 Tensor Core sweet spot) |
| Gradient accumulation | 1 (true batch = 1024) |
| DataLoader workers | 4 (async collation, persistent) |
| AMP dtype | bf16 (A100 native, 312 TFLOPS) |
| TF32 | enabled (cuDNN + matmul) |
| cudnn.benchmark | True (auto-tunes kernel selection) |

**After the first epoch**, the script prints estimated time remaining and peak GPU memory so you can verify the batch size fits.

---

### Resuming after a disconnect

Just re-run steps 2 and 5. The training loop automatically detects `*_last.pt` in `Drive/checkpoints/` and resumes from the last completed epoch — no progress is lost.

```bash
bash setup_colab.sh                   # re-clone + re-install deps
cd /content/nlp-chatbot
python run.py train --drive-dir /content/drive/MyDrive/nlp-chatbot-v2 --mode full
```

---

### What gets saved to Drive

```
MyDrive/nlp-chatbot-v2/
├── artifacts/
│   ├── stage5_spm.model              ← SentencePiece BPE model
│   ├── stage6_train_ids.jsonl        ← ~1.1M training pairs (BPE token IDs)
│   ├── stage6_val_ids.jsonl          ← ~47K validation pairs
│   ├── stage6_test_ids.jsonl         ← ~47K test pairs
│   └── stage8_embedding_matrix.npy   ← 16000 × 300 FastText embedding matrix
├── checkpoints/
│   ├── baseline_best.pt              ← best val-loss checkpoint
│   ├── baseline_last.pt              ← most recent epoch (used for resume)
│   ├── baseline_history.json         ← per-epoch loss / LR / TF history
│   ├── attention_best.pt
│   ├── attention_last.pt
│   └── attention_history.json
├── tb_logs/
│   ├── baseline/                     ← TensorBoard event files
│   └── attention/
└── logs/
    └── train_*.log
```

---

### Monitoring training with TensorBoard

From a Colab notebook cell (can run while training continues in the terminal):

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/nlp-chatbot-v2/tb_logs
```

---

### run.py reference

```
python run.py {phase1,train} [--drive-dir PATH] [--mode {full,sample}]
```

| Command | What it does |
|---|---|
| `python run.py phase1` | Data pipeline, local paths |
| `python run.py phase1 --drive-dir PATH` | Data pipeline, save to Drive |
| `python run.py train --mode sample` | Smoke test (10K pairs, 2 epochs), local paths |
| `python run.py train --mode full` | Full training, local paths |
| `python run.py train --drive-dir PATH --mode sample` | Smoke test → Drive |
| `python run.py train --drive-dir PATH --mode full` | Full training → Drive |

---

## Local development

```bash
# Create venv with Python 3.11 (required — 3.14 incompatible with gensim)
python3.11 -m venv venv
source venv/bin/activate
pip install torch          # Apple Silicon: MPS backend
pip install -r requirements.txt

# Run locally (no Drive, uses local artifact/ and checkpoints/ directories)
python run.py phase1
python run.py train --mode sample
python run.py train --mode full
```

---

## Branches

| Branch | Purpose |
|---|---|
| `master` | Stable baseline — conservative settings for local GPU |
| `A100` | Optimised for Google Colab A100 80GB — this branch |

---

## Requirements

See `requirements.txt`. Key dependencies: `torch>=2.1`, `sentencepiece`, `gensim`, `sacrebleu`, `rouge-score`, `bert-score`, `tensorboard`.
