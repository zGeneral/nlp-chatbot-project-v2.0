# NLP Chatbot — Seq2Seq Ablation Study

A production-ready sequence-to-sequence chatbot trained on the **Ubuntu IRC Dialogue Corpus** (~1.1 M conversation pairs). The project runs a controlled ablation comparing two architectures under identical conditions:

| Model | Architecture |
|---|---|
| **Baseline** | 2-layer BiLSTM encoder + LSTM decoder (fixed context vector) |
| **Attention** | Same encoder + Bahdanau attention decoder |

**Goal:** isolate the contribution of attention — every other hyperparameter is identical between the two models.

---

## Project Structure

```
nlp-chatbot-project-v2.0/
├── phase1.py           # Data pipeline  (download → tokenise → embeddings)
├── analyze_data.py     # Corpus statistics and vocabulary analysis
├── train.py            # Model training  (baseline + attention, sequential)
├── evaluate.py         # Post-training evaluation (BLEU, ROUGE, BERTScore)
├── chatv2.py           # Interactive chat  (both models, beam + greedy)
├── models.py           # Architecture definitions (encoder, decoders, seq2seq)
├── dataset.py          # DataLoader, BucketBatchSampler, collation
├── config.py           # Single source of truth for all hyperparameters
├── logging_utils.py    # Structured run logging
├── run.py              # Convenience launcher (wraps phase1 + train)
├── report/
│   └── plot_training.py  # Publication-ready training figures (6 plots)
├── data/               # Raw corpus (Ubuntu Dialogue Corpus CSV)
├── artifacts/          # Generated artifacts (vocab, SPM model, embeddings, IDs)
├── checkpoints/        # Model checkpoints (*_best.pt, *_last.pt)
├── logs/               # Text run logs
└── tb_logs/            # TensorBoard event files
```

---

## Hardware

Developed and tested on **Windows 11, NVIDIA RTX 3080 12 GB (CUDA 11.8)**.

- bf16 AMP is enabled automatically on Ampere+ GPUs (RTX 3080, RTX 3090, A100)
- Falls back to fp32 on older hardware
- `torch.compile` is disabled on Windows (Triton not available); it activates automatically on Linux/macOS CUDA

---

## Installation

### 1. Python version

Python **3.11** is required. Python 3.14 breaks gensim (C extension incompatibility).

```bash
# Verify
python --version   # should print Python 3.11.x
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch (choose your CUDA version)

```bash
# CUDA 12.4 (recommended, newer drivers):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (RTX 3080 with older drivers):
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only / Apple Silicon:
pip install torch
```

Check your CUDA version: `nvidia-smi` (shown in the top-right corner).

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Data Pipeline (`phase1.py`)

Processes the Ubuntu Dialogue Corpus from raw CSV into training-ready token ID files. Run **once** before training.

```bash
python phase1.py
```

**What it does (8 stages):**

| Stage | Output |
|---|---|
| 1–3 | Parse CSV, filter by length, build dialogue chains |
| 4 | Train/val/test split (90 / 5 / 5) |
| 5 | SentencePiece BPE model (`artifacts/stage5_spm.model`, vocab=16 000) |
| 6 | Tokenise all pairs → integer ID JSONL files (`stage6_*.jsonl`) |
| 7 | Train FastText 300d skip-gram embeddings on the tokenised corpus |
| 8 | Build embedding matrix (`stage8_embedding_matrix.npy`) |

**Artifacts produced** (required for training):
```
artifacts/
  stage5_spm.model
  stage6_vocab.json
  stage6_train_ids.jsonl   (~1.1 M pairs)
  stage6_val_ids.jsonl     (~47 K pairs)
  stage6_test_ids.jsonl    (~47 K pairs)
  stage8_embedding_matrix.npy
```

---

### Step 2 — Corpus Analysis (`analyze_data.py`)

Optional. Prints vocabulary statistics, token length distributions, and dialogue turn analysis. Useful for understanding the dataset before training.

```bash
python analyze_data.py
```

---

### Step 3 — Training (`train.py`)

Trains **both** models sequentially (baseline first, then attention). Checkpoints, logs, and TensorBoard events are saved automatically.

```bash
# Smoke test — validates full code path in < 5 minutes (2 epochs, 10 K pairs):
python train.py --mode sample

# Full training run (20 epochs, ~1.1 M pairs):
python train.py
# or equivalently:
python train.py --mode full
```

**Training configuration** (from `config.py`):

| Setting | Value |
|---|---|
| Effective batch size | 256 × 2 accum = 512 |
| Epochs | 20 |
| LR | 3e-4 → ReduceLROnPlateau (patience=3, factor=0.5) |
| Label smoothing | 0.0 |
| Early stopping patience | 4 epochs (active after Phase 1) |
| TF schedule | Phase 1 (1–5): 1.0 → Phase 2 (6–12): 0.9→0.5 → Phase 3 (13+): 0.5 |
| Mixed precision | bf16 AMP (Ampere+) |
| Data loading | BucketBatchSampler (groups by length) + prefetch_factor=4 |

**Checkpoints saved to `checkpoints/`:**
- `baseline_best.pt` — best validation loss
- `baseline_last.pt` — end of last completed epoch
- `attention_best.pt` / `attention_last.pt` — same for attention model

**Monitor training:**
```bash
tensorboard --logdir tb_logs/
```

**Alternative launcher (`run.py`):**
```bash
python run.py train --mode sample
python run.py train --mode full
python run.py phase1
```

---

### Step 4 — Evaluation (`evaluate.py`)

Evaluates trained checkpoints on the held-out test set. Computes BLEU-1/2/3/4, ROUGE-L, and BERTScore F1.

```bash
python evaluate.py
```

Results are printed to stdout and saved to `logs/`.

---

### Step 5 — Training Plots (`report/plot_training.py`)

Generates 6 publication-ready figures from training history for your report. Run after training completes.

```bash
python report/plot_training.py
```

Figures saved to `report/figures/` as both PNG (300 DPI) and PDF (vector):

| Figure | Description |
|---|---|
| `fig1_loss_curves` | Train + validation loss — both models, phase bands |
| `fig2_perplexity` | Validation perplexity with fill |
| `fig3_lr_tf_schedule` | LR decay (log scale) + TF ratio (2-panel) |
| `fig4_overfit_gap` | Generalisation gap (val − train loss) |
| `fig5_model_comparison` | Per-epoch val loss + cumulative best (2-panel) |
| `fig6_summary_table` | Key metrics table with best model highlighted |

**Preview without checkpoints (uses synthetic data):**
```bash
python report/plot_training.py --demo
```

---

### Step 6 — Chat Inference (`chatv2.py`)

Interactive chat comparing both trained models side by side.

```bash
python chatv2.py
```

Optional arguments:
```
--checkpoint-dir PATH   Directory containing *_best.pt files (default: checkpoints/)
--artifact-dir   PATH   Directory containing stage5_spm.model (default: artifacts/)
--beam-width     INT    Beam search width (default: 5; set to 1 for greedy)
```

---

## Architecture

```
Input tokens (context)
       │
  [Embedding 300d]  ← FastText pretrained, fine-tuned, shared encoder/decoder
       │
  [BiLSTM × 2]      ← enc_hidden_dim=512/dir → 1024 effective
       │
  [EncoderDecoderBridge]  ← projects (h_n, c_n) to decoder initial state
       │
  ┌────┴────────────────────────┐
  │ Baseline Decoder            │  Attention Decoder
  │ fixed context = enc_out[-1] │  Bahdanau: dynamic context per step
  └─────────────────────────────┘
       │
  [Output projection → vocab logits]
       │
  [SentencePiece BPE detokenise]
```

**Key design choices:**
- Shared embedding (weight tying) between encoder and decoder
- 3-phase teacher forcing schedule (prevents exposure bias)
- BucketBatchSampler: groups sequences by length → reduces padding by ~67%
- Phase 2 best_val_loss reset: prevents premature early stopping at TF transition

---

## Configuration

All hyperparameters live in `config.py`. Override any value with `CONFIG.update({...})` before training.

Key sections:
- `_ARCHITECTURE` — model dimensions, dropout
- `_TRAINING` — batch size, LR, epochs, patience
- `_TF_SCHEDULE` — teacher forcing phase boundaries
- `_LR_SCHEDULER` — ReduceLROnPlateau patience and factor
- `_PATHS` — all output directories (auto-created)

---

## Reproducibility

All runs use `seed=42`. Set via `set_seed()` in `config.py`, which seeds Python `random`, NumPy, and PyTorch. DataLoader workers are seeded via `worker_init_fn` in `dataset.py`.
