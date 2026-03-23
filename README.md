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
├── phase1.py           # Data pipeline  (CSV → tokenise → BPE → embeddings, 8 stages)
├── analyze_data.py     # Corpus statistics, token fertility, UNK forensics
├── train.py            # Model training  (baseline + attention, sequential, 20 epochs)
├── finetune.py         # Fine-tune trained checkpoints with lower TF floor (→ 0.0)
├── infer.py            # Greedy + beam search inference on 12 test prompts
├── models.py           # Architecture definitions (encoder, bridge, decoders, seq2seq)
├── dataset.py          # DataLoader, BucketBatchSampler, collation
├── config.py           # Single source of truth for all hyperparameters + GPU profiles
├── notebooks/          # Jupyter notebooks covering pipeline stages and training analysis
├── data/               # Raw corpus (Ubuntu Dialogue Corpus CSV)
├── artifacts/          # Generated artifacts (vocab, SPM model, embeddings, IDs)
├── checkpoints/        # Model checkpoints (*_best.pt, *_last.pt, *_history.json)
├── reports/            # Inference results (JSON / text)
└── logs/               # Text run logs
```

---

## Hardware

Developed and tested on two platforms — `config.py` auto-detects GPU and applies the appropriate profile:

| Platform | GPU | Batch size | Grad accum | Effective batch |
|---|---|---|---|---|
| OpenShift container (Linux) | NVIDIA A100 80 GB SXM4 | 1024 | ×1 | 1024 |
| Windows 11 development | NVIDIA RTX 3080 12 GB | 256 | ×2 | 512 |

- bf16 AMP is enabled automatically on Ampere+ GPUs
- Falls back to fp32 on older hardware
- `torch.compile` is disabled on Windows (Triton not available); activates automatically on Linux CUDA

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
| 1–3 | Parse CSV, filter by length/quality/domain, build dialogue chains |
| 4 | Train/val/test split (90 / 5 / 5) — temporal split |
| 5 | SentencePiece BPE model (`artifacts/stage5_spm.model`, vocab=32 000) |
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

Optional. Prints vocabulary statistics, token fertility, UNK forensics, and length distributions. Useful for understanding the dataset before training.

```bash
python analyze_data.py \
  --jsonl artifacts/stage6_train_ids.jsonl \
  --spm   artifacts/stage5_spm.model \
  --sample 5000 --unk-sample 50000
```

---

### Step 3 — Training (`train.py`)

Trains **both** models sequentially (baseline first, then attention). Checkpoints and logs are saved automatically.

```bash
python train.py
```

**Training configuration** (from `config.py`):

| Setting | Value |
|---|---|
| Batch size | 1024 (A100) / 256 (RTX 3080) — auto-detected |
| Effective batch size | 1024 (A100) / 512 (RTX 3080, grad accum ×2) |
| Epochs | 20 |
| Optimizer | AdamW (LR=3e-4, weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (patience=2, factor=0.5, min_lr=1e-5) |
| Label smoothing | 0.1 |
| Early stopping patience | 6 epochs (active from Phase 2 onward, epoch > 3) |
| TF schedule | Phase 1 (1–3): 1.0 → Phase 2 (4–12): 0.9→0.5 → Phase 3 (13–20): 0.5 |
| Gradient clipping | 1.0 |
| Mixed precision | bf16 AMP (Ampere+) |
| Data loading | BucketBatchSampler (groups by length, reduces padding overhead) |

**Checkpoints saved to `checkpoints/`:**
- `baseline_best.pt` — best validation loss
- `baseline_last.pt` — end of last completed epoch
- `attention_best.pt` / `attention_last.pt` — same for attention model
- `*_history.json` — full per-epoch metric history (loss, BLEU, Token F1, attn entropy, LR, TF ratio)

---

### Step 4 — Fine-tuning (`finetune.py`)

Fine-tunes a trained checkpoint with a progressively lower teacher-forcing floor to address exposure bias. Run after `train.py` completes.

```bash
# Fine-tune baseline with TF annealing 0.5 → 0.3:
python finetune.py --model baseline --tf-floor 0.30 --lr 1e-4 --epochs 12

# Fine-tune attention all the way to TF=0.0:
python finetune.py --model attention --tf-floor 0.0 --lr 5e-5 --epochs 15

# Fine-tune both models:
python finetune.py --model both --lr-schedule cosine
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `baseline` | `baseline`, `attention`, or `both` |
| `--tf-floor` | `0.30` | TF ratio to anneal to |
| `--tf-start` | `0.50` | TF ratio at start of fine-tuning |
| `--anneal-epochs` | `5` | Epochs over which TF decays |
| `--lr` | `1e-4` | Learning rate (lower than training) |
| `--epochs` | `12` | Total fine-tune epochs |
| `--lr-schedule` | `cosine` | `cosine`, `constant`, or `plateau` |
| `--patience` | `0` | Early stopping patience (0 = disabled) |

**Fine-tune checkpoints saved to `checkpoints/`:**
- `baseline_ft_best.pt` / `baseline_ft_best_gen.pt` / `baseline_ft_last.pt`
- `attention_ft_best.pt` / `attention_ft_best_gen.pt` / `attention_ft_last.pt`

---

### Step 5 — Inference (`infer.py`)

Runs greedy and beam search decoding on 12 held-out test prompts and saves results.

```bash
# Print results as a formatted table:
python infer.py --beam_size 5 --output text

# Save results as JSON (for programmatic use):
python infer.py --output json
```

Results saved to `reports/inference_results.json`.

---

### Step 6 — Training Analysis (Notebook)

Open `notebooks/nb_train_pipeline.ipynb` to visualise training metrics and generate publication-ready figures:

- Train / validation loss curves (both models, phase bands)
- Generation loss (TF=0 evaluation)
- TF ratio + LR schedule overlay
- BLEU-4 and Token F1 per epoch
- Prediction length and active tokens
- Attention entropy (attention model only)
- Checkpoint selection scatter (val_loss vs gen_loss)

---

## Architecture

```
Input tokens (context)
       │
  [Embedding 300d]  ← FastText pretrained, fine-tuned, shared encoder/decoder
       │
  [BiLSTM × 2]      ← enc_hidden_dim=512/dir → 1024 effective
       │
  [EncoderDecoderBridge]  ← projects (h_n, c_n) to decoder initial state via learned linear layers
       │
  ┌────┴──────────────────────────────┐
  │ Baseline Decoder                  │  Attention Decoder
  │ fixed context = last encoder step │  Bahdanau additive: dynamic context per step
  └───────────────────────────────────┘
       │
  [Projection bottleneck: 2048 → 512 → vocab]   ← 73% param reduction vs direct projection
       │
  [SentencePiece BPE detokenise]
```

**Key design choices:**
- Shared embedding (weight tying) between encoder and decoder
- 3-phase teacher forcing schedule prevents exposure bias while keeping fair ablation
- BucketBatchSampler: groups sequences by length → reduces padding overhead
- Phase 2 best_val_loss reset: prevents premature LR halving at TF transition
- Attention precomputation: keys cached once per decode loop for efficiency
- Orthogonal LSTM weight_hh initialisation — preserves gradient magnitudes
- Forget gate bias = 1.0 (Jozefowicz, 2015) — improves long-range memory retention
- Atomic checkpointing: `.tmp` → `os.replace()` prevents corruption on crash
- Post-EOS masking in generation loss: only counts tokens before first EOS

**Parameter counts (approximate):**

| Component | Params |
|---|---|
| Shared embeddings (32k × 300) | ~9.6 M |
| Encoder (BiLSTM × 2) | ~12.6 M |
| Bridge (linear projections) | ~2.1 M |
| Decoder LSTM + projection | ~19.7 M |
| Attention mechanism | ~0.5 M |
| **Total (attention model)** | **~44 M** |

---

## Configuration

All hyperparameters live in `config.py`. Override any value with `CONFIG.update({...})` before training.

Key settings:
- `batch_size` / `grad_accum_steps` — overridden by GPU profile (A100 or RTX 3080)
- `epochs`, `patience`, `label_smoothing` — training controls
- `lr`, `lr_scheduler_patience`, `lr_scheduler_factor` — learning rate schedule
- `phase1_end`, `phase2_end` — TF schedule phase boundaries (epochs 3 and 12)
- All output directories are auto-created on first run

---

## Results

Training results from `checkpoints/*_history.json` (20 epochs each):

| Metric | Baseline | Attention |
|---|---|---|
| Best val loss | 5.131 | **5.083** |
| Best gen loss (TF=0) | 5.946 | **5.926** |
| Best BLEU-4 | 0.865 | **1.073** |
| Best Token F1 | 15.99 | 15.97 |
| Mean attn entropy | — | ~2.03 nats |

Attention model achieves lower validation loss and higher BLEU-4, isolating the contribution of the attention mechanism under identical training conditions.

---

## Reproducibility

All runs use `seed=42`. Set via `set_seed()` in `config.py`, which seeds Python `random`, NumPy, and PyTorch. DataLoader workers are seeded via `worker_init_fn` in `dataset.py`.
