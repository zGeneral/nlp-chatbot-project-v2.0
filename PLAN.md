# Clean-From-Scratch Implementation Plan
## MSc AI Final Project — Seq2Seq Chatbot (Ubuntu Dialogue Corpus)

---

## Why From Scratch

The prior codebase (`phase1.py`, `models.py`, `train.py`, `dataset.py`) evolved through
multiple fix iterations and accumulated incompatible design decisions. Rather than patching,
we are building cleanly with all lessons learned baked in from line 1.

Key lessons:
1. **TF decay killed quality** — teacher forcing decay to 0.25 caused perplexity-quality
   inversion. The model learned to predict average responses. Solution: TF=1.0 for ≥75%
   of training; only a gentle taper in the final epochs.
2. **30k word vocab is too large** — 30,000-word softmax floods generic tokens.
   Solution: SentencePiece BPE at 16k–20k vocab (half the output space, zero OOV).
3. **Unidirectional encoder misses context** — bidirectional encoder reads context
   forwards AND backwards, producing richer representations for attention to align to.
4. **LR was too aggressive** — 3e-3 with complex warmup/schedule caused the
   "Sisyphus trap". Solution: flat 3e-4 with AdamW; simple ReduceLROnPlateau only.
5. **Gradient clipping was too tight** — clip=0.5 suppressed valid learning signals.
   Solution: clip=1.0.

What we carry forward (confirmed working):
- Temporal split (methodologically superior to random split)
- Quality filters (paste, IRC, temporal gap, bot, non-English, echo)
- FastText pretrained embeddings (trained on BPE-tokenised corpus)
- Bahdanau attention (additive, richer alignment than Luong dot-product)
- Projection bottleneck (1024→512→vocab, halves output projection params)
- Granular dropout (embed=0.3, lstm=0.5, out=0.4)
- AdamW optimizer + AMP bf16
- Top-p sampling + n-gram/stutter blocking in inference

---

## File Map

```
new/
├── PLAN.md              ← This file. Architecture decisions + implementation guide.
├── config.py            ← Single source of truth for ALL hyperparameters.
├── phase1.py            ← Data pipeline: clean, filter, split, BPE vocab, FastText.
├── dataset.py           ← PyTorch Dataset + DataLoader with BPE token IDs.
├── models.py            ← Encoder (bidirectional), Bahdanau Attention, Decoder, Seq2Seq.
├── train.py             ← Training loop, evaluation, checkpointing.
├── evaluate.py          ← BLEU corpus evaluation + manual sample generation.
├── chat.py              ← Interactive inference (greedy / top-p).
└── notebooks/
    ├── train_baseline.ipynb   ← Colab: baseline model (no attention).
    └── train_attention.ipynb  ← Colab: attention model.
```

---

## Phase 1 — Data Pipeline (`phase1.py`)

### Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Tokenization | SentencePiece BPE | Subword vocab handles technical terms naturally; no OOV |
| Vocab size | 16,000–20,000 pieces | Half of prior 30k softmax difficulty; 2× more expressive than osamadev's 8k |
| User-defined symbols | `<pad>`, `<unk>`, `<sos>`, `<eos>` | Same special tokens as before, compatible with models.py |
| Embedding training | FastText 300d on BPE-tokenised corpus | Pretrained advantage; trained on the same BPE text the model will see |
| Data split | Temporal (Ubuntu v2.0 date boundaries) | Prevents future-leakage; methodologically correct for dialogue |
| Train pairs | ~1.5M (down from 4.57M) | 4.57M flooded generic exchanges; quality > quantity |
| Quality filters | Keep all 10+ existing filters | Paste, IRC, temporal gap, echo, bot, non-English, repetitive |
| Additional filter | Response diversity filter | Caps how many times any identical response appears in training set |
| Max context | 100 tokens, 8 turns | Tighter window forces specificity; long contexts dilute attention |
| Max response | 40 tokens | Shorter targets reduce chance of generic padding |

### Stages

```
Stage 1 — Load raw Ubuntu Dialogue Corpus (CSV files → pickle)
Stage 2 — Clean text + apply all quality filters (parallel, chunked)
Stage 3 — Temporal split (train / val / test by date boundary)
Stage 4 — Generate context-response pairs + response diversity filter
Stage 5 — Train SentencePiece BPE model on training text (16k–20k vocab)
Stage 6 — Encode all pairs to BPE token IDs + build vocab JSON
Stage 7 — Train FastText 300d on BPE-tokenised training corpus
Stage 8 — Build embedding matrix aligned to BPE vocab
```

### Key change from old Phase 1
Old: word-level vocab built first, then FastText trained on raw text.
New: SentencePiece trained first (on raw text), then FastText trained on
**BPE-tokenised text** so embedding vectors correspond to BPE pieces, not words.

---

## Phase 2 — Architecture (`models.py`)

### Encoder
- **Bidirectional** 2-layer LSTM, `hidden_dim=512` per direction (→ 1024 total)
- `enforce_sorted=False` (no manual sort-gather-unsort)
- Granular dropout: embed=0.3, lstm=0.5, output=0.4
- Final hidden state: concat forward+backward → bridge linear → decoder hidden

### Attention
- **Bahdanau additive attention** (kept — richer than Luong dot-product)
- Energy: `tanh(W_enc * enc_out + W_dec * dec_hidden)`
- Softmax over source sequence → context vector

### Decoder
- Unidirectional 2-layer LSTM
- Input: `[embedded_token ; context_vector]` (concat before LSTM step)
- Projection bottleneck: `hidden_dim(1024) → proj_dim(512) → vocab_size`
- Shared embedding with encoder (weight tying)

### Baseline (no-attention) Decoder
- Same decoder but context = encoder final hidden state only (no alignment)
- Otherwise identical for fair comparison

### Parameter count estimate (R4 — corrected from smoke test)
- Bidirectional encoder: ~14.4M (includes 300×16k embedding + 4-layer bidir LSTM)
- EncoderDecoderBridge: ~2.1M (Linear 1024→1024 × 4 directions × layers)
- Attention decoder: ~32.6M (LSTM + projection bottleneck + attn mechanism)
- **Total: ~44.3M** (attention) / **~43.8M** (baseline)
- Confirmed by smoke test output from `models.py`

---

## Phase 2 — Training (`train.py`)

### CONFIG

```python
CONFIG = {
    # Tokenization
    "vocab_size":         16000,   # or 20000 — set after Phase 1

    # Architecture
    "hidden_dim":         512,     # per LSTM direction (bidir → 1024 effective)
    "num_layers":         2,
    "projection_dim":     512,
    "dropout_embed":      0.3,
    "dropout_lstm":       0.5,
    "dropout_out":        0.4,
    "embed_dim":          300,

    # Training
    "learning_rate":      3e-4,    # flat, conservative, no warmup needed
    "weight_decay":       1e-5,
    "max_grad_norm":      1.0,     # gentler than prior 0.5
    "batch_size":         256,
    "grad_accum_steps":   2,       # effective batch = 512
    "num_epochs":         20,
    "amp_dtype":          "bfloat16",

    # Teacher Forcing schedule (KEY DECISION)
    # Epochs 1–15:  TF = 1.0  (burn in sharp patterns)
    # Epochs 16–18: TF = 0.8  (gentle awareness of own outputs)
    # Epochs 19–20: TF = 0.5  (minimal adaptation, patterns already locked)
    "tf_schedule": {
        "phase1_end":   15,    # TF=1.0 until this epoch
        "phase2_end":   18,    # TF=0.8 until this epoch
        "phase3_tf":    0.5,   # TF for final epochs
    },

    # LR scheduler
    "lr_scheduler_patience": 3,
    "lr_scheduler_factor":   0.5,

    # Loss
    "label_smoothing":    0.0,     # OFF during TF=1.0; optionally 0.05 in final epochs
    "pad_idx":            0,

    # Data
    "max_ctx_tokens":     100,
    "max_resp_tokens":    40,
    "num_workers":        4,

    # Checkpoints
    "checkpoint_dir":     "phase2_checkpoints",
    "tensorboard_dir":    "phase2_tb_logs",
}
```

### Training loop principles
1. No warmup — LR=3e-4 is already conservative; warmup would waste epochs
2. NaN-triggered LR halving retained (safety net)
3. Periodic checkpoints every 2000 steps
4. Best checkpoint saved only when val loss improves
5. Atomic saves (`tmp` → `rename`) to prevent corruption
6. TF schedule is a pure lookup by epoch, no decay formula

---

## Phase 3 — Evaluation (`evaluate.py`)

- BLEU-1/2/3/4 (corpus-level, sacrebleu 13a tokeniser — not word-level)
  - Note (AC2-I2): BLEU-1/2/3 each have **independent brevity penalties** — they are NOT
    the n-gram precision components of BLEU-4. This is consistent with how the osamadev
    reference benchmark was computed. Documented in `evaluate.py` and this write-up.
- ROUGE-L F1, Distinct-1/2 (response diversity), BERTScore F1 (semantic similarity)
- All corpus metrics use **greedy decoding** for benchmark comparability (osamadev also uses greedy)
- Manual samples use **top-p decoding** (diversity, qualitative quality)
  - Note (AC2-I3): The decode-strategy asymmetry is intentional and documented in `bleu_results.json`
    under the `"decode_strategy"` key. The write-up must state: "BLEU/ROUGE/Distinct/BERTScore
    are reported under greedy decoding for benchmark comparability; human evaluation uses
    top-p sampling to assess best-case response quality."
- Manual rubric (G3): each sample rated on: **Fluency** (1–5) · **Coherence** (1–5) · **Relevance** (1–5) · **Specificity** (1–5, penalise generic "ok/yes" replies)
  - Limitation (AC2-I4): single annotator — cite as methodological limitation in write-up.
    Ideally: second rater on 20–25 samples with Cohen's κ. At minimum: acknowledge in §Limitations.
- Attention heatmap: single-example Bahdanau weight visualisation (attention model only)
- Side-by-side comparison table: baseline vs attention + osamadev reference

## Research Hypotheses (AC2-I5)

**H1:** The Bahdanau attention model will outperform the parameter-near baseline on BLEU-4,
ROUGE-L, and BERTScore because the attention mechanism allows the decoder to selectively
focus on relevant context tokens at each step, avoiding the information bottleneck of a
single fixed context vector.

**H2:** The attention model will achieve higher Distinct-2 than the baseline because richer,
position-sensitive context representations reduce the tendency toward generic/repetitive responses.

**Null hypothesis:** Both models perform equivalently on all metrics (no statistically significant
difference). Confirmation or rejection will be reported in the evaluation chapter.

## Benchmark Reference (R1/R2 — framing)

**Reference model:** osamadev/seq2seq-chatbot (LSTM+ATTN, 8k SentencePiece, greedy)  
BLEU-4 = 0.1386 · BLEU-1 = 0.4400 · ROUGE-L = 0.0922

**Important disclaimer:** direct numeric comparison is invalid (F3) — different  
tokeniser (word-level vs 13a subword), different vocab size (8k vs 16k), different  
data split. We use this as a *directional* reference, not a strict threshold.

**Our target:** BLEU-4 ≥ 0.14, Distinct-2 ≥ 0.10, BERTScore F1 ≥ 0.80  
Justified by: bidirectional encoder + BPE 16k + FastText + TF=1.0 regime

## TF Schedule Rationale (AC2-M1)

The 15/3/2 epoch split for TF=1.0/0.8/0.5 was chosen as follows:
- **75% at TF=1.0 (epochs 1–15):** mirrors osamadev (10/10 = 100% TF=1.0) while leaving room for a
  brief taper to reduce exposure bias before inference. With 20 epochs, 15 gives the model the same
  sharp-pattern burn-in period as osamadev's full run.
- **3 epochs at TF=0.8 (16–18):** gentle introduction to own-prediction errors. Fewer than 3 epochs
  is unlikely to shift learned behaviour; more than 5 risks re-introducing the quality-inversion seen
  in prior runs where TF decayed too early.
- **2 epochs at TF=0.5 (19–20):** minimum viable final adaptation. Patterns are locked in by epoch 15;
  these epochs provide marginal robustness to inference-time distribution shift.

## Response Diversity Cap (AC2-M2)

`max_response_occurrences = 500`: cap applied per unique response string in the training set.
Responses like "yes", "ok", "try it" appeared 5000+ times in the raw Ubuntu corpus, creating a
strong mode-collapse attractor. The value 500 was chosen to be ≥10× the mean response frequency
while still removing the most extreme repeaters. Stage 4 saves `stage4_stats.json` which includes
`"diversity_cap_removed"` and `"unique_responses_capped"` counts — fill these in with the actual
numbers after running Phase 1.

## Key References (R3)

- Bahdanau et al. (2015) — Neural Machine Translation by Jointly Learning to Align and Translate  
- Liu et al. (2016) — How NOT To Evaluate Your Dialogue System (BLEU inadequacy for dialogue)  
- Bojanowski et al. (2017) — Enriching Word Vectors with Subword Information (FastText)  
- Li et al. (2016) — A Diversity-Promoting Objective Function for Neural Dialogue (Distinct-N)  
- Zhang et al. (2020) — BERTScore: Evaluating Text Generation with BERT

---

## Implementation Order

1. `config.py` — define all constants first; everything else imports from here
2. `phase1.py` — data pipeline (run once, produces artifacts)
3. `dataset.py` — DataLoader (depends on Phase 1 artifacts)
4. `models.py` — architecture (depends on config)
5. `train.py` — training loop (depends on models + dataset)
6. `evaluate.py` — evaluation (depends on trained checkpoints)
7. `chat.py` — inference (depends on trained checkpoints)
8. `notebooks/train_baseline.ipynb` — Colab notebook for baseline
9. `notebooks/train_attention.ipynb` — Colab notebook for attention model
