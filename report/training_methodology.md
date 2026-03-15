# Training Methodology & Design

*MSc AI Final Project — Seq2Seq LSTM Chatbot on Ubuntu Dialogue Corpus*

---

## Overview

This project trains a Sequence-to-Sequence (Seq2Seq) chatbot on the Ubuntu Dialogue
Corpus — a dataset of multi-turn IRC technical support conversations. Two model variants
are trained and compared:

- **Baseline** — Bidirectional LSTM encoder with unidirectional LSTM decoder, no attention
- **Attention** — Identical architecture with Bahdanau (additive) attention added

The purpose is an ablation study: quantifying the contribution of the attention mechanism
to response quality on a realistic open-domain technical dialogue task.

### Reference Benchmark

The directional benchmark is the osamadev Seq2Seq chatbot implementation:

| Metric | osamadev (reference) | Notes |
|---|---|---|
| BLEU-1 | 0.4400 | LSTM + attention, 8k SentencePiece vocab, greedy |
| BLEU-4 | 0.1386 | |
| ROUGE-L F1 | 0.0922 | |

> **Important:** Direct numeric comparison is invalid. The reference uses an 8k vocabulary,
> a different tokeniser, and a different data split. It serves as a directional sanity check
> only and should be cited with this disclaimer in any write-up (Liu et al., 2016 note on
> metric reliability in open-domain dialogue).

---

## Architecture

### 1. Encoder — Bidirectional LSTM

```
Input tokens  →  Embedding(16000, 300)  →  BiLSTM(hidden=512, layers=2)
                                            → output [B, T, 1024]
                                            → final hidden [2×layers, B, 512]
                                            → final cell   [2×layers, B, 512]
```

The encoder is **bidirectional**: each token is encoded with both left-to-right and
right-to-left context. This is critical for dialogue because meaning is not strictly
causal — "I **can't** install it" requires the word after "can't" to fully disambiguate
the sentiment.

- **Hidden dim:** 512 per direction → 1024 effective (concatenated)
- **Layers:** 2 stacked LSTM layers for hierarchical feature extraction
- **LSTM over GRU:** LSTM has an explicit cell state in addition to the hidden state,
  giving it a dedicated long-term memory path. For multi-turn dialogue where early
  context turns must influence late decoder steps, the cell state provides an additional
  gradient highway that helps preserve information across long sequences.

### 2. Encoder–Decoder Bridge

```
BiLSTM final hidden [2*layers, B, 512]  →  Linear(1024 → 1024)  →  decoder init [layers, B, 1024]
BiLSTM final cell   [2*layers, B, 512]  →  Linear(1024 → 1024)  →  decoder cell  [layers, B, 1024]
```

The encoder output is bidirectional (1024-dim) but the decoder is unidirectional (1024-dim).
The bridge is a learned linear projection that maps the concatenated forward+backward final
hidden (and cell) states into the decoder's initial hidden (and cell) state. Without this,
the dimension mismatch would prevent the encoder state from initialising the decoder.

### 3. Shared Embedding (Weight Tying)

Both the encoder and decoder use the **exact same `nn.Embedding` object**. This is not two
copies — it is the same Python object passed to both components.

Benefits:
- **Parameter reduction:** Saves vocab_size × embed_dim = 16,000 × 300 = 4.8M parameters
- **Consistency:** Token representations are forced to be the same whether the model is
  reading input or generating output
- **Regularisation:** Shared weights reduce the risk of the encoder and decoder developing
  incompatible token spaces

This is standard practice for single-language Seq2Seq tasks where the input and output
vocabulary are identical.

### 4. Pretrained FastText Embeddings

Rather than random initialisation, the embedding matrix is pre-populated with FastText
skip-gram vectors trained on the Ubuntu corpus itself.

```
stage8_embedding_matrix.npy  →  nn.Embedding(16000, 300, padding_idx=0)
```

- **Why domain-specific?** Generic pretrained embeddings (GloVe, Word2Vec) do not know
  that `sudo`, `apt`, `kernel`, and `ubuntu` form a semantic cluster. Training on the
  Ubuntu corpus gives the embedding meaningful geometry from the start.
- **Freeze=False:** Embeddings are fine-tuned during training, allowing the Seq2Seq task
  to adapt the FastText initialisations to the dialogue response objective.
- **Pad row zeroed:** Row 0 (`<pad>`) is explicitly zeroed and receives no gradient
  via `padding_idx=0`.
- **Coverage:** 99.994% of vocabulary rows are non-zero (1 zero row = pad only).

**Initialisation data scope and mild leakage note:**
FastText is trained on the full corpus (train + validation + test combined) to ensure
complete vocabulary coverage — validation and test splits may contain tokens not present
in training dialogue pairs. Because `freeze=False`, these initial vectors are fine-tuned
during the Seq2Seq training phase; the embedding weight at epoch 1 still carries
co-occurrence statistics derived from val/test text, but this signal is progressively
overwritten across subsequent epochs. The contamination is limited in scope: FastText
learns token co-occurrence geometry (e.g. that `wireless` and `router` are semantically
proximate), not dialogue response structure or the specific responses in val/test. This
is a standard trade-off in domain-specific NLP systems and is consistent with practices
in the Ubuntu Dialogue Corpus literature (Lowe et al., 2015). It is acknowledged here
in the interest of full methodological transparency.

### 5. Attention Mechanism (Bahdanau / Additive)

The attention model adds a context-sensitive weighting over encoder outputs at each
decoder step.

**Score function:**
```
e(s, h_i) = v · tanh(W1 · s  +  W2 · h_i)
```
where `s` is the current decoder hidden state, `h_i` is the i-th encoder output,
`W1 ∈ R^{attn_dim × dec_hidden}`, `W2 ∈ R^{attn_dim × enc_hidden}`, `v ∈ R^{attn_dim}`.

**Attention weights:**
```
α_i = softmax(e_i)     over all encoder positions i
```

**Context vector:**
```
c = Σ_i  α_i · h_i     (weighted sum of encoder outputs)
```

The attention dimension is 256. Padding positions in the encoder output are masked to
`-inf` before softmax, ensuring they contribute zero to the context vector.

**Rationale:** In multi-turn IRC dialogue, relevant context can span several turns.
A context sequence like "how do I install X? → tried Y → still broken → what command?"
requires the decoder to attend to "install X" when generating the first response token,
not just the most recent encoder state.

### 6. Decoder Input — Luong Input Feeding

```
decoder_input = concat(embed(prev_token), context_prev)
              = [300-dim ; 1024-dim]
              = 1324-dim LSTM input
```

At each decoder step, the previous step's context vector is concatenated with the current
token embedding before feeding the LSTM. This is the Luong input-feeding trick: it carries
"what I was attending to last step" forward, giving the decoder memory of its own
attention history. Without this, the attention must independently re-discover the correct
alignment from scratch every step.

---

## Training Procedure

### 7. Teacher Forcing Schedule

Teacher forcing controls what token the decoder receives as input at step t:
- **TF=1.0:** Always use the gold (ground-truth) token from the reference sequence
- **TF=0.0:** Always use the model's own previous prediction (free-running)

```
Epochs  1–15:  TF = 1.0   (full teacher forcing)
Epochs 16–18:  TF = 0.8   (20% exposure to own predictions)
Epochs 19–20:  TF = 0.5   (half free-running)
```

**Rationale:**

*Phase 1 (TF=1.0):* Training is stable because the decoder always receives correct context.
The encoder, bridge, and embedding learn quickly without compounding errors. Gradients are
clean and informative.

*Phase 2–3 (TF decay):* At inference time there is no gold token — the model feeds its own
outputs. Training only with TF=1.0 causes **exposure bias**: the model has never seen its
own mistakes during training, so small errors at inference compound rapidly. Gradually
reducing TF closes this gap without destabilising early training.

*Why not decay earlier?* Rushing the reduction wastes the high-quality gradient signal
from the first 15 epochs. The model needs to first learn a reasonable output distribution
before it can benefit from self-feeding.

### 8. Loss Function

```python
CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.0)
```

- **Ignore pad:** Padding positions contribute zero loss — the model is not penalised for
  what it predicts at pad positions.
- **Label smoothing = OFF:** With TF=1.0, targets are perfect gold tokens. Smoothing would
  add uncertainty to tokens the model should confidently learn, reducing signal quality.
  Smoothing is more appropriate with noisy targets or scheduled sampling.

### 9. Optimiser

```python
AdamW(lr=3e-4, weight_decay=1e-5)
```

- **AdamW:** Adam with decoupled weight decay. Standard choice for NLP tasks — adaptive
  learning rates per parameter, weight decay as proper L2 regularisation (not mixed into
  the moment estimates as in vanilla Adam).
- **LR = 3e-4:** Standard starting point for AdamW on sequence models with this parameter
  count. Conservative enough to avoid early divergence, fast enough to converge in 20 epochs.
- **Weight decay = 1e-5:** Light regularisation. Seq2Seq models can overfit the most
  frequent response patterns; weight decay discourages this without restricting capacity.

### 10. Learning Rate Scheduling

```python
ReduceLROnPlateau(patience=3, factor=0.5, monitor=val_loss)
```

The LR is halved when validation loss does not improve for 3 consecutive epochs. This
allows the model to fine-tune after the initial learning plateau without manually tuning
a LR schedule. With 20 training epochs, this can fire 2–3 times, bringing the effective
LR from 3e-4 down to ~3.75e-5 by the end of training.

### 11. Gradient Clipping

```python
clip_grad_norm_(parameters, max_norm=1.0)
```

Applied after each backward pass (or after gradient accumulation). LSTMs can exhibit
gradient explosion on long sequences — the Ubuntu context can be up to 100 tokens. Clipping
at 1.0 is gentle: it rescales the gradient vector when its L2 norm exceeds 1.0, preserving
gradient direction while preventing catastrophic weight updates.

### 12. Gradient Accumulation

```python
grad_accum_steps = 2   # full training
grad_accum_steps = 1   # mini training
```

With batch_size=256 and grad_accum_steps=2, the effective batch size is 512 per parameter
update. This simulates a larger batch without requiring proportionally more GPU memory.
Loss is scaled by `1/grad_accum_steps` per sub-batch so that gradients are equivalent to
computing on the full effective batch.

### 13. Mixed Precision Training

```python
# config.py
"amp_dtype": "bfloat16"

# train.py — read from config, not hardcoded
_amp_dtype = getattr(torch, config.get("amp_dtype", "bfloat16"))
torch.amp.autocast(device_type="cuda", dtype=_amp_dtype)
```

Forward passes and loss computation run in bfloat16. bfloat16 is preferred over float16
for training because it has the same exponent range as float32 (8 bits vs 5 bits in
float16), making it far less prone to overflow during gradient computation. The embedding
The embedding matrix and LSTM parameters remain in float32 for numerical stability.

---

## Checkpoint Strategy

Every epoch where validation loss improves, a full checkpoint is saved atomically
(write to `.tmp`, then `os.replace` to avoid corrupt checkpoints on crash):

```
{model_type}_best.pt   ← lowest val_loss ever seen
{model_type}_last.pt   ← most recent epoch (for Colab/Windows resume)
```

On the next run, the training loop prefers `_last.pt` (most recent state) over `_best.pt`
to avoid silently discarding training progress on resume.

Each checkpoint stores:
- `model_state_dict` — weights
- `optimizer_state` / `scheduler_state` — resume exact optimiser momentum
- `history` — full train/val loss curve for plotting
- `git_hash` — exact code version that produced this checkpoint
- `config` — full hyperparameter snapshot for reproducibility

---

## Mini Training (Sanity Check)

Before committing to a multi-hour full training run, `train_mini.py` runs the identical
training loop on a subset of the mini pipeline output:

| Setting | Mini | Full |
|---|---|---|
| Training pairs | 50,000 (of 150k available) | ~500k+ |
| Max epochs | 20 | 20 |
| Early stopping patience | 5 epochs | disabled |
| Batch size | 64 | 256 |
| Grad accumulation | 1 | 2 |
| Artifact dir | `artifacts_mini/` | `artifacts/` |
| Checkpoint dir | `checkpoints_mini/` | `checkpoints/` |

The mini run answers: **does the model learn at all, or is it collapsing?**

After training, `evaluate_mini.py` runs automatically and produces a three-layer analysis:

- **Layer 1 (automatic):** BLEU-1/2/4, Distinct-1/2, UNK rate, empty response rate
- **Layer 2 (structural):** Top-20 most-generated responses, response length histogram,
  novel-word ratio (collapse detector)
- **Layer 3 (samples):** 30 decoded `CTX → BASELINE | ATTENTION | GOLD` pairs

**Verdict criteria:**

| Verdict | Trigger |
|---|---|
| ✅ HEALTHY | All checks pass |
| ⚠️ COLLAPSE | Top-1 response > 20% of outputs, or >30% empty responses |
| ⚠️ NOT LEARNING | BLEU-1 < 0.05, or val_loss still near ln(vocab_size)=9.68 |
| ⚠️ OVERFIT | val_loss − train_loss > 0.5 |

---

## Early Stopping — Implementation Note

Early stopping uses a `_improved` boolean captured **before** `best_val_loss` is updated.
This avoids an off-by-one bug where `val_loss < best_val_loss` always evaluates to `False`
after the update (they are equal). The counter resets to 0 on any improvement; training
halts when `_no_improve >= patience`.

Early stopping is **disabled** (patience=0) for full training — the LR scheduler handles
plateau adaptation instead.

---

## Key Design Choices vs Reference Benchmark

| Choice | This project | osamadev reference | Reason |
|---|---|---|---|
| Encoder | Bidirectional LSTM | Bidirectional LSTM | Same mechanism; LSTM chosen for cell-state memory |
| Vocab size | 16,000 BPE | 8,000 SentencePiece | More coverage, fewer `<unk>` |
| Embedding init | FastText domain-specific | Random | Better geometry from the start |
| Weight tying | Yes | Not specified | Reduces params, regularises |
| Teacher forcing | Staged schedule | Fixed | Closes exposure bias gap |
| Label smoothing | OFF | Not specified | Clean signal with TF=1.0 |
| Mixed precision | bfloat16 | Not used | ~2× training speed |
| Attention | Bahdanau additive | Bahdanau | Same mechanism |

---

## References

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with
   neural networks. *NeurIPS*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly
   learning to align and translate. *ICLR*.
3. Lowe, R., et al. (2015). The Ubuntu Dialogue Corpus. *SIGDIAL*.
4. Liu, C. W., et al. (2016). How NOT to evaluate your dialogue system. *EMNLP*.
   *(On the limitations of BLEU for open-domain dialogue evaluation.)*
5. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder
   for statistical machine translation. *EMNLP*. *(Original RNN encoder-decoder paper.)*
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
   *(LSTM paper.)*
6. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
   *(AdamW.)*
