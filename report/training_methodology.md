# Training Methodology

Both models (baseline and attention) are trained with identical
hyperparameters on the same 1 103 539 pair dataset, ensuring that any
performance difference isolates the architectural contribution of
attention rather than training regime differences.

---

## Hardware & Runtime

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 12 GB (Ampere, CC 8.6) |
| Precision | bfloat16 (AMP via `torch.amp.autocast`) |
| Peak GPU memory | 5.42 GB / 12 GB |
| Effective batch | 256 × 2 accumulation steps = **512** |
| Epoch duration | ~11 min · 20 epochs · 2 models ≈ **~7.5 hours** |

---

## Configuration Reference

| Hyperparameter | Value | Key |
|---|---|---|
| Optimizer | AdamW | — |
| Peak LR | 3e-4 | `learning_rate` |
| Weight decay | 1e-5 | `weight_decay` |
| Adam β₁, β₂, ε | 0.9, 0.999, 1e-8 | (PyTorch defaults) |
| LR scheduler | ReduceLROnPlateau | — |
| Scheduler factor | 0.5 | `lr_scheduler_factor` |
| Scheduler patience | 3 epochs | `lr_scheduler_patience` |
| Minimum LR | 1e-5 | `lr_min` |
| Loss function | CrossEntropyLoss | — |
| Label smoothing | 0.0 (off) | `label_smoothing` |
| Pad ignore index | 0 | `pad_idx` |
| Gradient clip | max_norm = 1.0 (L2) | `max_grad_norm` |
| Grad accumulation | 2 steps | `grad_accum_steps` |
| Epochs | 20 | `num_epochs` |
| Batch size | 256 | `batch_size` |
| DataLoader workers | 4 | `num_workers` |
| Early stopping | patience = 4 | `patience` |
| Seed | 42 | `seed` |

---

## Training Loop Overview

```
+- EPOCH LOOP (1 -> 20) -----------------------------------------------------+
|                                                                             |
|  tf_ratio  <-  get_tf_ratio(epoch)   (see Appendix A)                     |
|  optimizer.zero_grad()                                                      |
|                                                                             |
|  +- BATCH LOOP --------------------------------------------------------+   |
|  |                                                                      |   |
|  |  with autocast(cuda, bf16):                                         |   |
|  |      logits = model(src, src_len, trg, tf_ratio)                   |   |
|  |      loss   = CrossEntropyLoss(logits, trg[:,1:])  <- pad masked   |   |
|  |                                                                      |   |
|  |  if loss is NaN -> zero_grad, skip batch        <- NaN guard       |   |
|  |                                                                      |   |
|  |  (loss / accum_steps).backward()               <- scale grad       |   |
|  |                                                                      |   |
|  |  if (step % 2 == 0) or last_batch:                                 |   |
|  |      clip_grad_norm_(model, max_norm=1.0)                          |   |
|  |      if grad_norm is NaN -> zero_grad, skip step                   |   |
|  |      optimizer.step()                                               |   |
|  |      optimizer.zero_grad(set_to_none=True)                         |   |
|  |      if global_step % 2000 == 0 -> save step checkpoint            |   |
|  |                                                                      |   |
|  +----------------------------------------------------------------------+   |
|                                                                             |
|  val_loss <- evaluate(model, val_loader, tf=0.0)   <- no TF in val        |
|  scheduler.step(val_loss)                          <- may halve LR        |
|                                                                             |
|  if epoch == 6: reset best_val_loss, no_improve    <- Phase 2 reset       |
|  if val_loss < best -> save best.pt                                        |
|  else: no_improve += 1; if >= 4 -> early stop                             |
|  save last.pt  (always)                            <- resume guard        |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## Optimiser & Regularisation

**AdamW** is used with peak LR 3e-4, no warm-up. Weight decay (1e-5) is
applied via AdamW's decoupled L2 penalty rather than standard Adam's
gradient coupling, preventing decay from distorting the adaptive
learning rate estimates. Gradient clipping (max_norm = 1.0) prevents
exploding gradients common in deep RNNs.

**ReduceLROnPlateau** halves the learning rate after 3 consecutive
epochs of non-improving validation loss, with a floor of 1e-5. The
scheduler is called once per epoch after validation.

---

# Appendix

## A. Teacher Forcing Schedule

Teacher forcing (TF) controls how much the decoder relies on
ground-truth tokens as inputs during training. At TF = 1.0 the decoder
always receives the correct previous token; at TF = 0.0 it uses its
own previous prediction (autoregressive / free-running).

### A.1 Three-Phase Schedule

```
TF
1.0 |*-----*
    |       \  Phase 2: linear decay 0.9->0.5
0.9 |        *
    |         \
0.7 |          *
    |            \
0.5 |             *-----------* Phase 3: floor
    +---------------------------------- epoch
      1  2  3  4  5  6  7  8  9 10 11 12 13 14 ... 20
      |<-- Phase 1 -->|<---- Phase 2 ---->|<-- Phase 3 -->|
```

| Phase | Epochs | TF Ratio | Purpose |
|---|---|---|---|
| 1 | 1 - 5 | 1.00 (fixed) | Stable early convergence with full supervision |
| 2 | 6 - 12 | 0.90 -> 0.50 (linear) | Gradually shift to free-running; close exposure bias |
| 3 | 13 - 20 | 0.50 (fixed floor) | Half ground-truth, half autoregressive |

### A.2 Phase 2 Interpolation Formula

```
tf(epoch) = 0.9 - (epoch - 6) x (0.4 / 6)
```

| Epoch | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|
| TF | 0.900 | 0.833 | 0.767 | 0.700 | 0.633 | 0.567 | 0.500 |

### A.3 Sampling Behaviour

TF ratio is computed once per epoch and applied **per token**
stochastically inside the decoder's forward pass:

```
use_teacher = random() < teacher_forcing_ratio
decoder_input = trg[t]  if use_teacher  else  top1_prediction
```

This means within a single batch, some tokens receive ground-truth
input and others receive the model's own prediction — creating a
mixed-supervision signal that smooths the transition.

---

## B. Gradient Accumulation & Mixed Precision

### B.1 Effective Batch Construction

Physical batch size is constrained by GPU VRAM. Two accumulation steps
double the effective batch without exceeding memory:

```
Batch 0 --> forward -> (loss/2).backward()  <- gradients accumulate
Batch 1 --> forward -> (loss/2).backward()  <- gradients accumulate
             clip -> optimizer.step() -> zero_grad
             effective batch = 256 x 2 = 512 samples
```

Loss is divided by `grad_accum_steps = 2` before `.backward()` so
accumulated gradients equal the mean over 512 samples. The unscaled
loss is used for all logging and reporting.

### B.2 bfloat16 vs float16

bfloat16 (bf16) is chosen over fp16 for three reasons:

| Property | bf16 | fp16 |
|---|---|---|
| Exponent bits | 8 (same as fp32) | 5 |
| Mantissa bits | 7 | 10 |
| Dynamic range | Same as fp32 | 65504 max |
| Underflow risk | Very low | High (needs GradScaler) |
| Ampere support | Native (TF32 path) | Supported |

bf16's wider dynamic range eliminates the need for a `GradScaler`.
`torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps
both forward pass and loss computation; the backward pass and
optimizer step run in fp32 automatically.

---

## C. LR Scheduling & Early Stopping

### C.1 ReduceLROnPlateau Mechanics

```
Epoch    Val Loss    Plateau ctr   LR
------------------------------------------
  1       7.44          0         3.0e-4  <- new best
  2       7.70          1         3.0e-4
  3       7.79          2         3.0e-4
  4       7.82          3         3.0e-4  <- patience hit -> LR x0.5
  5       7.83          4         1.5e-4  <- LR halved
  6  [Phase 2 reset: best_val_loss = inf, no_improve = 0]
  ...
```

The plateau counter resets to 0 whenever a new best validation loss is
found. Min LR floor of 1e-5 prevents the scheduler from reducing LR
below a useful training signal.

### C.2 Phase 2 Counter Reset Rationale

At epoch 6, both `best_val_loss` and the early-stopping `no_improve`
counter are reset to their initial values. Without this reset, the
early stopping counter from Phase 1 (where TF=1.0 creates artificially
easy training, inflating the train-val gap) would carry over and
potentially terminate training before Phase 2 can reduce the exposure
bias. The reset grants Phase 2 its own independent convergence window.

### C.3 Early Stopping Timeline

```
Phase 1 (ep 1-5):  early stopping DISABLED (counter not checked)
Phase 2+ (ep 6+):  counter active

     best found -> counter = 0
     no improve  -> counter + 1
     counter >= 4 -> training halted, best.pt already saved
```

---

## D. Formal Equations

### D.1 AdamW Update Rule

```
m_t = B1*m_(t-1) + (1-B1)*g_t
v_t = B2*v_(t-1) + (1-B2)*g_t^2
m_hat = m_t / (1-B1^t),  v_hat = v_t / (1-B2^t)
theta_t = theta_(t-1) - lr * ( m_hat / (sqrt(v_hat)+eps) + lambda*theta_(t-1) )
```

Parameters: lr=3e-4, B1=0.9, B2=0.999, eps=1e-8, lambda=1e-5.
The weight decay term lambda*theta is applied **after** the adaptive
step (decoupled), unlike standard Adam where decay is folded into the
gradient.

### D.2 Gradient Clipping

```
g_clipped = g                    if ||g||_2 <= 1.0
            g / ||g||_2          if ||g||_2 >  1.0
```

Clipping is applied to the concatenated parameter vector across all
model parameters before each `optimizer.step()`.

### D.3 Training Loss (with gradient accumulation)

```
L_batch = - (1/|T|) * sum_{t in T} log p(y_t | y_<t, X)
```

where T excludes padding positions. For A=2 accumulation steps,
backward receives L_batch/A so accumulated gradients equal the mean
over the full effective batch.

### D.4 Perplexity

```
PPL = exp( min(L_val, 20) )
```

The cap at 20 prevents numerical overflow when the model diverges
early in training (PPL would otherwise exceed 485 million at
val_loss=20).

---

## E. Checkpoint Strategy

Three complementary checkpoints are maintained throughout training:

| Type | Filename | Trigger | Purpose |
|---|---|---|---|
| **Best** | `{model}_best.pt` | Val loss improves | Production model; used for evaluation |
| **Step** | `{model}_step_{n}.pt` | Every 2 000 opt steps | Mid-epoch recovery; fine-grained audit |
| **Last** | `{model}_last.pt` | Every epoch end | Resume after disconnect/crash |

### E.1 Best Checkpoint Contents

```
epoch, global_step, model_type
model_state_dict           <- weights
optimizer_state            <- Adam moment buffers
scheduler_state            <- plateau counter & LR state
val_loss, val_ppl
train_loss, tf_ratio
config                     <- full hyperparameter dict
history                    <- per-epoch loss/LR/grad arrays
git_hash, torch_version, run_timestamp
```

### E.2 Resume Logic

On restart, `last.pt` is preferred over `best.pt` to avoid silently
discarding epochs completed after the last best checkpoint. The
scheduler state is **not** restored — the current LR is read directly
from `optimizer.param_groups` to avoid scheduler reset artefacts.

---

## F. Glossary

| Term | Definition |
|---|---|
| **AdamW** | Adam with decoupled weight decay (Loshchilov & Hutter, 2019); prevents weight decay from distorting adaptive LR estimates |
| **Teacher forcing** | Feeding the ground-truth previous token as decoder input during training; speeds convergence but causes exposure bias |
| **Exposure bias** | Train/inference mismatch: model conditioned on gold tokens during training but its own outputs at inference |
| **Gradient accumulation** | Summing gradients over multiple forward passes before an optimizer step; simulates a larger effective batch without extra VRAM |
| **ReduceLROnPlateau** | LR scheduler that halves the learning rate when a monitored metric stops improving for `patience` epochs |
| **bfloat16 (bf16)** | 16-bit float with 8 exponent bits (same as fp32); preserves dynamic range, eliminating the need for a GradScaler |
| **GradScaler** | PyTorch utility that scales loss to prevent fp16 underflow; not needed with bf16 |
| **Gradient clipping** | Rescaling the gradient vector to L2-norm <= max_norm; prevents exploding gradients in deep RNNs |
| **PPL (Perplexity)** | exp(cross-entropy loss); lower is better; indicates how many equally likely tokens the model considers at each step |
| **Early stopping** | Halting training when validation loss fails to improve for `patience` consecutive epochs |
| **best.pt / last.pt** | Best = saved on val improvement (production model); last = saved every epoch (resume guard) |
| **Phase 2 reset** | At epoch 6, best_val_loss and no_improve counter reset to allow Phase 2 its own convergence window |
| **TF floor** | Minimum teacher forcing ratio (0.5) held fixed in Phase 3 |
| **Effective batch size** | Physical batch x accumulation steps; determines gradient noise and convergence behaviour |
| **NaN guard** | Per-batch check: if loss or grad_norm is non-finite, the batch is skipped without updating weights |

---

## G. Design Decisions and Rationale

**Why AdamW and not Adam?**
Standard Adam applies L2 weight decay by adding it to the gradient,
which interferes with the per-parameter adaptive learning rate scale.
AdamW applies decay directly to the weights (decoupled), making the
regularisation effect predictable regardless of gradient magnitude.
At weight_decay = 1e-5 this is a light constraint, but the decoupled
form avoids underdecaying large-gradient parameters (embeddings) and
overdecaying small-gradient ones (output layer).

**Why no learning rate warm-up?**
Warm-up is most beneficial for Transformers, where large initial
updates to attention weights can destabilise training. LSTM models are
less sensitive — the recurrent inductive bias and orthogonal weight
init provide natural stability from step 1. Adding warm-up would
introduce an additional hyperparameter with minimal expected benefit.

**Why ReduceLROnPlateau and not cosine annealing?**
Cosine annealing decays LR on a fixed schedule regardless of whether
the model is still improving. ReduceLROnPlateau is adaptive: it only
reduces LR when validation loss genuinely stalls. Given the 3-phase
teacher forcing schedule (which itself causes planned transitions in
loss behaviour), a fixed cosine schedule would require careful
alignment with TF phase boundaries. ReduceLROnPlateau handles this
automatically.

**Why label smoothing = 0.0 (disabled)?**
Label smoothing encourages the model to assign some probability to all
vocabulary tokens, reducing overconfidence. In this experiment,
teacher forcing ratio already acts as curriculum regularisation. At
TF = 0.5 in Phase 3 the model already faces a harder, noisier
objective. Enabling smoothing simultaneously would confound two
regularisation strategies, making ablation analysis ambiguous. It is
reserved as a separate ablation.

**Why greedy decoding at validation and not beam search?**
Beam search is a decoding strategy, not a training objective. At
validation time, the goal is a fast, consistent generalisation signal,
not optimal output quality. Greedy decoding is deterministic, fast,
and produces a consistent lower bound epoch-to-epoch. Beam search is
applied only at final evaluation and inference.

**Why bf16 and not fp32 or fp16?**
fp32 would consume ~2x the VRAM and compute for marginal quality
benefit at this model size. fp16 requires a GradScaler to prevent
underflow. bf16 offers identical dynamic range to fp32 (8 exponent
bits) with half the memory footprint, and is natively accelerated on
Ampere hardware (RTX 3080 CC 8.6). The only trade-off is slightly
lower mantissa precision (7 vs 23 bits), which is negligible for
gradient descent.
