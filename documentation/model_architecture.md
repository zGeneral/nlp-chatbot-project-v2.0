# Model Architecture: Attention vs. Baseline Seq2Seq

This document describes the two sequence-to-sequence (Seq2Seq) architectures implemented in `models.py` for an open-domain NLP chatbot, and explains how they differ in a controlled ablation study on the contribution of attention to response quality. The general encoder–decoder framework follows Sutskever, Vinyals and Le (2014), wherein an encoder network reads the source sequence and a decoder network generates the target sequence conditioned on the encoder's output.

---

## Shared: BiLSTM Encoder

Both models begin with the same input pipeline and bidirectional LSTM (BiLSTM) encoder. Raw text is first tokenised into subword units using a SentencePiece BPE model (vocabulary size: 16,000), and each token index is mapped to a 300-dimensional dense vector via a pretrained FastText embedding matrix. The Long Short-Term Memory (LSTM) unit, introduced by Hochreiter and Schmidhuber (1997), addresses the vanishing gradient problem of standard recurrent networks through gated cell states that can carry information across arbitrarily long sequences. Unlike a unidirectional LSTM — which processes tokens left-to-right and compresses the full source sequence into a single fixed-length vector — the BiLSTM (Schuster and Paliwal, 1997) runs a forward and a backward pass simultaneously. Every encoder position therefore has access to both past and future context, producing richer token representations.

The final hidden states of both directions are merged through a learned linear bridge (`EncoderDecoderBridge`) to initialise the decoder with full bidirectional coverage.

```
Raw text
      │
      ▼  SentencePiece BPE tokenisation
      │  vocab_size = 16,000 tokens
      ▼
┌──────────────────────────────────────────┐
│  Shared Embedding  (FastText pretrained) │
│  [16,000 × 300]  — fine-tuned, not frozen│
│  embed_dropout = 0.3                     │
└────────────────────┬─────────────────────┘
                     │  token vectors  [batch, src_len, 300]
                     ▼
┌──────────────────────────────────────────┐
│  BiLSTM Encoder                          │
│  2 layers · 512 hidden units/direction   │
│  output: [batch, src_len, 1,024]         │
│  (fwd 512 ⊕ bwd 512 per timestep)        │
│  lstm_dropout = 0.5 · out_dropout = 0.4  │
└────────────────────┬─────────────────────┘
                     │  h_n, c_n  [4, batch, 512]
                     ▼
┌──────────────────────────────────────────┐
│  EncoderDecoderBridge                    │
│  concat(fwd, bwd) → Linear(1,024→1,024)  │
│  tanh · separate projections for h and c │
└────────────────────┬─────────────────────┘
                     │  h₀, c₀  [2, batch, 1,024]
                     ▼
           Decoder initial state
```

---

## Branch: Where the Models Differ

From this point the two models diverge **only** in how they form the context vector fed to the decoder LSTM at each step.

```
                  ┌─────────────────┴──────────────────┐
                  │                                    │
   ┌──────────────▼──────────────┐   ┌────────────────▼────────────────┐
   │     Attention Decoder       │   │        Baseline Decoder         │
   │  BahdanauAttention          │   │  Fixed context (no attention)   │
   │  attn_dim = 256             │   │  c_t = encoder_outputs[:, -1, :]│
   │  enc_dim=1,024 dec_dim=1,024│   │  shape: [batch, 1,024]          │
   │  dynamic c_t per step       │   │  constant across all steps      │
   │  LSTM input: 300+1,024=1,324│   │  LSTM input: 300+1,024=1,324    │
   │  2 layers · hidden=1,024    │   │  2 layers · hidden=1,024        │
   └──────────────┬──────────────┘   └────────────────┬────────────────┘
                  │                                    │
                  └─────────────────┬──────────────────┘
```

### Attention Decoder (primary model)

The attention decoder uses Bahdanau additive attention (Bahdanau, Cho and Bengio, 2015). At each decoding step *t*, it computes a scalar energy score for every encoder output position using a learned alignment function:

$$e_{t,i} = \mathbf{v}^\top \cdot \tanh\!\left(W_\text{enc}\, h_i + W_\text{dec}\, s_t\right)$$

A softmax over these scores yields a normalised attention distribution:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

The resulting context vector is a **dynamic**, input-conditioned weighted sum of all encoder states:

$$c_t = \sum_i \alpha_{t,i}\, h_i$$

This allows the decoder to selectively focus on the most relevant source tokens at each generation step — particularly valuable for longer or structurally complex utterances.

### Baseline Decoder (no attention)

The baseline decoder replaces dynamic attention with a **fixed** context vector — the final encoder output timestep — held constant across all decoding steps:

$$c_t = h_{\text{enc},\,-1} \quad \forall\, t$$

This is the classic bottleneck design described by Sutskever, Vinyals and Le (2014): the entire source sequence must be summarised into a single static vector before decoding begins.

---

## Shared: Output Stage

Beyond the context vector, both decoders are identical. Dropout (Srivastava et al., 2014) is applied at three points — after embedding lookup, between LSTM layers, and before the output projection — to regularise both models equivalently. The encoder and decoder share a single embedding matrix through weight tying (Press and Wolf, 2017), which reduces parameter count and acts as an additional regulariser given the single shared vocabulary.

| Component | Value |
|---|---|
| LSTM input size | `embed_dim + enc_hidden_dim` = 1,324 |
| Projection bottleneck | 2,048 → 512 → `vocab_size` |
| Dropout schedule | embed 0.3 · LSTM 0.5 · output 0.4 (Srivastava et al., 2014) |
| Embedding | Shared weight tying, encoder ↔ decoder (Press and Wolf, 2017) |

```
   [LSTM output ; context vector]
   [batch, 1,024] ⊕ [batch, 1,024] = [batch, 2,048]
             │
      out_dropout = 0.4
             │
   Linear projection  2,048 → 512
             │
            tanh
             │
   fc_out  512 → 16,000  (vocab_size)
             │
         logits  [batch, 16,000]
```

The two models are **parameter-near** (<1.2% total parameter difference). Any performance gap between them can therefore be attributed directly to the presence or absence of the attention mechanism, providing a clean and interpretable ablation.

---

## 3.4 Training Configuration

Both models are trained with identical hyperparameters; the sole architectural difference is the attention mechanism, ensuring any performance gap is attributable exclusively to attention.

| Component | Configuration |
|---|---|
| Optimiser | AdamW (Loshchilov and Hutter, 2019) |
| Learning Rate | 3×10⁻⁴ |
| Weight Decay | 1×10⁻⁵ |
| Grad Clipping | norm 1.0 |
| Dropout | Embed: 0.3 · LSTM: 0.5 · Proj: 0.4 |
| Batch | 256 × accum ×2 → effective 512 |
| Sampler | BucketBatchSampler (padding minimised) |
| Precision | bfloat16 AMP (no GradScaler) |
| Loss | Cross-entropy (non-padding); label smoothing = 0.0 |
| LR Schedule | ReduceLROnPlateau: ×0.5, patience 3, min = 1×10⁻⁵; reset at epoch 6 |
| Early Stopping | Patience 4; active from Phase 2 onward |
| Checkpointing | Best-val per epoch; step checkpoint every 2,000 steps |
| Logging | TensorBoard every step; seed = 42 |
| Inference | Greedy (default); beam width 5; nucleus p=0.9, T=0.8 via `--decoding` |

| Phase | Epochs | TF Ratio | Purpose |
|:---:|:---:|---|---|
| 1 | 1–5 | 1.0 | Burn-in; loss ~5.5 → ~4.25 |
| 2 | 6–12 | 0.9 → 0.5 (linear) | Exposure bias annealing |
| 3 | 13–20 | 0.5 (floor) | Semi-autoregressive maturation |

Floor = 0.5 for both models; baseline unstable below this threshold.

| | Attention Model | Baseline Model |
|---|---|---|
| Encoder | BiLSTM — Schuster and Paliwal (1997) | BiLSTM — Schuster and Paliwal (1997) |
| Tokenisation | SentencePiece BPE, vocab = 16,000 | SentencePiece BPE, vocab = 16,000 |
| Embedding | FastText 300-dim, corpus-trained + fine-tuned — Bojanowski et al. (2017) | FastText 300-dim, corpus-trained + fine-tuned — Bojanowski et al. (2017) |
| Context vector | Dynamic — Bahdanau, Cho and Bengio (2015) | Fixed — last encoder timestep, Sutskever, Vinyals and Le (2014) |
| Decoder LSTM input | `[embed ; c_t]` (1,324-dim) | `[embed ; c_t]` (1,324-dim) |
| Output projection | Shared bottleneck | Shared bottleneck |
| Regularisation | Dropout — Srivastava et al. (2014) | Dropout — Srivastava et al. (2014) |
| Embedding | Weight tying — Press and Wolf (2017) | Weight tying — Press and Wolf (2017) |
| Parameters | ~42M | ~41.5M |

---

## References

Bahdanau, D., Cho, K. and Bengio, Y. (2015) 'Neural machine translation by jointly learning to align and translate', in *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*, San Diego, CA. Available at: https://arxiv.org/abs/1409.0473 (Accessed: 17 March 2026).

Bojanowski, P., Grave, E., Joulin, A. and Mikolov, T. (2017) 'Enriching word vectors with subword information', *Transactions of the Association for Computational Linguistics*, 5, pp. 135–146. Available at: https://arxiv.org/abs/1607.04606 (Accessed: 17 March 2026).

Loshchilov, I. and Hutter, F. (2019) 'Decoupled weight decay regularization', in *Proceedings of the 7th International Conference on Learning Representations (ICLR 2019)*, New Orleans, LA. Available at: https://arxiv.org/abs/1711.05101 (Accessed: 17 March 2026).

Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory', *Neural Computation*, 9(8), pp. 1735–1780. Available at: https://www.bioinf.jku.at/publications/older/2604.pdf (Accessed: 17 March 2026).

Press, O. and Wolf, L. (2017) 'Using the output embedding to improve language models', in *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017)*, Valencia, Spain, pp. 157–163. Available at: https://arxiv.org/abs/1608.05859 (Accessed: 17 March 2026).

Schuster, M. and Paliwal, K.K. (1997) 'Bidirectional recurrent neural networks', *IEEE Transactions on Signal Processing*, 45(11), pp. 2673–2681. Available at: https://ieeexplore.ieee.org/document/650093 (Accessed: 17 March 2026).

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014) 'Dropout: A simple way to prevent neural networks from overfitting', *Journal of Machine Learning Research*, 15(56), pp. 1929–1958. Available at: http://jmlr.org/papers/v15/srivastava14a.html (Accessed: 17 March 2026).

Sutskever, I., Vinyals, O. and Le, Q.V. (2014) 'Sequence to sequence learning with neural networks', in *Advances in Neural Information Processing Systems (NeurIPS 2014)*, Montreal, Canada, pp. 3104–3112. Available at: https://arxiv.org/abs/1409.3215 (Accessed: 17 March 2026).

---

## Glossary

| Term | Definition |
|---|---|
| **Ablation study** | An experiment in which one component is removed or replaced in isolation to measure its individual contribution to performance. |
| **AdamW** | Optimiser (Loshchilov and Hutter, 2019) that decouples L2 weight decay from adaptive gradient estimates, correcting a flaw in standard Adam. Used here with lr = 3×10⁻⁴, weight decay = 1×10⁻⁵. |
| **AMP / bfloat16** | Automatic Mixed Precision: forward passes run in bfloat16 to save memory and accelerate computation while master weights stay in float32. bfloat16 avoids float16 underflow, eliminating the need for a GradScaler. |
| **Attention mechanism** | A learned alignment function that, at each decoder step, computes a weighted distribution over all encoder states, allowing the decoder to focus dynamically on the most relevant source tokens. |
| **Bahdanau (additive) attention** | Attention formulation (Bahdanau, Cho and Bengio, 2015) scoring each encoder position as *e*_{t,i} = **v**⊤ · tanh(*W*_enc *h*_i + *W*_dec *s*_t). Named *additive* to distinguish it from dot-product attention. |
| **Bidirectional LSTM (BiLSTM)** | LSTM variant (Schuster and Paliwal, 1997) that processes the sequence in both directions simultaneously, concatenating forward and backward hidden states at each position for full left–right context. |
| **Bottleneck problem** | The constraint in standard encoder–decoder models where the full source sequence must be compressed into a single fixed-length vector before decoding, motivating the use of attention. |
| **BPE (Byte Pair Encoding)** | Subword tokenisation that iteratively merges frequent character pairs until a target vocabulary size is reached. Implemented here via SentencePiece with a vocabulary of 16,000 tokens. |
| **BucketBatchSampler** | Batching strategy that sorts sequences by length within buckets before forming batches, minimising padding waste and reducing effective LSTM unroll depth per step. |
| **Context vector (*c*_t)** | A vector summarising source information for the decoder at step *t*. Dynamic (attention-weighted sum) in the attention model; fixed (last encoder output) in the baseline. |
| **Decoder** | The recurrent network that generates the target sequence one token at a time, conditioned on the encoder output and its own previous predictions. |
| **Dropout** | Regularisation (Srivastava et al., 2014) that randomly zeroes activations during training. Applied at three points: embedding (0.3), LSTM layers (0.5), projection (0.4). |
| **Early stopping** | Halts training when validation loss fails to improve for a set number of epochs (patience = 4). Active only from Phase 2 onward, as Phase 1 validation loss is not meaningful under autoregressive evaluation. |
| **Encoder** | The recurrent network that reads the source sequence and produces contextual hidden states. A two-layer BiLSTM is used here. |
| **EncoderDecoderBridge** | Learned linear projection that converts the interleaved BiLSTM final states into the decoder's initial hidden and cell states, correctly merging forward and backward directional signals. |
| **Exposure bias** | Training–inference mismatch where a teacher-forced model receives ground-truth inputs during training but its own (potentially erroneous) predictions at inference. Mitigated by annealing TF ratio from 1.0 to 0.5. |
| **FastText** | Embedding method (Bojanowski et al., 2017) representing tokens as sums of character n-gram vectors. Trained from scratch on the BPE-tokenised Ubuntu Dialogue Corpus (300-dim, skip-gram, 10 epochs); fine-tuned during seq2seq training. |
| **Gradient accumulation** | Accumulates gradients over multiple micro-batches before a single optimiser update. Physical batch 256 × 2 steps = effective batch 512, without additional memory cost. |
| **Gradient clipping** | Caps the gradient norm before each optimiser step (norm 1.0) to prevent divergence while permitting valid large gradients. |
| **Hidden state (*h*_t)** | LSTM internal representation at timestep *t*. In a BiLSTM, forward and backward hidden states are concatenated at each position. |
| **Label smoothing** | Loss regularisation that softens one-hot targets to discourage over-confidence. Set to 0.0 here — omitted to preserve sharp token learning in Phase 1 and kept off for consistency throughout. |
| **LSTM (Long Short-Term Memory)** | Recurrent unit (Hochreiter and Schmidhuber, 1997) with input, forget, and output gates that selectively retain information across long sequences, mitigating the vanishing gradient problem. |
| **Nucleus sampling (top-p)** | Stochastic decoding that samples from the smallest vocabulary subset whose cumulative probability exceeds *p*. Available via `--decoding nucleus` with p = 0.9, T = 0.8. |
| **Parameter-near baseline** | Baseline model within <1.2% of the primary model's parameter count, ensuring performance differences reflect the design choice (attention) rather than model capacity. |
| **Projection bottleneck** | Intermediate linear layer (2,048 → 512) between LSTM output and vocabulary projection, reducing output-stage parameters by 72% (32.8M → 9.2M). |
| **ReduceLROnPlateau** | LR scheduler that halves the learning rate (×0.5) when validation loss plateaus for 3 consecutive epochs. Reset at epoch 6 to avoid premature reduction during Phase 1. Minimum lr = 1×10⁻⁵. |
| **Seq2Seq (Sequence-to-Sequence)** | Neural architecture (Sutskever, Vinyals and Le, 2014) mapping variable-length input sequences to variable-length outputs via an encoder–decoder framework. |
| **Teacher forcing** | Training strategy feeding ground-truth target tokens as decoder input at each step, stabilising training but introducing exposure bias. A mixed ratio annealed 1.0 → 0.5 is used here. |
| **Vanishing gradient** | Training pathology where gradients decay exponentially through many layers or timesteps, preventing early layers from learning. Addressed by LSTM gating. |
| **Weight tying** | Sharing one embedding matrix across encoder input, decoder input, and decoder output projection (Press and Wolf, 2017), reducing ~9.6M parameters and enforcing consistent token representations. |

---

## Architecture Key Decisions

This section explains the *why* behind the principal design choices in this architecture — the reasoning that motivated each decision rather than simply what was chosen.

---

### Why BiLSTM and not a unidirectional LSTM?

A unidirectional LSTM reads the source sequence strictly left-to-right. At position *i* it has seen tokens 1…*i* but knows nothing about what follows. For a conversational input such as *"how do I mount a USB drive"*, the hidden state at *"mount"* is computed before *"USB drive"* has been seen, so the representation of *"mount"* carries no information about the object it relates to.

A BiLSTM runs a second LSTM in the reverse direction (right-to-left) in parallel. The hidden state at position *i* is the concatenation of the forward state (context from the left) and the backward state (context from the right), giving every token a representation grounded in the full sentence. This is the single most impactful encoder upgrade available without moving to a Transformer: it doubles the effective context available at each position at the cost of roughly doubling encoder computation, which remains negligible relative to the decoder.

---

### Why Bahdanau (additive) attention and not dot-product attention?

Two common attention formulations exist:

- **Dot-product (multiplicative):** *e*_{t,i} = *s*_t · *h*_i — fast, parameter-free, but scales poorly when the encoder and decoder hidden dimensions differ.
- **Additive (Bahdanau):** *e*_{t,i} = **v**⊤ · tanh(*W*_enc *h*_i + *W*_dec *s*_t) — introduces learned projection matrices *W*_enc and *W*_dec that map encoder and decoder states into a shared alignment space before scoring.

Here the encoder output dimension is 1,024 (bidirectional, 512 per direction) and the decoder hidden dimension is also 1,024, so dot-product attention would technically be applicable. However, additive attention was chosen for two reasons. First, the learned projections into the 256-dim alignment space act as a bottleneck that regularises the alignment, preventing the model from overfitting spurious token-level correlations on a relatively small chatbot dataset. Second, the encoder key projections (*W*_enc applied to all encoder outputs) are constant across decoder steps and can be precomputed once before the decode loop — a practical efficiency gain (saves *trg\_len* − 1 redundant matrix multiplications per sequence).

---

### Why greedy decoding as the default?

At inference time the decoder must select one token per step from the full vocabulary distribution. Three strategies are available in this codebase: greedy, top-p (nucleus) sampling, and beam search.

**Greedy decoding** selects the single highest-probability token at each step (`argmax`). It is deterministic, requires no additional memory, and runs in *O*(1) per step. For a chatbot evaluated on short-to-medium responses, greedy decoding is a sensible default because the quality gap between greedy and beam search narrows significantly when the model is well-trained and the vocabulary is constrained. It also provides a reproducible, stable baseline — responses do not vary across runs — which is necessary for consistent BLEU and perplexity evaluation.

**Beam search** maintains multiple hypotheses in parallel and is available as an alternative (`--decoding beam`), but it multiplies inference cost by the beam width and can favour short, generic responses due to length penalties. **Top-p sampling** introduces controlled stochasticity and is offered for more varied conversational output, but makes automated evaluation noisier. Greedy therefore represents the best trade-off between quality, speed, and evaluation reproducibility for the primary experiments.

---

### Why is a bridge needed between the encoder and decoder?

The bidirectional encoder produces final hidden states with a specific layout that is incompatible with the decoder's expected input format. PyTorch stores the BiLSTM final states as an interleaved tensor of shape `[num_layers × 2, batch, hidden_dim]`, where layers alternate between forward and backward: `[layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd]`. The unidirectional decoder expects initial states of shape `[num_layers, batch, dec_hidden_dim]`.

A naive reshape or slice would either discard half the information (using only forward states) or produce an incorrectly shaped tensor. The `EncoderDecoderBridge` solves this by:

1. Separating forward and backward states using stride-2 indexing (`[0::2]` and `[1::2]`).
2. Concatenating them along the hidden dimension per layer, forming a `[num_layers, batch, enc_hidden_dim × 2]` tensor.
3. Passing this through a learned linear projection (with tanh) to produce `[num_layers, batch, dec_hidden_dim]`.

The learned projection is preferable to a fixed reshape because it allows the model to emphasise whichever directional signal is more useful for initialising each decoder layer — a choice the network makes during training rather than one imposed by architecture.

---

### Why a projection bottleneck before the vocabulary layer?

At each decoder step, the model concatenates the LSTM output and the context vector to form a 2,048-dim combined representation before projecting to vocabulary logits. Without a bottleneck, this single linear layer would have 2,048 × 16,000 ≈ 32.8M parameters — nearly as large as the rest of the model combined. A 512-dim intermediate projection reduces this to (2,048 × 512) + (512 × 16,000) ≈ 9.2M parameters, a 72% reduction. The bottleneck also applies a tanh non-linearity, giving the output stage expressive power to combine LSTM and attention signals non-linearly before scoring against the vocabulary.

---

### Why shared embeddings (weight tying)?

The encoder input, decoder input, and decoder output projection all operate over the same vocabulary. Using three separate embedding matrices of shape `[vocab_size, embed_dim]` would introduce redundant parameters and allow the input and output representations of the same token to diverge during training. Weight tying (Press and Wolf, 2017) forces all three to share a single matrix, reducing parameter count by approximately 2 × vocab_size × embed_dim ≈ 9.6M parameters for a 16,000-token vocabulary with 300-dim embeddings. It also ensures that the model must represent each token consistently whether it appears as input or output, acting as a strong inductive bias and regulariser.

---

### Why per-sample teacher forcing with a mixed ratio?

Pure teacher forcing (always feeding ground-truth tokens as decoder input during training) trains the decoder on a distribution of inputs it never sees at inference — a discrepancy known as *exposure bias*. Pure scheduled sampling (always feeding predicted tokens) is unstable early in training when predictions are poor. A stochastic mixed ratio — where each sample in a batch independently uses ground-truth or predicted input with probability *p* — balances these extremes. The ratio is annealed during training (decaying from 1.0 towards a lower value across phases) so early training benefits from stable gradients while later training builds robustness to the model's own prediction errors.

---

### Why SentencePiece BPE tokenisation (vocabulary size 16,000)?

Text must be converted to discrete token indices before it can be embedded. Word-level tokenisation produces a large, sparse vocabulary with poor coverage of rare words and out-of-vocabulary (OOV) terms. Character-level tokenisation avoids OOV entirely but produces very long sequences, increasing computational cost and making it harder for the LSTM to learn long-range dependencies.

Byte Pair Encoding (BPE), as implemented by SentencePiece, is a subword tokenisation scheme that iteratively merges the most frequent character pairs until a target vocabulary size is reached. Common words remain as single tokens; rare or compound words are split into meaningful subword pieces. A vocabulary of 16,000 strikes a balance: large enough that frequent words and short common phrases are represented atomically, small enough that the output projection layer (512 × 16,000) remains computationally tractable. BPE also handles informal language, spelling variants, and domain-specific terms gracefully — properties particularly valuable in open-domain dialogue.

---

### Why FastText pretrained embeddings (300-dim)?

Rather than initialising the embedding matrix randomly, the model loads a 300-dimensional embedding matrix produced in Stage 7–8 of the Phase 1 pipeline. Importantly, this FastText model is **trained from scratch on the project's own corpus** — the full Ubuntu Dialogue Corpus, BPE-tokenised with SentencePiece — not loaded from a general-purpose external source. Training is performed using Gensim's FastText implementation with skip-gram (`sg=1`), a context window of 5, minimum count of 3, and 10 epochs over all available pairs (train + val + test combined) to maximise embedding coverage.

FastText (Bojanowski et al., 2017) represents each token as the sum of its character n-gram vectors. This is particularly well-suited here because the input tokens are BPE subword pieces (e.g. `▁mount`, `USB`, `▁drive`) rather than whole words. FastText can produce meaningful vectors for rare or previously unseen BPE pieces by composing their character n-gram components — something standard Word2Vec cannot do. Training on the project corpus directly also means the 300-dim vectors reflect the vocabulary, register, and topic distribution of the Ubuntu technical dialogue domain rather than a generic web corpus.

The embeddings are initialised from this corpus-trained matrix but are **not frozen** — they are fine-tuned throughout seq2seq training (`freeze=False`). The embedding dropout (0.3) is deliberately lighter than the LSTM dropout (0.5) to preserve as much of the domain-trained signal as possible during fine-tuning.
