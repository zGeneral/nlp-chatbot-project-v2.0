# Developer Agent — Senior PyTorch Developer

## Role
Senior PyTorch Developer. Identify every place where the plan is underspecified,
where the API details are wrong or missing, and where a developer would get stuck
or make a silent bug.

## Files to Read Before Running
```
new/PLAN.md
new/INTERFACES.md
new/config.py
new/phase1.py
new/dataset.py
new/models.py
new/train.py
new/evaluate.py
new/chat.py
models.py          (old — reference for what worked)
train.py           (old — reference for NaN handling, AMP, checkpointing)
dataset.py         (old — reference for collate_fn)
```

## Phase 1 Pipeline Design Intent

### Purpose
8-stage pipeline: raw Ubuntu Dialogue Corpus CSV → BPE-encoded JSONL pairs + FastText embedding matrix. Runs once; resumable via _stage_done().

### Stage Summary
- Stage 1: Load CSVs; unique key = `folder/dialogueID`
- Stage 2: Dialogue filters — bots, paste, IRC actions, repetitive; then dyadic, speaker dominance (0.80), alternation ratio (0.15), temporal gap ceiling (3600s) + ratio (0.30)
- Stage 3: Temporal split by THREAD first-turn date (train<2014-01-01, val<2014-06-01, test≥2014-06-01); three zero-overlap assertions
- Stage 4: Context = last 8 turns (≤100 words rough estimate); Response = 3–40 words; diversity cap 500/response (train only); cap 1.5M train pairs
- Stage 5: SentencePiece BPE 16k; CRITICAL explicit pad_id=0,unk_id=1,bos_id=2,eos_id=3; post-train assertions verify IDs
- Stage 6: Encode — context = raw BPE IDs (NO <sos>); response = [sos=2]+IDs+[eos=3]; saves stage6_vocab.json {word2idx, idx2word}
- Stage 7: FastText sg=1, epochs=10, min_count=3, window=5, 300d; trains on ALL 3 splits combined; input = ▁-prefixed BPE pieces
- Stage 8: Embedding matrix [vocab_size×300] float32; row 0=zeros; atomic write

### Key Invariants
- Token IDs: pad=0, unk=1, sos=2, eos=3 — enforced by SPM trainer + 4 post-train assertions
- mp.get_context("spawn") for Colab safety; atomic saves via .tmp→os.replace()
- PHASE1_CONFIG in phase1.py; paths relative to project root

## Prompt

```
You are a Senior PyTorch Developer reviewing an implementation plan for an MSc AI
Final Project — a Seq2Seq chatbot. Identify every place where the plan is underspecified,
where the API details are wrong or missing, and where a developer would get stuck or
make a silent bug.

Read all files listed in new/agents/developer.md under "Files to Read Before Running".

Background:
- PyTorch 2.1+, CUDA 12.1, A100 GPU, bf16 AMP
- SentencePiece BPE tokenization (replaces word-level vocab)
- Bidirectional LSTM encoder (NEW — old code was unidirectional)
- EncoderDecoderBridge: h_n[num_layers*2, batch, hidden_dim] → [num_layers, batch, hidden_dim*2]
- Shared embedding between encoder and decoder
- Teacher forcing schedule: TF=1.0 (ep1-15), 0.8 (16-18), 0.5 (19-20)
- Gradient accumulation, periodic checkpoints, atomic saves

Review checklist:
1. pack_padded_sequence with enforce_sorted=False + bidirectional LSTM — any issues?
   Does total_length need to be specified?
2. EncoderDecoderBridge h_n slicing: indices 0,2 forward; 1,3 backward — is h_n[0::2] correct?
3. AttentionDecoder LSTM input_size = embed_dim + enc_hidden_dim — what value is enc_hidden_dim?
4. W_enc precomputation in BahdanauAttention — is this optimization noted?
5. collate_fn sort requirement vs enforce_sorted=False contradiction
6. Loss computation: trg[:, 1:] shift — is trg_len the padded or actual length?
7. GradScaler: needed for bf16 on A100?
8. global_step counting: optimizer steps vs forward passes with grad_accum_steps=2
9. top_p_decode n-gram blocking algorithm specification
10. SentencePiece encode method: which API call to use?
11. Any other developer-level gaps.

Produce a structured report: BLOCKING (won't work), AMBIGUOUS (developer will guess wrong),
MISSING (needs to be added). Cite file and line numbers.
```

---

## Findings

### Run 1 — 2026-03-14 (agent-1)

**7 BLOCKING | 6 AMBIGUOUS | 9 MISSING**

#### BLOCKING
| ID | Issue | Location |
|---|---|---|
| B1 | `pad_packed_sequence` needs `total_length=src.size(1)` — without it attention gets shape mismatch | `models.py Encoder.forward` |
| B2 | SPM default IDs won't match INTERFACES.md — must use `pad_id/unk_id/bos_id/eos_id` params in `SentencePieceTrainer.train()` | `phase1.py stage5` |
| B3 | CONFIG is empty dict — missing `embedding_matrix_path`, `spm_model_path`, `sos_idx`, `eos_idx`, `unk_idx`, `max_ctx_turns`, `artifact_dir`, `attn_dim` | `config.py` |
| B4 | Bidirectional h_n slicing `h_n[0::2]`/`h_n[1::2]` never specified — common wrong guesses exist | `models.py EncoderDecoderBridge` |
| B5 | `chat.py` line 80: `tf_schedule["phase1_end"]`=15 (epoch count) passed as `max_turns` instead of `max_ctx_turns`=8 | `chat.py:80` |
| B6 | `AttentionDecoder` LSTM `input_size` needs `enc_hidden_dim=1024` (bidir total), not 512 | `models.py AttentionDecoder` |
| B7 | `src_mask` construction and propagation path completely absent from plan and INTERFACES.md | `models.py; INTERFACES.md` |

#### AMBIGUOUS
| ID | Issue | Location |
|---|---|---|
| A1 | `collate_fn` TODO says "sort descending required" but plan chose `enforce_sorted=False` | `dataset.py:65` |
| A2 | `global_step` increment timing with `grad_accum_steps=2` not stated | `train.py` |
| A3 | Validation TF=0.0 still receives `trg` for step count — semantics unclear | `train.py evaluate_epoch` |
| A4 | n-gram blocking: "zero out" not specified as pre-softmax logit masking; fallback missing | `evaluate.py top_p_decode` |
| A5 | SPM encode method unspecified; `sp.Load()` in INTERFACES.md is old API | `phase1.py; INTERFACES.md` |
| A6 | `dec_hidden_dim` never stated as 1024; PLAN.md param estimate uses wrong 512 | `models.py; PLAN.md` |

#### MISSING
| ID | Issue | Location |
|---|---|---|
| M1 | bf16 → no GradScaler; use `torch.amp.GradScaler` not deprecated `cuda.amp` | `train.py` |
| M2 | `W_enc` precomputation optimization not mentioned | `models.py BahdanauAttention` |
| M3 | `<sos>` on encoder input unusual, unjustified, wastes a context slot | `phase1.py stage6; INTERFACES.md` |
| M4 | `src_lengths.cpu()` required by `pack_padded_sequence` not in `Encoder.forward` TODO | `models.py` |
| M5 | FastText lookup must use `▁`-prefixed SPM piece strings verbatim | `phase1.py stage8` |
| M6 | `build_dataloaders` new call signature not shown in `train_model` pseudocode | `train.py` |
| M7 | `optimizer_state` vs `optimizer_state_dict` naming — must be consistent save/load | `train.py; INTERFACES.md` |
| M8 | Validation uses full padded `trg` length for steps — `trg_lengths.max()` is more efficient | `train.py evaluate_epoch` |
| M9 | `dataset.py` docstring says "no in-memory loading" but TODO says "load all into self.pairs" | `dataset.py` |

---
