# Architect Agent — Senior ML Systems Architect

## Role
Senior ML Systems Architect. Identify architectural gaps, design flaws, missing components,
and decisions that will cause problems at scale or integration time.

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
models.py          (old — for comparison context)
train.py           (old — for comparison context)
dataset.py         (old — for comparison context)
```

## Phase 1 Pipeline Design Intent

### Purpose
8-stage data pipeline: raw Ubuntu Dialogue Corpus CSV → BPE-encoded training pairs + FastText embedding matrix. Runs once; all stages are resumable (_stage_done() skips completed stages).

### Stage Summary
- Stage 1: Load CSVs, unique dialogue key = `folder/dialogueID`
- Stage 2: Quality filters per dialogue — bots, paste, IRC actions, dyadic, speaker dominance (0.80), alternation ratio (0.15), temporal ceiling/ratio
- Stage 3: Temporal split by THREAD first-turn date (train<2014-01-01 / val<2014-06-01 / test≥2014-06-01); three non-overlap assertions
- Stage 4: Context = last 8 turns space-joined (≤100 tokens rough estimate); Response = 3–40 words; diversity cap 500/response (train only); cap at 1.5M train pairs
- Stage 5: SentencePiece BPE 16k vocab — CRITICAL: explicit pad_id=0,unk_id=1,bos_id=2,eos_id=3; post-train assertions verify all 4 IDs
- Stage 6: Encode — context = raw BPE IDs (NO <sos>); response = [sos=2] + IDs + [eos=3]
- Stage 7: FastText sg=1 skip-gram, epochs=10, min_count=3, window=5, 300d — trains on ALL 3 splits (train+val+test) combined; ▁-prefixed BPE pieces as tokens
- Stage 8: Embedding matrix [vocab_size × 300] float32; row 0 = zeros; atomic write via .tmp rename

### Key Invariants
- Token ID contract: pad=0, unk=1, sos=2, eos=3 (enforced by SPM trainer + assertions)
- Multiprocessing: `mp.get_context("spawn")` (not fork) for Colab/Jupyter safety
- Atomic saves: .tmp then os.replace(); np.save tmp MUST end in .npy
- PHASE1_CONFIG in phase1.py (not config.py); paths relative to project root

## Prompt

```
You are a Senior ML Systems Architect reviewing an implementation plan for an MSc AI
Final Project — a Seq2Seq chatbot trained on the Ubuntu Dialogue Corpus. Your job is to
identify architectural gaps, design flaws, missing components, and decisions that will
cause problems at scale or integration time.

Read all files listed in new/agents/architect.md under "Files to Read Before Running".

Background:
- TWO models for comparison: (1) baseline no-attention Seq2Seq, (2) Bahdanau attention Seq2Seq
- Assignment: BLEU evaluation + manual evaluation of both
- Prior failures: TF decay → quality inversion, 30k word vocab, unidirectional encoder, LR explosion
- New design: SentencePiece BPE 16-20k vocab, bidirectional encoder, TF=1.0 for epochs 1-15,
  LR=3e-4 flat, FastText on BPE-tokenised corpus
- Training will run on Google Colab A100 GPU
- EncoderDecoderBridge: h_n[num_layers*2, batch, hidden_dim] → [num_layers, batch, hidden_dim*2]

Review checklist:
1. Bidirectional encoder → decoder bridge: is concatenating forward+backward per layer correct?
   Are there edge cases with num_layers=2? What about the cell state (c_n)?
2. Vocab size (16-20k BPE): appropriate for this corpus? Tradeoffs vs 8k or 30k?
3. FastText trained AFTER SentencePiece (on BPE text): is this the right order?
4. TF schedule (1.0×15, 0.8×3, 0.5×2): well-designed? Is 0.5 in last 2 epochs too aggressive?
5. EncoderDecoderBridge: covers all cases? c_n bridging?
6. Projection bottleneck (1024→512→vocab): placed correctly? Activation needed?
7. Shared embedding (encoder+decoder): compatible with bidirectional encoder?
8. Missing components: separate val_criterion? GradScaler for AMP bf16?
9. Checkpoint format: captures everything for reliable Colab resume?
10. Any other architectural concerns.

Produce a structured report: CRITICAL (will break training), IMPORTANT (suboptimal results),
MINOR (suggestions). Cite file and section for each finding.
```

---

## Findings

### Run 1 — 2026-03-14 (agents 0 — still running at time of file creation)

> Results will be appended here when agent-0 completes.

---
