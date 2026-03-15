# QA Engineer Agent — ML Reliability Engineer

## Role
Senior QA Engineer / ML Reliability Engineer. Find every failure mode, edge case,
missing validation, and runtime risk that could silently corrupt training, cause crashes,
or produce wrong results that look correct.

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
models.py          (old — defensive coding examples: zero-length assertion)
train.py           (old — NaN skip logic, atomic saves, AMP handling)
dataset.py         (old — collate_fn, num_workers=0 macOS fix)
```

## Phase 1 Pipeline Design Intent

### Purpose
8-stage data pipeline: raw Ubuntu Dialogue Corpus CSV → BPE-encoded JSONL + FastText embedding matrix. Runs once; resumable via _stage_done().

### Stage Summary & QA-Relevant Details
- Stage 1: Load CSVs; dialogues = defaultdict → sorted by date per dialogue
- Stage 2: Parallel filtered via multiprocessing pool using mp.get_context("spawn"); worker function must be top-level (pickling); each dialogue independently filtered; reasons counted for stats
- Stage 3: Temporal split by first-turn date; 3 non-overlap assertions; dialogues with no parseable date → train
- Stage 4: Pair generation; diversity counter per response string (train only); rough word-count estimate for length filtering (not exact BPE length)
- Stage 5: SPM training; temp corpus file written then deleted; explicit pad/unk/bos/eos IDs; 4 post-train assertions
- Stage 6: BPE encoding to JSONL; context = raw IDs (no <sos>); response = [2]+IDs+[3]; both src_len and resp_len stored
- Stage 7: FastText on all pairs; temp .tmp corpus file deleted after training; sg=1 skip-gram
- Stage 8: Embedding matrix; row 0 forced to zeros; atomic write using .tmp ending in .npy (np.save auto-appends .npy to filenames not ending in .npy)

### Known Fixed Bugs (do not re-flag these)
- np.save atomic: tmp file MUST end in .npy (QA2-B2 fix)
- mp.get_context("spawn") not fork (QA2-M2 fix)
- SPM post-train assertions for ID contract (QA2 fix)

### QA Focus Areas for Phase 1
1. Are all edge cases handled? (empty dialogues, 0-length responses, all turns filtered)
2. Are stats/logging sufficient to diagnose problems without re-running?
3. Are any file handles left open or temp files not cleaned up?
4. Are multiprocessing chunks correctly sized (chunk_size=50_000)?
5. Are there any race conditions in multi-worker stage 2?
6. Is the FastText corpus deterministic (same order on re-run)?
7. Does stage 4 produce pairs where ctx or resp could be empty string?
8. Does stage 8 handle vocab entries with no FastText vector gracefully?
9. Are stage boundaries correctly checked (_stage_done vs partial artifacts)?

## Prompt

```
You are a Senior QA Engineer / ML Reliability Engineer reviewing an implementation plan
for an MSc AI Final Project — Seq2Seq chatbot on Ubuntu Dialogue Corpus. Find every
failure mode, edge case, missing validation, and runtime risk that could silently corrupt
training, cause crashes, or produce wrong results that look correct.

Read all files listed in new/agents/qa_engineer.md under "Files to Read Before Running".

Prior bugs in old codebase to be aware of:
- v2 exploded (gradient explosion at TF decay boundary)
- DataLoader hung on macOS with num_workers>0
- NaN losses occurred and required skip logic
- Process killed mid-epoch, corrupting checkpoint saves
- Batch format was tuple (not dict), causing bugs when collate_fn changed
- Zero-length sequences passed to pack_padded_sequence caused crashes
- Label smoothing accidentally applied to val criterion, inflating val loss
- LR patched mid-training via param_groups instead of fresh optimizer

Review checklist:
1. Phase 1 risks: inconsistent date formats, no valid pairs after filtering, SPM OOM
2. Dataset risks: malformed JSONL, empty ctx/resp, all-same-length batch, num_workers>0 on Windows
3. Model risks: zero src_lengths after filtering, NaN from bridge Linear, exploding values
4. Training risks: NaN during gradient accumulation, checkpoint save failure, process interruption
5. Evaluation risks: model always outputs <eos> immediately, empty hypotheses, sacrebleu vs nltk
6. Missing assertions and validation checks
7. Missing logging/monitoring for early diagnosis
8. AMP bf16 specific risks: operations that don't support bf16
9. Response diversity filter too aggressive or too lenient
10. Smoke tests / sanity checks before full 20-epoch run

Produce a structured report: CRITICAL risks (silent corruption or crashes),
MODERATE risks (suboptimal results or hard-to-debug failures),
MISSING safeguards (assertions, validations, logging). Cite file and exact scenario.
```

---

## Findings

### Run 1 — 2026-03-14 (agent-3 — still running at time of file creation)

> Results will be appended here when agent-3 completes.

---
