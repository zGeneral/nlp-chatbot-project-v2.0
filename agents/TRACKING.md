# Implementation Tracking Dashboard
## Last updated: 2026-03-14 | Branch: clean_from_scratch_again

> This file is updated after every coding session and every manager agent run.
> It is the single source of truth for "what still needs to be fixed before coding starts."

---

## Status Summary

| Severity | Total | ✅ Resolved | 🔄 In Progress | ❌ Open |
|---|---|---|---|---|
| 🔴 Critical / Blocking | 11 | 0 | 0 | **11** |
| 🟠 Ambiguous | 6 | 0 | 0 | **6** |
| 🟡 Significant | 8 | 0 | 0 | **8** |
| 🔵 Missing | 9 | 0 | 0 | **9** |
| ⚪ Minor | 3 | 0 | 0 | **3** |
| **Total** | **37** | **0** | **0** | **37** |

> Agents still running: architect (agent-0), qa_engineer (agent-3) — findings TBD

---

## 🔴 Critical / Blocking — Must Resolve Before Writing Any Code

| ID | Reviewer | File | Finding | Status |
|---|---|---|---|---|
| B1 | developer | `models.py` | `pad_packed_sequence` needs `total_length=src.size(1)` — attention shape mismatch without it | ❌ Open |
| B2 | developer | `phase1.py` | SPM token IDs won't match INTERFACES.md — must use `pad_id/unk_id/bos_id/eos_id` in `SentencePieceTrainer.train()` | ❌ Open |
| B3 | developer | `config.py` | CONFIG is empty — missing `embedding_matrix_path`, `spm_model_path`, `sos_idx`, `eos_idx`, `unk_idx`, `max_ctx_turns`, `artifact_dir`, `attn_dim` | ❌ Open |
| B4 | developer | `models.py` | BiDir `h_n` slicing never specified — correct: `h_n[0::2]` fwd, `h_n[1::2]` bwd | ❌ Open |
| B5 | developer | `chat.py` | Line 80: `tf_schedule["phase1_end"]`=15 (epoch) passed as `max_turns` — should be `max_ctx_turns`=8 | ❌ Open |
| B6 | developer | `models.py` | `AttentionDecoder` LSTM `input_size` must use `enc_hidden_dim=1024` (bidir total), not 512 | ❌ Open |
| B7 | developer | `models.py` | `src_mask` construction and propagation path absent from plan | ❌ Open |
| F1 | academic | `PLAN.md` | Baseline vs Attention NOT parameter-fair — different LSTM input sizes (300 vs 1324) | ❌ Open |
| F2 | academic | `models.py` | BiDir encoder 1024 output vs decoder hidden 512 — BahdanauAttention dim mismatch | ❌ Open |
| F3 | academic | `evaluate.py` | BLEU comparison to osamadev invalid without sacrebleu 13a tokeniser | ❌ Open |
| G1 | academic | `evaluate.py` | BLEU inadequate for dialogue — Distinct-1/2 and BERTScore missing | ❌ Open |

---

## 🟠 Ambiguous — Developer Will Guess Wrong

| ID | Reviewer | File | Finding | Status |
|---|---|---|---|---|
| A1 | developer | `dataset.py` | `collate_fn` TODO says sort descending but `enforce_sorted=False` chosen — contradictory | ❌ Open |
| A2 | developer | `train.py` | `global_step` with `grad_accum_steps=2` — count optimizer steps not forward passes | ❌ Open |
| A3 | developer | `train.py` | Validation TF=0.0 still receives `trg` for step count — semantics unclear | ❌ Open |
| A4 | developer | `evaluate.py` | n-gram blocking: "zero out" = pre-softmax logit masking; fallback case unspecified | ❌ Open |
| A5 | developer | `INTERFACES.md` | SPM encode method unspecified; `sp.Load()` is old API | ❌ Open |
| A6 | developer | `models.py` | `dec_hidden_dim` never stated as 1024 in CONFIG; PLAN.md param count uses wrong 512 | ❌ Open |

---

## 🟡 Significant — Important for Quality or Academic Validity

| ID | Reviewer | File | Finding | Status |
|---|---|---|---|---|
| F4 | academic | `train.py` | TF schedule justified by correlation not causal evidence | ❌ Open |
| G2 | academic | `evaluate.py` | Perplexity-quality inversion not measurable without Distinct-N | ❌ Open |
| G3 | academic | `INTERFACES.md` | 50 manual samples insufficient; no rubric defined | ❌ Open |
| G4 | academic | `evaluate.py` | No attention weight heatmap visualisation | ❌ Open |
| G5 | academic | `evaluate.py` | No bootstrap CIs on BLEU scores | ❌ Open |
| G6 | academic | `phase1.py` | Thread boundary integrity at temporal split not guaranteed | ❌ Open |
| R1 | academic | `PLAN.md` | No ablation studies | ❌ Open |
| R5 | academic | `evaluate.py` | `compute_distinct_n()` not implemented | ❌ Open |

---

## 🔵 Missing Details — Must Be Documented Before Implementation

| ID | Reviewer | File | Finding | Status |
|---|---|---|---|---|
| M1 | developer | `train.py` | bf16 → no GradScaler needed — not documented | ❌ Open |
| M2 | developer | `models.py` | `W_enc` precomputation opportunity not noted | ❌ Open |
| M3 | developer | `phase1.py` | `<sos>` on encoder input unjustified, wastes context slot | ❌ Open |
| M4 | developer | `models.py` | `src_lengths.cpu()` required by `pack_padded_sequence` not in docstring | ❌ Open |
| M5 | developer | `phase1.py` | FastText lookup must use `▁`-prefixed SPM piece strings verbatim | ❌ Open |
| M6 | developer | `train.py` | `build_dataloaders` new call signature not shown in pseudocode | ❌ Open |
| M7 | developer | `train.py` | `optimizer_state` vs `optimizer_state_dict` naming — pick one | ❌ Open |
| M8 | developer | `train.py` | Validation uses full padded `trg` length — use `trg_lengths.max()` instead | ❌ Open |
| M9 | developer | `dataset.py` | Docstring contradicts TODO: "no in-memory loading" vs "load all" | ❌ Open |

---

## ⚪ Minor

| ID | Reviewer | File | Finding | Status |
|---|---|---|---|---|
| R2 | academic | `PLAN.md` | Benchmark comparison overclaims vs osamadev | ❌ Open |
| R3 | academic | `PLAN.md` | Missing citations: Sutskever 2014, Liu 2016, Li 2016, Kudo 2018, Bengio 2015 | ❌ Open |
| R4 | academic | `PLAN.md` | Parameter count arithmetic wrong (says 16.7M, correct ~5.4M) | ❌ Open |

---

## ⏳ Pending (Agents To Re-Run)

| Agent | Status | Notes |
|---|---|---|
| architect | ⛔ Terminated | Re-run when ready — use `new/agents/architect.md` prompt |
| qa_engineer | ⛔ Terminated | Re-run when ready — use `new/agents/qa_engineer.md` prompt |

---

## 🗂️ Findings by File (implementation grouping)

| File | Critical | Ambiguous | Significant | Missing | Minor | Total |
|---|---|---|---|---|---|---|
| `config.py` | 1 | — | — | — | — | **1** |
| `models.py` | 5 | 1 | — | 3 | — | **9** |
| `phase1.py` | 1 | — | 1 | 3 | 1 | **6** |
| `evaluate.py` | 2 | 1 | 4 | — | — | **7** |
| `train.py` | — | 2 | 1 | 4 | — | **7** |
| `dataset.py` | — | 1 | — | 1 | — | **2** |
| `chat.py` | 1 | — | — | — | — | **1** |
| `PLAN.md` | 1 | — | 2 | — | 2 | **5** |
| `INTERFACES.md` | — | 1 | 1 | — | — | **2** |

---

## 📋 Recommended Implementation Order

Fix in this order — earlier files unblock later ones:

1. **`config.py`** — B3: fill CONFIG dict completely. Everything imports from here.
2. **`INTERFACES.md`** — A5, G3: fix SPM API reference; add manual eval rubric spec.
3. **`PLAN.md`** — F1, R2, R4: fix parameter-fair design decision; fix param count.
4. **`models.py`** — B1, B4, B6, B7, F2, M2, M4: all architecture fixes before dataset/train.
5. **`phase1.py`** — B2, G6, M3, M5: data pipeline correctness before training.
6. **`dataset.py`** — A1, M9: clean up contradictions.
7. **`train.py`** — A2, A3, M1, M6, M7, M8: training loop clarity.
8. **`evaluate.py`** — F3, G1, G2, G4, G5, R5, A4: evaluation completeness.
9. **`chat.py`** — B5: quick single-line fix.

---

## ✅ Resolved Findings

*(none yet — implementation has not started)*

---

## Workflow

**When you fix a finding:**
```
Tell Copilot: "Mark B1 as resolved"
→ Copilot updates this file + SQL: UPDATE review_findings SET status='resolved', resolved_in_commit='abc123' WHERE id='B1'
```

**When a new agent run comes back:**
```
Tell Copilot: "Append architect findings to TRACKING.md"
→ Copilot adds new rows, updates the summary table, re-runs manager agent
```

**Before starting a coding session:**
```
Tell Copilot: "Run the manager agent"
→ Copilot spawns manager agent → reads TRACKING.md + all code → produces fresh status report
```
