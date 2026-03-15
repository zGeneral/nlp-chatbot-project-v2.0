# Academic Reviewer Agent — NLP Research Reviewer

## Role
NLP Research Reviewer / Academic Assessor. Evaluate whether the plan meets academic
standards, whether the experimental design is sound, and whether there are methodological
gaps that would undermine the results or the report.

## Files to Read Before Running
```
new/PLAN.md
new/INTERFACES.md
new/evaluate.py
new/train.py
new/phase1.py
NLP Week 7.md
models.py          (old — architecture reference)
train.py           (old — training reference)
```

## Phase 1 Pipeline Design Intent (for academic review context)

### Data Pipeline Summary
- 8-stage pipeline producing 1.5M train / ~150k val / ~150k test context-response pairs
- Ubuntu Dialogue Corpus (IRC chat logs) — technically-focused English text
- SentencePiece BPE 16k vocab replaces word-level 30k vocab (zero OOV, smaller softmax)
- Quality filters: dyadic (2-speaker only), speaker dominance (≤80% one speaker), alternation ratio (≥15% turn changes), temporal gaps, paste/bot/echo/placeholder filters
- Temporal split on THREAD first-turn date — no thread boundary leakage; three non-overlap assertions

### Token ID Contract
- pad=0, unk=1, sos=2, eos=3 — guaranteed by explicit SPM trainer params + post-train assertions
- Context encoding: raw BPE IDs (NO <sos>) — encoder receives plain token sequence
- Response encoding: [sos=2] + BPE_IDs + [eos=3]

### Embedding Strategy
- FastText 300d trained on ALL pairs (train+val+test) tokenised with BPE
- Skip-gram (sg=1), epochs=10, min_count=3, window=5
- BPE piece strings used as FastText tokens (▁-prefixed word-initial markers)
- No OOV issue: FastText subword fallback handles any BPE piece

### Split Dates
- train < 2014-01-01, val < 2014-06-01, test ≥ 2014-06-01
- These are temporal boundaries based on Ubuntu corpus data distribution

### Known Deliberate Choices (not bugs)
- max_ctx_tokens=100 / max_resp_tokens=40: BPE subwords are more information-dense than words
- Diversity cap 500: prevents "I don't know" / "thank you" from dominating train set
- 1.5M train pairs cap: quality > quantity (vs prior 4.57M word-level pairs)

## Prompt

```
You are an NLP Research Reviewer / Academic Assessor reviewing an MSc AI Final Project
implementation plan. Evaluate whether the plan meets academic standards, whether the
experimental design is sound, and whether there are methodological gaps.

Read all files listed in new/agents/academic_reviewer.md under "Files to Read Before Running".

Background:
- Assignment: Seq2Seq WITHOUT attention (baseline) vs WITH Bahdanau attention, compared
  using BLEU + manual evaluation on Ubuntu Dialogue Corpus
- External benchmark: osamadev BLEU-4=0.1386 (LSTM+ATTN, 8k SentencePiece, Luong attention)
- New plan: 16-20k BPE, bidirectional encoder, Bahdanau attention, FastText embeddings
- Both models share same architecture except attention mechanism — intentional for fair comparison

Review checklist:
1. Is the baseline vs attention comparison parameter-fair?
2. Is BLEU appropriate for dialogue? Missing metrics?
3. Is the BLEU benchmark comparison to osamadev valid?
4. Is the TF schedule rationale sound or just copying osamadev?
5. Does temporal split prevent all data leakage (thread boundary integrity)?
6. Is 50 manual samples sufficient? Is there a rubric?
7. Is ~38-42M parameters appropriate for 1.5M pairs? Underfitting risk?
8. Does the plan address perplexity-quality inversion (Li et al. 2016)?
9. Are ablation studies present?
10. Are required citations present (Bahdanau 2014, Sutskever 2014, Liu 2016, etc.)?
11. Any other academic/methodological concerns.

Produce a structured report: METHODOLOGICAL flaws (would fail peer review),
GAPS in evaluation (missing metrics/analyses), RECOMMENDATIONS for strengthening.
```

---

## Findings

### Run 1 — 2026-03-14 (agent-2)

**3 Critical Flaws | 6 Evaluation Gaps | 6 Recommendations**

#### Critical Methodological Flaws
| ID | Issue | Location |
|---|---|---|
| F1 | Baseline vs Attention: different LSTM input sizes (300 vs 1324) — NOT parameter-fair | `PLAN.md §Phase 2; models.py` |
| F2 | BiDir encoder output=1024 vs decoder hidden=512 — BahdanauAttention dim mismatch; param count wrong | `PLAN.md §Architecture` |
| F3 | BLEU comparison to osamadev invalid — different tokeniser, vocab, attention type; use sacrebleu 13a | `PLAN.md §Benchmark; evaluate.py` |

#### Evaluation Gaps
| ID | Issue | Location |
|---|---|---|
| G1 | BLEU inadequate for dialogue (Liu et al. 2016); Distinct-1/2 and BERTScore missing | `evaluate.py` |
| G2 | Perplexity-quality inversion not measurable with current metrics — need Distinct-N | `evaluate.py` |
| G3 | 50 manual samples insufficient; no rubric, no inter-rater agreement | `PLAN.md §Phase 3` |
| G4 | No attention weight heatmap visualisation — expected output for Bahdanau project | `evaluate.py` |
| G5 | No bootstrap confidence intervals on BLEU scores | `evaluate.py` |
| G6 | Thread boundary integrity at temporal split not guaranteed — IRC threads can span boundary | `phase1.py Stage 3` |

#### Recommendations
| ID | Recommendation |
|---|---|
| R1 | Add TF schedule ablation study (minimum) — validates core claim |
| R2 | Reframe benchmark: osamadev as reference point, not direct comparison |
| R3 | Add missing citations: Sutskever 2014, Liu 2016, Li 2016, Kudo 2018, Bengio 2015 |
| R4 | Fix parameter count arithmetic (PLAN.md says 16.7M, correct is ~5.4M for that formula) |
| R5 | Implement `compute_distinct_n()` in `evaluate.py` |
| R6 | Implement attention weight heatmap in `evaluate.py` |

---
