# Data Readiness Analysis — Seq2Seq GRU Chatbot
**Date:** 2026-03-14  
**Dataset:** Ubuntu Dialogue Corpus (Lowe et al., 2015)  
**Pipeline:** Phase 1 — 8-stage preprocessing (phase1.py)  
**Verified on:** Mini run (10% corpus, seed=42) — artifacts_mini/  
**Splits:** Train 150,000 / Val 6,206 / Test 6,436 pairs

---

## Summary

All eight scientific criteria for Seq2Seq readiness were evaluated programmatically
and through LLM-agent manual inspection of 150 decoded sample pairs (50 per split).

**Overall verdict: ✅ DATA IS SCIENTIFICALLY READY FOR SEQ2SEQ TRAINING.**

Two known monitoring points flagged (non-blocking):
- Test ctx is ~16% longer on average (distribution shift — later IRC messages are longer)
- Val/test URL rate is ~4pp higher than train (minor, same root cause)

---

## Criterion 1 — Sequence Contract

*What a GRU Seq2Seq model requires from its input/output integer sequences.*

| Requirement | Expectation | Result | Verdict |
|---|---|---|---|
| PAD token ID | 0 (`nn.Embedding(padding_idx=0)`) | `<pad> = 0` | ✅ |
| UNK token ID | 1 (OOV fallback) | `<unk> = 1` | ✅ |
| SOS token ID | 2 (decoder seed) | `<sos> = 2`; every resp[0]==2 (500/500 train, 200/200 val, 200/200 test) | ✅ |
| EOS token ID | 3 (generation stop) | `<eos> = 3`; every resp[-1]==3 (500/500, 200/200, 200/200) | ✅ |
| No PAD in middle | pack_padded_sequence requires no mid-sequence PAD | 0 violations across all splits | ✅ |
| No zero-length sequences | RNN crashes on length=0 | min ctx=2 tokens, min resp=7 tokens, zeros=0 | ✅ |

---

## Criterion 2 — Vocabulary Quality

*Fixed mapping from BPE token to integer ID, consistent across all splits.*

| Requirement | Expectation | Result | Verdict |
|---|---|---|---|
| Vocab size | Fixed, known at model init | 16,000 BPE tokens | ✅ |
| UNK rate in responses | Should be near 0 — UNK in target is unlearnable loss | **0.000%** across train, val, test | ✅ |
| Domain special tokens as single IDs | Multi-token tags split gradient signal | All 7 tags are single IDs (4–10): `__url__`=4, `__path__`=5, `__ip__`=6, `__cmd__`=7, `__number__`=8, `__eot__`=9, `__user__`=10 | ✅ |
| Consistent IDs across splits | Same vocabulary file used for all splits | Single SPM model (stage5_spm.model) encodes all splits in stage 6 | ✅ |

The 0.000% UNK rate is a direct result of BPE subword tokenisation: even rare or
unseen words are decomposed into known subword pieces, eliminating OOV in practice.

---

## Criterion 3 — Embedding Initialisation

*Pre-trained FastText vectors warm-start the encoder's embedding layer.*

| Requirement | Expectation | Result | Verdict |
|---|---|---|---|
| Pre-trained weights | Better convergence than random init | FastText skip-gram (sg=1), dim=300, 5 epochs, 164,851 pairs | ✅ |
| PAD row = zero vector | No gradient should flow through PAD | Row 0 = `[0, 0, ..., 0]` confirmed | ✅ |
| Full vocab coverage | No silent zero vectors that look like PAD | 15,999 / 16,000 rows filled (**99.99%**) | ✅ |
| Healthy norm distribution | Vectors not degenerate (all same direction or near-zero) | norm mean = 3.60, std = 0.591 (uniform, healthy) | ✅ |
| Special tokens have vectors | `__eot__`, `__user__`, etc. need semantic initialisation | All 7 tags present in FastText training corpus → have non-zero vectors | ✅ |

The single unfilled row is row 0 (`<pad>`), which is intentionally zeroed and excluded
from gradient updates via `padding_idx=0`. This is correct behaviour.

---

## Criterion 4 — Sequence Length Distribution

*Lengths must fit within the model's encoding capacity without excessive truncation.*

| Metric | Train | Val | Test | Note |
|--------|-------|-----|------|------|
| ctx mean (tokens) | 58.6 | 59.4 | 67.9 | ⚠️ test +16% |
| ctx p50 | 63 | 66 | 67 | consistent |
| ctx max (cap=100) | 100 | 100 | 100 | ✅ cap active |
| resp mean (tokens) | 16.2 | 17.4 | 17.8 | ✅ uniform |
| resp p50 | 14 | 15 | 15 | ✅ uniform |
| resp max (cap=42) | 42 | 42 | 42 | ✅ cap active |
| resp at length cap | 1.8% | 1.9% | 2.1% | ✅ negligible |

**Known distribution shift:** Test context sequences are ~16% longer on average.
This is inherent to the temporal split: data from 2012 (test/val window) contains
longer messages than the 2004–2012 training average. This is a known risk factor
to monitor via per-epoch val loss during training.

**Response length distribution** (word counts, text-level):

| Bucket | Train | Val | Test |
|--------|-------|-----|------|
| 5–9 words | 41.2% | 39.1% | 39.3% |
| 10–19 words | 38.6% | 38.5% | 38.3% |
| 20–29 words | 13.3% | 14.5% | 15.0% |
| 30–39 words | 6.4% | 7.6% | 7.1% |
| 40+ words | 0.5% | 0.4% | 0.3% |

Distributions are virtually identical across splits — no length bias introduced
by the temporal split or downsampling.

---

## Criterion 5 — Split Purity (No Data Leakage)

*Train, val, test must be statistically independent.*

| Requirement | Expectation | Result | Verdict |
|---|---|---|---|
| Zero thread overlap | No dialogue thread appears in two splits | T∩V=0, T∩Te=0, V∩Te=0 (verified in stage 3) | ✅ |
| Temporal ordering respected | Model trains on past, evaluates on future | Temporal split: train <2012-04-27, val <2012-08-07, test >2012-08-07 | ✅ |
| Echo pairs removed | Response must not be copied from context | 0 echo pairs confirmed across all three splits | ✅ |

The temporal split is intentionally strict: all dialogues from the same thread are
assigned to exactly one split, preventing any sentence-level overlap.

---

## Criterion 6 — Response Quality

*Target sequences must represent genuine, learnable human utterances.*

| Requirement | Expectation | Result | Verdict |
|---|---|---|---|
| Min response length | Avoid 1–4 token "noise targets" | min_resp_tokens=5 (raised from 3); min=7 incl. SOS+EOS | ✅ |
| No bot/automated responses | Model must not learn scripted responses | 28-pattern blacklist; 1,009 bot resp filtered in train | ✅ |
| No moderator boilerplate | Channel management text is not dialogue | Blacklist includes moderation patterns (watch language, offtopic, etc.) | ✅ |
| No echo pairs | Target ≠ source | 0 echo pairs, text-level verified | ✅ |
| No incoherent pairs | Context and response must be topically related | Backward-walk coherence filter removed 75,360 pairs (21% of train raw) | ✅ |
| Response diversity | Model must not collapse to top-N generic responses | Diversity ratio=0.992; max frequency=84× ("how do i do that") | ✅ |
| No responses at diversity cap | No resp hit 500× cap | 0 responses at cap across all splits | ✅ |

---

## Criterion 7 — Distribution Uniformity Across Splits

*Train, val, test must represent the same underlying distribution.*

| Feature | Train | Val | Test | Uniform? |
|---------|-------|-----|------|----------|
| 1-turn context | 26.8% | 27.4% | 27.2% | ✅ |
| 2-turn context | 16.6% | 16.7% | 16.0% | ✅ |
| 8-turn context (max) | 22.9% | 20.4% | 20.8% | ✅ |
| Short resp (5–9 words) | 41.2% | 39.1% | 39.3% | ✅ |
| Medium resp (10–19 words) | 38.6% | 38.5% | 38.3% | ✅ |
| `__eot__` rate in ctx | 73.2% | 72.6% | 72.8% | ✅ |
| `__user__` masking rate | 3.2% | 2.6% | 2.6% | ✅ |
| `__url__` rate | 12.7% | 17.0% | 17.5% | ⚠️ minor |
| `__path__` rate | 16.3% | 15.9% | 16.5% | ✅ |
| `__ip__` rate | 1.3% | 1.8% | 1.6% | ✅ |
| Bot name leakage (200 sample) | 0/200 | 0/200 | 0/200 | ✅ |
| Resp type/token ratio | 0.102 | 0.084 | 0.084 | ✅ |

**URL rate note:** val/test have ~4pp more `__url__` tokens than train. This is the
same temporal distribution shift — later IRC users referenced URLs more frequently.
The model will see fewer URL-containing responses during training than it will
encounter at evaluation time. Likely impact: slightly lower BLEU on URL-containing
test pairs. No code change required.

---

## Criterion 8 — Masking Integrity (PII / Identity Removal)

*Personally identifiable information must be masked so the model does not overfit
to specific identities or memorise personal data.*

Manually inspected 150 decoded text-level samples (50 per split) for leakage.

| Mask type | Token | Found unmasked? | Examples |
|-----------|-------|-----------------|---------|
| IRC usernames (handles) | `__user__` | ✅ None | Val[33]: `"best __user__"` — bot ref masked correctly |
| Known bot names (ubottu, chanserv, etc.) | `__user__` | ✅ None | 0/200 per split in bot-name check |
| IPv4 addresses | `__ip__` | ✅ None | Test[30]: `"nameserver __ip__"` — masked correctly |
| File system paths | `__path__` | ✅ None (masked) | Present as `__path__` placeholder |
| URLs | `__url__` | ✅ None (masked) | Present as `__url__` placeholder |

Masking approach:
- `_RE_URL`, `_RE_PATH`, `_RE_IP` applied in `_clean_text()` (stage 2)
- IRC addressee stripping: `^nick: message` pattern in `_clean_text()`
- IRC handle masking in stage 4: names with digits/underscores/hyphens or len>9
- Bot name masking: `_RE_BOT_NAMES` regex covers all 11 known IRC bots including possessive form (`ubottu's` → `__user__`)

---

## Fixes Applied During Data Preparation

The following issues were identified by the data analyst agent during iterative
pipeline development and resolved before finalising the dataset:

| Run | Issue | Fix applied |
|-----|-------|-------------|
| 1→2 | No turn delimiters in multi-turn ctx | `__eot__` separator between turns (Ubuntu corpus standard) |
| 1→2 | Unmasked IRC usernames in ctx/resp | Addressee strip regex + stage 4 IRC-handle masking |
| 1→2 | Bot/moderator scripted responses | `_BOT_RESPONSE_BLACKLIST` expanded from 5 to 28 patterns |
| 1→2 | Fragments below quality threshold | `min_resp_tokens` raised 3 → 5 |
| 2→3 | Stage 6 ctx kept oldest tokens | `ctx[:max_ctx_len]` → `ctx[-max_ctx_len:]` |
| 3→4 | BPE fragmented all 7 `__tags__` | All 7 tags added to SPM `user_defined_symbols` |
| 3→4 | Biased train downsampling (first-N) | `random.shuffle()` before `[:max_pairs]` cap |
| 4→5 | Coherence filter blind spot (short last turn) | Backward-walk through ctx turns, threshold 6→5 content words |

---

## Dataset Statistics (Mini Run — 10% of full corpus)

| Stage | Metric | Value |
|-------|--------|-------|
| 1 | Raw dialogues loaded | 1,852,868 |
| 1 | After 10% subsample (seed=42) | 185,286 |
| 2 | After dialogue-level filters | 131,445 (70.9% kept) |
| 3 | Train / Val / Test dialogues | 125,949 / 2,743 / 2,753 |
| 4 | Train / Val / Test pairs | 150,000 / 6,206 / 6,436 |
| 5 | BPE vocabulary size | 16,000 tokens |
| 6 | BPE-encoded split sizes | 150,000 / 6,206 / 6,436 |
| 7 | FastText embedding dim | 300 |
| 8 | Embedding coverage | 15,999 / 16,000 (99.99%) |

**Full run estimates** (scale by 10×):  
Train ~1,500,000 / Val ~62,000 / Test ~64,000 pairs

---

## References

- Lowe, R. et al. (2015). *The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems.* SIGDIAL 2015.
- Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. (BPE tokenisation)
- Bojanowski, P. et al. (2017). *Enriching Word Vectors with Subword Information.* TACL 2017. (FastText)
- Bahdanau, D. et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. (Attention mechanism)
