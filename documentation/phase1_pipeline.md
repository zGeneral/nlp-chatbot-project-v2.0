# Phase 1: Data Preparation Pipeline

## 1. Overview

`phase1.py` is a run-once, eight-stage data pipeline that transforms the raw Ubuntu
Dialogue Corpus (Lowe et al., 2015) into domain-filtered, BPE-tokenised
(context, response) pairs and a 300-dimensional corpus-trained embedding matrix ready
for Seq2Seq training. Each stage is idempotent — checking for its output artefacts on
disk before executing — enabling safe resumption after interruption. Quality filtering,
temporal data splitting, subword tokenisation, domain restriction, and embedding
initialisation are each encapsulated as discrete, independently verifiable units.

---

## 2. Pipeline Architecture

```
Ubuntu Dialogue Corpus (CSV files)
           │
    ┌──────▼──────┐
    │   Stage 1   │  Load raw CSV files → folder/dialogueID-keyed dialogue dicts
    │  Load Corpus│  Sorted turns; undated turns discarded
    └──────┬──────┘
           │  raw dialogue dicts
    ┌──────▼──────┐
    │   Stage 2   │  11 quality filters (dialogue-level)
    │ Clean+Filter│  Parallel: spawn pool, chunk_size=50,000, workers=cpu_count−2
    └──────┬──────┘
           │  clean dialogues
    ┌──────▼──────┐
    │   Stage 3   │  Split by thread first-turn date (no random shuffle)
    │  Temporal   │  train: <2012-04-27 (~80%)
    │    Split    │  val:  <2012-08-07 (~10%)
    └──────┬──────┘  test:  ≥2012-08-07 (~10%)   zero-overlap asserted
           │  train / val / test dialogue splits
    ┌──────▼──────┐
    │   Stage 4   │  (context, response) string pairs; __eot__ turn delimiter
    │   Generate  │  max_ctx_turns=8, max_ctx_tokens=100, max_resp_tokens=40
    │    Pairs    │  min_resp_tokens=5, train cap=1,500,000, diversity_cap=500
    └──────┬──────┘  8 pair-level quality filters
           │  ≤1.5M train pairs + val/test pairs
    ┌──────▼──────┐
    │  Stage 4.5  │  Domain-focused filter  strategy=union
    │   Domain    │  Strategy A: Linux command regex OR __path__ token
    │   Filter    │  Strategy B: question pattern on last substantive ctx turn
    └──────┬──────┘  ~73% of pairs retained (both train and eval splits)
           │  domain-filtered pairs
    ┌──────▼──────┐
    │   Stage 5   │  SentencePiece BPE  vocab_size=16,000
    │  Train SPM  │  character_coverage=0.9999
    │             │  pad=0  unk=1  sos=2  eos=3  (7 user-defined symbols)
    └──────┬──────┘
           │  stage5_spm.model  +  stage5_spm.vocab
    ┌──────▼──────┐
    │   Stage 6   │  ctx  → sp.encode(text)[-100:]       (last 100 BPE tokens)
    │  Encode IDs │  resp → [sos=2] + sp.encode(text)[:40] + [eos=3]
    └──────┬──────┘  JSONL  +  vocab.json  +  idx2word.json
           │  stage6_*.jsonl
    ┌──────▼──────┐
    │   Stage 7   │  Gensim FastText  skip-gram (sg=1)
    │   FastText  │  dim=300, epochs=10, min_count=3, window=5, workers=8
    │  Training   │  Corpus: ALL pairs (train + val + test)
    └──────┬──────┘
           │  stage7_fasttext.model
    ┌──────▼──────┐
    │   Stage 8   │  [16,000 × 300]  float32  numpy matrix
    │  Embedding  │  Exact ▁-prefixed BPE piece strings as lookup keys
    │   Matrix    │  <pad> row (index 0) forced to zeros
    └─────────────┘
           │
    stage8_embedding_matrix.npy
```

**Figure 1: Phase 1 Eight-Stage Data Preparation Pipeline**

---

## 3. Stage Summary

**Table 1: Stage Summary**

| Stage | Function | Key Output Artefacts |
|---|---|---|
| 1 | Load Ubuntu Dialogue Corpus CSV files into structured dialogue dicts | `stage1_dialogues.pkl` |
| 2 | Apply 11 dialogue-level quality filters; clean utterance text in parallel | `stage2_clean_dialogues.pkl`, `stage2_stats.json` |
| 3 | Temporal split by thread first-turn date into train / val / test | `stage3_{train,val,test}.pkl`, `stage3_stats.json` |
| 4 | Generate `(context, response)` string pairs with 8 pair-level filters | `stage4_{train,val,test}_pairs.json`, `stage4_stats.json` |
| 4.5 | Domain-focused filter: retain command-related or question-style pairs | `stage4_5_{train,val,test}_pairs.json`, `stage4_5_filter_stats.json` |
| 5 | Train SentencePiece BPE model on filtered training text | `stage5_spm.model`, `stage5_spm.vocab` |
| 6 | Encode all pairs to BPE token ID sequences; save vocab mappings | `stage6_{train,val,test}_ids.jsonl`, `stage6_vocab.json`, `stage6_idx2word.json` |
| 7 | Train Gensim FastText skip-gram model on BPE-tokenised full corpus | `stage7_fasttext.model` |
| 8 | Build `[16,000 × 300]` float32 embedding matrix aligned to BPE vocab | `stage8_embedding_matrix.npy`, `stage8_stats.json` |

---

## 4. Configuration Parameters

**Table 2: Phase 1 Configuration Parameters**

| Parameter | Value | Description |
|---|---|---|
| `max_ctx_turns` | 8 | Maximum preceding turns included in context window |
| `max_ctx_tokens` | 100 | Maximum BPE tokens retained from context (rightmost) |
| `max_resp_tokens` | 40 | Maximum BPE tokens in response |
| `min_resp_tokens` | 5 | Minimum response length (eliminates short fragments) |
| `min_ctx_tokens` | 3 | Minimum context word count (drops degenerate pairs) |
| `max_train_pairs` | 1,500,000 | Training pair cap (randomly sampled; 0 = no cap) |
| `max_response_occurrences` | 500 | Diversity cap: max identical response occurrences (train) |
| `train_cutoff_date` | 2012-04-27 | Thread first-turn date threshold for train split |
| `val_cutoff_date` | 2012-08-07 | Thread first-turn date threshold for val split |
| `spm_vocab_size` | 16,000 | SentencePiece BPE vocabulary size |
| `spm_character_coverage` | 0.9999 | Fraction of characters covered by the vocabulary |
| `fasttext_dim` | 300 | FastText embedding dimensionality |
| `fasttext_epochs` | 10 | FastText training epochs |
| `fasttext_window` | 5 | FastText context window size |
| `fasttext_sg` | 1 | 1 = skip-gram; 0 = CBOW |
| `fasttext_min_count` | 3 | Minimum piece frequency to include in FastText vocabulary |
| `domain_filter_strategy` | union | Retain pairs matching command regex OR question pattern |
| `max_turn_gap_seconds` | 3,600 | Hard ceiling on single-turn timestamp gap |
| `max_single_speaker_ratio` | 0.80 | Max fraction of turns from one speaker before dropping |
| `min_alternation_ratio` | 0.15 | Minimum fraction of consecutive speaker changes |

---

## 5. Text Normalisation

All utterances pass through `_clean_text()` before any stage. Steps are applied in order.

**Table 3: Text Cleaning Steps Applied to Every Utterance**

| Step | Operation | Example |
|---|---|---|
| 1 | URL normalisation | `https://askubuntu.com` → `__url__` |
| 2 | Filesystem path normalisation | `/etc/apt/sources.list` → `__path__` |
| 3 | IPv4 address masking | `192.168.1.1:8080` → `__ip__` |
| 4 | IRC `<nick>` prefix stripping | `<actionparsnip> try this` → `try this` |
| 5 | IRC addressee removal | `nick: message` → `message` |
| 6 | Lowercasing | `Ubuntu` → `ubuntu` |
| 7 | Contraction expansion (53 mappings) | `can't` → `cannot`, `it's` → `it is` |
| 8 | Non-alphanumeric removal (preserves `'`, `-`, `_`, `.`) | `?!@#` → ` ` |
| 9 | Known IRC bot name masking | `ubottu`, `chanserv`, etc. → `__user__` |
| 10 | Whitespace collapse | multiple spaces → single space |

---

## 6. Quality Filters

**Table 4: Dialogue-Level Quality Filters (Stage 2)**

| Filter | Drop Condition | Threshold |
|---|---|---|
| Valid dates | Fewer than 2 turns with parseable timestamps | < 2 |
| Bot speaker removal | Turn speaker in known IRC bot list | 10 bots |
| IRC action drops | Turn matches `/me` emote pattern | regex |
| Paste detection | Alpha ratio < 0.30, special-char density > 0.15, or ≥ 3 colons in < 200 chars | see values |
| Repetitive turn | Single token accounts for > 50% of turn words | > 0.50 |
| Dyadic-only | Dialogue has ≠ 2 unique speakers | ≠ 2 |
| Minimum turns | Fewer than 2 cleaned turns remain | < 2 |
| Speaker dominance | One speaker accounts for > 80% of turns | > 0.80 |
| Alternation ratio | Fewer than 15% of consecutive turns change speaker | < 0.15 |
| Temporal hard ceiling | Any single inter-turn gap exceeds 3,600 s | > 3,600 s |
| Temporal gap ratio | More than 30% of gaps exceed 600 s | > 0.30 |

**Table 5: Pair-Level Quality Filters (Stage 4)**

| Filter | Drop Condition |
|---|---|
| Context length | Fewer than 3 words in full context |
| Response too short | Fewer than 5 words |
| Response too long | More than 40 words |
| Non-English | ASCII alpha ratio < 80% |
| Bot response | Response matches 21-entry boilerplate blacklist |
| Echo pair | Response text appears verbatim inside context |
| Placeholder-only | Response contains only `__url__`, `__path__`, etc. |
| Incoherent pair | Zero content-word overlap between last substantive context turn and response (both ≥ 5 content words) |
| Diversity cap | Same response string appears > 500 times in training set |

---

## 7. Architecture Key Decisions

**7.1 Temporal split over random shuffle**
Splitting dialogues by thread first-turn date prevents thread-boundary leakage: a single IRC thread spanning a random cut would supply semantically identical turns to both train and test sets, artificially inflating evaluation scores. Zero-overlap is asserted programmatically at runtime.

**7.2 SentencePiece BPE and 16,000 vocabulary**
BPE (Sennrich et al., 2016; Kudo & Richardson, 2018) eliminates out-of-vocabulary tokens, reduces the softmax from ~50,000 word-level to 16,000 subword pieces, and handles Ubuntu-specific identifiers compositionally. The 16k size achieves character coverage of 0.9999 on the filtered corpus while remaining computationally tractable for a 44-million-parameter model.

**7.3 Corpus-trained FastText over external embeddings**
External embeddings trained on general web text lack Ubuntu-specific subword pieces such as `▁apt-get` and `▁systemctl`. Corpus-trained FastText (Bojanowski et al., 2017) produces vectors aligned to the exact BPE vocabulary. Skip-gram (Mikolov et al., 2013) is preferred over CBOW because it learns better representations for rare pieces that carry high technical signal in this domain.

**7.4 FastText trained on all three splits**
Training FastText on train + val + test ensures that every token encountered at evaluation time receives a meaningful initialisation rather than a zero vector. Because embeddings are unfrozen during Seq2Seq training (`freeze=False`), any mild co-occurrence leakage is progressively overwritten.

**7.5 Domain filtering**
The Ubuntu Dialogue Corpus contains substantial off-topic IRC chatter. The union-strategy domain filter retains ~73% of pairs — those containing Linux commands or question patterns — sharpening vocabulary distribution and reducing the averaging signal that leads to generic responses.

**7.6 1.5 million pair cap**
The pre-filtering corpus yields approximately 4.57 million pairs, many near-duplicate or generic. Capping at 1.5 million randomly sampled quality-filtered pairs reduces averaging noise, avoids temporal bias from alphabetical ordering, and fits within GPU memory constraints.

**7.7 `__eot__` delimiter and encoder input contract**
`__eot__` is declared as a `user_defined_symbols` entry in SentencePiece, guaranteeing it is never split into sub-pieces by BPE. The encoder receives no `<sos>` prefix, preserving all 100 context token positions for dialogue content; `<sos>` and `<eos>` are used exclusively on the decoder side.

---

## References

Bojanowski, P., Grave, E., Joulin, A. and Mikolov, T. (2017) 'Enriching Word Vectors with Subword Information', *Transactions of the Association for Computational Linguistics*, 5, pp. 135–146. Available at: https://arxiv.org/abs/1607.04606 (Accessed: 17 March 2026).

Kudo, T. and Richardson, J. (2018) 'SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing', *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 66–71. Available at: https://arxiv.org/abs/1808.06226 (Accessed: 17 March 2026).

Lowe, R., Pow, N., Serban, I. and Pineau, J. (2015) 'The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems', *Proceedings of the 16th Annual Meeting of the Special Interest Group on Discourse and Dialogue*, pp. 285–294. Available at: https://arxiv.org/abs/1506.08909 (Accessed: 17 March 2026).

Mikolov, T., Chen, K., Corrado, G. and Dean, J. (2013) 'Efficient Estimation of Word Representations in Vector Space', *Proceedings of the International Conference on Learning Representations (ICLR) Workshop*. Available at: https://arxiv.org/abs/1301.3781 (Accessed: 17 March 2026).

Řehůřek, R. and Sojka, P. (2010) 'Software Framework for Topic Modelling with Large Corpora', *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, pp. 45–50. Available at: https://radimrehurek.com/gensim/ (Accessed: 17 March 2026).

Sennrich, R., Haddow, B. and Birch, A. (2016) 'Neural Machine Translation of Rare Words with Subword Units', *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, pp. 1715–1725. Available at: https://arxiv.org/abs/1508.07909 (Accessed: 17 March 2026).

---

## Glossary

**Table 6: Glossary of Key Terms**

| Term | Definition |
|---|---|
| Alternation ratio | Fraction of consecutive dialogue turns where the speaker changes; used to filter non-alternating flood-style dialogues. |
| Artefact-idempotent pipeline | A pipeline where each stage checks for its output files before running; allows safe re-entry after interruption without recomputing prior stages. |
| BPE (Byte Pair Encoding) | Subword tokenisation algorithm that iteratively merges the most frequent character pair, producing a compact vocabulary that handles rare and compound words without OOV tokens (Sennrich et al., 2016). |
| CBOW (Continuous Bag of Words) | Word embedding training objective that predicts a centre word from its surrounding context; contrast with skip-gram. |
| Context-response pair | A training sample consisting of a multi-turn dialogue context and the next-turn response, used as input–output for Seq2Seq training. |
| Corpus-trained embeddings | Word vectors trained from scratch on the project's own dataset rather than loaded from an externally pre-trained source. |
| Diversity cap | A ceiling on how many times any identical response string may appear in the training set; prevents high-frequency generic responses from dominating training. |
| Domain filter | Stage 4.5 filter that retains only (context, response) pairs containing Linux commands or question patterns, restricting the corpus to Ubuntu-relevant exchanges. |
| Dyadic dialogue | A conversation involving exactly two participants; multi-party threads are discarded to reduce conversational ambiguity. |
| `__eot__` token | A user-defined SentencePiece token marking the boundary between dialogue turns within a concatenated context sequence. |
| FastText | Subword-aware word embedding model (Bojanowski et al., 2017) that represents each token as the sum of its character n-gram vectors, enabling meaningful representations for unseen or rare subword pieces. |
| Gensim | Open-source Python library for unsupervised topic and vector space modelling, used here to train and serve the FastText model (Řehůřek & Sojka, 2010). |
| Idempotent | Property of an operation that produces the same result whether applied once or multiple times; each pipeline stage is idempotent via artefact-existence checks. |
| OOV (Out-of-Vocabulary) | A token not present in the model's vocabulary; eliminated by BPE, which decomposes any string into known subword pieces. |
| Paste detection | Heuristic filter that drops utterances likely to be copied terminal output, identified by low alphabetic ratio (< 0.30) or high special-character density (> 0.15). |
| Response diversity cap | See *Diversity cap*. |
| SentencePiece | Language-independent subword tokeniser that trains directly on raw text and enforces explicit special-token IDs via trainer parameters (Kudo & Richardson, 2018). |
| Skip-gram | Word embedding training objective that predicts surrounding context words from a centre word; preferred over CBOW for rare tokens (Mikolov et al., 2013). |
| Temporal split | Data partitioning strategy that assigns dialogues to train/val/test based on chronological order of the thread's first turn, preventing information leakage across split boundaries. |
| Thread-boundary leakage | Data leakage that occurs when a single IRC thread (dialogue) contributes turns to more than one split; prevented here by assigning each thread to exactly one split. |
| Ubuntu Dialogue Corpus | A large English conversational dataset derived from Ubuntu IRC technical support channels (Lowe et al., 2015), used as the training corpus for this project. |
| Union strategy | Domain filter mode that retains a pair if it satisfies Strategy A (command/path signal) OR Strategy B (question pattern); retains ~73% of pairs. |
| User-defined symbols | SentencePiece trainer parameter that forces specified strings (e.g. `__eot__`, `__url__`) to be treated as indivisible single tokens, never split by the BPE algorithm. |
| Vocabulary coverage | Fraction of all characters in the training corpus representable by the chosen vocabulary; set to 0.9999 to handle rare Unicode characters without OOV. |
