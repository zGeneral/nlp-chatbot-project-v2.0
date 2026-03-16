# Model Evaluation

**Project:** NLP Seq2Seq LSTM Chatbot  
**Script:** `evaluate.py`  
**Test set:** 47,377 pairs — Ubuntu Dialogue Corpus (temporal split: post 2012-08-07)  
**Decoding:** Greedy (corpus metrics) | Top-p nucleus sampling (manual samples)  
**Hardware:** RTX 3080 12 GB, CUDA 12.4, torch 2.6.0+cu124

---

## Overview

Both trained models — the no-attention baseline and the Bahdanau attention model — are evaluated on the held-out test set using four complementary automatic metrics. BLEU alone is insufficient for open-domain dialogue (Liu et al., 2016); the full suite captures n-gram precision, recall, semantic similarity, and response diversity.

| Metric | What it measures | Scope |
|---|---|---|
| BLEU-1/2/3/4 | N-gram precision against reference responses | Surface |
| ROUGE-L | Longest common subsequence recall | Surface |
| BERTScore F1 | Contextual semantic similarity (DistilBERT) | Semantic |
| Distinct-1/2 | Unique unigram/bigram ratio — response diversity | Diversity |

Corpus metrics use **greedy decoding** for reproducibility. Manual evaluation samples use **top-p nucleus sampling** (p=0.9, temperature=0.8) to surface more representative conversational outputs.

---

## Results

### Automatic Metrics — Baseline vs Attention

| Metric | Baseline | Attention | Δ | Δ% |
|---|---|---|---|---|
| BLEU-1 | 0.0494 | **0.0535** | +0.0041 | +8.3% |
| BLEU-2 | 0.0136 | **0.0157** | +0.0021 | +15.4% |
| BLEU-3 | 0.0061 | **0.0074** | +0.0013 | +21.3% |
| BLEU-4 | 0.0037 | **0.0046** | +0.0009 | +24.3% |
| ROUGE-L F1 | 0.1240 | **0.1289** | +0.0049 | +3.9% |
| Distinct-1 | 0.0033 | **0.0048** | +0.0015 | +45.5% |
| Distinct-2 | 0.0166 | **0.0266** | +0.0100 | +60.2% |
| BERTScore F1 | 0.7041 | **0.7058** | +0.0017 | +0.2% |
| Test samples | 47,377 | 47,377 | — | — |
| Decode strategy | greedy | greedy | — | — |

The attention model outperforms the baseline across **all eight metrics**. The improvement is monotonically consistent — no metric regresses — providing a clean ablation result.

![Figure 6 — Full metric comparison across all 8 evaluation metrics](figures/fig6_metric_comparison.png)

*Figure 6: All eight metrics side-by-side. Attention (red) exceeds baseline (blue) on every measure.*

---

## Figure Breakdown

### BLEU-N Scores

BLEU measures the precision of n-gram matches between the model's output and the reference response. Four independent BLEU-N scores are reported (each computed as a separate corpus_bleu call with its own brevity penalty), following the sacrebleu 13a tokenisation standard.

The BLEU-N scores are low in absolute terms — this is expected and normal for open-domain dialogue on noisy IRC text. Liu et al. (2016) demonstrated that BLEU correlates poorly with human judgment in dialogue; a valid response to "how do I update ubuntu?" may be `sudo apt-get update && apt-get dist-upgrade`, `run dist-upgrade`, or `open software updater` — all correct, none sharing n-grams. The **relative improvement** from baseline to attention is what matters: BLEU-4 improves by +24.3%, indicating that the attention model produces longer stretches of consecutive correct tokens.

![Figure 7 — BLEU-1/2/3/4 breakdown](figures/fig7_bleu_breakdown.png)

*Figure 7: BLEU-N breakdown. The absolute gap widens with N (longer correct n-gram chains require coherence, which attention provides).*

The widening gap with N is diagnostic: BLEU-1 improves by 8% but BLEU-4 improves by 24%. This means the attention model is not just matching more individual words — it is generating more coherent multi-word sequences. Without attention, the decoder tends to repeat or drift after the first few tokens, breaking 4-gram chains. With attention, it can re-read the encoder's representation of the input at every step, maintaining topical coherence throughout the response.

### Response Diversity (Distinct-N)

Distinct-N measures the proportion of unique n-grams across all generated responses. Low Distinct-N indicates the model is collapsing to a small set of generic phrases.

![Figure 8 — Distinct-1/2 response diversity](figures/fig8_diversity_metrics.png)

*Figure 8: Response diversity. Attention produces 45% more unique unigrams and 60% more unique bigrams than the baseline.*

This is the most analytically significant result. The baseline's Distinct-2 of 0.0166 reflects severe **repetition collapse** — the decoder, receiving only a fixed encoder summary vector with no per-step input access, defaults to high-frequency tokens after the first step. This was directly observable in chat testing (`"vim is vim vim vim"`, repeated `sudo apt-get install openssh-server`). The attention model's Distinct-2 of 0.0266 (+60.2%) confirms that the Bahdanau mechanism structurally prevents this: the decoder queries the encoder output sequence at each step, obtaining a fresh context vector that depends on what has already been generated.

### BERTScore

BERTScore evaluates semantic similarity using contextual embeddings from DistilBERT (Sanh et al., 2019). Unlike BLEU, it does not require exact string matches — it rewards responses that are semantically equivalent to the reference even if differently phrased.

Both models achieve BERTScore F1 ≈ 0.70–0.71. The small delta (+0.17%) reflects that both models have learned the same Ubuntu domain vocabulary and semantics from the same training data. BERTScore is relatively insensitive to repetition and coherence compared to Distinct-N; its near-parity indicates that the models differ in *how* they express knowledge (diversity, coherence) rather than *what* domain knowledge they possess.

### ROUGE-L

ROUGE-L measures the longest common subsequence (LCS) between hypothesis and reference. A +3.9% improvement in ROUGE-L means the attention model generates longer in-order sequences that match the reference — consistent with the BLEU-4 finding.

---

## Attention Heatmap

The heatmap below visualises the Bahdanau attention weights for a single decoded response. Each row corresponds to one decoder step (one output token); each column corresponds to one encoder position (one input token). Bright cells indicate high attention weight — the decoder was focusing strongly on that input position when generating the output token.

![Figure 9 — Bahdanau attention weight heatmap](figures/fig9_attention_heatmap.png)

*Figure 9: Bahdanau attention heatmap. Rows = decoder steps (output tokens); columns = encoder positions (input tokens). Diagonal-dominant patterns indicate the decoder is tracking input position sequentially; off-diagonal focus indicates it is attending to specific earlier context tokens.*

The heatmap provides qualitative evidence that the attention mechanism is functioning correctly: the decoder does not assign uniform weight across all input positions (which would degenerate to the same fixed-vector behaviour as the baseline), but selectively focuses on different input positions depending on which output token is being generated.

---

## Metric Interpretation Notes

**Why are BLEU scores low?**
Open-domain dialogue BLEU is structurally bounded by reference diversity. The Ubuntu IRC corpus contains thousands of valid responses to any given question, but evaluation uses exactly one reference per test pair. Even a perfect model would achieve low BLEU against a single arbitrary reference. Low absolute BLEU is the expected operating point for this task; the *relative improvement* between models is the meaningful quantity.

**Why is BERTScore near-parity?**
Both models trained on the same 1.1M pairs for 20 epochs. They have learned identical domain knowledge — the Ubuntu vocabulary, command syntax, and common problem patterns. BERTScore at the semantic level is therefore expected to be similar. The structural advantage of attention (reduced repetition, longer coherent sequences) shows in Distinct-N and BLEU-4, not BERTScore.

**Why Distinct-N is the primary diagnostic**
For a chatbot, the most damaging failure mode is not low BLEU but repetition collapse — producing the same handful of generic responses regardless of input. Distinct-N directly measures this. The +60% Distinct-2 improvement is the strongest evidence that Bahdanau attention solves the core limitation of the fixed-vector seq2seq architecture.

---

## Appendix A — Metric Definitions

### A.1 BLEU (Bilingual Evaluation Understudy)

BLEU-N computes the geometric mean of n-gram precision scores up to order N, with a brevity penalty (BP) for short hypotheses:

```
BLEU-N = BP × exp( Σ wₙ log pₙ )

where:  pₙ = clipped n-gram matches / total n-grams in hypothesis
        wₙ = 1/N  (uniform weights)
        BP = min(1, exp(1 − r/c))  where r = reference length, c = hypothesis length
```

Each BLEU-1/2/3/4 is computed as an independent corpus-level score (separate brevity penalty per order) using sacrebleu 2.6.0 with 13a tokenisation and `force=True`.

### A.2 ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

```
Precision_LCS = LCS(H, R) / len(H)
Recall_LCS    = LCS(H, R) / len(R)
ROUGE-L F1    = (1 + β²) × P × R / (β² × P + R)   where β = 1.2
```

LCS is computed at sentence level. No stemming applied (`use_stemmer=False`).

### A.3 BERTScore

For each token tᵢ in hypothesis H and tⱼ in reference R, cosine similarity is computed between contextual embeddings from DistilBERT (distilbert-base-uncased):

```
Precision_BS = (1/|H|) Σᵢ max_j cos(hᵢ, rⱼ)
Recall_BS    = (1/|R|) Σⱼ max_i cos(rⱼ, hᵢ)
F1_BS        = 2 × P × R / (P + R)
```

Aggregated as mean F1 over all 47,377 test pairs, batch size 64, on CUDA device.

### A.4 Distinct-N

```
Distinct-N = |unique N-grams across all hypotheses| / |total N-grams across all hypotheses|
```

Computed at corpus level (not averaged per response). Values range 0–1; higher = more diverse.

---

## Appendix B — Model Configuration at Evaluation

| Parameter | Value |
|---|---|
| Test set size | 47,377 pairs |
| Corpus metric decoding | Greedy (argmax at each step) |
| Manual sample decoding | Top-p nucleus (p=0.9, temperature=0.8, 3-gram block) |
| Max decode length | 40 tokens |
| BERTScore model | distilbert-base-uncased |
| BERTScore batch | 64 |
| sacrebleu tokeniser | 13a |
| SentencePiece model | stage5_spm.model (16k BPE) |
| Baseline checkpoint | checkpoints/baseline_best.pt (epoch 18, val loss 5.9122) |
| Attention checkpoint | checkpoints/attention_best.pt |

---

## Appendix C — Design Decisions

### C.1 Why greedy decoding for corpus metrics?

Beam search and nucleus sampling are stochastic (top-p) or heuristic (beam). Greedy decoding is deterministic and reproducible: the same model always produces the same output, enabling fair comparison. Corpus-level BLEU, ROUGE-L, BERTScore, and Distinct-N are all computed on greedy output.

Manual evaluation samples use top-p to surface the model's conversational range — greedy frequently produces the safest/most generic response, which would underrepresent quality during manual review.

### C.2 Why DistilBERT for BERTScore?

DistilBERT (Sanh et al., 2019) is a distilled version of BERT-base with 66M parameters — fast enough to score 47k pairs in ~90 seconds on the RTX 3080. Full BERT-base or RoBERTa would provide marginally higher correlation with human judgment but would take 3–5× longer. For an ablation study between two architecturally similar models, DistilBERT is sufficient to detect the expected effect sizes.

### C.3 Why four metrics and not just BLEU-4?

Liu et al. (2016) demonstrated that BLEU, METEOR, and ROUGE fail to correlate with human judgment for open-domain dialogue because they measure n-gram overlap against a single reference while valid responses are highly diverse. The evaluation suite addresses this by reporting:
- **BLEU-4**: standard benchmark metric for cross-paper comparability
- **ROUGE-L**: recall-oriented LCS metric (different failure mode to BLEU)
- **BERTScore**: semantic similarity beyond surface n-gram matching
- **Distinct-1/2**: direct measurement of the repetition collapse failure mode

No single metric is definitive; the four together provide a multi-dimensional view of model quality.

---

## References

- Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL 2004 Workshop.
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020.
- Li, J. et al. (2016). *A Diversity-Promoting Objective Function for Neural Conversation Models.* NAACL 2016. (Distinct-N)
- Liu, C.-W. et al. (2016). *How NOT To Evaluate Your Dialogue System.* EMNLP 2016.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS 2019 Workshop.
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* WMT 2018. (sacrebleu)
