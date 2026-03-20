# Training Results Report
## Seq2Seq Chatbot — Ubuntu IRC Dialogue Corpus

---

## Overview

This report presents the training results of two sequence-to-sequence (seq2seq) neural dialogue models — a baseline BiLSTM encoder-decoder and a Bahdanau attention-augmented variant — trained on the Ubuntu Dialogue Corpus (Lowe et al., 2015) [1]. Both models were trained on an NVIDIA A100-80GB GPU with a batch size of 1024 over a maximum of 20 epochs, using a two-phase curriculum: Phase 1 (epochs 1–5) with full teacher forcing (TF=1.0), and Phase 2 (epochs 6–13) with linear teacher-forcing decay from 0.90 to 0.50 to mitigate exposure bias (Bengio et al., 2015) [2].

## Results

Both models were evaluated using teacher-forced validation loss (Val TF1), perplexity (PPL), corpus BLEU-4 (sacrebleu), and token-level F1 on 1,024 validation samples. Early stopping (patience = 4) fired at epoch 13 for both models, with best checkpoints saved at epoch 9.

| Model | Best Val (TF1) | PPL | Peak BLEU | Peak F1 | Best Epoch |
|---|---|---|---|---|---|
| Baseline (no attention) | 5.2015 | 181.5 | 0.70 | 14.61 | 9 |
| Attention (Bahdanau) | **5.1636** | **174.8** | **0.89** | **14.87** | 9 |

The attention model outperforms the baseline across all metrics. Val loss improvement of 0.038 nats corresponds to a 3.7% perplexity reduction. BLEU improved by 27% relative (0.70 → 0.89), reflecting the attention mechanism's ability to align decoder outputs to relevant source tokens — particularly important for technical terms such as command names and package identifiers in the Ubuntu IRC domain.

## Discussion

The modest absolute improvement (3.7% PPL) is consistent with findings on Ubuntu IRC (Kosovan et al., 2017) [3] and with Bahdanau et al.'s original observation that attention gains scale with output sequence length [4]. With average response lengths of 7–9 tokens, the fixed context vector already captures sufficient information for short responses, leaving limited room for alignment gains. Both models converged with identical structural patterns — Phase 2 TF decay was the dominant factor governing convergence shape rather than architecture, consistent with Mihaylova and Martins (2019) [5].

Token F1 showed a consistent upward trend throughout Phase 2 (1.03 → 14.87 for attention), confirming genuine learning of response structure in this technical support domain. GenLoss — computed under fully autoregressive decoding with post-EOS masking — declined steadily from epoch 1 to epoch 12 for both models, indicating improving free-running generation quality even as teacher-forced validation loss plateaued. This divergence is a known artefact of the TF mismatch between training and evaluation regimes.

The baseline PPL of 181.5 and attention PPL of 174.8 are well aligned with published seq2seq baselines on the Ubuntu corpus (Lowe et al., 2017) [6], validating the correctness of the training pipeline. The encoder bottleneck remains the primary architectural ceiling; future work with a stronger BiLSTM encoder or transformer-based architecture is expected to yield more substantial gains.

---

## References

[1] Lowe, R., Pow, N., Serban, I. V., & Pineau, J. (2015). *The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems*. SIGDIAL 2015. https://arxiv.org/abs/1506.08909

[2] Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*. NeurIPS 2015. https://arxiv.org/abs/1506.03099

[3] Kosovan, S., Lehmann, J., & Fischer, A. (2017). *Dialogue Response Generation using Neural Networks with Attention and Background Knowledge*. CSCUBS 2017. http://jens-lehmann.org/files/2017/cscubs_dialogues.pdf

[4] Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015. https://arxiv.org/abs/1409.0473

[5] Mihaylova, T., & Martins, A. F. T. (2019). *Scheduled Sampling for Transformers*. ACL 2019. https://aclanthology.org/P19-2049/

[6] Lowe, R., Pow, N., Serban, I. V., Charlin, L., Liu, C., & Pineau, J. (2017). *Training End-to-End Dialogue Systems with the Ubuntu Dialogue Corpus*. Dialogue & Discourse. https://www.cs.toronto.edu/~lcharlin/papers/ubuntu_dialogue_dd17.pdf

[7] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS 2014. https://arxiv.org/abs/1409.3215
