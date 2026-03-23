# Run 2 — Checkpoints & Results

## Baseline (BiLSTM seq2seq, no attention)
- Best epoch: 20/20  Val(TF1)=5.1735  PPL=176.54  F1=14.46  Len=6.7
- LR never reduced (model improved every epoch, patience=2 never exhausted)
- No early stopping (improved every epoch through 20)
- Checkpoint: baseline_best.pt (NOT in git — 651 MB)

## Attention (Bahdanau)
- Best epoch: 20/20  Val(TF1)=5.1410  PPL=170.88  F1=13.91  Len=6.8
- Entropy: 2.967 (ep1) → 2.186 (ep12, fast phase) → 2.125 (ep20, slow plateau)
- Entropy plateau from ep12 onward = attention saturated on current data
- LR never reduced, no early stopping, improved every epoch
- Checkpoint: attention_best.pt (NOT in git — 657 MB)

## Run 2 vs Run 1
| Model            | Val(TF1) | PPL    |
|------------------|----------|--------|
| Run1 baseline    | 5.2015   | 181.5  |
| Run1 attention   | 5.1636   | 174.8  |
| Run2 baseline    | 5.1735   | 176.5  |
| Run2 attention   | 5.1410   | 170.9  |

## Config changes vs Run 1
- phase1_end: 5 → 3
- patience: 4 → 6
- lr_scheduler_patience: 4 → 2
- Added: log_decoded_samples(), compute_attention_entropy()
- DataLoader deadlock fix (num_workers=0 in eval functions)

## Diagnosis
- LR never fired because model improved every epoch (patience=2 never exhausted)
- Entropy plateau at ~2.13 signals data quality ceiling AND mild encoder weakness
- Root cause: 30% of training pairs were short noisy responses (<7 words)
- Fix: Run 3 uses cleaned data (min_resp_tokens=7, diversity_cap=5, filter-first-cap)
