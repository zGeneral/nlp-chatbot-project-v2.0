# Checkpoint Backup — Run 1 (Baseline)

**Date:** 2026-03-19
**Branch:** A100-80GB
**Hardware:** NVIDIA A100-SXM4-80GB

## Config
| Parameter | Value |
|---|---|
| Batch size | 1024 × 1 accum = 1024 effective |
| LR | 3.0e-04 |
| Epochs max | 20 |
| Phase 1 end | 5 (TF=1.0) |
| Phase 2 TF decay | 1.0 → 0.50 linear |
| Early stop patience | 4 |
| Embed dim | 300 |
| Enc hidden | 512 (1024 bidir) |
| Dec hidden | 1024 |
| Attn dim | 256 |
| Layers | 2 |
| Dropout (embed/lstm/out) | 0.3 / 0.5 / 0.4 |

## Results
| Model | Best Epoch | Val(TF1) | PPL | Peak BLEU | Peak F1 | Stopped |
|---|---|---|---|---|---|---|
| baseline | 9 | 5.2015 | 181.5 | 0.70 | 14.61 | Epoch 13 |
| attention | 9 | 5.1636 | 174.8 | 0.89 | 14.87 | Epoch 13 |

## Files
- `baseline_best.pt` — best checkpoint (epoch 9)
- `baseline_last.pt` — last checkpoint (epoch 12, before early stop)
- `baseline_history.json` — full per-epoch metrics
- `attention_best.pt` — best checkpoint (epoch 9)
- `attention_last.pt` — last checkpoint (epoch 12, before early stop)
- `attention_history.json` — full per-epoch metrics
