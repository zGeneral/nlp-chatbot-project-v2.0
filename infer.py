"""
infer.py — Greedy and Beam Search inference for both seq2seq models.

Loads best checkpoints for baseline and attention models, runs both
greedy and beam search decoding on representative Ubuntu IRC prompts,
and prints a comparative results table for inclusion in the report.

Usage:
    python infer.py                     # default beam_size=5
    python infer.py --beam_size 3
    python infer.py --output json       # emit JSON for programmatic use
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

from config import CONFIG
from models import build_model


# ── Constants ────────────────────────────────────────────────────────────────

PAD  = CONFIG["pad_idx"]    # 0
UNK  = CONFIG["unk_idx"]    # 1
SOS  = CONFIG["sos_idx"]    # 2
EOS  = CONFIG["eos_idx"]    # 3
MAX_DECODE = 60             # hard cap for both greedy and beam
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Representative Ubuntu IRC test prompts ───────────────────────────────────
# Chosen to cover: package management, file system, network, permissions,
# error recovery, multi-word commands, and short/long context.

TEST_PROMPTS = [
    # Package management
    "how do i install vim",
    "apt-get is broken and i cant install anything",
    "how do i remove a package completely including config files",
    # File system
    "my hard drive is full what can i do",
    "how do i find large files on my system",
    "permission denied when i try to write to a file",
    # Network
    "my wifi is not working after upgrade",
    "how do i check my ip address",
    # Process / system
    "my system is very slow what should i check",
    "how do i see what processes are running",
    # Errors
    "i get a segfault when running my program",
    "bash command not found error how to fix",
]


# ── Encoding helpers ─────────────────────────────────────────────────────────

def encode_prompt(
    text: str,
    sp: spm.SentencePieceProcessor,
    max_len: int = CONFIG["max_ctx_tokens"],
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Encode text → (src [1, T], lengths [1])."""
    ids = sp.encode(text.strip(), out_type=int)
    ids = ids[:max_len]
    src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=DEVICE)
    return src, lengths


def decode_ids(
    ids: List[int],
    sp: spm.SentencePieceProcessor,
    strip_special: bool = True,
) -> str:
    """Decode token IDs → string, optionally stripping special tokens."""
    if strip_special:
        ids = [i for i in ids if i not in (PAD, UNK, SOS, EOS)]
    return sp.decode(ids)


# ── Encoder pass (shared by both decoding strategies) ────────────────────────

@torch.no_grad()
def encode(
    model,
    src: torch.LongTensor,
    src_lengths: torch.LongTensor,
):
    """Run encoder + bridge. Returns (encoder_outputs, h0, c0, src_mask)."""
    encoder_outputs, (h_n, c_n) = model.encoder(src, src_lengths)
    src_mask = (src == PAD)
    h0, c0 = model.bridge(h_n, c_n)
    return encoder_outputs, h0, c0, src_mask


# ── Greedy Decoding ───────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model,
    src: torch.LongTensor,
    src_lengths: torch.LongTensor,
    max_steps: int = MAX_DECODE,
) -> Tuple[List[int], float]:
    """
    Greedy (argmax) decoding for a single source sequence.

    Returns:
        token_ids: list of generated token IDs (without SOS, up to and
                   including first EOS if produced within max_steps)
        avg_log_prob: mean per-token log probability (lower = less confident)
    """
    model.eval()
    encoder_outputs, hidden, cell, src_mask = encode(model, src, src_lengths)

    context = torch.zeros(1, encoder_outputs.size(-1), device=DEVICE)
    input_token = torch.tensor([SOS], device=DEVICE)

    # Precompute attention keys once (attention model only; baseline ignores).
    keys_proj = None
    if hasattr(model.decoder, "attention"):
        keys_proj = model.decoder.attention.W_enc(encoder_outputs)

    generated = []
    log_probs  = []

    for _ in range(max_steps):
        if hasattr(model.decoder, "attention"):
            logits, hidden, cell, context, _ = model.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, context, src_mask,
                keys_proj=keys_proj,
            )
        else:
            logits, hidden, cell, context, _ = model.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, context, src_mask,
            )

        log_prob = F.log_softmax(logits, dim=-1)
        next_token = log_prob.argmax(dim=-1)           # [1]
        token_lp   = log_prob[0, next_token.item()].item()

        generated.append(next_token.item())
        log_probs.append(token_lp)

        if next_token.item() == EOS:
            break
        input_token = next_token

    avg_lp = sum(log_probs) / len(log_probs) if log_probs else 0.0
    return generated, avg_lp


# ── Beam Search Decoding ──────────────────────────────────────────────────────

@dataclass
class _Beam:
    tokens:  List[int]
    score:   float          # sum of log probs (unnormalised)
    hidden:  torch.Tensor
    cell:    torch.Tensor
    context: torch.Tensor
    done:    bool = False


@torch.no_grad()
def beam_decode(
    model,
    src: torch.LongTensor,
    src_lengths: torch.LongTensor,
    beam_size: int = 5,
    max_steps: int = MAX_DECODE,
    length_penalty: float = 0.7,
) -> Tuple[List[int], float]:
    """
    Beam search decoding for a single source sequence.

    Uses length normalisation: score = sum_log_prob / len ^ length_penalty
    (Wu et al., 2016 — Google NMT). This prevents beams from being
    systematically favoured for shorter lengths.

    Returns:
        best_tokens: token IDs of the highest-scoring complete beam
        norm_score:  length-normalised log probability score
    """
    model.eval()
    encoder_outputs, h0, c0, src_mask = encode(model, src, src_lengths)

    # Tile encoder states for beam_size independent beams.
    # [1, T, H] → [B, T, H]
    enc = encoder_outputs.repeat(beam_size, 1, 1)
    msk = src_mask.repeat(beam_size, 1)
    ctx = torch.zeros(beam_size, encoder_outputs.size(-1), device=DEVICE)

    # Tile bridge outputs along batch dim.
    h = h0.repeat(1, beam_size, 1)     # [num_layers, B, dec_hidden]
    c = c0.repeat(1, beam_size, 1)

    keys_proj = None
    if hasattr(model.decoder, "attention"):
        keys_proj = model.decoder.attention.W_enc(enc)

    # Initialise with one beam holding only SOS.
    beams: List[_Beam] = [
        _Beam(
            tokens=[SOS],
            score=0.0,
            hidden=h[:, i:i+1, :],
            cell=c[:, i:i+1, :],
            context=ctx[i:i+1, :],
            done=False,
        )
        for i in range(beam_size)
    ]
    # Only one active beam at start — collapse later ones after first step.
    # Simpler: run first step with batch=1, then expand.

    # --- Step 0: one-beam first step ---
    input_token = torch.tensor([SOS], device=DEVICE)
    if hasattr(model.decoder, "attention"):
        kp0 = model.decoder.attention.W_enc(encoder_outputs)
        logits, h0s, c0s, ctx0, _ = model.decoder.forward_step(
            input_token, h0, c0, encoder_outputs, ctx[0:1], src_mask, keys_proj=kp0,
        )
    else:
        logits, h0s, c0s, ctx0, _ = model.decoder.forward_step(
            input_token, h0, c0, encoder_outputs, ctx[0:1], src_mask,
        )

    log_probs_0 = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab]
    topk_lp, topk_ids = log_probs_0.topk(beam_size)

    # Seed beams from top-k tokens of step 0.
    beams = []
    for rank in range(beam_size):
        tok  = topk_ids[rank].item()
        lp   = topk_lp[rank].item()
        beams.append(_Beam(
            tokens=[SOS, tok],
            score=lp,
            hidden=h0s.clone(),
            cell=c0s.clone(),
            context=ctx0.clone(),
            done=(tok == EOS),
        ))

    completed: List[_Beam] = []

    # --- Main beam loop ---
    for step in range(1, max_steps):
        active = [b for b in beams if not b.done]
        if not active:
            break

        # Batch all active beams into one forward pass.
        n = len(active)
        batch_input  = torch.tensor([b.tokens[-1] for b in active], device=DEVICE)  # [n]
        batch_hidden = torch.cat([b.hidden for b in active], dim=1)  # [layers, n, H]
        batch_cell   = torch.cat([b.cell   for b in active], dim=1)
        batch_ctx    = torch.cat([b.context for b in active], dim=0)  # [n, H_enc]
        batch_enc    = encoder_outputs.repeat(n, 1, 1)
        batch_mask   = src_mask.repeat(n, 1)

        if hasattr(model.decoder, "attention"):
            kp = model.decoder.attention.W_enc(batch_enc)
            logits, new_h, new_c, new_ctx, _ = model.decoder.forward_step(
                batch_input, batch_hidden, batch_cell, batch_enc, batch_ctx,
                batch_mask, keys_proj=kp,
            )
        else:
            logits, new_h, new_c, new_ctx, _ = model.decoder.forward_step(
                batch_input, batch_hidden, batch_cell, batch_enc, batch_ctx, batch_mask,
            )

        log_probs = F.log_softmax(logits, dim=-1)  # [n, vocab]
        topk_lp, topk_ids = log_probs.topk(beam_size, dim=-1)  # [n, beam_size]

        candidates: List[_Beam] = []
        for i, beam in enumerate(active):
            for rank in range(beam_size):
                tok = topk_ids[i, rank].item()
                lp  = topk_lp[i, rank].item()
                new_beam = _Beam(
                    tokens=beam.tokens + [tok],
                    score=beam.score + lp,
                    hidden=new_h[:, i:i+1, :].clone(),
                    cell=new_c[:, i:i+1, :].clone(),
                    context=new_ctx[i:i+1, :].clone(),
                    done=(tok == EOS),
                )
                candidates.append(new_beam)

        # Keep top beam_size by length-normalised score.
        def _norm(b: _Beam) -> float:
            l = max(len(b.tokens) - 1, 1)   # exclude SOS from length
            return b.score / (l ** length_penalty)

        candidates.sort(key=_norm, reverse=True)
        beams = candidates[:beam_size]

        # Move completed beams to finished list.
        for b in beams:
            if b.done:
                completed.append(b)
        beams = [b for b in beams if not b.done]

        if len(completed) >= beam_size:
            break

    all_finished = completed + beams
    if not all_finished:
        return [EOS], 0.0

    best = max(all_finished, key=lambda b: b.score / max(len(b.tokens) - 1, 1) ** length_penalty)
    norm_score = best.score / max(len(best.tokens) - 1, 1) ** length_penalty
    # Strip SOS from output.
    return best.tokens[1:], norm_score


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_type: str) -> torch.nn.Module:
    """Load best checkpoint for the given model type."""
    ckpt_path = (
        f"/opt/app-root/src/nlp-chatbot-project-v2.0/checkpoints/{model_type}_best.pt"
    )
    model = build_model(model_type, CONFIG, DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(beam_size: int = 5, output_fmt: str = "text") -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(CONFIG["spm_model_path"])

    print(f"\nLoading models on {DEVICE} …", flush=True)
    baseline  = load_model("baseline")
    attention = load_model("attention")
    print("Models loaded.\n")

    results = []

    for prompt in TEST_PROMPTS:
        src, src_len = encode_prompt(prompt, sp)

        # Greedy
        base_g_ids,  base_g_lp  = greedy_decode(baseline,  src, src_len)
        attn_g_ids,  attn_g_lp  = greedy_decode(attention, src, src_len)

        # Beam
        base_b_ids,  base_b_sc  = beam_decode(baseline,  src, src_len, beam_size=beam_size)
        attn_b_ids,  attn_b_sc  = beam_decode(attention, src, src_len, beam_size=beam_size)

        results.append({
            "prompt":         prompt,
            "base_greedy":    decode_ids(base_g_ids,  sp),
            "base_greedy_lp": round(base_g_lp, 4),
            "base_beam":      decode_ids(base_b_ids,  sp),
            "base_beam_sc":   round(base_b_sc, 4),
            "attn_greedy":    decode_ids(attn_g_ids,  sp),
            "attn_greedy_lp": round(attn_g_lp, 4),
            "attn_beam":      decode_ids(attn_b_ids,  sp),
            "attn_beam_sc":   round(attn_b_sc, 4),
        })

    if output_fmt == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # ── Pretty-print table ───────────────────────────────────────────────────
    sep  = "─" * 120
    sep2 = "═" * 120

    print(sep2)
    print(f"  INFERENCE RESULTS  |  Beam size = {beam_size}")
    print(sep2)

    for r in results:
        print(f"\n  PROMPT : {r['prompt']}")
        print(sep)
        print(f"  {'':30s}  {'BASELINE':42s}  {'ATTENTION':42s}")
        print(f"  {'':30s}  {'─'*42}  {'─'*42}")
        print(f"  {'Greedy':30s}  {r['base_greedy'][:42]:42s}  {r['attn_greedy'][:42]:42s}")
        print(f"  {'Greedy avg log-prob':30s}  {r['base_greedy_lp']:<42.4f}  {r['attn_greedy_lp']:<42.4f}")
        print(f"  {f'Beam (k={beam_size})':30s}  {r['base_beam'][:42]:42s}  {r['attn_beam'][:42]:42s}")
        print(f"  {'Beam norm score':30s}  {r['base_beam_sc']:<42.4f}  {r['attn_beam_sc']:<42.4f}")
        print(sep)

    print(sep2)
    print("  SUMMARY")
    print(sep2)

    # Greedy beam agreement rate
    base_agree = sum(1 for r in results if r["base_greedy"] == r["base_beam"])
    attn_agree = sum(1 for r in results if r["attn_greedy"] == r["attn_beam"])
    # Attention vs baseline agreement on greedy
    model_agree = sum(1 for r in results if r["base_greedy"] == r["attn_greedy"])

    print(f"  Baseline  — greedy/beam agreement : {base_agree}/{len(results)}")
    print(f"  Attention — greedy/beam agreement : {attn_agree}/{len(results)}")
    print(f"  Greedy output match (base vs attn): {model_agree}/{len(results)}")
    print(sep2)

    # Save JSON alongside
    out_path = "/opt/app-root/src/nlp-chatbot-project-v2.0/reports/inference_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy + Beam inference for seq2seq chatbot")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam width for beam search (default: 5)")
    parser.add_argument("--output", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    args = parser.parse_args()
    main(beam_size=args.beam_size, output_fmt=args.output)
