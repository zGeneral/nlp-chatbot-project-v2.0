"""
analyze_data.py — Post-pipeline data analysis utilities.

Provides token fertility measurement (H-3) and UNK forensics (nb03-G-1).
Run directly or import individual functions into notebooks / CI checks.

Usage:
    python analyze_data.py --jsonl artifacts/stage6_train_ids.jsonl \
                           --spm   artifacts/stage5_spm.model
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterator


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_jsonl_sample(jsonl_path: Path, sample_n: int, seed: int) -> Iterator[dict]:
    """Yield up to sample_n records from a JSONL file in a reproducible order."""
    rng = random.Random(seed)
    # Reservoir-sample to avoid loading the full file into memory.
    reservoir: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if i < sample_n:
                reservoir.append(record)
            else:
                j = rng.randint(0, i)
                if j < sample_n:
                    reservoir[j] = record
    rng.shuffle(reservoir)
    yield from reservoir


# ── H-3 — Token fertility ─────────────────────────────────────────────────────

def analyse_token_fertility(                                                     # FIX: H-3
    sp_model_path: Path,
    jsonl_path: Path,
    sample_n: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Measure subword fertility: average number of BPE tokens per whitespace word.

    Well-trained BPE at 16K on English typically gives 1.2–1.8 tokens/word.
    Values above 2.5 indicate over-fragmentation.

    Reads raw 'ctx' / 'resp' text fields from the JSONL (not the pre-encoded
    ID sequences) so the JSONL must contain both text and ID fields, or a
    separate raw-text JSONL can be provided.
    """
    import sentencepiece as spm_module
    sp = spm_module.SentencePieceProcessor(model_file=str(sp_model_path))

    char_counts, tok_counts, word_counts = [], [], []
    for rec in _load_jsonl_sample(jsonl_path, sample_n, seed):
        raw_ctx  = rec.get("ctx_text",  rec.get("ctx",  ""))
        raw_resp = rec.get("resp_text", rec.get("resp", ""))
        for text in [raw_ctx, raw_resp]:
            if not text or not isinstance(text, str):
                continue
            words = text.split()
            toks  = sp.encode(text, out_type=str)
            if words:
                char_counts.append(sum(len(w) for w in words))
                tok_counts.append(len(toks))
                word_counts.append(len(words))

    avg_toks_per_word = sum(tok_counts)  / max(sum(word_counts), 1)
    avg_chars_per_tok = sum(char_counts) / max(sum(tok_counts),  1)

    result = {
        "avg_tokens_per_word": round(avg_toks_per_word, 3),
        "avg_chars_per_token": round(avg_chars_per_tok, 3),
        "sample_size":         sample_n,
    }

    if not (1.0 <= avg_toks_per_word <= 2.5):
        print(f"  ⚠  Fertility {avg_toks_per_word:.3f} tok/word is outside "
              f"expected range [1.0, 2.5] — check SPM vocab size and training data")
    else:
        print(f"  ✓  Fertility {avg_toks_per_word:.3f} tok/word  "
              f"({avg_chars_per_tok:.2f} chars/tok) — within normal range")

    return result


# ── nb03-G-1 — UNK forensics ─────────────────────────────────────────────────

def analyse_unk_tokens(                                                          # FIX: nb03-G-1
    jsonl_path: Path,
    sp_model_path: Path,
    unk_id: int = 1,
    sample_n: int = 50_000,
    seed: int = 42,
    top_k: int = 20,
) -> dict:
    """
    Count UNK token occurrences across all sequences and identify which
    original text fragments most commonly produce UNK.

    Reads pre-encoded 'ctx' / 'resp' ID sequences from the JSONL, and the
    raw text fields ('ctx_text' / 'resp_text' or 'ctx' / 'resp' as strings)
    for word-level forensics.
    """
    import sentencepiece as spm_module
    sp = spm_module.SentencePieceProcessor(model_file=str(sp_model_path))

    total_tokens = 0
    unk_count    = 0
    unk_triggers = Counter()

    for rec in _load_jsonl_sample(jsonl_path, sample_n, seed):
        # Count UNK in pre-encoded ID sequences
        for seq_key in ("ctx", "resp"):
            ids = rec.get(seq_key, [])
            if isinstance(ids, list):
                total_tokens += len(ids)
                for tok_id in ids:
                    if tok_id == unk_id:
                        unk_count += 1

        # Forensics: encode raw text words to find UNK triggers
        for text_key in ("ctx_text", "resp_text", "ctx", "resp"):
            text = rec.get(text_key, "")
            if not text or not isinstance(text, str):
                continue
            for word in text.split():
                word_ids = sp.encode(word, out_type=int)
                if unk_id in word_ids:
                    unk_triggers[word] += 1
            break  # only use first available text field per role

    unk_rate = unk_count / max(total_tokens, 1)
    result   = {
        "total_tokens":  total_tokens,
        "unk_count":     unk_count,
        "unk_rate":      round(unk_rate, 8),
        "top_unk_words": unk_triggers.most_common(top_k),
    }

    print(f"  UNK rate : {unk_rate:.2e}  ({unk_count:,} / {total_tokens:,} tokens)")
    if unk_rate > 1e-3:
        print(f"  ⚠  UNK rate > 0.1% — investigate top triggers below")
    print(f"  Top {top_k} UNK-triggering words:")
    for word, cnt in result["top_unk_words"][:top_k]:
        print(f"    {word:<30} {cnt:>8,}")

    return result


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NLP data analysis utilities")
    parser.add_argument("--jsonl",    required=True, type=Path, help="Stage-6 JSONL path")
    parser.add_argument("--spm",      required=True, type=Path, help="Stage-5 SPM model path")
    parser.add_argument("--sample",   type=int, default=5000,   help="Sample size for fertility")
    parser.add_argument("--unk-sample", type=int, default=50_000, help="Sample size for UNK analysis")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--top-k",   type=int, default=20,      help="Top-K UNK words to show")
    args = parser.parse_args()

    print("\n── Token fertility ─────────────────────────────────────────────")
    fertility = analyse_token_fertility(args.spm, args.jsonl, args.sample, args.seed)
    print(f"  avg_tokens_per_word : {fertility['avg_tokens_per_word']}")
    print(f"  avg_chars_per_token : {fertility['avg_chars_per_token']}")

    print("\n── UNK forensics ───────────────────────────────────────────────")
    unk_stats = analyse_unk_tokens(
        args.jsonl, args.spm,
        sample_n=args.unk_sample, seed=args.seed, top_k=args.top_k,
    )
    print(f"  unk_rate : {unk_stats['unk_rate']:.2e}")


if __name__ == "__main__":
    main()
