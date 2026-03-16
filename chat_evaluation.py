"""
chat_evaluation.py — Qualitative evaluation: batch inference on a curated question set.

Sends 20 questions to both models (baseline + attention) using two decode modes
(greedy and beam=7), producing 80 responses total.  Results are saved to
checkpoints/chat_eval_results.json for human/LLM review and report generation.

Usage:
    python chat_evaluation.py
    python chat_evaluation.py --beam-width 7 --out checkpoints/chat_eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import sentencepiece as spm

from config import CONFIG
from models import build_model
from phase1 import _clean_text
from chatv2 import beam_decode, build_context, load_model_from_checkpoint
from evaluate import greedy_decode


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation question set — 20 questions across 5 categories
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = [
    # Category: Package management
    {"id": "q01", "category": "Package management",
     "text": "how do i install a package in ubuntu"},
    {"id": "q02", "category": "Package management",
     "text": "how do i remove a package completely"},
    {"id": "q03", "category": "Package management",
     "text": "how do i update all packages"},
    {"id": "q04", "category": "Package management",
     "text": "i cannot install python because of a dependency error"},

    # Category: File system
    {"id": "q05", "category": "File system",
     "text": "how do i find a file by name"},
    {"id": "q06", "category": "File system",
     "text": "how do i change file permissions"},
    {"id": "q07", "category": "File system",
     "text": "what does the df command do"},
    {"id": "q08", "category": "File system",
     "text": "how do i mount a usb drive"},

    # Category: Users & permissions
    {"id": "q09", "category": "Users & permissions",
     "text": "how do i add a new user"},
    {"id": "q10", "category": "Users & permissions",
     "text": "how do i reset my root password"},
    {"id": "q11", "category": "Users & permissions",
     "text": "how do i give a user sudo access"},

    # Category: Networking
    {"id": "q12", "category": "Networking",
     "text": "how do i check my ip address"},
    {"id": "q13", "category": "Networking",
     "text": "how do i connect to a remote server using ssh"},
    {"id": "q14", "category": "Networking",
     "text": "my wifi is not working after upgrading ubuntu"},

    # Category: Editors & tools
    {"id": "q15", "category": "Editors & tools",
     "text": "what is the difference between vim and nano"},
    {"id": "q16", "category": "Editors & tools",
     "text": "how do i open a file in nano"},
    {"id": "q17", "category": "Editors & tools",
     "text": "how do i run a bash script"},

    # Category: System & hardware
    {"id": "q18", "category": "System & hardware",
     "text": "how do i check how much ram i have"},
    {"id": "q19", "category": "System & hardware",
     "text": "my system is very slow what should i do"},
    {"id": "q20", "category": "System & hardware",
     "text": "how do i install nvidia drivers on ubuntu"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def respond(model, question: str, sp: spm.SentencePieceProcessor,
            mode: str, beam_width: int, config: dict,
            device: torch.device) -> str:
    """Encode question, decode response, return decoded string."""
    cleaned = _clean_text(question)
    src_tensor = build_context(
        history=[cleaned],
        sp_processor=sp,
        max_ctx_tokens=config.get("max_ctx_tokens", 100),
        max_turns=config.get("max_ctx_turns", 8),
    ).to(device)
    src_lengths = torch.tensor([src_tensor.size(1)], dtype=torch.long).to(device)

    sos  = config.get("sos_idx", 2)
    eos  = config.get("eos_idx", 3)
    mlen = config.get("max_decode_len", 40)

    with torch.inference_mode():
        if mode == "greedy":
            ids_batch = greedy_decode(
                model, src_tensor, src_lengths,
                sos_idx=sos, eos_idx=eos, max_len=mlen, device=device,
            )
            toks = ids_batch[0]
        else:  # beam
            ids_batch = beam_decode(
                model, src_tensor, src_lengths,
                sos_idx=sos, eos_idx=eos, max_len=mlen, device=device,
                beam_width=beam_width,
            )
            toks = ids_batch[0] if ids_batch else []

    toks = [t for t in toks if t not in (sos, eos, 0)]
    return sp.decode(toks) if toks else "…"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch qualitative evaluation")
    parser.add_argument("--beam-width", type=int, default=7)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--artifact-dir",   default="artifacts")
    parser.add_argument("--out", default="checkpoints/chat_eval_results.json")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    art_dir  = Path(args.artifact_dir)

    # Load SPM
    sp = spm.SentencePieceProcessor(
        model_file=str(art_dir / "stage5_spm.model")
    )
    print(f"SPM loaded. Vocab size: {sp.get_piece_size()}")

    # Load both models
    models = {}
    for mtype in ("baseline", "attention"):
        ckpt_path = ckpt_dir / f"{mtype}_best.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found — skipping {mtype}")
            continue
        print(f"  Loading {mtype} from {ckpt_path} …")
        model, _ = load_model_from_checkpoint(ckpt_path, mtype, CONFIG, device)
        model.eval()
        models[mtype] = model
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {mtype}: {n:,} params  |  device: {device}")

    if not models:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # Run inference
    results = []
    total = len(QUESTIONS) * len(models) * 2  # 2 decode modes
    done  = 0

    for q in QUESTIONS:
        entry = {
            "id":       q["id"],
            "category": q["category"],
            "question": q["text"],
            "responses": {},
        }
        for mtype, model in models.items():
            entry["responses"][mtype] = {}
            for mode in ("greedy", f"beam{args.beam_width}"):
                decode_mode = "greedy" if mode == "greedy" else "beam"
                resp = respond(model, q["text"], sp, decode_mode,
                               args.beam_width, CONFIG, device)
                entry["responses"][mtype][mode] = resp
                done += 1
                print(f"  [{done:3d}/{total}] {q['id']} | {mtype:9s} | {mode:7s} → {resp[:80]}")

        results.append(entry)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"beam_width": args.beam_width, "results": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} entries → {out_path}")


if __name__ == "__main__":
    main()
