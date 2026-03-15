"""
dataset_stats.py — Full dataset statistics for nlp-chatbot-project-v2.0
Analyses stage6 train/val/test JSONL files (token-ID arrays) and prints
a comprehensive report to help decide on context window / filtering strategy.

Usage:
    python dataset_stats.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARTIFACTS = Path(__file__).parent / "artifacts"
SPLITS = {
    "train": ARTIFACTS / "stage6_train_ids.jsonl",
    "val":   ARTIFACTS / "stage6_val_ids.jsonl",
    "test":  ARTIFACTS / "stage6_test_ids.jsonl",
}

MAX_CTX  = 100   # current model config
MAX_RESP = 40


# ── helpers ───────────────────────────────────────────────────────────────────
def percentile(sorted_data, p):
    if not sorted_data:
        return 0
    k = (len(sorted_data) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def bar(pct, width=28):
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def hline(w=72):
    return "─" * w


def histogram(data_sorted, bins):
    total = len(data_sorted)
    rows, prev = [], 0
    for b in bins:
        count = sum(1 for x in data_sorted if prev <= x < b)
        rows.append((f"{prev}-{b-1}", count, count / total * 100))
        prev = b
    tail = sum(1 for x in data_sorted if x >= prev)
    rows.append((f"{prev}+", tail, tail / total * 100))
    return rows


# ── load one split ─────────────────────────────────────────────────────────────
def load_split(path: Path):
    ctx_lens, resp_lens, pairs = [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cl, rl = len(obj["ctx"]), len(obj["resp"])
            ctx_lens.append(cl)
            resp_lens.append(rl)
            pairs.append((cl, rl))
    return sorted(ctx_lens), sorted(resp_lens), pairs


# ── per-split report ───────────────────────────────────────────────────────────
def report_split(name, ctx_s, resp_s, pairs):
    n = len(pairs)
    print(f"\n{'═'*72}")
    print(f"  SPLIT: {name.upper()}   ({n:,} pairs)")
    print(f"{'═'*72}")

    avg_ctx  = sum(ctx_s)  / n
    avg_resp = sum(resp_s) / n

    # ── CTX length distribution ───────────────────────────────────────────────
    p50_c  = percentile(ctx_s, 50)
    p75_c  = percentile(ctx_s, 75)
    p90_c  = percentile(ctx_s, 90)
    p95_c  = percentile(ctx_s, 95)
    p99_c  = percentile(ctx_s, 99)

    print(f"\n  CTX TOKEN LENGTH")
    print(f"  {hline(68)}")
    print(f"  min={ctx_s[0]}  mean={avg_ctx:.1f}  "
          f"p50={p50_c:.0f}  p75={p75_c:.0f}  p90={p90_c:.0f}  "
          f"p95={p95_c:.0f}  p99={p99_c:.0f}  max={ctx_s[-1]}")

    ctx_bins = [10, 20, 40, 60, 80, 100, 120, 150, 200, 250]
    print(f"\n  Histogram  (current max_ctx_tokens = {MAX_CTX}  marked ◄):")
    cumulative = 0
    for label, count, pct in histogram(ctx_s, ctx_bins):
        cumulative += pct
        lo = int(label.split("-")[0].replace("+", ""))
        trunc_mark = "  ◄ TRUNCATED" if lo >= MAX_CTX else ""
        print(f"    {label:>8}  {bar(pct)}  {count:>9,}  {pct:5.1f}%  "
              f"(cum {cumulative:5.1f}%){trunc_mark}")

    ctx_trunc  = sum(1 for c, _ in pairs if c > MAX_CTX)
    resp_trunc = sum(1 for _, r in pairs if r > MAX_RESP)
    clean      = sum(1 for c, r in pairs if c <= MAX_CTX and r <= MAX_RESP)
    both_trunc = sum(1 for c, r in pairs if c > MAX_CTX and r > MAX_RESP)

    print(f"\n  Truncation summary:")
    print(f"    ctx  > {MAX_CTX} tokens : {ctx_trunc:>9,}  ({ctx_trunc/n*100:.1f}%)")
    print(f"    resp > {MAX_RESP} tokens : {resp_trunc:>9,}  ({resp_trunc/n*100:.1f}%)")
    print(f"    both truncated       : {both_trunc:>9,}  ({both_trunc/n*100:.1f}%)")
    print(f"    fully within limits  : {clean:>9,}  ({clean/n*100:.1f}%)")

    # ── RESP length distribution ──────────────────────────────────────────────
    p90_r = percentile(resp_s, 90)
    p95_r = percentile(resp_s, 95)
    print(f"\n  RESP TOKEN LENGTH")
    print(f"  {hline(68)}")
    print(f"  min={resp_s[0]}  mean={avg_resp:.1f}  "
          f"p50={percentile(resp_s,50):.0f}  p75={percentile(resp_s,75):.0f}  "
          f"p90={p90_r:.0f}  p95={p95_r:.0f}  p99={percentile(resp_s,99):.0f}  max={resp_s[-1]}")

    resp_bins = [5, 10, 15, 20, 25, 30, 40, 60]
    print(f"\n  Histogram  (current max_resp_tokens = {MAX_RESP}  marked ◄):")
    cumulative = 0
    for label, count, pct in histogram(resp_s, resp_bins):
        cumulative += pct
        lo = int(label.split("-")[0].replace("+", ""))
        trunc_mark = "  ◄ TRUNCATED" if lo >= MAX_RESP else ""
        print(f"    {label:>8}  {bar(pct)}  {count:>9,}  {pct:5.1f}%  "
              f"(cum {cumulative:5.1f}%){trunc_mark}")

    # ── Window scenario table ─────────────────────────────────────────────────
    print(f"\n  CONTEXT WINDOW SCENARIOS:")
    print(f"  {hline(68)}")
    print(f"    {'Window':>8}  {'Fits':>12}  {'Truncated':>12}  {'vs current':>14}")
    base_fits = clean
    for w in [80, 100, 120, 150, 200, 250, 300]:
        fits    = sum(1 for c, r in pairs if c <= w and r <= MAX_RESP)
        dropped = n - fits
        delta   = fits - base_fits
        mark    = "  ◄ NOW" if w == MAX_CTX else ""
        print(f"    {w:>8}  {fits:>8,} ({fits/n*100:4.1f}%)  "
              f"{dropped:>8,} ({dropped/n*100:4.1f}%)  "
              f"{delta:>+9,} ({delta/n*100:+5.1f}%){mark}")

    return {
        "n": n,
        "avg_ctx": avg_ctx,
        "avg_resp": avg_resp,
        "p75_ctx": p75_c,
        "p90_ctx": p90_c,
        "p95_ctx": p95_c,
        "p99_ctx": p99_c,
        "p90_resp": p90_r,
        "p95_resp": p95_r,
        "ctx_trunc_pct": ctx_trunc / n * 100,
        "resp_trunc_pct": resp_trunc / n * 100,
        "clean_pct": clean / n * 100,
    }


# ── recommendations ────────────────────────────────────────────────────────────
def recommendations(stats):
    tr = stats["train"]
    print(f"\n{'═'*72}")
    print(f"  RECOMMENDATIONS  (based on train split)")
    print(f"{'═'*72}\n")

    # Context
    ct = tr["ctx_trunc_pct"]
    if ct > 40:
        sev = "🔴 HIGH"
    elif ct > 15:
        sev = "🟡 MODERATE"
    else:
        sev = "✅ LOW"

    print(f"  Context truncation: {sev} — {ct:.1f}% of train pairs exceed {MAX_CTX} tokens")
    print(f"  Response truncation: {tr['resp_trunc_pct']:.1f}% of train pairs exceed {MAX_RESP} tokens")
    print(f"  Pairs fully clean:  {tr['clean_pct']:.1f}%\n")

    print(f"  Suggested max_ctx_tokens values:")
    print(f"    {int(tr['p75_ctx'])+1:>4}  → covers 75th percentile")
    print(f"    {int(tr['p90_ctx'])+1:>4}  → covers 90th percentile  (good balance)")
    print(f"    {int(tr['p95_ctx'])+1:>4}  → covers 95th percentile  (conservative)")
    print(f"    {int(tr['p99_ctx'])+1:>4}  → covers 99th percentile  (near-complete)")

    print(f"\n  Options:")
    print(f"  {'─'*68}")
    print(f"  A) Increase window to ~{int(tr['p90_ctx'])+1}")
    print(f"       Pros: more complete context, better multi-turn understanding")
    print(f"       Cons: longer seqs → slower LSTM, more VRAM (~linear scaling)")

    keep_pct = 100 - ct
    print(f"\n  B) Filter out pairs with ctx > {MAX_CTX} (keep {keep_pct:.1f}%)")
    print(f"       Pros: every sample fully usable, cleanest training signal")
    print(f"       Cons: loses {ct:.1f}% of data — {int(tr['n']*ct/100):,} train pairs dropped")

    print(f"\n  C) Keep current window={MAX_CTX} (truncate to last {MAX_CTX} tokens)")
    print(f"       Pros: zero data loss, recent context preserved")
    print(f"       Cons: beginning of long conversations silently dropped")

    print(f"\n  D) Hybrid — raise to {int(tr['p90_ctx'])+1}, drop remainder")
    print(f"       Best balance: keeps ~90% of data with full context visible")
    print(f"       VRAM impact for LSTM: +{int(tr['p90_ctx'])+1-MAX_CTX} tokens ≈ "
          f"+{((int(tr['p90_ctx'])+1)/MAX_CTX-1)*100:.0f}% sequence memory overhead\n")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  DATASET STATISTICS — nlp-chatbot-project-v2.0")
    print(f"  Current config: max_ctx_tokens={MAX_CTX}  max_resp_tokens={MAX_RESP}")
    print("=" * 72)

    all_stats = {}
    for split, path in SPLITS.items():
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"\nLoading {split} ({size_mb:.1f} MB) ...", end=" ", flush=True)
        ctx_s, resp_s, pairs = load_split(path)
        print(f"{len(pairs):,} pairs loaded.")
        all_stats[split] = report_split(split, ctx_s, resp_s, pairs)

    if all_stats:
        recommendations(all_stats)


if __name__ == "__main__":
    main()
