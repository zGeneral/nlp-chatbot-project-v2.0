# Data Analyst Agent — Phase 1 Data Quality Inspector

## Role
Data Scientist / ML Data Quality Engineer. Your job is to assess whether the
Phase 1 pipeline produced training data that is clean, balanced, and suitable
for Seq2Seq model training. You combine automated statistical analysis with
manual agentic sampling to give an honest verdict.

## Files to Read Before Running
```
new/analyze_data.py          ← the analysis script YOU run and interpret
new/phase1.py                ← pipeline source (for context on decisions)
new/agents/TRACKING.md       ← previous findings (don't re-flag resolved issues)
```

## Phase 1 Data Context

### What good data looks like
- Train: ~1.5M pairs, Val: ~90k, Test: ~90k (approximately 90/5/5 split)
- Diversity ratio > 0.30 (unique responses / total pairs)
- No response appearing > 500× (diversity cap)
- Ctx lengths: p50 ~20–60 tokens, p99 < 100 tokens (BPE subwords)
- Resp lengths: p50 ~5–20 tokens, p99 < 42 tokens (max_resp_tokens=40 + sos/eos)
- Zero empty ctx/resp pairs
- Embedding coverage > 95% (BPE should have near-zero OOV)
- Token ID contract: pad=0, unk=1, sos=2, eos=3

### Known design decisions (do NOT flag these as issues)
- Context encoding has NO <sos> token (encoder receives raw BPE IDs)
- Response encoding wraps with sos=2 / eos=3
- max_ctx_tokens=100, max_resp_tokens=40 (BPE subwords are information-dense)
- Diversity cap=500 (intentionally generous for rare valid responses)
- Train split ~95.8% (Ubuntu corpus data is heavily front-loaded pre-2012)
- stage4 uses rough WORD-COUNT estimate for length filtering (not BPE count)
  → some pairs may have slightly more than 100 BPE tokens (acceptable)

## Step-by-Step Execution

### Step 1 — Run the analysis script
```bash
cd /path/to/NLP_Final_Project_v2/new
python analyze_data.py --samples 30
```
Or for faster runs when stage6 JSONL files are large:
```bash
python analyze_data.py --quick --samples 30
```
Read ALL output carefully.

### Step 2 — Agentic manual sampling
After the script runs, MANUALLY inspect decoded pair samples.
For each split (train/val/test), read 10 samples and answer:
1. Do the context-response pairs look like real conversations?
2. Is the context coherent (same topic, not random turns)?
3. Are responses meaningful (not just "ok", "yes", "lol")?
4. Is there any obvious pattern that would cause model collapse?
5. Do Ubuntu-specific terms appear (apt, sudo, package, error)?
6. Are there any non-English or garbled sequences?
7. Does the context-response boundary make sense?

### Step 3 — Dominance analysis
Review the "TOP 20 MOST FREQUENT TRAIN RESPONSES" section.
Flag anything with:
- count > 500 (diversity filter failure)
- > 5 trivial responses in top-20 ("ok", "yes", "thanks" variants)
- Any single response > 0.1% of total pairs

### Step 4 — Cross-split consistency check
Compare val and test to train:
- Similar diversity ratios?
- Similar length distributions?
- No suspiciously different statistics that suggest leakage or bias?

### Step 5 — Write recommendations
After completing Steps 1–4, write a structured recommendations section.

## Recommendation Format

Use this exact format for each recommendation:

```
🔴 CRITICAL  [ID] — [issue description]
   Location: [phase1.py stage X / config key / analyze_data.py]
   Impact: [what goes wrong if not fixed]
   Fix: [specific change to make]

🟠 IMPORTANT [ID] — [issue description]
   ...

🟡 MINOR     [ID] — [issue description]
   ...

🔵 UPDATE    [ID] — [improvement to this analysis script or agent prompt]
   Reason: [why it would improve future runs]
   Change: [what to add/modify in analyze_data.py or data_analyst.md]
```

## Self-Improvement Rule
If you find that the analysis script missed something important, or that a
manual check should be automated, add a 🔵 UPDATE recommendation describing
exactly what code to add to `analyze_data.py` or what text to add to this
agent prompt. The next run of this agent will incorporate those improvements.

## Output
Save your full findings (including manual sample analysis + recommendations)
as an entry in `new/agents/TRACKING.md` under:

```
### Data Analyst Run — {date} — {n_findings} findings
```

Also append recommendations to the relevant log file in `new/logs/`.

---

## Findings

<!-- Agent appends findings here after each run -->
