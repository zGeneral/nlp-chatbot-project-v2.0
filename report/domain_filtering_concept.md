# Domain-Focused Filtering: Concept Report
## Ubuntu IRC Corpus — Command-Line and Question-Pattern Extraction

**Status:** Concept under evaluation — not yet implemented  
**Relates to:** Phase 1 data pipeline (`phase1.py`)

---

## 1. Motivation

The Ubuntu IRC Dialogue Corpus (~3.7M dialogues, ~1.5M selected pairs in current pipeline) is extremely broad. A large fraction of turns are:

- Social chit-chat ("thanks", "lol", "gtg")
- Off-topic IRC noise (greetings, nick changes, flood)
- Non-technical discussions unrelated to Ubuntu/Linux

A Seq2Seq chatbot trained on this noise learns to produce generic, low-information responses — which is exactly what the mini training results revealed: both models collapsed to high-frequency tokens ("i you you…", "i is you the the"). While the primary cause was temporal distribution shift at 50k scale, reducing noise in the training corpus is an independently valid strategy for improving response quality.

Two complementary filtering strategies have been proposed:

---

## 2. Strategy A: Command-Line Pattern Filtering

### Concept
Retain only Q&A pairs where **either the context or the response** contains evidence of command-line content. This grounds the model in Ubuntu's primary use-case: technical Linux support.

### Signal Sources

**Regex patterns** (strong signal — exact syntax):
```
sudo \w+          apt-get \w+       chmod / chown
grep / cat / ls   mkdir / rm / cp   wget / curl
tar / ssh         systemctl         ./executable
export / echo     df / du / kill    mount / umount
pip install       nano / vim / vi   ifconfig / ip addr
```

**Keyword signals** (weaker — contextual):
```
terminal  bash  shell  cli  command line  console
script    pipe  redirect  /etc/  /var/  /usr/  /home/
stderr    stdout  #!/bin/bash
```

### Estimated Yield
The Ubuntu IRC corpus is explicitly a *technical support* channel. Prior work (Lowe et al., 2015) notes ~60–70% of turns contain technical content. A conservative estimate suggests **40–60% of pairs** would survive command-line filtering, giving approximately 600k–900k pairs from 1.5M — sufficient for full training.

### Strengths
- High precision: regex matches on actual command syntax rarely produce false positives
- Directly aligned with the corpus domain (Ubuntu = Linux terminal support)
- Responses containing commands are more objectively evaluable (BLEU on exact tokens)

### Weaknesses
- May discard valid Q&A pairs where a question is answered in natural language without an explicit command (e.g., "You need to edit the sudoers file" — no command shown)
- Regex patterns require maintenance as the command set is not exhaustive
- Risk of length bias: longer responses are more likely to contain a command even incidentally

---

## 3. Strategy B: Question-Pattern Filtering

### Concept
Retain only pairs where the **context (last speaker turn) contains a question-like utterance**. This enforces the Q&A structure the Seq2Seq model is being trained for, discarding conversational noise that isn't question-answer shaped.

### Pattern Categories

| Category | Example patterns | Example |
|---|---|---|
| How-to | `how (do\|can\|to) I` | "how do I install chrome?" |
| What | `what (is\|are\|does)` | "what's the command for…" |
| Where | `where (is\|can\|do)` | "where is the config file?" |
| Why | `why (is\|does\|can't)` | "why won't it start?" |
| Troubleshoot | `can't\|cannot\|problem\|error` | "I can't mount the drive" |
| Help request | `help\|trying to\|need to` | "I'm trying to install…" |
| Yes/No | `^(is\|can\|do\|will)` | "is there a way to…" |
| Catch-all | `.+\?$` | Any sentence ending in `?` |

### Multi-Turn Handling
The context in our pipeline is a concatenated string of turns separated by `__eot__` tokens. Question-pattern filtering should scan **all turns in the context** and require at least one to match — not just the final turn — since the question may have been asked several turns earlier and the most recent turn is a follow-up.

**Important note on the catch-all pattern `.+\?$`:** This is deliberately broad and should be used only as a *fallback* after the specific patterns have been applied, or with a minimum-content guard (e.g., at least 3 words before the `?`). Standalone `?` replies (common on IRC as a prompt for repetition) would otherwise be incorrectly included.

### Estimated Yield
Question patterns are pervasive in IRC support channels. A rough estimate based on the pattern coverage suggests **50–70% of pairs** would survive, giving ~750k–1.05M pairs from 1.5M. The catch-all `?` pattern alone would capture a large fraction.

### Strengths
- High recall: covers many natural ways users ask for help
- Language-agnostic within English: does not require command-specific knowledge
- Produces a dataset with clear Q→A structure, directly matching the Seq2Seq training objective
- Categorisation (how_to, troubleshoot, etc.) enables stratified analysis of model performance

### Weaknesses
- Lower precision than command filtering: "I can't believe it worked!" matches `can't` (troubleshoot pattern)
- Catch-all `?` is extremely noisy on IRC (single-character `?` responses, rhetorical questions)
- Question in context does not guarantee a helpful answer in the response

---

## 4. Combined Strategy: Question + Command Answer

### Concept
The most targeted subset: pairs where the **context contains a question** AND the **response contains a command or technical answer**. This gives the "gold" dataset for a technical Q&A assistant.

```
Pairs where: is_question(context) AND is_command_related(response)
```

### Estimated Yield
This intersection will be significantly smaller. A rough estimate:
- ~50% survive question filter × ~40% of responses are command-containing ≈ **~20% of original pairs, ~300k**
- This is still well above the 50k used in mini training and sufficient for a meaningful full training run

### Strengths
- Highest signal-to-noise ratio of all three strategies
- Natural alignment of input type (question) and output type (command/answer)
- Most likely to produce meaningful BLEU improvements and reduce collapse

### Weaknesses
- Smallest dataset — may reintroduce overfitting risk that larger datasets mitigate
- Answers that explain rather than demonstrate (natural language explanations of concepts) are filtered out, potentially biasing the model toward command-only responses

---

## 5. Pipeline Integration

### Where filtering belongs
Filtering must occur **after Stage 1** (dialogue loading) and **before Stage 5** (SentencePiece training), specifically between Stage 2 (cleaning) and Stage 4 (pair generation), or as a post-Stage 4 filter on the generated pairs.

**Recommended insertion point: post-Stage 4 (on the `stage4_train_pairs` list)**

```
Stage 1: Load raw dialogues
Stage 2: Clean (remove bots, normalize text)
Stage 3: Temporal split (train/val/test by date)          ← KEEP: temporal integrity
Stage 4: Generate (ctx, resp) pairs                       ← Filter output of this stage
Stage 4.5: Domain filter (NEW) ← apply here
Stage 5: Train SentencePiece BPE on FILTERED train pairs  ← vocabulary reflects new domain
Stage 6: Encode all pairs to BPE token IDs
Stage 7: Train FastText on FILTERED pairs
Stage 8: Build embedding matrix
```

### Implications for Phase 1 rerun
**Yes, Phase 1 must be rerun** if this filtering is adopted:
- Stage 5 (SentencePiece): vocabulary must be built on filtered text — filtering changes token frequency distributions, and the BPE model must reflect the actual training domain
- Stage 7 (FastText): the embedding space must be trained on the filtered corpus to capture command-line terminology correctly
- Stage 8: embedding matrix must be rebuilt from the new FastText model

The existing `artifacts_mini/` would become stale. Full `artifacts/` would also need regenerating.

### What does NOT change
- The model architecture (`models.py`) — no changes required
- The training loop (`train.py`) — no changes required
- The evaluation pipeline (`evaluate.py`) — no changes required
- The existing special tokens (`__url__`, `__cmd__`, `__path__`, `__ip__`, `__number__`, `__eot__`, `__user__`) — these were already added in phase1 and remain valid

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Dataset too small after filtering | Medium | Combine strategies rather than intersecting; use Strategy B alone for maximum retention |
| Vocabulary shift breaks pretrained embeddings | Low | FastText retraining in Stage 7 handles this automatically |
| Temporal split integrity violated | None | Filtering is applied within each split, not across splits |
| Command regex doesn't cover all commands | Medium | Supplement with keyword signals; treat as soft filter (OR logic, not AND) |
| Over-fitting on filtered domain | Low-Medium | Monitor train/val gap; use full 500k+ pairs with Strategy B to maintain dataset size |
| Phase 1 rerun time cost | Low | Estimated 1–2 hours on Windows GPU; mini mode available for validation first |

---

## 7. Agent Evaluation Findings (Automated Code Audit)

Two independent agents audited the proposed strategies against the actual `phase1.py` pipeline. The findings below supersede any assumptions in Sections 2–4 above.

### Critical Finding 1 — `__cmd__` token does NOT exist in the pipeline

The concept report assumed commands are pre-replaced with `__cmd__` before pairs are generated. **This is wrong.** `_clean_text` (phase1.py lines 305–328) performs these substitutions in order:

```
__url__   ← URLs
__path__  ← filesystem paths with ≥2 segments (e.g. /etc/fstab → __path__)
__ip__    ← IP addresses
```

There is no `__cmd__` substitution. The token is reserved in the SentencePiece symbol list (line 1023) but never generated. **Command words like `sudo`, `apt-get`, `chmod` survive `_clean_text` intact** — they consist entirely of `[a-z0-9-]` characters which pass the `_RE_NONALPHA` filter. Strategy A's regex patterns WILL match on Stage 4 text.

### Critical Finding 2 — `?` is stripped by `_clean_text`

`_RE_NONALPHA = re.compile(r"[^a-z0-9 '\-_.]+")` (phase1.py line 324) removes every character not in `[a-z0-9 '\-_.]`. The `?` character is not in the allowed set. **Every `?` in every turn is replaced with a space during Stage 2 cleaning.** Consequences for Strategy B:

- The catch-all pattern `.+\?\s*$` matches **zero pairs** — dead on arrival
- All patterns relying on `?` (what is?, where is?, etc.) are dead
- All `?`-terminated question patterns must be rewritten as vocabulary-only patterns (no punctuation)

### Critical Finding 3 — Path keywords are dead at Stage 4

The keyword signals `/etc/`, `/var/`, `/usr/`, `/home/`, `#!/bin/bash` are all non-functional:
- Multi-segment paths (e.g. `/etc/fstab`) → replaced by `__path__` at Stage 2 (line 314)
- Remaining slashes are removed by `_RE_NONALPHA` (line 324)
- By Stage 4, none of these literal strings exist in any pair

**Replacement signal (already live, zero implementation cost):** `"__path__" in pair["ctx"] or "__path__" in pair["resp"]` — the existing `__path__` token is the correct filter signal for filesystem-related pairs.

### Critical Finding 4 — `can't` is expanded before Stage 4

The contraction map (phase1.py lines 246–261) expands `can't` → `cannot` before `_RE_NONALPHA` runs. The pattern `can't` is dead on Stage 4 text. Only `cannot` is live. Similarly: `won't` → `will not`, `don't` → `do not`, `isn't` → `is not`.

### Critical Finding 5 — "Scan all turns" harms training quality

The report proposed scanning all turns in `ctx` for question patterns. The Stage 4 `ctx` is:
```
turn_{i-k} __eot__ ... __eot__ turn_{i-2} __eot__ turn_{i-1}
```
`resp` = `turn_i`  

The model is trained to produce `turn_i` in response to `turn_{i-1}` (contextualised by earlier turns). If a question appeared in `turn_{i-3}` but `turn_{i-1}` is `"ok thanks"`, the model learns `"ok thanks" → ...` — exactly the generic response problem filtering was meant to fix. **Filtering must check the last 1–2 substantive context turns only** (mirror the existing coherence filter backward-walk at lines 864–881).

### Critical Finding 6 — A∩B at ~300k is undersized for 44M parameters

At ~40 tokens per pair average, 300k pairs ≈ 12M training tokens. For a 44M parameter supervised seq2seq model, empirical minimums suggest 5–10× the parameter count in training examples as a floor for diversity coverage (~220k–440k pairs). 300k sits at the low end and, combined with the narrowed output space (command-only responses), risks vocabulary undercoverage and increased overfitting. The A+B union at ~1M pairs is substantially safer.

### Corrected Signal Inventory

| Signal | Type | Status | Stage 4 text |
|---|---|---|---|
| `sudo`, `apt-get`, `chmod`, `grep`, `wget`, `curl`, `tar`, `ssh`, `systemctl`, `echo`, `export`, `df`, `du`, `kill`, `mount`, `nano`, `vim` | Command regex | ✅ Live | Commands are not substituted |
| `apt` (bare), `mv`, `find`, `sed`, `awk`, `dpkg`, `snap`, `service`, `ufw`, `ps`, `ping`, `crontab`, `adduser`, `passwd` | Command regex (missing from proposal) | ✅ Live — **must be added** | Same as above |
| `terminal`, `bash`, `shell`, `cli`, `script`, `pipe`, `redirect`, `stderr`, `stdout`, `console` | Keyword | ✅ Live | Plain alpha — survives cleaning |
| `__path__` | Special token | ✅ Live — **best path signal** | Generated by `_RE_PATH` at Stage 2 |
| `/etc/`, `/var/`, `/usr/`, `/home/`, `#!/bin/bash` | Keyword | ❌ Dead | Replaced by `__path__` or slashes stripped |
| `__cmd__` | Special token | ❌ Never generated | Token reserved but never substituted |
| `.+\?`, `how do I?`, `what is?` | Question pattern (with `?`) | ❌ Dead | `?` stripped by `_RE_NONALPHA` |
| `can't` | Contraction | ❌ Dead | Expanded to `cannot` |
| `how do i`, `what is`, `where is`, `why does`, `cannot`, `problem`, `error` | Question pattern (no `?`) | ✅ Live | Pure alpha/spaces survive |

### Corrected Filtering Logic

```python
# Strategy A — Command filter (post-Stage 4, on pair["ctx"] and pair["resp"])
COMMAND_PATTERNS = re.compile(
    r"\b(sudo|apt-get|apt|dpkg|snap|chmod|chown|grep|cat|ls|mkdir|rm|cp|mv|"
    r"wget|curl|tar|ssh|df|du|kill|mount|umount|pip|nano|vim|vi|sed|awk|"
    r"find|locate|ps|top|ping|service|systemctl|ufw|crontab|adduser|passwd|"
    r"netstat|ifconfig|export|echo|source|make|gcc)\b",
    re.IGNORECASE,
)

def is_command_related(text: str) -> bool:
    return bool(COMMAND_PATTERNS.search(text)) or "__path__" in text

# Strategy B — Question filter (last substantive context turn only)
QUESTION_PATTERNS = [
    re.compile(r"\bhow (do|can|to|would|should) (i|you|we)\b"),
    re.compile(r"\bwhat (is|are|does|do|was|should)\b"),
    re.compile(r"\bwhere (is|are|can|do)\b"),
    re.compile(r"\bwhy (is|does|do|will|cannot|wont|didnt)\b"),
    re.compile(r"\bcannot\b"),          # can't → cannot via contraction map
    re.compile(r"\b(problem|error|fail|broken|issue)\b"),
    re.compile(r"^(is|can|do|will|does|has|have|should|would)\b"),
    # NO ?-based catch-all — ? stripped by _RE_NONALPHA in phase1 _clean_text
]

def _last_substantive_turn(ctx: str, max_lookback: int = 2) -> str:
    turns = ctx.split(" __eot__ ")
    for turn in reversed(turns[-max_lookback:]):
        if len(turn.split()) >= 4:
            return turn
    return turns[-1]

def is_question_pair(pair: dict) -> bool:
    turn = _last_substantive_turn(pair["ctx"])
    return any(p.search(turn) for p in QUESTION_PATTERNS)
```

### Revised Recommendations

| Strategy | Yield (est.) | Verdict | Notes |
|---|---|---|---|
| A (command) | ~700k | ✅ Implement with corrected regex + `__path__` | Commands ARE in text; remove dead path keywords |
| B (question) | ~900k | ✅ Implement with corrected patterns | Remove all `?` patterns; use `cannot` not `can't`; last turn only |
| A+B union | ~1M | ✅ **Recommended** | Best coverage, near-original volume, both signals complement each other |
| A∩B intersection | ~300k | ❌ Not recommended | Too small for 44M params; bias toward command-only outputs |

---

## 8. Evaluation Questions for Agent Review

The following questions should be answered by automated evaluation:

1. **Yield estimation**: How many pairs from the existing `stage4_train_pairs.json` would survive each of the three filter strategies? What is the category distribution under Strategy B?

2. **Regex coverage audit**: Do the proposed command patterns cover the actual commands appearing in the training data? What percentage of command-containing turns in the current data would be captured?

3. **False positive audit (Strategy B)**: What fraction of pairs matched by the question patterns contain genuine questions vs. noisy IRC artefacts (e.g., single `?`, rhetorical statements)?

4. **Response quality correlation**: Do question-filtered pairs have shorter, lower-entropy responses (suggesting better-quality Q&A structure) compared to unfiltered pairs?

5. **Interaction with temporal split**: Does question/command prevalence vary between train, val, and test splits? (If val/test have a higher question density, filtering may help close the distribution gap that caused mini collapse.)

---

## 8. Academic Context

Domain-focused filtering of dialogue corpora is a well-established technique:

- **Xu et al. (2016)** "Incorporating Loose-Structured Knowledge into LSTM with Recall Gate" — demonstrates that topic-focused subsets of dialogue data improve task-specific response quality.
- **Henderson et al. (2019)** "A Repository of Conversational Datasets" — emphasises that domain specificity and response relevance are stronger determinants of trained model quality than raw corpus size.
- **Lowe et al. (2015)** — original Ubuntu Dialogue Corpus paper notes the corpus's density of technical exchanges but acknowledges significant social chatter contamination.

The proposed filtering strategies operationalise standard dataset curation practice: **prefer quality and domain alignment over volume**, especially when training capacity is fixed (44M parameter model, 20-epoch budget).

---

## 9. Recommendation Summary

| Strategy | Yield (est.) | Phase 1 rerun | Recommended for |
|---|---|---|---|
| A: Command filter | ~700k pairs | Yes | Maximising technical precision |
| B: Question filter | ~900k pairs | Yes | Maximising Q&A structure quality |
| A+B combined (OR) | ~1M pairs | Yes | Best coverage, low extra cost |
| A∩B (AND — gold set) | ~300k pairs | Yes | Highest signal, risk of underfitting |

**Current recommendation:** Implement Strategy B (question-pattern filter) OR the A+B union as a new `stage4_5_domain_filter` function in `phase1.py`. The union gives near-original volume with significantly reduced noise. Evaluate yield on mini data before committing to a full phase1 rerun.

---

*Last updated: 2026-03-14*  
*Author: NLP Final Project — MSc AI*
