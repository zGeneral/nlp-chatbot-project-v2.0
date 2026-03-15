"""
phase1.py — Data pipeline for clean-from-scratch Ubuntu Dialogue Corpus processing.

Stages (run once, produces artifacts in ARTIFACT_DIR):
  Stage 1 — Load raw Ubuntu Dialogue Corpus CSV files into structured dialogues
  Stage 2 — Clean text + apply quality filters (parallel, chunked)
  Stage 3 — Temporal split (train / val / test) by THREAD first-turn date (G6 fix)
  Stage 4 — Generate context-response pairs + response diversity filter
  Stage 5 — Train SentencePiece BPE model on raw training text (16k–20k vocab)
  Stage 6 — Encode all pairs to BPE token IDs + save vocab JSON
  Stage 7 — Train FastText 300d on BPE-tokenised training corpus
  Stage 8 — Build embedding matrix aligned to BPE vocab → .npy matrix

Key decisions vs old phase1.py:
  - SentencePiece BPE replaces word-level vocab (smaller softmax, zero OOV)
  - FastText trained on BPE-tokenised text (▁-prefixed pieces), not raw words
  - Response diversity filter caps repeat responses (reduces "averaging signal")
  - Max context: 100 tokens / 8 turns (tighter = more specific)
  - Max response: 40 tokens
  - Target ~1.5M train pairs (quality > quantity vs prior 4.57M)
  - Temporal split on THREAD (dialogue) first-turn date — no thread boundary leakage

Quality filters:
  - filter_paste:           drop pasted terminal output blocks
  - filter_irc_actions:     drop /me emote turns
  - filter_repetitive:      drop "help help help" style turns
  - filter_temporal:        drop dialogues with large timestamp gaps
  - dyadic_only:            only 2-speaker conversations
  - filter_echo_pairs:      skip verbatim context-response copies
  - filter_placeholder:     skip placeholder-only responses
  - filter_non_english:     drop mostly non-ASCII responses
  - filter_bot_responses:   drop known boilerplate canned responses
  - filter_response_diversity: cap max occurrences of any identical response

Token ID contract (must match config.py):
  pad=0, unk=1, sos=2, eos=3  (enforced via explicit SPM trainer parameters)
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Resolve paths relative to this file (works from any working directory) ───
_NEW_DIR = Path(__file__).resolve().parent          # .../NLP_Final_Project_v2/new/
_PROJECT_DIR = _NEW_DIR.parent                      # .../NLP_Final_Project_v2/

# ── CONFIG ────────────────────────────────────────────────────────────────────

PHASE1_CONFIG = {
    # Corpus source — data/ lives inside new/ alongside this file
    "corpus_dir":                    str(_NEW_DIR / "data" / "Ubuntu-dialogue-corpus"),

    # Dialogue-level quality filters (set False to disable individually)
    "min_turns":                     2,
    "dyadic_only":                   True,
    "filter_paste":                  True,
    "filter_irc_actions":            True,
    "filter_repetitive":             True,
    "filter_temporal":               True,
    "max_turn_gap_seconds":          3600,   # hard ceiling per gap; 0 = disabled
    "large_gap_threshold":           600,    # soft threshold (seconds) for ratio check
    "max_large_gap_ratio":           0.3,    # fraction of gaps allowed to exceed threshold

    # Pair-level quality filters
    "filter_echo_pairs":             True,
    "filter_placeholder":            True,
    "filter_non_english":            True,
    "filter_bot_responses":          True,

    # Response diversity cap (train set only)
    "filter_response_diversity":     True,
    "max_response_occurrences":      500,

    # Pair generation dimensions
    "max_ctx_tokens":                100,
    "max_ctx_turns":                 8,
    "max_resp_tokens":               40,
    "min_resp_tokens":               5,    # raised from 3 — eliminates 3-4 token fragments
    "max_train_pairs":               1_500_000,   # 0 = no cap

    # Temporal split boundaries — split by THREAD first-turn date.
    # Calibrated to the Ubuntu Dialogue Corpus (2004-08 → 2012-11).
    # These dates give ~80% train / ~10% val / ~10% test.
    "train_cutoff_date":             "2012-04-27",
    "val_cutoff_date":               "2012-08-07",

    # SentencePiece BPE — special token IDs guaranteed via explicit params
    "spm_vocab_size":                16000,
    "spm_model_type":                "bpe",
    "spm_character_coverage":        0.9999,
    "spm_input_sentence_size":       2_000_000,   # cap SPM training lines (avoids "too many" warning)
    # NOTE: do NOT add spm_user_defined_symbols — use explicit pad_id/bos_id/eos_id
    # Dialogue-level speaker quality filters
    "max_single_speaker_ratio":      0.80,  # drop monologue-style dialogues (1 speaker > 80%)
    "min_alternation_ratio":         0.15,  # drop non-alternating floods (A,A,A,B,B,B style)

    # FastText
    "fasttext_dim":                  300,
    "fasttext_epochs":               10,    # 10 matches original; 5 was too few
    "fasttext_min_count":            3,     # 3 filters hapax noise; 1 was too inclusive
    "fasttext_window":               5,     # explicit context window
    "fasttext_sg":                   1,     # skip-gram > CBOW for rare BPE pieces
    "fasttext_workers":              8,

    # Pipeline internals
    "artifact_dir":                  str(_NEW_DIR / "artifacts"),
    "log_dir":                       str(_NEW_DIR / "logs"),
    "num_workers":                   max(1, mp.cpu_count() - 2),
    "chunk_size":                    50_000,

    # Mini-mode subsample — fraction of stage 1 dialogues to keep.
    # Set to 0.10 in phase1_mini.py for a ~10x faster pipeline run.
    # 1.0 = use all dialogues (default full run).
    "stage1_subsample_frac":         1.0,
}

# ── Stage expected artifacts ──────────────────────────────────────────────────

STAGE_ARTIFACTS = {
    1: ["stage1_dialogues.pkl"],
    2: ["stage2_clean_dialogues.pkl", "stage2_stats.json"],
    3: ["stage3_train.pkl", "stage3_val.pkl", "stage3_test.pkl", "stage3_stats.json"],
    4: ["stage4_train_pairs.json", "stage4_val_pairs.json",
        "stage4_test_pairs.json", "stage4_stats.json"],
    5: ["stage5_spm.model", "stage5_spm.vocab"],
    6: ["stage6_train_ids.jsonl", "stage6_val_ids.jsonl",
        "stage6_test_ids.jsonl", "stage6_vocab.json", "stage6_stats.json"],
    7: ["stage7_fasttext.model"],
    8: ["stage8_embedding_matrix.npy", "stage8_stats.json"],
}

# ── Known IRC bots (turns from these speakers are dropped in stage 2) ─────────

_BOTS = frozenset({
    "ubottu", "ubotu", "ubot5", "ubot3", "ubot4", "ubot2",
    "chanserv", "nickserv", "memoserv", "operserv",
    "logbot", "meetingology", "supybot",
})

# ── Bot/boilerplate response blacklist (matched against cleaned, lowercased text)

_BOT_RESPONSE_BLACKLIST = frozenset({
    # Flood / paste warnings (ubottu)
    "please do not flood use __url__ to paste do not use enter as punctuation",
    "please do not flood the channel use __url__ to paste",
    "please do not flood",
    "i am a bot please do not message me",
    "you can use __url__ to paste",
    # Moderation / op scripted responses
    "watch your language",
    "please keep the channel family friendly",
    "please keep this channel on-topic",
    "this is not the right channel for that",
    "this is ubuntu support only",
    "this channel is for ubuntu support",
    "please take offtopic chat",
    "please use #ubuntu-offtopic",
    "try #ubuntu-offtopic",
    "not ubuntu related",
    "that is not ubuntu related",
    # Join / leave / mode announcements (occasionally leak through)
    "has joined",
    "has quit",
    "has left",
    "was kicked",
    "changed the topic",
    # ubottu factoid pattern prefix
    "ubuntu is",
    "please see",
})

# ── Compiled regex patterns ───────────────────────────────────────────────────

_RE_URL      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_PATH     = re.compile(r"(?:/[a-zA-Z0-9._~-]+){2,}")
_RE_IP       = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b")  # IPv4 + optional port
_RE_IRC_NICK = re.compile(r"^<[^>]+>\s*")
_RE_NONALPHA = re.compile(r"[^a-z0-9 '\-_.]+")
_RE_MULTI_SP = re.compile(r"\s{2,}")
_RE_ACTION   = re.compile(r"^\*\s*\S+\s+")   # IRC /me emotes: "* nick does thing"
_RE_PLACEHOLDER_ONLY = re.compile(
    r"^(__url__|__path__|__ip__|__cmd__|__number__|__user__|\s)+$"
)
# Matches known IRC bot names (including possessive 's) in message text.
# Stage 2 drops bot *turns*, but humans still reference bot names (e.g.
# "follow ubottu's message"), so we mask these in stage 4 text as __user__.
_RE_BOT_NAMES = re.compile(
    r"\b(?:" + "|".join(re.escape(b) for b in sorted(_BOTS, key=len, reverse=True)) + r")(?:'s)?\b",
    re.IGNORECASE,
)

# ── Coherence filter stopwords ────────────────────────────────────────────────
# Used in stage 4 to discard pairs where the last ctx turn and response share
# no content words — a signal of dialogue-window misalignment.
_COHERENCE_STOPWORDS = frozenset({
    "that", "this", "with", "have", "from", "they", "will", "been", "would",
    "could", "should", "there", "their", "what", "when", "which", "your",
    "also", "into", "more", "some", "such", "about", "just", "then", "than",
    "here", "does", "like", "very", "even", "most", "over", "only", "same",
    "each", "make", "much", "come", "want", "need", "know", "think", "look",
    "time", "back", "after", "where", "while", "through", "however", "because",
    "these", "those", "other", "them", "been", "were", "have", "will", "shall",
    "still", "again", "being", "going", "doing", "using", "something", "anything",
})

# ── Contraction map ───────────────────────────────────────────────────────────

_CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "he's": "he is", "he'll": "he will",
    "he'd": "he would", "she's": "she is", "she'll": "she will",
    "she'd": "she would", "it's": "it is", "it'll": "it will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "we'd": "we would", "they're": "they are", "they've": "they have",
    "they'll": "they will", "they'd": "they would",
    "that's": "that is", "that'll": "that will",
    "who's": "who is", "who'll": "who will", "who'd": "who would",
    "what's": "what is", "what'll": "what will",
    "where's": "where is", "where'd": "where did",
    "when's": "when is", "when'd": "when did",
    "why's": "why is", "why'd": "why did",
    "how's": "how is", "how'd": "how did",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not",
    "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not",
    "let's": "let us", "there's": "there is", "here's": "here is",
    "ain't": "is not",
}

# ── Utility helpers ───────────────────────────────────────────────────────────

def _elapsed(t0: float) -> str:
    """Return human-readable elapsed time since t0."""
    s = time.time() - t0
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def _save_json(obj: object, path: Path) -> None:
    """Atomically write obj as JSON (tmp-then-rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)
    print(f"  saved {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def _load_json(path: Path) -> object:
    """Load JSON from path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_pickle(obj: object, path: Path) -> None:
    """Atomically write obj as pickle (tmp-then-rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"  saved {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")


def _load_pickle(path: Path) -> object:
    """Load pickle from path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _stage_done(stage: int, artifact_dir: Path) -> bool:
    """Return True if all artifacts for stage already exist."""
    return all((artifact_dir / f).exists() for f in STAGE_ARTIFACTS[stage])


# ── Text cleaning helpers ─────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Clean a single utterance string.

    Steps: URL/path normalisation, IRC nick stripping, lowercase,
    contraction expansion, non-alphanumeric removal, whitespace collapse.
    """
    if not text:
        return ""
    text = _RE_URL.sub(" __url__ ", text)
    text = _RE_PATH.sub(" __path__ ", text)
    text = _RE_IP.sub(" __ip__ ", text)      # mask IPv4 addresses (e.g. 173.224.120.70)
    text = _RE_IRC_NICK.sub("", text)
    text = text.lower()
    # Strip IRC addressee pattern at message start: "nick: message" or "nick, message"
    # e.g. "actionparsnip: try this" → "try this"
    text = re.sub(r"^[a-z][a-z0-9_\-\[\]\\^{}|`]{1,25}\s*[:,]\s*", "", text)
    words = text.split()
    words = [_CONTRACTIONS.get(w, w) for w in words]
    text = " ".join(words)
    text = _RE_NONALPHA.sub(" ", text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'([a-z0-9])\.((?:\s|$))', r'\1 .\2', text)
    text = " ".join(w.lstrip("'") for w in text.split())
    return _RE_MULTI_SP.sub(" ", text).strip()


def _is_likely_paste(text: str) -> bool:
    """Detect pasted terminal output / log lines / config dumps.

    Heuristics: low alphabetic ratio, high special-character density,
    or many colons in a short line (timestamp / key-value patterns).
    """
    if not text or len(text) < 10:
        return False
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    special_count = sum(1 for c in text if c in r'[]{}()=<>|;:@#$%^&*\/')
    special_density = special_count / len(text)
    colon_count = text.count(":")
    if alpha_ratio < 0.30:
        return True
    if special_density > 0.15:
        return True
    if colon_count >= 3 and len(text) < 200:
        return True
    return False


def _is_repetitive(text: str) -> bool:
    """Detect repetitive turns like 'help help help help'."""
    tokens = text.split()
    if len(tokens) < 4:
        return False
    most_common_count = Counter(tokens).most_common(1)[0][1]
    return (most_common_count / len(tokens)) > 0.50


def _is_english_response(text: str) -> bool:
    """Return True if text is predominantly ASCII (English).

    Non-English IRC users typically have enough non-ASCII characters to
    trip this check. Responses with fewer than 3 alpha chars are passed
    through — other filters handle trivially short responses.
    """
    clean = re.sub(r"__url__|__path__|__cmd__|__number__", " ", text)
    alpha_chars = [c for c in clean if c.isalpha()]
    if len(alpha_chars) < 3:
        return True
    ascii_count = sum(1 for c in alpha_chars if ord(c) < 128)
    return (ascii_count / len(alpha_chars)) >= 0.80


def _is_placeholder_only(text: str) -> bool:
    """Return True if response contains only placeholder tokens."""
    return bool(_RE_PLACEHOLDER_ONLY.match(text.strip()))


def _is_echo_pair(resp_text: str, ctx_text: str) -> bool:
    """Return True if the response appears verbatim inside the context."""
    if len(resp_text) < 6:
        return False
    ctx_norm = re.sub(r"__eot__|__user__", " ", ctx_text.lower())
    ctx_norm = re.sub(r"\s+", " ", ctx_norm).strip()
    return resp_text.lower() in ctx_norm


def _is_bot_response(text: str) -> bool:
    """Return True if text matches a known bot/boilerplate string."""
    for entry in _BOT_RESPONSE_BLACKLIST:
        if text.startswith(entry):
            return True
    return False


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> Optional[datetime]:
    """Try several common date formats; return UTC datetime or None."""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def _dialogue_first_date(dlg: Dict) -> Optional[datetime]:
    """Return the parsed date of the first turn that has a valid date."""
    for t in dlg["turns"]:
        d = _parse_date(t.get("date", ""))
        if d is not None:
            return d
    return None


# ── Same-speaker turn merging ─────────────────────────────────────────────────

def _merge_same_speaker_turns(turns: List[Dict]) -> List[Dict]:
    """Merge consecutive turns by the same speaker into one turn.

    Utterances are joined with a single space. Duplicate text within the
    same merged turn is silently deduplicated.
    """
    if not turns:
        return []
    merged = [dict(turns[0])]
    for t in turns[1:]:
        if t["from"].lower() == merged[-1]["from"].lower():
            existing = {u.strip() for u in merged[-1]["text"].split(" ")}
            candidate = t["text"].strip()
            if candidate and candidate not in existing:
                merged[-1]["text"] += " " + candidate
        else:
            merged.append(dict(t))
    return merged


# ── Stage 2 worker (top-level for multiprocessing pickling) ──────────────────

def _filter_worker(args: Tuple) -> Tuple[List[Dict], Dict]:
    """Filter and clean one chunk of dialogues.

    Top-level function required for multiprocessing pickling.
    Returns (kept_dialogues, reason_counts).
    """
    chunk, cfg = args
    kept: List[Dict] = []
    counts: Dict[str, int] = defaultdict(int)

    for dlg in chunk:
        result, reason = _filter_dialogue(dlg, cfg)
        counts[reason] += 1
        if result is not None:
            kept.append(result)

    return kept, dict(counts)


def _filter_dialogue(dlg: Dict, cfg: dict) -> Tuple[Optional[Dict], str]:
    """Apply all dialogue-level quality filters and clean turn text.

    Returns (cleaned_dialogue, reason) where reason is 'kept' on success
    or a short label describing why the dialogue was dropped.
    """
    # Sort turns by date; require at least 2 parseable dates
    valid = []
    for t in dlg["turns"]:
        dt = _parse_date(t.get("date", ""))
        if dt is not None:
            valid.append((dt, t))
    if len(valid) < 2:
        return None, "too_few_valid_dates"
    valid.sort(key=lambda x: x[0])
    sorted_turns = [t for _dt, t in valid]

    # Per-turn filtering
    cleaned_turns: List[Dict] = []
    for t in sorted_turns:
        speaker = t["from"].lower().strip()
        if speaker in _BOTS:
            continue
        raw = t["text"].strip()
        if cfg.get("filter_irc_actions", True) and _RE_ACTION.match(raw):
            continue
        if cfg.get("filter_paste", True) and _is_likely_paste(raw):
            continue
        ctext = _clean_text(raw)
        if not ctext:
            continue
        if cfg.get("filter_repetitive", True) and _is_repetitive(ctext):
            continue
        cleaned_turns.append({
            "date": t.get("date", ""),
            "from": speaker,
            "text": ctext,
        })

    if len(cleaned_turns) < cfg.get("min_turns", 2):
        return None, "too_few_turns"

    # Dyadic check
    if cfg.get("dyadic_only", True):
        speakers = {t["from"] for t in cleaned_turns}
        if len(speakers) != 2:
            return None, "not_dyadic"

    # Speaker dominance filter — drop monologue-style threads where one speaker
    # dominates >max_single_speaker_ratio of turns (e.g. help-flooding bots).
    max_sr = cfg.get("max_single_speaker_ratio", 1.0)
    if max_sr < 1.0:
        speaker_counts = Counter(t["from"] for t in cleaned_turns)
        dominant_ratio = max(speaker_counts.values()) / len(cleaned_turns)
        if dominant_ratio > max_sr:
            return None, "speaker_dominance"

    # Alternation ratio filter — drop non-alternating floods like A,A,A,B,B,B.
    # Requires at least min_alternation_ratio fraction of consecutive-turn speaker changes.
    min_alt = cfg.get("min_alternation_ratio", 0.0)
    if min_alt > 0.0 and len(cleaned_turns) >= 2:
        alt_count = sum(
            1 for i in range(1, len(cleaned_turns))
            if cleaned_turns[i]["from"] != cleaned_turns[i - 1]["from"]
        )
        alt_ratio = alt_count / (len(cleaned_turns) - 1)
        if alt_ratio < min_alt:
            return None, "low_alternation"

    # Temporal coherence: hard ceiling on single gap
    if cfg.get("filter_temporal", True) and cfg.get("max_turn_gap_seconds", 0) > 0:
        times = [_parse_date(t["date"]) for t in cleaned_turns]
        times = [x for x in times if x is not None]
        if len(times) >= 2:
            max_gap = max(
                abs((times[i] - times[i - 1]).total_seconds())
                for i in range(1, len(times))
            )
            if max_gap > cfg["max_turn_gap_seconds"]:
                return None, "temporal_hard_ceiling"

    # Temporal coherence: large-gap ratio check
    if cfg.get("filter_temporal", True) and cfg.get("max_large_gap_ratio", 1.0) < 1.0:
        threshold = cfg.get("large_gap_threshold", 600)
        times = [_parse_date(t["date"]) for t in cleaned_turns]
        times = [x for x in times if x is not None]
        if len(times) >= 2:
            gaps = [
                abs((times[i] - times[i - 1]).total_seconds())
                for i in range(1, len(times))
            ]
            large_frac = sum(1 for g in gaps if g > threshold) / len(gaps)
            if large_frac > cfg["max_large_gap_ratio"]:
                return None, "temporal_gap_ratio"

    return {"id": dlg["id"], "turns": cleaned_turns}, "kept"


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage1_load_corpus(cfg: dict) -> List[Dict]:
    """Load raw Ubuntu Dialogue Corpus CSV files → list of dialogue dicts.

    Reads all CSV files matching '*.csv' in cfg['corpus_dir'].
    Each dialogue is structured as:
        {"id": str, "turns": [{"date": str, "from": str, "text": str}]}
    Turns within each dialogue are sorted by date before return.
    The unique dialogue key is 'folder/dialogueID' to avoid merging
    unrelated IRC sessions from different folders.
    """
    csv.field_size_limit(2 ** 24)
    source_dir = Path(cfg["corpus_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"corpus_dir not found: {source_dir}")

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")
    print(f"  Found {len(csv_files)} CSV file(s) in {source_dir}")

    dialogues: Dict[str, List[Dict]] = defaultdict(list)
    total_turns = 0

    for csv_path in csv_files:
        print(f"  Reading {csv_path.name} …")
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                did = row.get("dialogueID", "").strip()
                fld = row.get("folder", "").strip()
                if not did:
                    continue
                unique_id = f"{fld}/{did}" if fld else did
                date_val = row.get("date", "").strip()
                from_val = row.get("from", "").strip()
                text_val = row.get("text", "").strip()
                if not from_val or not text_val:
                    continue
                dialogues[unique_id].append({
                    "date": date_val,
                    "from": from_val,
                    "text": text_val,
                })
                total_turns += 1
                if total_turns % 5_000_000 == 0:
                    print(f"    … {total_turns:,} turns, {len(dialogues):,} dialogues")

    # Sort turns by date within each dialogue; skip undated turns
    result: List[Dict] = []
    for uid, turns in dialogues.items():
        dated = []
        undated = []
        for t in turns:
            dt = _parse_date(t["date"])
            if dt is not None:
                dated.append((dt, t))
            else:
                undated.append(t)
        dated.sort(key=lambda x: x[0])
        sorted_turns = [t for _dt, t in dated] + undated
        if sorted_turns:
            result.append({"id": uid, "turns": sorted_turns})

    print(f"  Total turns: {total_turns:,}   Dialogues: {len(result):,}")
    return result


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def stage2_clean_and_filter(dialogues: List[Dict], cfg: dict) -> Tuple[List[Dict], Dict]:
    """Apply all quality filters and text cleaning in parallel chunks.

    Returns (clean_dialogues, stats) where stats includes per-filter
    discard counts. Uses multiprocessing.Pool with fallback to sequential
    processing if the pool fails.
    """
    total_in = len(dialogues)
    chunk_size = cfg.get("chunk_size", 50_000)
    num_workers = cfg.get("num_workers", max(1, mp.cpu_count() - 2))

    chunks = [dialogues[i: i + chunk_size] for i in range(0, total_in, chunk_size)]
    print(f"  {total_in:,} dialogues → {len(chunks)} chunks (size {chunk_size:,}), {num_workers} workers")

    kept_all: List[Dict] = []
    reason_totals: Dict[str, int] = defaultdict(int)

    worker_args = [(chunk, cfg) for chunk in chunks]

    try:
        # Use "spawn" context to avoid deadlocks when forking inside Jupyter/Colab
        # (fork inherits background threads; spawn starts clean) (QA2-M2).
        _mp_ctx = mp.get_context("spawn")
        with _mp_ctx.Pool(num_workers) as pool:
            for ci, (kept_chunk, reason_counts) in enumerate(
                pool.imap_unordered(_filter_worker, worker_args), 1
            ):
                kept_all.extend(kept_chunk)
                for r, n in reason_counts.items():
                    reason_totals[r] += n
                if ci % max(1, len(chunks) // 10) == 0 or ci == len(chunks):
                    print(f"    chunk {ci}/{len(chunks)}  kept={len(kept_all):,}")
    except Exception as exc:
        print(f"  Pool failed ({exc}); falling back to sequential processing …")
        kept_all = []
        reason_totals = defaultdict(int)
        for ci, (chunk, _cfg) in enumerate(worker_args, 1):
            kept_chunk, reason_counts = _filter_worker((chunk, _cfg))
            kept_all.extend(kept_chunk)
            for r, n in reason_counts.items():
                reason_totals[r] += n
            if ci % max(1, len(chunks) // 10) == 0 or ci == len(chunks):
                print(f"    chunk {ci}/{len(chunks)}  kept={len(kept_all):,}")

    total_out = len(kept_all)
    total_disc = total_in - total_out

    print(f"\n  Filter breakdown (input: {total_in:,}):")
    for reason, count in sorted(reason_totals.items(), key=lambda x: -x[1]):
        tag = "kept" if reason == "kept" else "disc"
        print(f"    {tag}  {reason:<28}  {count:>8,}  ({count/total_in*100:.1f}%)")
    print(f"  TOTAL KEPT: {total_out:,} ({total_out/total_in*100:.1f}%)")

    stats = {
        "stage": 2,
        "n_input": total_in,
        "n_output": total_out,
        "n_discarded": total_disc,
        "filter_breakdown": dict(reason_totals),
    }
    return kept_all, stats


# ── Stage 3 ───────────────────────────────────────────────────────────────────

def stage3_temporal_split(dialogues: List[Dict], cfg: dict) -> Tuple[List, List, List, Dict]:
    """Split dialogues by THREAD first-turn date → (train, val, test, stats).

    Thread boundary safety (G6): each dialogue is assigned to exactly one
    split based on its FIRST turn's date. This prevents a single IRC thread
    from appearing in both train and test (data leakage).

    Dialogues with no parseable date are assigned to train.
    """
    train_cutoff = datetime.strptime(
        cfg["train_cutoff_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)
    val_cutoff = datetime.strptime(
        cfg["val_cutoff_date"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)

    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []
    no_date = 0

    for dlg in dialogues:
        first_date = _dialogue_first_date(dlg)
        if first_date is None:
            no_date += 1
            train.append(dlg)
            continue
        if first_date < train_cutoff:
            train.append(dlg)
        elif first_date < val_cutoff:
            val.append(dlg)
        else:
            test.append(dlg)

    total = len(train) + len(val) + len(test)

    # Verify thread boundary integrity
    train_ids = {d["id"] for d in train}
    val_ids   = {d["id"] for d in val}
    test_ids  = {d["id"] for d in test}
    tv_overlap = len(train_ids & val_ids)
    tt_overlap = len(train_ids & test_ids)
    vt_overlap = len(val_ids & test_ids)
    assert tv_overlap == 0, f"Thread boundary violation: {tv_overlap} dialogues in train ∩ val"
    assert tt_overlap == 0, f"Thread boundary violation: {tt_overlap} dialogues in train ∩ test"
    assert vt_overlap == 0, f"Thread boundary violation: {vt_overlap} dialogues in val ∩ test"

    stats = {
        "stage": 3,
        "n_train_dialogues": len(train),
        "n_val_dialogues":   len(val),
        "n_test_dialogues":  len(test),
        "n_no_date":         no_date,
        "train_pct":         round(len(train) / total * 100, 1) if total else 0,
        "val_pct":           round(len(val) / total * 100, 1) if total else 0,
        "test_pct":          round(len(test) / total * 100, 1) if total else 0,
        "overlap_train_val":  tv_overlap,
        "overlap_train_test": tt_overlap,
        "overlap_val_test":   vt_overlap,
        "zero_overlap_confirmed": True,
    }
    print(f"  Train: {len(train):,} ({stats['train_pct']}%)  "
          f"Val: {len(val):,} ({stats['val_pct']}%)  "
          f"Test: {len(test):,} ({stats['test_pct']}%)  "
          f"No-date: {no_date:,}")
    print(f"  Overlaps — T∩V: {tv_overlap}, T∩Te: {tt_overlap}, V∩Te: {vt_overlap}  ✓ zero overlap confirmed")

    return train, val, test, stats


# ── Stage 4 ───────────────────────────────────────────────────────────────────

def _generate_pairs_for_split(
    dialogues: List[Dict],
    cfg: dict,
    apply_diversity_filter: bool,
) -> Tuple[List[Dict], Dict]:
    """Generate (ctx, resp) string pairs from a list of dialogues.

    For each dialogue, same-speaker turns are merged, then for each
    response position i the context is the max_ctx_turns most-recent
    preceding turns joined by a space.

    The response diversity filter (apply_diversity_filter=True) limits
    how many times any identical response string can appear.

    Returns (pairs, discard_counts).
    """
    max_ctx_turns  = cfg["max_ctx_turns"]
    max_resp_tokens = cfg["max_resp_tokens"]
    min_resp_tokens = cfg["min_resp_tokens"]

    resp_counter: Counter = Counter()
    max_occ = cfg.get("max_response_occurrences", 500)
    do_diversity = apply_diversity_filter and cfg.get("filter_response_diversity", True)

    disc = defaultdict(int)
    pairs: List[Dict] = []

    for dlg in dialogues:
        merged = _merge_same_speaker_turns(dlg["turns"])
        if len(merged) < 2:
            continue

        for i in range(1, len(merged)):
            start = max(0, i - max_ctx_turns)
            # Join context turns with the Ubuntu corpus-standard __eot__ delimiter
            # (Lowe et al. 2015).  The SPM model is trained with __eot__ in
            # the corpus so it is guaranteed a single dedicated token ID.
            ctx_text  = " __eot__ ".join(merged[j]["text"] for j in range(start, i))
            resp_text = merged[i]["text"]

            # Mask bare speaker-name references (e.g. "gordonjcp i think …")
            # Only mask names that look like IRC handles (have digits, underscores,
            # hyphens, or are unusually long) to avoid clobbering common words.
            for sp in {t["from"].lower() for t in merged}:
                if re.search(r"[\d_\-\[\]\\^{}|`]", sp) or len(sp) > 9:
                    pattern = r"\b" + re.escape(sp) + r"\b"
                    ctx_text  = re.sub(pattern, "__user__", ctx_text)
                    resp_text = re.sub(pattern, "__user__", resp_text)

            # Always mask known IRC bot names appearing in message text —
            # stage 2 drops bot turns but humans still write "follow ubottu's
            # message", "ubotu can help", etc.  Possessive 's is consumed.
            ctx_text  = _RE_BOT_NAMES.sub("__user__", ctx_text)
            resp_text = _RE_BOT_NAMES.sub("__user__", resp_text)

            # Rough token estimate (word count) for filtering
            resp_words = resp_text.split()
            if len(resp_words) < min_resp_tokens:
                disc["resp_too_short"] += 1
                continue
            if len(resp_words) > max_resp_tokens:
                disc["resp_too_long"] += 1
                continue

            # Pair-level quality filters
            if cfg.get("filter_non_english", True) and not _is_english_response(resp_text):
                disc["non_english"] += 1
                continue
            if cfg.get("filter_bot_responses", True) and _is_bot_response(resp_text):
                disc["bot_response"] += 1
                continue
            if cfg.get("filter_echo_pairs", True) and _is_echo_pair(resp_text, ctx_text):
                disc["echo_pair"] += 1
                continue
            if cfg.get("filter_placeholder", True) and _is_placeholder_only(resp_text):
                disc["placeholder_only"] += 1
                continue

            # Turn-coherence filter: discard pairs where no recent substantive
            # ctx turn shares content words with the response — catches
            # dialogue-window misalignments.
            #
            # Blind-spot fix: the last turn is often a short ack ("yes", "ok",
            # "sure") which has < 5 content words.  We now walk BACKWARD through
            # ctx turns to find the most recent one that is substantive (≥ 5
            # content words) before running the overlap check.  This catches the
            # pattern: [long question] __eot__ [cingular] __eot__ [yes] / [XP resp]
            if cfg.get("filter_incoherent_pairs", True):
                ctx_turns = ctx_text.split(" __eot__ ")
                ctx_content: set = set()
                for turn in reversed(ctx_turns):
                    words = {
                        w for w in turn.split()
                        if w.isalpha() and len(w) >= 4 and w not in _COHERENCE_STOPWORDS
                    }
                    if len(words) >= 5:
                        ctx_content = words
                        break
                resp_content = {
                    w for w in resp_text.split()
                    if w.isalpha() and len(w) >= 4 and w not in _COHERENCE_STOPWORDS
                }
                if len(ctx_content) >= 5 and len(resp_content) >= 5:
                    if not (ctx_content & resp_content):
                        disc["incoherent_pair"] += 1
                        continue

            # Response diversity cap (train only)
            if do_diversity:
                if resp_counter[resp_text] >= max_occ:
                    disc["diversity_cap"] += 1
                    continue
                resp_counter[resp_text] += 1

            pairs.append({"ctx": ctx_text, "resp": resp_text})

    return pairs, dict(disc)


def _write_stage4_samples(
    splits: Dict[str, List[Dict]],
    artifact_dir: Path,
    n: int = 200,
    seed: int = 42,
) -> None:
    """Write n randomly-sampled pairs from each split as a readable text file.

    Files are saved alongside the main stage 4 JSONs as:
        stage4_train_samples.txt
        stage4_val_samples.txt
        stage4_test_samples.txt

    Each pair is formatted as:
        ── Pair 001 ──────────────────────────────────────
        CTX : <context with __eot__ separators>
        RESP: <response>

    Re-running always overwrites so the samples stay fresh.
    """
    rng = random.Random(seed)
    for split_name, pairs in splits.items():
        sample = rng.sample(pairs, min(n, len(pairs)))
        out_path = artifact_dir / f"stage4_{split_name}_samples.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Stage 4 — {split_name.upper()} samples  "
                    f"({len(sample)} of {len(pairs):,} pairs, seed={seed})\n")
            f.write("=" * 70 + "\n\n")
            for idx, pair in enumerate(sample, 1):
                f.write(f"── Pair {idx:03d} {'─' * 50}\n")
                f.write(f"CTX : {pair['ctx']}\n")
                f.write(f"RESP: {pair['resp']}\n\n")
        print(f"  Saved {len(sample)} {split_name} samples → {out_path.name}")


def stage4_generate_pairs(
    train_dialogues: List[Dict],
    val_dialogues: List[Dict],
    test_dialogues: List[Dict],
    cfg: dict,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Extract (context, response) string pairs from all three splits.

    Applies the response diversity filter on train only.
    Caps train to cfg['max_train_pairs'] if > 0.
    Returns (train_pairs, val_pairs, test_pairs, stats).
    """
    print("  Generating train pairs …")
    train_pairs, train_disc = _generate_pairs_for_split(train_dialogues, cfg, apply_diversity_filter=True)
    print(f"    train raw: {len(train_pairs):,}  discards: {train_disc}")

    max_pairs = cfg.get("max_train_pairs", 0)
    if max_pairs > 0 and len(train_pairs) > max_pairs:
        # Random shuffle before cap to avoid temporal/alphabetical bias.
        # Without this, we'd only see dialogues from the earliest part of
        # the corpus — identical to the old phase1.py approach (ported back).
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:max_pairs]
        print(f"    train capped to {max_pairs:,} (randomly sampled)")

    if len(train_pairs) == 0:
        raise ValueError("Stage 4: 0 train pairs produced — check filters and corpus path.")

    print("  Generating val pairs …")
    val_pairs, val_disc = _generate_pairs_for_split(val_dialogues, cfg, apply_diversity_filter=False)
    print(f"    val: {len(val_pairs):,}  discards: {val_disc}")

    print("  Generating test pairs …")
    test_pairs, test_disc = _generate_pairs_for_split(test_dialogues, cfg, apply_diversity_filter=False)
    print(f"    test: {len(test_pairs):,}  discards: {test_disc}")

    stats = {
        "stage": 4,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs":   len(val_pairs),
        "n_test_pairs":  len(test_pairs),
        "train_discards": train_disc,
        "val_discards":   val_disc,
        "test_discards":  test_disc,
    }
    return train_pairs, val_pairs, test_pairs, stats


# ── Stage 5 ───────────────────────────────────────────────────────────────────

def stage5_train_spm(train_pairs: List[Dict], cfg: dict) -> str:
    """Train SentencePiece BPE model on training context + response text.

    Uses explicit pad_id/unk_id/bos_id/eos_id parameters to guarantee
    the token ID contract: pad=0, unk=1, sos=2, eos=3.
    Returns the path to the saved .model file.
    """
    import sentencepiece as spm

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(artifact_dir / "stage5_spm")

    # Write corpus to a temp file (tmp-then-replace not needed — SPM writes directly)
    corpus_path = artifact_dir / "stage5_spm_corpus.tmp"
    print(f"  Writing SPM corpus ({len(train_pairs):,} pairs) …")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            ctx_line = pair["ctx"].strip()
            resp_line = pair["resp"].strip()
            if ctx_line:
                f.write(ctx_line + "\n")
            if resp_line:
                f.write(resp_line + "\n")

    print(f"  Training SentencePiece BPE  vocab_size={cfg['spm_vocab_size']} …")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=cfg["spm_vocab_size"],
        model_type=cfg.get("spm_model_type", "bpe"),
        character_coverage=cfg.get("spm_character_coverage", 0.9999),
        input_sentence_size=cfg.get("spm_input_sentence_size", 2_000_000),
        shuffle_input_sentence=True,
        pad_id=0,    pad_piece="<pad>",
        unk_id=1,    unk_piece="<unk>",
        bos_id=2,    bos_piece="<sos>",
        eos_id=3,    eos_piece="<eos>",
        user_defined_symbols=[
            # All __PLACEHOLDER__ tokens must be guaranteed single pieces.
            # Without this, BPE splits them into fragments (e.g. __url__ →
            # ['▁__','url','__']) which wastes token budget and makes the
            # turn delimiter __eot__ noisy rather than a clean boundary signal.
            "__url__", "__path__", "__ip__", "__cmd__", "__number__",
            "__eot__", "__user__",
        ],
    )

    model_path = model_prefix + ".model"

    # Verify token ID contract
    sp = spm.SentencePieceProcessor(model_file=model_path)
    assert sp.piece_to_id("<pad>") == 0, f"<pad> ID mismatch: got {sp.piece_to_id('<pad>')}"
    assert sp.piece_to_id("<unk>") == 1, f"<unk> ID mismatch: got {sp.piece_to_id('<unk>')}"
    assert sp.piece_to_id("<sos>") == 2, f"<sos> ID mismatch: got {sp.piece_to_id('<sos>')}"
    assert sp.piece_to_id("<eos>") == 3, f"<eos> ID mismatch: got {sp.piece_to_id('<eos>')}"
    print(f"  ✓ Token ID contract verified: pad=0, unk=1, sos=2, eos=3")

    # Clean up temp corpus file
    corpus_path.unlink(missing_ok=True)

    print(f"  SPM model saved → {model_path}")
    return model_path


# ── Stage 6 ───────────────────────────────────────────────────────────────────

def _encode_split(
    pairs: List[Dict],
    sp,
    out_path: Path,
    max_ctx_tokens: int,
    max_resp_tokens: int,
    sos_id: int,
    eos_id: int,
) -> int:
    """Encode one split to JSONL and return number of lines written."""
    tmp = out_path.with_suffix(".tmp")
    written = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for pair in pairs:
            ctx_ids_full = sp.encode(pair["ctx"], out_type=int)
            # Keep the LAST max_ctx_tokens BPE tokens (most recent dialogue turns)
            ctx_ids = ctx_ids_full[-max_ctx_tokens:]
            resp_ids = (
                [sos_id]
                + sp.encode(pair["resp"], out_type=int)[:max_resp_tokens]
                + [eos_id]
            )
            f.write(json.dumps({"ctx": ctx_ids, "resp": resp_ids}) + "\n")
            written += 1
    os.replace(tmp, out_path)
    return written


def stage6_encode_pairs(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    test_pairs: List[Dict],
    spm_model_path: str,
    cfg: dict,
) -> Tuple[str, str, str, Dict, Dict]:
    """Encode all pairs to BPE token ID sequences and write JSONL files.

    Context encoding (M3 — no <sos> on encoder):
        ctx_ids = sp.encode(ctx_text)[-max_ctx_tokens:]   # keep LAST N tokens (most recent context)

    Response encoding:
        resp_ids = [sos_id] + sp.encode(resp_text)[:max_resp_tokens] + [eos_id]

    Returns (train_path, val_path, test_path, vocab_dict, stats).
    """
    import sentencepiece as spm_module

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sp = spm_module.SentencePieceProcessor(model_file=spm_model_path)

    sos_id = sp.piece_to_id("<sos>")   # == 2
    eos_id = sp.piece_to_id("<eos>")   # == 3
    vocab_size = sp.get_piece_size()

    max_ctx_tokens  = cfg["max_ctx_tokens"]
    max_resp_tokens = cfg["max_resp_tokens"]

    train_path = artifact_dir / "stage6_train_ids.jsonl"
    val_path   = artifact_dir / "stage6_val_ids.jsonl"
    test_path  = artifact_dir / "stage6_test_ids.jsonl"

    print(f"  Encoding train ({len(train_pairs):,} pairs) …")
    n_train = _encode_split(train_pairs, sp, train_path, max_ctx_tokens, max_resp_tokens, sos_id, eos_id)

    print(f"  Encoding val ({len(val_pairs):,} pairs) …")
    n_val = _encode_split(val_pairs, sp, val_path, max_ctx_tokens, max_resp_tokens, sos_id, eos_id)

    print(f"  Encoding test ({len(test_pairs):,} pairs) …")
    n_test = _encode_split(test_pairs, sp, test_path, max_ctx_tokens, max_resp_tokens, sos_id, eos_id)

    # Build vocab dict {piece_str: id}
    vocab: Dict[str, int] = {sp.id_to_piece(i): i for i in range(vocab_size)}

    vocab_path = artifact_dir / "stage6_vocab.json"
    tmp_vocab  = vocab_path.with_suffix(".tmp")
    with open(tmp_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    os.replace(tmp_vocab, vocab_path)
    print(f"  Vocab saved → {vocab_path.name}  ({vocab_size:,} entries)")

    stats = {
        "stage": 6,
        "vocab_size":  vocab_size,
        "n_train":     n_train,
        "n_val":       n_val,
        "n_test":      n_test,
        "sos_id":      sos_id,
        "eos_id":      eos_id,
    }
    stats_path = artifact_dir / "stage6_stats.json"
    _save_json(stats, stats_path)

    return str(train_path), str(val_path), str(test_path), vocab, stats


# ── Stage 7 ───────────────────────────────────────────────────────────────────

def stage7_train_fasttext(spm_model_path: str, all_pairs: List[Dict], cfg: dict) -> str:
    """Tokenise all corpus pairs with SPM and train a FastText model.

    Uses all available pairs (train + val + test combined) for richer
    embedding coverage — mirrors the original phase1 strategy of training
    FastText on the full pre-downsampled corpus (~4.8M lines) rather than
    only the capped 1.5M train set.

    The FastText model is trained on BPE-tokenised text (piece strings
    including ▁ word-initial markers), so every embedding vector
    corresponds directly to a BPE piece.

    Args:
        spm_model_path: Path to the trained SentencePiece .model file.
        all_pairs:      Combined list of dicts with 'ctx' and 'resp' keys
                        (typically train + val + test pairs from stage 4).
        cfg:            Pipeline config dict.

    Returns path to the saved FastText .model file.
    """
    import sentencepiece as spm_module
    from gensim.models import FastText
    from gensim.models.word2vec import LineSentence

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    sp = spm_module.SentencePieceProcessor(model_file=spm_model_path)

    corpus_path = artifact_dir / "stage7_bpe_corpus.tmp"
    print(f"  Tokenising {len(all_pairs):,} pairs with SPM …")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            for text in (pair["ctx"], pair["resp"]):
                pieces = sp.encode(text.strip(), out_type=str)
                if pieces:
                    f.write(" ".join(pieces) + "\n")

    print(f"  Training FastText  dim={cfg['fasttext_dim']}  epochs={cfg['fasttext_epochs']}  sg={cfg.get('fasttext_sg', 1)} …")
    sentences = LineSentence(str(corpus_path))
    model = FastText(
        sentences,
        vector_size=cfg.get("fasttext_dim", 300),
        epochs=cfg.get("fasttext_epochs", 10),
        min_count=cfg.get("fasttext_min_count", 3),
        window=cfg.get("fasttext_window", 5),
        sg=cfg.get("fasttext_sg", 1),
        workers=cfg.get("fasttext_workers", 8),
    )

    model_path = str(artifact_dir / "stage7_fasttext.model")
    model.save(model_path)
    print(f"  FastText model saved → {model_path}")

    corpus_path.unlink(missing_ok=True)
    return model_path


# ── Stage 8 ───────────────────────────────────────────────────────────────────

def stage8_build_embedding_matrix(
    vocab: Dict[str, int],
    fasttext_model_path: str,
    cfg: dict,
) -> Tuple[str, Dict]:
    """Build numpy embedding matrix [vocab_size × embed_dim] from FastText.

    For each vocab piece (by integer ID order), the embedding is looked up
    using the EXACT piece string including the ▁ word-initial prefix (M5 fix).
    FastText's subword mechanism handles any piece not directly in training.
    The <pad> row (index 0) is forced to all zeros.

    Returns (matrix_path, stats).
    """
    from gensim.models import FastText

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading FastText model from {fasttext_model_path} …")
    ft_model = FastText.load(fasttext_model_path)

    vocab_size = len(vocab)
    embed_dim  = cfg.get("fasttext_dim", 300)
    print(f"  Building embedding matrix  [{vocab_size} × {embed_dim}] …")

    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    n_found = 0
    for piece_str, idx in vocab.items():
        if idx == 0:
            # <pad> row stays all zeros
            continue
        try:
            vec = ft_model.wv[piece_str]   # exact ▁-prefixed piece string
            matrix[idx] = vec
            n_found += 1
        except KeyError:
            pass   # FastText subword fallback is automatic; this shouldn't happen

    # Guarantee pad row is zero (in case FastText returned something)
    matrix[0] = 0.0

    assert matrix[0].sum() == 0.0, "<pad> row is not all zeros"
    print(f"  Vectors filled: {n_found:,}/{vocab_size:,}  (pad row forced to zeros)")

    matrix_path = artifact_dir / "stage8_embedding_matrix.npy"
    tmp_path    = artifact_dir / "stage8_embedding_matrix_tmp.npy"   # must end .npy so np.save writes here
    np.save(str(tmp_path), matrix)
    os.replace(tmp_path, matrix_path)
    print(f"  Embedding matrix saved → {matrix_path.name}  shape={matrix.shape}")

    stats = {
        "stage": 8,
        "vocab_size":  vocab_size,
        "embed_dim":   embed_dim,
        "n_filled":    n_found,
        "pad_row_sum": float(matrix[0].sum()),
        "matrix_shape": list(matrix.shape),
    }
    _save_json(stats, artifact_dir / "stage8_stats.json")
    return str(matrix_path), stats


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main(cfg: Optional[Dict] = None, script_name: str = "phase1") -> None:
    """Run all pipeline stages sequentially, skipping completed ones.

    Each stage checks whether all its STAGE_ARTIFACTS exist; if so it
    loads from disk and skips computation. Otherwise it runs, saves
    artifacts atomically, and logs timing statistics.

    Args:
        cfg: Config dict; defaults to PHASE1_CONFIG.
        script_name: Used for the run log filename (e.g. "phase1_mini").
    """
    if cfg is None:
        cfg = PHASE1_CONFIG

    from logging_utils import setup_run_logging
    setup_run_logging(script_name, log_dir=cfg.get("log_dir", "new/logs"))

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    s1_path = artifact_dir / "stage1_dialogues.pkl"
    if _stage_done(1, artifact_dir):
        print("✓ Stage 1 already complete — loading dialogues …")
        dialogues = _load_pickle(s1_path)
    else:
        print("=" * 60)
        print("STAGE 1 — Load corpus")
        print("=" * 60)
        t0 = time.time()
        dialogues = stage1_load_corpus(cfg)
        _save_pickle(dialogues, s1_path)
        print(f"Stage 1 done ({_elapsed(t0)})  {len(dialogues):,} dialogues\n")

    # Optional subsample — used by phase1_mini.py to work on 10% of dialogues.
    # Applied after the stage 1 pickle so a full run can share its stage 1
    # cache and mini only re-runs stages 2-8 on the subset.
    subsample_frac = float(cfg.get("stage1_subsample_frac", 1.0))
    if subsample_frac < 1.0:
        n_keep = max(1, int(len(dialogues) * subsample_frac))
        rng = random.Random(42)
        dialogues = rng.sample(dialogues, n_keep)
        print(f"  [mini] Subsampled to {len(dialogues):,} dialogues "
              f"({subsample_frac:.0%} of corpus, seed=42)\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    s2_path       = artifact_dir / "stage2_clean_dialogues.pkl"
    s2_stats_path = artifact_dir / "stage2_stats.json"
    if _stage_done(2, artifact_dir):
        print("✓ Stage 2 already complete — loading clean dialogues …")
        clean_dialogues = _load_pickle(s2_path)
    else:
        print("=" * 60)
        print("STAGE 2 — Clean and filter")
        print("=" * 60)
        t0 = time.time()
        clean_dialogues, s2_stats = stage2_clean_and_filter(dialogues, cfg)
        s2_stats["elapsed"] = _elapsed(t0)
        _save_pickle(clean_dialogues, s2_path)
        _save_json(s2_stats, s2_stats_path)
        del dialogues
        gc.collect()
        print(f"Stage 2 done ({_elapsed(t0)})  {len(clean_dialogues):,} dialogues kept\n")

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    s3_train_path = artifact_dir / "stage3_train.pkl"
    s3_val_path   = artifact_dir / "stage3_val.pkl"
    s3_test_path  = artifact_dir / "stage3_test.pkl"
    s3_stats_path = artifact_dir / "stage3_stats.json"
    if _stage_done(3, artifact_dir):
        print("✓ Stage 3 already complete — loading splits …")
        train_dlg = _load_pickle(s3_train_path)
        val_dlg   = _load_pickle(s3_val_path)
        test_dlg  = _load_pickle(s3_test_path)
    else:
        print("=" * 60)
        print("STAGE 3 — Temporal split (by thread first-turn date)")
        print("=" * 60)
        t0 = time.time()
        train_dlg, val_dlg, test_dlg, s3_stats = stage3_temporal_split(clean_dialogues, cfg)
        s3_stats["elapsed"] = _elapsed(t0)
        _save_pickle(train_dlg, s3_train_path)
        _save_pickle(val_dlg,   s3_val_path)
        _save_pickle(test_dlg,  s3_test_path)
        _save_json(s3_stats, s3_stats_path)
        del clean_dialogues
        gc.collect()
        print(f"Stage 3 done ({_elapsed(t0)})\n")

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    s4_train_path = artifact_dir / "stage4_train_pairs.json"
    s4_val_path   = artifact_dir / "stage4_val_pairs.json"
    s4_test_path  = artifact_dir / "stage4_test_pairs.json"
    s4_stats_path = artifact_dir / "stage4_stats.json"
    if _stage_done(4, artifact_dir):
        print("✓ Stage 4 already complete — loading pairs …")
        train_pairs = _load_json(s4_train_path)
        val_pairs   = _load_json(s4_val_path)
        test_pairs  = _load_json(s4_test_path)
        _write_stage4_samples(
            {"train": train_pairs, "val": val_pairs, "test": test_pairs},
            artifact_dir,
            n=200,
            seed=42,
        )
    else:
        print("=" * 60)
        print("STAGE 4 — Generate context-response pairs")
        print("=" * 60)
        t0 = time.time()
        train_pairs, val_pairs, test_pairs, s4_stats = stage4_generate_pairs(
            train_dlg, val_dlg, test_dlg, cfg
        )
        s4_stats["elapsed"] = _elapsed(t0)
        _save_json(train_pairs, s4_train_path)
        _save_json(val_pairs,   s4_val_path)
        _save_json(test_pairs,  s4_test_path)
        _save_json(s4_stats,    s4_stats_path)
        del train_dlg, val_dlg, test_dlg
        gc.collect()
        print(f"Stage 4 done ({_elapsed(t0)})  train={len(train_pairs):,}  val={len(val_pairs):,}  test={len(test_pairs):,}\n")

    # Write 200-pair human-readable sample files for each split so you can
    # inspect data quality without opening the full multi-hundred-MB JSONs.
    # Samples are drawn randomly each run (reproducible via seed below).
    _write_stage4_samples(
        {"train": train_pairs, "val": val_pairs, "test": test_pairs},
        artifact_dir,
        n=200,
        seed=42,
    )

    # ── Stage 5 ──────────────────────────────────────────────────────────────
    s5_model_path = artifact_dir / "stage5_spm.model"
    if _stage_done(5, artifact_dir):
        print("✓ Stage 5 already complete — SPM model exists")
        spm_model_path = str(s5_model_path)
    else:
        print("=" * 60)
        print("STAGE 5 — Train SentencePiece BPE")
        print("=" * 60)
        t0 = time.time()
        spm_model_path = stage5_train_spm(train_pairs, cfg)
        print(f"Stage 5 done ({_elapsed(t0)})\n")

    # ── Stage 6 ──────────────────────────────────────────────────────────────
    s6_vocab_path = artifact_dir / "stage6_vocab.json"
    if _stage_done(6, artifact_dir):
        print("✓ Stage 6 already complete — loading vocab …")
        vocab = _load_json(s6_vocab_path)
    else:
        print("=" * 60)
        print("STAGE 6 — Encode pairs to BPE IDs")
        print("=" * 60)
        t0 = time.time()
        _, _, _, vocab, s6_stats = stage6_encode_pairs(
            train_pairs, val_pairs, test_pairs, spm_model_path, cfg
        )
        del train_pairs, val_pairs, test_pairs
        gc.collect()
        print(f"Stage 6 done ({_elapsed(t0)})\n")

    # ── Stage 7 ──────────────────────────────────────────────────────────────
    s7_model_path = artifact_dir / "stage7_fasttext.model"
    if _stage_done(7, artifact_dir):
        print("✓ Stage 7 already complete — FastText model exists")
        ft_model_path = str(s7_model_path)
    else:
        print("=" * 60)
        print("STAGE 7 — Train FastText on BPE-tokenised corpus")
        print("=" * 60)
        t0 = time.time()
        # Load all 3 splits so FastText sees full vocabulary coverage.
        # NOTE: val/test token co-occurrence statistics are encoded in the initial
        # embedding vectors. Since freeze=False, this is progressively overwritten
        # during Seq2Seq fine-tuning, but the initialisation carries mild leakage.
        # This is a deliberate trade-off: train-only FastText risks OOV tokens in
        # val/test that would receive random (zero) initialisations instead.
        ft_pairs = _load_json(s4_train_path)
        ft_pairs += _load_json(s4_val_path)
        ft_pairs += _load_json(s4_test_path)
        print(f"  FastText corpus: {len(ft_pairs):,} pairs (train + val + test)")
        ft_model_path = stage7_train_fasttext(spm_model_path, ft_pairs, cfg)
        del ft_pairs
        gc.collect()
        print(f"Stage 7 done ({_elapsed(t0)})\n")

    # ── Stage 8 ──────────────────────────────────────────────────────────────
    if _stage_done(8, artifact_dir):
        print("✓ Stage 8 already complete — embedding matrix exists")
    else:
        print("=" * 60)
        print("STAGE 8 — Build embedding matrix")
        print("=" * 60)
        t0 = time.time()
        # Ensure spm_model_path is set in cfg for stage8 lookup
        cfg_with_spm = dict(cfg)
        cfg_with_spm["spm_model_path"] = spm_model_path
        matrix_path, s8_stats = stage8_build_embedding_matrix(vocab, ft_model_path, cfg_with_spm)
        print(f"Stage 8 done ({_elapsed(t0)})  matrix shape: {s8_stats['matrix_shape']}\n")

    print("=" * 60)
    print(f"✓ Phase 1 pipeline complete  ({_elapsed(t_total)})")
    print(f"  Artifacts in: {artifact_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
