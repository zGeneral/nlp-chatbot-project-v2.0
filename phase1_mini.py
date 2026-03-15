"""phase1_mini.py — 10% mini pipeline run for fast iteration.

Runs all 8 phases of the Phase 1 data pipeline on a 10% random sample
of the Ubuntu Dialogue Corpus.  Artifacts are written to artifacts_mini/
so they never conflict with a full run.

Purpose
-------
Validate pipeline fixes and inspect data quality in ~12 minutes instead
of ~70 minutes.  After spotting issues in artifacts_mini/, apply the fix
to both phase1.py and phase1_mini.py (they share all code), then re-run
phase1_mini.py to verify before committing the full re-run.

Usage
-----
    python phase1_mini.py

Analyse results
---------------
    python analyze_data.py --artifact-dir artifacts_mini --samples 50

Design
------
This file is intentionally thin (~50 lines).  All logic lives in
phase1.py — no duplication.  We simply import phase1, build a config
override dict, and call phase1.main() with it.

Config changes vs full run
--------------------------
  stage1_subsample_frac : 0.10   → ~185k dialogues (random, seed=42)
  max_train_pairs       : 150_000 → 10% of 1.5M full cap
  spm_input_sentence_size: 200_000 → proportional SPM corpus cap
  fasttext_epochs       : 5       → halved (still enough for 10% data)
  artifact_dir          : artifacts_mini/
  (all quality filters, temporal splits, BPE vocab size unchanged)
"""

from __future__ import annotations

import copy
from pathlib import Path

# Import phase1 from the same directory as this file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1

_NEW_DIR = Path(__file__).resolve().parent


def main() -> None:
    cfg = copy.deepcopy(phase1.PHASE1_CONFIG)

    # ── Mini-specific overrides ───────────────────────────────────────────────
    cfg["stage1_subsample_frac"]    = 0.10        # 10% of dialogues
    cfg["max_train_pairs"]          = 150_000     # proportional train cap
    cfg["spm_input_sentence_size"]  = 200_000     # proportional SPM corpus cap
    cfg["fasttext_epochs"]          = 5           # halved; still good for 10% data
    cfg["artifact_dir"]             = str(_NEW_DIR / "artifacts_mini")
    cfg["domain_filter"]            = True        # domain filter ON in mini (matches full run)
    # log_dir stays the same — log filename uses "phase1_mini" prefix

    phase1.main(cfg, script_name="phase1_mini")


if __name__ == "__main__":
    main()
