#!/usr/bin/env bash
# setup_colab.sh — Run this once at the start of every Colab A100 session.
#
# What it does:
#   1. Mounts Google Drive (prompts for auth in a notebook cell — see NOTE below)
#   2. Clones the A100 branch (skips if already cloned)
#   3. Installs PyTorch + all pip requirements
#   4. Creates the Drive output folders
#   5. Copies the Ubuntu corpus CSV from Drive into the repo (if stored there)
#
# Usage (from a Colab terminal):
#   bash setup_colab.sh
#
# NOTE: Google Drive must be mounted BEFORE running this script.
# In a notebook cell, run this first:
#   from google.colab import drive; drive.mount('/content/drive')
# Then open a terminal and run this script.
#
# If you are working entirely in the terminal (no notebook), mount Drive with:
#   python3 -c "from google.colab import drive; drive.mount('/content/drive')"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration — edit these two lines if needed ───────────────────────────
DRIVE_ROOT="/content/drive/MyDrive/nlp-chatbot-v2"
REPO_DIR="/content/nlp-chatbot"
GITHUB_REPO="https://github.com/zGeneral/nlp-chatbot-project-v2.0.git"
BRANCH="A100"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Colab A100 setup — nlp-chatbot-project-v2.0"
echo "  Drive root : $DRIVE_ROOT"
echo "  Repo dir   : $REPO_DIR"
echo "============================================================"

# ── 1. Verify Drive is mounted ────────────────────────────────────────────────
if [ ! -d "/content/drive/MyDrive" ]; then
    echo ""
    echo "ERROR: Google Drive is not mounted at /content/drive/MyDrive"
    echo "Mount it first:"
    echo "  python3 -c \"from google.colab import drive; drive.mount('/content/drive')\""
    exit 1
fi
echo "[1/5] Google Drive detected ✓"

# ── 2. Clone repo (skip if already present) ───────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[2/5] Repo already cloned — pulling latest changes..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[2/5] Cloning branch '$BRANCH'..."
    git clone --branch "$BRANCH" --single-branch "$GITHUB_REPO" "$REPO_DIR"
fi

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/5] Installing dependencies..."
# Colab already has a compatible PyTorch with CUDA; upgrade to ensure bf16 support.
pip install torch --upgrade -q
pip install -r "$REPO_DIR/requirements.txt" -q
echo "      Dependencies installed ✓"

# ── 4. Create Drive output folders ───────────────────────────────────────────
echo "[4/5] Creating Drive output folders..."
mkdir -p \
    "$DRIVE_ROOT/artifacts" \
    "$DRIVE_ROOT/checkpoints" \
    "$DRIVE_ROOT/tb_logs" \
    "$DRIVE_ROOT/logs"
echo "      Folders ready ✓"
echo "      $DRIVE_ROOT/"
echo "        ├── artifacts/    ← SPM model, embeddings, stage6 JSONL files"
echo "        ├── checkpoints/  ← best/last model checkpoints + history JSON"
echo "        ├── tb_logs/      ← TensorBoard event files"
echo "        └── logs/         ← run logs"

# ── 5. Link corpus CSV from Drive ─────────────────────────────────────────────
echo "[5/5] Checking Ubuntu corpus CSV..."
CORPUS_FILE="dialogueText_301.csv"
CORPUS_IN_DRIVE="$DRIVE_ROOT/data/Ubuntu-dialogue-corpus/$CORPUS_FILE"
CORPUS_DEST="$REPO_DIR/data/Ubuntu-dialogue-corpus"
CORPUS_IN_REPO="$CORPUS_DEST/$CORPUS_FILE"

mkdir -p "$CORPUS_DEST"

if [ -f "$CORPUS_IN_REPO" ] || [ -L "$CORPUS_IN_REPO" ]; then
    echo "      Corpus already linked/present in repo ✓"
elif [ -f "$CORPUS_IN_DRIVE" ]; then
    echo "      Corpus found in Drive — symlinking into repo..."
    ln -sf "$CORPUS_IN_DRIVE" "$CORPUS_IN_REPO"
    echo "      Symlinked ✓"
    echo "      $CORPUS_IN_DRIVE → $CORPUS_IN_REPO"
else
    echo ""
    echo "  ERROR: Ubuntu corpus CSV not found at expected Drive location:"
    echo "    $CORPUS_IN_DRIVE"
    echo ""
    echo "  Make sure the file is at:"
    echo "    MyDrive/nlp-chatbot-v2/data/Ubuntu-dialogue-corpus/dialogueText_301.csv"
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  cd $REPO_DIR"
echo ""
echo "  # Run data pipeline (phase1) — ~30-60 min on A100:"
echo "  python run.py phase1 --drive-dir $DRIVE_ROOT"
echo ""
echo "  # Smoke test — verify training works end-to-end (~5 min):"
echo "  python run.py train --drive-dir $DRIVE_ROOT --mode sample"
echo ""
echo "  # Full A100 training run (~8-12 hours):"
echo "  python run.py train --drive-dir $DRIVE_ROOT --mode full"
echo "============================================================"
