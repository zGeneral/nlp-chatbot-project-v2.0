"""
monitor.py — Live training log analyser for nlp-chatbot-project-v2.0
Tails the log file, parses epoch summaries, and prints advisory diagnostics.

Usage:
    python monitor.py                          # auto-detect latest log
    python monitor.py logs/train_XXXX.log      # specify log file
"""

import re
import sys
import time
import os
from pathlib import Path
from collections import deque
from datetime import datetime

# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── ANSI colours ──────────────────────────────────────────────────────────────
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
G  = "\033[92m"   # green
B  = "\033[94m"   # blue
C  = "\033[96m"   # cyan
M  = "\033[95m"   # magenta
DIM = "\033[2m"
BOLD = "\033[1m"
RST = "\033[0m"

# ── Regex patterns ─────────────────────────────────────────────────────────────
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)"
    r"\s+\|\s+PPL:\s+([\d.]+)\s+\|\s+LR:\s+([\de.\-+]+)"
    r"\s+\|\s+TF:\s+([\d.]+)\s+\|\s+Grad:\s+([\d.]+)\s+\|\s+([\d.]+)s"
)
RE_EARLY_STOP = re.compile(
    r"\[(\w+)\]\s+Early stopping at epoch\s+(\d+)\s+\(no improvement for\s+(\d+)\s+epochs\)"
)
RE_PHASE2 = re.compile(
    r"\[(\w+)\]\s+Phase 2 begins\s+\(epoch\s+(\d+)\)"
)
RE_BATCH = re.compile(
    r"Epoch\s+(\d+)\s+train:\s+(\d+)%.*?\|\s+(\d+)/(\d+)\s+\[.*?loss=([\d.]+).*?grad_norm=([\d.]+)"
)
RE_PEAK_MEM = re.compile(
    r"Peak GPU memory epoch\s+(\d+):\s+([\d.]+)\s+GB"
)
RE_ETA = re.compile(
    r"ETA:\s+~(\d+)\s+min for\s+(\d+)\s+remaining epoch"
)

# ── Thresholds for diagnostics ─────────────────────────────────────────────────
STALL_DELTA        = 0.001   # val_loss improvement considered stalling
COLLAPSE_RATIO     = 1.15    # val_loss spike > 15% above recent min → collapse warning
GRAD_EXPLODE       = 10.0    # grad norm above this → explosion warning
GRAD_VANISH        = 0.01    # grad norm below this → vanishing warning
OVERFIT_MARGIN     = 0.30    # val_loss - train_loss > this → overfitting warning
PPL_HIGH           = 500.0   # perplexity above this is very high


def find_latest_log() -> Path:
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir = Path(__file__).parent / "logs"
    logs = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print(f"{R}No training logs found in {log_dir}{RST}")
        sys.exit(1)
    return logs[-1]


def open_shared(path: Path):
    """Open a file that may be locked by another process (Windows-safe)."""
    import io
    fh = open(path, "rb")
    return io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")


def bar(value, min_val, max_val, width=20, fill="█", empty="░") -> str:
    if max_val == min_val:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    filled = int(round(frac * width))
    return fill * filled + empty * (width - filled)


def pct_change(new, old) -> str:
    if old == 0:
        return ""
    delta = (new - old) / abs(old) * 100
    arrow = "▼" if delta < 0 else "▲"
    colour = G if delta < 0 else R
    return f"{colour}{arrow}{abs(delta):.2f}%{RST}"


class Monitor:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.epochs: list[dict] = []          # parsed epoch records
        self.recent_losses: deque = deque(maxlen=5)
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.no_improve_count = 0
        self.last_lr = None
        self.lr_reductions = 0
        self.phase2_started = False
        self.current_epoch_batches: list[float] = []
        self.last_batch_info: dict = {}
        self.warned: set = set()
        self.start_time = time.time()

    # ── Epoch-level analysis ───────────────────────────────────────────────────
    def analyse_epoch(self, e: dict) -> list[str]:
        messages = []
        ep    = e["epoch"]
        total = e["total"]
        tl    = e["train_loss"]
        vl    = e["val_loss"]
        ppl   = e["ppl"]
        lr    = e["lr"]
        grad  = e["grad"]
        tf    = e["tf"]

        prev = self.epochs[-1] if self.epochs else None

        # ── Best / improvement ──────────────────────────────────────────────
        improved = vl < self.best_val_loss - STALL_DELTA
        if improved:
            self.best_val_loss = vl
            self.best_epoch = ep
            self.no_improve_count = 0
            messages.append(f"{G}✅ New best val_loss={vl:.4f}  (checkpoint saved){RST}")
        else:
            self.no_improve_count += 1

        self.recent_losses.append(vl)

        # ── LR change ───────────────────────────────────────────────────────
        if self.last_lr is not None and lr < self.last_lr - 1e-10:
            self.lr_reductions += 1
            messages.append(
                f"{Y}📉 LR reduced: {self.last_lr:.2e} → {lr:.2e}  "
                f"(total reductions: {self.lr_reductions}){RST}"
            )
        self.last_lr = lr

        # ── Stalling ────────────────────────────────────────────────────────
        if self.no_improve_count >= 2:
            messages.append(
                f"{Y}⚠️  Stalling: no val improvement for {self.no_improve_count} epoch(s)  "
                f"(best={self.best_val_loss:.4f} @ ep {self.best_epoch}){RST}"
            )

        # ── Collapse / spike ────────────────────────────────────────────────
        if len(self.recent_losses) >= 3:
            recent_min = min(list(self.recent_losses)[:-1])
            if vl > recent_min * COLLAPSE_RATIO:
                messages.append(
                    f"{R}🔴 Loss spike! val={vl:.4f} vs recent_min={recent_min:.4f} "
                    f"({(vl/recent_min - 1)*100:.1f}% increase){RST}"
                )

        # ── Overfitting ──────────────────────────────────────────────────────
        gap = vl - tl
        if gap > OVERFIT_MARGIN:
            messages.append(
                f"{Y}⚠️  Overfitting signal: val-train gap = {gap:.4f}  "
                f"(train={tl:.4f}, val={vl:.4f}){RST}"
            )

        # ── Gradient health ──────────────────────────────────────────────────
        if grad > GRAD_EXPLODE:
            messages.append(f"{R}💥 Gradient explosion: grad_norm={grad:.3f}{RST}")
        elif grad < GRAD_VANISH:
            messages.append(f"{R}🪫  Vanishing gradients: grad_norm={grad:.6f}{RST}")

        # ── High perplexity ──────────────────────────────────────────────────
        if ppl > PPL_HIGH:
            messages.append(f"{Y}📈 High perplexity: PPL={ppl:.1f}{RST}")

        # ── Progress overview ────────────────────────────────────────────────
        if prev:
            vl_trend = pct_change(vl, prev["val_loss"])
            tl_trend = pct_change(tl, prev["train_loss"])
        else:
            vl_trend = tl_trend = ""

        progress_bar = bar(ep, 1, total)
        messages.append(
            f"{C}📊 Progress: [{progress_bar}] {ep}/{total} epochs{RST}"
        )
        messages.append(
            f"   train={BOLD}{tl:.4f}{RST} {tl_trend}  "
            f"val={BOLD}{vl:.4f}{RST} {vl_trend}  "
            f"PPL={ppl:.1f}  LR={lr:.2e}  TF={tf:.2f}  grad={grad:.3f}"
        )

        # ── Overall health verdict ───────────────────────────────────────────
        issues = [m for m in messages if any(x in m for x in ["⚠️", "🔴", "💥", "🪫"])]
        if not issues:
            if improved:
                verdict = f"{G}✨ Healthy — improving nicely{RST}"
            elif self.no_improve_count == 1:
                verdict = f"{B}ℹ️  One flat epoch — watching...{RST}"
            else:
                verdict = f"{G}👍 Looking good{RST}"
        elif len(issues) == 1 and "Stalling" in issues[0]:
            verdict = f"{Y}👀 Plateauing — LR scheduler may trigger soon{RST}"
        else:
            verdict = f"{R}⚡ Needs attention — see warnings above{RST}"
        messages.append(f"   {verdict}")

        return messages

    # ── Print header ──────────────────────────────────────────────────────────
    def print_header(self):
        os.system("cls")
        now = datetime.now().strftime("%H:%M:%S")
        print(f"{BOLD}{C}{'═'*70}{RST}")
        print(f"{BOLD}{C}  🤖 Training Monitor  ·  {self.log_path.name}  ·  {now}{RST}")
        print(f"{BOLD}{C}{'═'*70}{RST}")

    # ── Print live batch status ───────────────────────────────────────────────
    def print_live_status(self):
        if not self.last_batch_info:
            return
        b = self.last_batch_info
        print(
            f"\n{DIM}⏳ Epoch {b['epoch']} in progress: "
            f"{b['pct']}%  batch {b['cur']}/{b['total']}  "
            f"loss={b['loss']:.4f}  grad={b['grad']:.3f}{RST}"
        )

    # ── Main tail loop ────────────────────────────────────────────────────────
    def run(self):
        print(f"{B}Monitoring: {self.log_path}{RST}")
        print(f"{DIM}Waiting for epoch summaries...{RST}\n")

        with open_shared(self.log_path) as f:
            f.seek(0, 2)  # jump to end — only watch new data as it arrives
            # Actually read from start so we catch already-logged epochs
            f.seek(0)

            while True:
                line = f.readline()

                if not line:
                    time.sleep(1)
                    continue

                line = line.strip()

                # ── Batch-level live update (no reprint, just track) ─────────
                m = RE_BATCH.search(line)
                if m:
                    self.last_batch_info = {
                        "epoch": int(m.group(1)),
                        "pct":   int(m.group(2)),
                        "cur":   int(m.group(3)),
                        "total": int(m.group(4)),
                        "loss":  float(m.group(5)),
                        "grad":  float(m.group(6)),
                    }
                    continue  # don't reprint for every batch line

                # ── Phase 2 transition ───────────────────────────────────────
                m = RE_PHASE2.search(line)
                if m:
                    self.phase2_started = True
                    self.print_header()
                    print(f"\n{M}🔀 Phase 2 began at epoch {m.group(2)} "
                          f"— TF schedule shifting, counters reset{RST}\n")
                    continue

                # ── Early stopping ────────────────────────────────────────────
                m = RE_EARLY_STOP.search(line)
                if m:
                    self.print_header()
                    self._print_epoch_history()
                    print(f"\n{R}{BOLD}🛑 EARLY STOPPING at epoch {m.group(2)} "
                          f"— no improvement for {m.group(3)} epochs{RST}")
                    print(f"{G}   Best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}{RST}\n")
                    return

                # ── ETA ────────────────────────────────────────────────────────
                m = RE_ETA.search(line)
                if m:
                    self._eta_min = int(m.group(1))
                    continue

                # ── Epoch summary ─────────────────────────────────────────────
                m = RE_EPOCH.search(line)
                if not m:
                    continue

                e = {
                    "epoch":      int(m.group(1)),
                    "total":      int(m.group(2)),
                    "train_loss": float(m.group(3)),
                    "val_loss":   float(m.group(4)),
                    "ppl":        float(m.group(5)),
                    "lr":         float(m.group(6)),
                    "tf":         float(m.group(7)),
                    "grad":       float(m.group(8)),
                    "elapsed_s":  float(m.group(9)),
                }

                msgs = self.analyse_epoch(e)
                self.epochs.append(e)

                self.print_header()
                self._print_epoch_history()

                print(f"\n{BOLD}── Epoch {e['epoch']}/{e['total']}  "
                      f"({e['elapsed_s']:.0f}s) ──────────────────────────{RST}")
                for msg in msgs:
                    print(f"  {msg}")

                self.print_live_status()
                print()

    def _print_epoch_history(self):
        if not self.epochs:
            return
        print(f"\n{DIM}  {'Ep':>3}  {'Train':>8}  {'Val':>8}  {'PPL':>8}  "
              f"{'LR':>10}  {'TF':>5}  {'Grad':>7}  {'Time':>6}{RST}")
        print(f"{DIM}  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*5}  {'─'*7}  {'─'*6}{RST}")
        for e in self.epochs:
            best_mark = f"{G}★{RST}" if e["epoch"] == self.best_epoch else " "
            print(
                f"{DIM}  {e['epoch']:>3}{RST}{best_mark} "
                f"{e['train_loss']:>8.4f}  {e['val_loss']:>8.4f}  "
                f"{e['ppl']:>8.1f}  {e['lr']:>10.2e}  "
                f"{e['tf']:>5.2f}  {e['grad']:>7.3f}  {e['elapsed_s']:>5.0f}s"
            )


def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = find_latest_log()

    if not log_path.exists():
        print(f"{R}Log file not found: {log_path}{RST}")
        sys.exit(1)

    monitor = Monitor(log_path)
    try:
        monitor.run()
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{RST}")


if __name__ == "__main__":
    main()
