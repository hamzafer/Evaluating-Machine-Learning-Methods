"""Re-score the archived tool-using equation answers (no API calls).

The equation experiment (`run_equation.py`) scores raw-API answers. Four more
answers were obtained through channels where the model could run code; they are
archived as text in `journal/llm/raw/equation_harness/` and described in
`journal/results/llm/harness_channel_note.md`. This script scores each one on the
same 100 held-out PC10-CMY patches, with the same parser and ΔE00 path as
`run_equation.py`, and writes `journal/results/llm/harness_summary.csv`.

    python -m journal.llm.score_harness
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from journal.llm.equation import parse_equation
from journal.llm.protocol import build_split
from journal.llm.run_equation import score_xyz

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "journal" / "llm" / "raw" / "equation_harness"
OUT = ROOT / "journal" / "results" / "llm" / "harness_summary.csv"

# (file, model, channel). Channels as recorded in harness_channel_note.md and
# in the message of commit 532c564 (web run).
RUNS = [
    ("claude-fable-5_web_equations.txt", "claude-fable-5",
     "web app with code interpreter"),
    ("claude-fable-5_claudecode_harness.txt", "claude-fable-5",
     "CLI harness, attempt 1 (ran Bash)"),
    ("claude-opus-5_claudecode_tools-flag.txt", "claude-opus-5",
     "CLI harness, attempt 2 (ran browser JavaScript)"),
    ("claude-fable-5_claudecode_tools-flag.txt", "claude-fable-5",
     "CLI harness, attempt 3 (called a deployed endpoint)"),
]


def main():
    _, test_df, spec = build_split("PC10-CMY", 400, 100, seed=42)
    X = test_df.loc[:, list(spec.input_cols)].to_numpy(dtype=float)
    Y = test_df.loc[:, list(spec.target_cols)].to_numpy(dtype=float)
    names = tuple(c.split("_")[1].lower() for c in spec.input_cols)

    rows = []
    for fname, model, channel in RUNS:
        eq = parse_equation((RAW / fname).read_text(), names)
        pred = np.nan_to_num(eq(X), nan=0.0, posinf=0.0, neginf=0.0)
        stats, n_clipped = score_xyz(pred, Y)
        rows.append({
            "source_file": fname, "model": model, "channel": channel,
            "code_execution": True, "n_terms": eq.n_terms,
            "max_total_degree": eq.max_total_degree,
            "max_var_exponent": eq.max_var_exponent,
            "median_de00": round(stats["median"], 3),
            "p95_de00": round(stats["p95"], 3),
            "max_de00": round(stats["max"], 3),
            "mean_de00": round(stats["mean"], 3),
            "n_test": stats["n"], "n_clipped": n_clipped,
        })
        print(f"{fname}: median {stats['median']:.3f}  p95 {stats['p95']:.3f}  "
              f"max {stats['max']:.3f}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
