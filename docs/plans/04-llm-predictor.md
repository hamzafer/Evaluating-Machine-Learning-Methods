# 04 — LLM as Direct Color Predictor Implementation Plan

> **SUPERSEDED BY PLAN 08** (multi-LLM). This plan's GPT-4o/mini run is DONE and kept as the preliminary single-provider baseline; the full multi-model comparison (Claude Fable/Opus, GPT, DeepSeek via OpenRouter) lives in `08-multi-llm.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **BEFORE implementing Task 2, load the `claude-api` skill** for current model ids, pricing, and SDK usage — do not code the API calls from memory.

**Goal:** Measure how well an LLM predicts XYZ from CMY(K) given training patches in-context, scored with the identical ΔE00 machinery as the 14 classical models. (Track requested by Phil / ICC GASIG: "Claude would do a better job than the ChatGPT models".)

**Architecture:** `journal/llm/` stays separate from `pipeline/` (API calls, cost, nondeterminism). A protocol module builds prompts from a train/test split; a runner calls the API, parses predictions, and writes results CSVs in the same schema as `journal/results/`. Classical comparison numbers come from the existing summaries.

**Tech Stack:** Python, `anthropic` SDK (install into `.venv` at execution time). **Gate: needs `ANTHROPIC_API_KEY` and a budget cap from Hamza — ask before the first paid call, not before writing code.**

## Global Constraints

- Same data discipline as the pipeline: PC10-CMY (818 K=0 rows) first; identical 90/10 split seeded 42 so the LLM sees the same task as a classical model would.
- Phil's overfitting concern (Apr 2026 notes): the in-context examples ARE the training set; test patches must never appear among them. Report train/test sizes explicitly.
- Deterministic decoding (temperature 0). Every raw API response is saved to disk (`journal/llm/raw/`) so results are re-scorable without re-spending.
- Cost control: hard cap via `--max-test-samples` (default 100) and a printed cost estimate BEFORE any call; abort unless `--yes`.

---

### Task 1: Protocol + prompt construction (no API needed)

**Files:**
- Create: `journal/llm/protocol.py`
- Test: `journal/llm/tests/test_protocol.py` (+ `__init__.py` files)

**Interfaces:**
- Produces: `build_split(dataset_name, n_train, n_test, seed=42) -> (train_df, test_df)`;
  `build_prompt(train_df, cmyk_row) -> str` (n-channel-generic: uses the dataset's input columns);
  `parse_xyz(text) -> tuple[float, float, float] | None` (tolerant of prose around a JSON object).

- [ ] **Step 1: Write the failing tests**

```python
# journal/llm/tests/test_protocol.py
from journal.llm.protocol import build_split, build_prompt, parse_xyz


def test_split_disjoint_and_sized():
    tr, te, spec = build_split('PC10-CMY', n_train=200, n_test=50)
    assert len(tr) == 200 and len(te) == 50
    overlap = tr.merge(te, on=list(spec.input_cols))   # same recipe both sides = leakage
    assert len(overlap) == 0


def test_prompt_contains_examples_and_query():
    tr, te, spec = build_split('PC10-CMY', 5, 1)
    p = build_prompt(tr, te.iloc[0], spec.input_cols)
    assert p.count('->') >= 5 and 'JSON' in p


def test_parse_xyz_tolerates_prose():
    assert parse_xyz('Sure! {"X": 32.71, "Y": 16.81, "Z": 11.32}') == (32.71, 16.81, 11.32)
    assert parse_xyz('no numbers here') is None
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest journal/llm/tests -v` → ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# journal/llm/protocol.py
"""Prompt protocol for LLM-as-color-predictor. Split is seeded and leakage-free."""
import json
import re

import pandas as pd

from journal.pipeline.datasets import registry


def build_split(dataset_name: str, n_train: int, n_test: int, seed: int = 42):
    spec = registry()[dataset_name]
    df = pd.read_csv(spec.csv)
    if spec.filter_k_zero:
        df = df[df['CMYK_K'] == 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=list(spec.input_cols))      # kill recipe twins
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_test], spec


def build_prompt(train_df, query_row, input_cols) -> str:
    lines = [
        "You are a printer color characterization model. Given ink percentages, "
        "predict the measured CIE XYZ tristimulus values (0-100 scale, D50).",
        "Training measurements from this printer:",
    ]
    for _, r in train_df.iterrows():
        ink = ", ".join(f"{c.split('_')[1]}={r[c]:g}" for c in input_cols)
        lines.append(f"{ink} -> X={r.XYZ_X:.2f}, Y={r.XYZ_Y:.2f}, Z={r.XYZ_Z:.2f}")
    ink = ", ".join(f"{c.split('_')[1]}={query_row[c]:g}" for c in input_cols)
    lines.append(
        f"Predict for: {ink}. Respond with ONLY a JSON object "
        '{"X": <number>, "Y": <number>, "Z": <number>}.')
    return "\n".join(lines)


def parse_xyz(text: str):
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return float(d['X']), float(d['Y']), float(d['Z'])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
```

(`build_split` returns `(train, test, spec)`; callers pass `spec.input_cols`
to `build_prompt` — the tests above already use these signatures.)

- [ ] **Step 4: Run tests to green**, then **Step 5: Commit**

```bash
git add journal/llm/
git commit -m "journal/llm: split + prompt protocol with leakage guard"
```

### Task 2: API runner + scoring  *(gate: ask Hamza for ANTHROPIC_API_KEY + budget)*

**Files:**
- Create: `journal/llm/run_llm.py`
- Output: `journal/llm/raw/<model>/<dataset>/<sample_id>.json`, `journal/results/llm/<dataset>_summary.csv`

**Load the `claude-api` skill first** for model ids/pricing; design points that are fixed regardless:

- Models: one strong + one small Claude (exact ids from the skill), temperature 0, max_tokens ~200.
- Defaults: `n_train=400` in-context examples, `n_test=100` queries, PC10-CMY; print estimated cost from the skill's pricing before running; require `--yes`.
- Failed parses count as missing — report `n_parsed` and score only parsed rows, and also report a worst-case variant where failures count as ΔE00 = the dataset's 99th percentile (so parse failures can't silently flatter the LLM).
- Scoring: `journal.pipeline.color.delta_e00(pred_xyz, true_xyz)`; summary via `journal.pipeline.evaluate.summarize`; CSV schema matches `journal/results/*/summary.csv` with an extra `model_id` column.
- Retry once on API error, then record failure. Sleep to respect rate limits.

- [ ] Step 1: load `claude-api` skill; confirm model ids + prices
- [ ] Step 2: implement `run_llm.py` per the design points above
- [ ] Step 3: dry run with `--max-test-samples 3` (needs key) and inspect raw responses
- [ ] Step 4: full run (100 samples, both models) after cost printout approved
- [ ] Step 5: commit code + results CSV (raw responses committed too — they are the provenance)

### Task 3: Comparison table

- [ ] Merge `journal/results/llm/*_summary.csv` with the classical `journal/results/PC10-CMY/summary.csv` into `journal/results/llm/comparison.csv` (one row per method, classical + LLM), commit. Expected narrative: LLM lands mid-field at best; wherever it lands, it answers the special issue's question directly.
