# 06 — CMYKOGV (n=7) Implementation Plan — **BLOCKED: awaiting data**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full model comparison on an n=7 (CMYKOGV) dataset — the paper's core "can AI handle n>4" question — and produce the degradation-with-dimensionality figure (n = 3 → 4 → 7).

**Status: BLOCKED.** Data is with Will (ex-Kodak) via Phil (Apr 2026 meeting note). Parked last by decision (Aug 2026). Written now so execution is same-day when the data lands. **If no data by ~15 Aug:** the paper reframes to "n≤4 answered thoroughly + newsprint generalization", with n>4 as future work — decide with Phil, then mark this plan SKIPPED in `00-execution-order.md`.

**Architecture:** Nothing new — the pipeline is n-channel-generic by design. This plan is: inspect format → ingest to the standard CSV schema → register a 7-input `DatasetSpec` → run → one figure script.

## Global Constraints

- Same reporting stats; same tripwire (if the data ships without measured Lab, derive at ingestion as IFRA does and note the tripwire limitation).
- GP at n=7: if sample count is ≫2k, fitting cost grows steeply — subsample training folds to ≤2,000 with a fixed seed and report that choice.

---

### Task 1: Inspect + ingest

- [ ] Inventory the delivered files (format unknown — likely CGATS like IFRA; reuse `journal/pipeline/ifra.py:parse_cgats`, which is format-generic).
- [ ] Identify the 7 input columns; record verbatim names in `journal/data/processed/cmykogv/README.md` (create).
- [ ] Convert to standard schema CSV(s) in `journal/data/processed/cmykogv/` — reuse `spectral_to_xyz` if spectral, else map existing XYZ/Lab columns directly.
- [ ] Register in `datasets.py`:

```python
        specs['CMYKOGV'] = DatasetSpec(
            name='CMYKOGV',
            csv=REPO_ROOT / 'journal' / 'data' / 'processed' / 'cmykogv' / 'main.csv',
            input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K',
                        'OGV_O', 'OGV_G', 'OGV_V'),   # ← replace with verbatim names
            filter_k_zero=False)
```

- [ ] Commit ingestion.

### Task 2: Run + degradation figure

- [ ] `.venv/bin/python -m journal.pipeline.run --datasets CMYKOGV` (all 14 models; background).
- [ ] Create `journal/figures/make_dimensionality_figure.py`: median ΔE00 vs n (3, 4, 7) per model family, from `journal/results/{PC10-CMY,PC10-CMYK,CMYKOGV}/summary.csv`. Follow the dataviz skill; generate from CSVs only (repo rule).
- [ ] Acceptance: the n=3→4 trend (poly3 3× worse, GP flat) either continues to n=7 or it doesn't — both are headline findings. Anomalies investigated before reported.
- [ ] Commit results + figure script + PNG.
