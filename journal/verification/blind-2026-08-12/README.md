# Independent blind verification — 12 Aug 2026

A Claude Sonnet agent re-implemented the pipeline **from scratch** on a second machine
(colourlab, Linux x86_64) with access to the raw source files and a prose spec only — never to
`journal/pipeline`, `docs/`, `.superpowers/`, or the git history. It wrote its own numbers to
`blind_results.csv` before being allowed to read any of our result CSVs.

- `SPEC.md` — the prose specification it was given (no code, no expected numbers).
- `blind_results.csv` — its results, 36 rows (4 models × 9 dataset variants).
- `BLIND_REPORT.md` — its full report: row counts, ink scales, duplicate analysis, colorimetric
  consistency checks, header oddities, comparison with our published values, verdicts.
- `work/` — every line of analysis code it wrote, plus its run logs.

## Outcome

**Pipeline validated.** Its independent implementation reproduces our Gaussian process medians to
within 0.003 ΔE00 on all coated datasets (PC11-CMY exact) and our poly3 medians to within 0.001–0.006,
having written its own CGATS parser, Lab→XYZ and spectral→XYZ conversions, CV, and ΔE00 path.

**It endorsed our CMYKOGV-7 deduplication** after initially using all 3534 rows: repeated
bit-identical patches double-count in a pooled median, and fold-grouping prevents leakage but not
double-counting.

**It found one real defect (see `docs/research/cv-leakage-2026-08-12.md`):** the n≤4 datasets ran
ungrouped, so byte-identical duplicate rows could straddle a train/test split, contradicting the
protocol we state in the paper.

## Caveats on the review, disclosed by the agent itself
1. The harness auto-injects the repo's root `CLAUDE.md` into every session, so it saw programme-level
   context (deadlines, dataset provenance) though no code, hyperparameters or results. It backed its
   central finding with its own experiment rather than that file.
2. It compared against the **remote clone's** result CSVs, which our x86 replication runs had
   overwritten — so the deltas in its report are against x86 values. The coordinator re-ran the
   comparison against the canonical arm64 values; conclusions unchanged (see the leakage note).
