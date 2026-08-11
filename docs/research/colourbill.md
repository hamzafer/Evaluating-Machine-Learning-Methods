# colourbill / CharData Viewer — external benchmark characterization (Plan 11, Task 1)

Investigated 11 Aug 2026 (site v1.18.0, © William Li) via Playwright-driven browser.
URLs: https://chardata.colourbill.com/ (viewer), https://chardata.colourbill.com/profiletool/
(companion ICC-profile inspector/editor), user manual at https://chardata.colourbill.com/help.html.

## What the tool is

**CharData Viewer** is a free, fully browser-resident (no upload, no login, no payment)
explorer/comparator for printer characterisation datasets and ICC profiles. It is a *viewer and
diagnostic tool*, not a library: there is no API and no batch mode, so everything below was
extracted through the UI (screenshots + verbatim readings in `journal/results/colourbill/`).

## Inputs

- Characterisation data: CGATS/IT8, CSV, CxF/X-3, CxF/X-4 — device colorant percentages
  (n-channel, incl. 7-colorant CMYKOGV) + measured L\*a\*b\* and/or spectral reflectance.
- ICC profiles (treated as virtual datasets through their A2B transform).
- 57 built-in "Standard datasets" served from `https://chardata.colourbill.com/standard-data/`,
  including **exactly the files our study uses** (verified numerically identical, see below):
  `APTEC_CMYKOGV_Coated_LinearCTV_2025_M1`, `APTEC_PC10_CardBoard_2023_v1`,
  `APTEC_PC11_CCNB_2023_v1`, `FOGRA51` — plus FOGRA27–54, ISO 15339 CRPC1–7, IFRA26L/S (the
  *standard* IFRA26 newsprint set — NOT our IFRA 2005 13-press-run research data, so no overlap
  on the IFRA track), JapanColor2011.

## Outputs relevant to us

1. **Explore → Estimate** (the number we benchmark against): fits a polynomial model
   colorants → L\*a\*b\* and reports **Mean ΔE, Min ΔE, Max ΔE, Std Dev, Points fitted** plus the
   chosen degree. Documented fitting strategy (help §4.5):
   - starts at degree 2, steps up to max degree = min(5, #colorants); keeps the best model,
     stopping when a higher degree fits worse (criterion: mean ΔE, then std dev, then max);
   - least squares on Lab; Settings→Model "Weighted: On" (default) = iteratively reweighted LS
     (robust, outliers down-weighted), "Off" = ordinary LS;
   - ΔE method selectable in Settings: ΔEab (default) / ΔE94 / **ΔE00** — we set ΔE00;
   - **statistics are resubstitution (training-set) errors** — the model is fitted on all loaded
     rows and evaluated on the same rows; there is no cross-validation or holdout.
2. **Compare mode**: row-by-row ΔE/ΔL\*/ΔC\*/ΔH\* between two datasets (or dataset vs ICC profile)
   matched by device inking, with mean/min/max/std **and P90/P95**, histogram, per-channel
   breakdowns, G7 grey-balance check.
3. Duplicate handling: on load it reports duplicate measurements and their repeatability ΔE;
   optional "Filter Duplicates" (median/mean) collapses them.
4. Other (not benchmarked): 3D gamut shells, 2D slices, tone value/dot gain, L\* reversal scan,
   near-neutral survey, subset extraction, image-gamut checks.

**profiletool** inspects/edits ICC profiles (header/tags, CLUT, round-trip edits). Our models are
not packaged as ICC profiles, so it is out of scope for the benchmark; noted for completeness.

## Same-data verification (what makes the comparison defensible)

Downloaded the tool's standard files and compared numerically against ours:
- `APTEC_CMYKOGV_Coated_LinearCTV_2025_M1.txt` ≡ our `journal/data/raw/ncolor/APTEC_CMYKOGV_7clr_xyzlab.txt`:
  3534/3534 rows, every numeric column identical, same order. (Bonus: colourbill's copy carries the
  corrected header `NUMBER_OF_SETS 3534`, confirming our finding that the ICC-registry header's
  1624 is stale.)
- PC10 / PC11 / FOGRA51 ≡ our `data/cleaned/*.csv`: 1617/1617 rows exact on CMYK+LAB.
- Loading CMYKOGV with Filter Duplicates on: "Loaded 3534 rows, reduced to **3302** rows" —
  independently reproducing our exact-dedup count (3534 → 3302), and its duplicate report
  ("232 duplicate measurements, repeatability ΔE mean 0.00, max 0.00") matches our
  byte-identical-duplicates finding.

## Comparison scope decision

- **In scope (like-for-like data, stated protocol caveat):** CMYKOGV-7 (3302 dedup + 3534 raw),
  PC10-CMYK, PC11-CMYK, FOGRA51-CMYK — colourbill Estimate ΔE00 fit statistics vs our 5-fold-CV
  results, bridged by evaluating the same model class (least-squares polynomial → Lab) under both
  protocols (`journal/figures/fig_vs_colourbill.py`).
- **Out of scope:** our CMY variants (818-row K=0 subsets — colourbill has no K=0 filter);
  KCMYG-5 / CMYKOGB-7 (Phil's non-public files; could be loaded manually via File Select, but the
  built-in standard-set provenance is the point of an *external* benchmark); IFRA 2005 press runs
  (tool ships IFRA26, a different dataset); profiletool (we produce no ICC profiles).

## Reproduction finding

CharData's polynomial fitter is a **compiled WASM module** (`gamut.js` marshals
JSON into `mod.fitModel(...)`) — the exact basis/weighting is not inspectable. Re-implementing
the *documented* strategy (full multivariate polynomial basis, OLS on Lab, auto degree) fits
tighter than the tool's displayed statistics on 3 of 4 datasets (PC10 mean 0.14 vs 0.31;
CMYKOGV-7 degree-4 max 4.81 vs 12.37). Ruled out as explanations by direct test: fitting XYZ then
converting (mean 0.64 on PC10), interaction-restricted bases (0.87–5.3), float32 normal equations
(0.165), Huber/Tukey IRLS (0.138/0.146). Conclusion: colourbill's displayed stats are a
*conservative* reference for the polynomial-fit class; the like-for-like row in our comparison is
our own implementation of the documented class. Its Weighted On/Off setting produced identical
displayed stats (2 dp) on CMYKOGV in both duplicate-filter states.

## Results snapshot (details in journal/results/colourbill/)

ΔE00, D50/2°. colourbill = in-sample fit of its degree-4 polynomial; ours = pooled 5-fold CV.

| dataset | colourbill mean/max (in-sample) | poly4-Lab our CV mean/max | GP our CV mean/max |
|---|---|---|---|
| PC10-CMYK (1617) | 0.31 / 2.19 | 0.16 / 2.41 | 0.16 / 7.10* |
| PC11-CMYK (1617) | 0.26 / 1.64 | 0.15 / 2.39 | 0.14 / 6.22* |
| FOGRA51-CMYK (1617) | 0.34 / 2.65 | 0.22 / 3.86 | 0.12 / 4.19* |
| CMYKOGV-7 (3302 dedup) | 0.58 / 12.37 | 0.44 / 18.55 | 1.17 / 49.76 |

\* GP rows read from summaries mid-Plan-10-refresh (11 Aug); regenerate
`journal/figures/fig_vs_colourbill.py` after Plan 10 commits.

Headline for the paper: even under the harder out-of-sample protocol, our models' mean ΔE00 on
the n=4 charts matches or beats the external tool's *in-sample* fit quality on identical data;
on the 7-ink chart the polynomial's typical error stays low but worst-case error grows, in line
with our n>4 findings.

**ΔE00 metric cross-check:** exporting our PC10 CV predictions as CGATS and running colourbill's
Compare mode (measured vs predicted) reproduces our statistics exactly at the tool's displayed
precision (mean 0.16, median 0.11, P95 0.38, max 2.41 vs our 0.157/0.110/0.380/2.412) — the
external tool independently confirms our CIEDE2000 arithmetic
(`journal/results/colourbill/compare_crosscheck.csv`).

## The one caveat that must always accompany the numbers

colourbill's Estimate statistics answer "how well can a robust polynomial *describe* this chart"
(resubstitution / goodness-of-fit). Our numbers answer "how well does a model *predict unseen
patches*" (5-fold CV, ΔE00 on denormalized XYZ). Resubstitution is systematically optimistic;
the two must never be put in one column without the protocol label. The bridge rows in
`journal/results/colourbill/comparison.csv` make the gap explicit.
