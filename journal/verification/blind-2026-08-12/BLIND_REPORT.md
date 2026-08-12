# Blind verification report — printer colour characterization

## 0. Blindness statement

I did not open, list, grep, or otherwise read `journal/pipeline`, `journal/llm`, `journal/figures`,
`docs/`, `.superpowers/`, or any git history of the authors' repository at any point before writing
`~/blind_verify/blind_results.csv`. All code in `~/blind_verify/work/` (parsers, colorimetry,
duplicate/consistency checks, cross-validation, model fitting) was written from scratch against
`SPEC.md` and the raw data files only. `blind_results.csv` was fully populated (27 cheap-model rows,
then all 9 `gaussian_process` rows) before I read a single byte of
`journal/results/**/summary.csv` or `journal/results/ifra/within_run.csv`.

One caveat, disclosed for completeness rather than as a breach of the letter of the rule: the Claude
Code session this review ran in had the project's root `CLAUDE.md` injected automatically as ambient
system context at the start of the conversation (this happens for every session in this repo,
independent of the task). It contains high-level programme narrative (deadlines, dataset provenance,
a general remark that "sklearn version drift moves results ~0.01-0.05") but no algorithmic detail,
no code, no result numbers, and nothing about CV/fold construction or model hyperparameters. I did not
navigate to it, and I deliberately did not use it to steer or "explain away" any of the analysis below
— all mechanism claims here (in particular the fold-grouping finding in §6) are backed by a from-scratch
empirical experiment I ran on my own side, not by anything in that file. I flag its existence because
true blindness would ideally exclude even this, and I'd rather over-disclose than let it pass silently.

With that caveat noted, the substantive rule — do not read their implementation — held.

## 1. Data as received: row counts, ink scales, format

| file | dataset | declared `NUMBER_OF_SETS` | actual data rows | ink channels | ink scale found | colorimetry given |
|---|---|---:|---:|---|---|---|
| APTEC_PC10_CardBoard_2023_v1.txt | PC10 | 1617 | 1617 | CMYK (4) | 0–100 | Lab (D50/2°) only |
| APTEC_PC11_CCNB_2023_v1.txt | PC11 | 1617 | 1617 | CMYK (4) | 0–100 | Lab (D50/2°) only |
| FOGRA51.txt | FOGRA51 | 1617 | 1617 | CMYK (4) | 0–100 | Lab (D50/2°) only |
| KCMYG_5clr_spectral.txt | KCMYG5 | 2214 | 2214 | 5CLR (5) | 0–100 | spectral only, 380–730nm/10nm (36 bands) |
| APTEC_CMYKOGV_7clr_xyzlab.txt | CMYKOGV7 | **1624** | **3534** | 7CLR (7) | 0–100 | native XYZ **and** native Lab |
| Apex_CMYKOGB_7clr_spectral.txt | CMYKOGB7 | 2000 | 2000 | 7CLR (7) | 0–100 | spectral only, 380–730nm/10nm |
| Age_64a_wb.txt | IFRA_Age64a | 1485 | 1485 | CMYK (4) | 0–100 | spectral only, 380–730nm/10nm |
| PressJ_158_wb.txt | IFRA_PressJ158 | 1485 | 1485 | CMYK (4) | 0–100 | spectral only, 380–730nm/10nm |

All eight files use a 0–100 ink percentage scale throughout (min 0.0, max 100.0 observed in every
file) — no rescaling from 0–1 was needed anywhere.

Where only Lab was given (PC10/PC11/FOGRA51), I derived XYZ via `colour.Lab_to_XYZ` under the CIE
1931 2° D50 white point (`96.42, 100.00, 82.51`), matching the header's own stated measurement
condition (`D50, 2 degree`). Where only spectral reflectance was given (KCMYG5, CMYKOGB7, both IFRA
runs), I integrated to XYZ under the same D50/2° combination using `colour.msds_to_XYZ` (ASTM E308
method), verified to be bit-identical to per-sample `colour.sd_to_XYZ` on the same data. For
CMYKOGV7, native XYZ was used directly as the regression target (see §3 for why, and for the one
header inconsistency this file has).

### CMYKOGV7 row-count anomaly

The header declares `NUMBER_OF_SETS 1624`, but there are 3534 contiguous data rows between a single
`BEGIN_DATA`/`END_DATA` pair (no second data block, no ragged rows). `SAMPLE_ID` runs 1..3534
without a break at 1624. This looks like a stale header field left over from an earlier version of
the file that was later extended with ~1910 more measured patches (verification wedges / additional
combinations) without updating `NUMBER_OF_SETS`. See §4/§6 for why this matters: the authors'
published `n=3302` for this dataset is consistent with deduplicating the file's exact-duplicate rows
(3534 − 3302 = 232, which is exactly the number of "extra" copies contributed by its 169
duplicate-recipe groups: 401 − 169 = 232).

## 2. Duplicate-row analysis

For each file I grouped rows by ink recipe (rounded to 1e-6) and checked whether rows sharing a
recipe also share identical measured values (Lab, XYZ+Lab, or the raw spectral vector, whichever the
file natively provides — I did not use my own derived XYZ for this comparison, to avoid laundering
any of my own conversion error into the "duplicate" judgement).

| dataset | rows | unique recipes | duplicate groups | rows in dup groups | groups: identical measurement | groups: differing measurement | max spread within a differing group |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC10 | 1617 | 1588 | 29 | 58 | 29 | 0 | — |
| PC11 | 1617 | 1588 | 29 | 58 | 29 | 0 | — |
| FOGRA51 | 1617 | 1588 | 29 | 58 | 29 | 0 | — |
| KCMYG5 | 2214 | 2214 | 0 | 0 | — | — | — |
| CMYKOGV7 | 3534 | 3302 | 169 | 401 | 169 | 0 | — |
| CMYKOGB7 | 2000 | 1981 | 1 | 20 | 0 | 1 | 0.031 (reflectance units) |
| IFRA_Age64a | 1485 | 1457 | 28 | 56 | 0 | 28 | 0.033 (reflectance units) |
| IFRA_PressJ158 | 1485 | 1457 | 28 | 56 | 0 | 28 | 0.073 (reflectance units) |

Two qualitatively different kinds of duplication show up, and they matter differently for CV design:

- **PC10/PC11/FOGRA51** (identical 29-group/58-row structure across all three, as expected — they're
  the same ISO 12642-2 target measured on three substrates) and **CMYKOGV7** (169 groups/401 rows):
  every duplicate group has **bit-identical** measured values. These are literal repeated chart
  patches, not independent remeasurements — no new information, pure redundancy.
- **CMYKOGB7**'s one duplicate group is the all-zero (paper white) patch, remeasured 20 times
  (`SampleID` 1 plus 1982–2000, named `2N32`..`2N50`) — a drift/QC check, with real (if tiny)
  measurement spread (max 0.031 reflectance units across bands).
- **Both IFRA runs** have 28 duplicate-recipe groups (identical 56 rows, same recipes, same order in
  both files — see §5) with **differing** measurements — genuine repeated physical measurements of
  the same nominal CMYK recipe at different sheet/press positions, with real press-condition noise
  (spread up to 0.033 / 0.073 reflectance units, i.e. actual newsprint variability, not duplication
  artifacts).
- **No `SAMPLE_ID`/`SampleID` collisions** in any file — all IDs are unique within their file.

Bonus finding (not required by spec but worth flagging): `Age_64a_wb.txt` and `PressJ_158_wb.txt`
have **exactly the same 1485 `SAMPLE_ID`s in exactly the same row order** with **exactly the same
CMYK recipes** — i.e. the two files are the same target chart printed on two different press runs,
row-aligned. This makes a paired within-vs-across-run comparison possible even though the spec only
asks for within-run CV here.

## 3. XYZ↔Lab and spectral↔native consistency

**Spectral-vs-native check**: the spec asks whether integrated-from-spectral XYZ agrees with a
file's own native XYZ/Lab, for files that provide both. **No file in this set provides both spectral
and native colorimetry** — PC10/PC11/FOGRA51 have Lab only, KCMYG5/CMYKOGB7/both IFRA runs have
spectral only, and CMYKOGV7 has native XYZ+Lab but no spectral. So this check is vacuously satisfied
for every file; I looked, and there is nothing to cross-check on this axis.

**XYZ↔Lab self-consistency**: only CMYKOGV7 has both native XYZ and native Lab in the same row,
letting me check whether the two fields (presumably from the same underlying instrument read) agree
under D50/2°. Converting native XYZ → Lab and comparing to the file's own Lab column:

| stat | value |
|---|---|
| median ΔE00 | 0.005 |
| p95 ΔE00 | 0.020 |
| max ΔE00 | 0.027 |
| mean \|ΔL\| | 1.7×10⁻⁵ |
| mean \|Δa\| | 0.008 |
| mean \|Δb\| | 0.007 |

This is excellent agreement (max 0.027 ΔE00 over 3534 rows) — well within float rounding of the
XYZ/Lab values as printed (XYZ given to 6 sig figs, Lab to 3 decimals), confirming both columns come
from one consistent D50/2° computation and I've reconstructed their white point convention correctly.
I used the native XYZ column directly as the regression target for CMYKOGV7 (rather than
re-deriving from Lab), since it's the more direct measurement and this check shows nothing is lost by
doing so.

## 4. Header/metadata oddities

- **`APTEC_PC10_CardBoard_2023_v1.txt` declares `DESCRIPTOR APTEC_PC11_CCNB_2023_v1`** — i.e. PC10's
  own header identifies it as PC11. This is very likely a copy-paste artifact (PC10 and PC11 share
  the same ISO 12642-2 target design and near-identical header boilerplate — only the `CREATED` date
  format, `PRINT_CONDITIONS` text, and the data itself actually differ between the two files). The
  filename and data content are correct for PC10 (data disagrees with PC11's own file — verified the
  first data row differs: PC10 row 1 Lab is `48.02, 72.76, 7.22` vs PC11 row 1 `44.13, 69.53, 6.71` —
  so this is a stray metadata line, not a swapped/duplicated data file).
- **CMYKOGV7's `NUMBER_OF_SETS` undercounts actual rows by more than 2×** (1624 vs 3534) — see §1.
- Both IFRA files carry a non-ASCII byte (`0xB0`, a Latin-1/Windows-1252 degree sign) in
  `MEASUREMENT_SOURCE` (`ObserverAngle=2°`) and CRLF line endings — not an error, just confirms the
  files are Windows/Latin-1 authored, not UTF-8; I read them as Latin-1 to avoid mojibake.
- KCMYG5 / CMYKOGB7 headers are X-Rite ProfileMaker/i1iO exports (`LGOMCCHANNEL*`, `Eye-One iO`) with
  embedded per-ink spectral swatch definitions and a `MinMetamerismLight01/02` block — informational,
  internally consistent, `ILLUMINATION_NAME`/`OBSERVER_ANGLE` correctly say D50/2° matching what I
  integrated under.
- Everything else (`NUMBER_OF_FIELDS` vs actual column count, `NUMBER_OF_SETS` vs row count for the
  other 7 files) is self-consistent.

## 5. My results (written to `blind_results.csv` before any comparison)

Protocol exactly as specified: MinMax-scale inputs and targets on the training fold only, grouped
5-fold CV (seed 42, identical-recipe rows forced into the same fold — see §6 for why this matters),
inverse-transform before scoring, clip negative XYZ, ΔE00 under D50/2°, pooled over all folds.

| dataset | variant | model | median | p95 | max | n |
|---|---|---|---:|---:|---:|---:|
| PC10 | CMY | poly3 | 0.280 | 1.074 | 7.316 | 818 |
| PC10 | CMY | svm | 0.767 | 2.046 | 6.172 | 818 |
| PC10 | CMY | knn | 1.948 | 4.245 | 10.344 | 818 |
| PC10 | CMY | gaussian_process | 0.047 | 0.157 | 2.093 | 818 |
| PC10 | CMYK | poly3 | 0.943 | 8.193 | 29.350 | 1617 |
| PC10 | CMYK | svm | 0.975 | 5.493 | 23.100 | 1617 |
| PC10 | CMYK | knn | 2.252 | 5.823 | 13.871 | 1617 |
| PC10 | CMYK | gaussian_process | 0.057 | 0.471 | 12.724 | 1617 |
| PC11 | CMY | poly3 | 0.245 | 0.827 | 7.543 | 818 |
| PC11 | CMY | svm | 0.723 | 1.803 | 4.987 | 818 |
| PC11 | CMY | knn | 1.881 | 4.061 | 10.255 | 818 |
| PC11 | CMY | gaussian_process | 0.044 | 0.140 | 1.755 | 818 |
| FOGRA51 | CMY | poly3 | 0.369 | 1.097 | 8.408 | 818 |
| FOGRA51 | CMY | svm | 0.789 | 1.890 | 5.016 | 818 |
| FOGRA51 | CMY | knn | 1.968 | 4.289 | 9.905 | 818 |
| FOGRA51 | CMY | gaussian_process | 0.058 | 0.161 | 0.485 | 818 |
| KCMYG5 | native (5 in) | poly3 | 1.462 | 8.437 | 53.986 | 2214 |
| KCMYG5 | native | svm | 1.199 | 6.694 | 45.048 | 2214 |
| KCMYG5 | native | knn | 1.814 | 5.195 | 41.049 | 2214 |
| KCMYG5 | native | gaussian_process | 0.851 | 2.326 | 32.424 | 2214 |
| CMYKOGV7 | native (7 in, all rows) | poly3 | 5.625 | 25.834 | 56.342 | 3534 |
| CMYKOGV7 | native | svm | 2.008 | 11.411 | 27.246 | 3534 |
| CMYKOGV7 | native | knn | 3.451 | 9.670 | 26.854 | 3534 |
| CMYKOGV7 | native | gaussian_process | 0.226 | 4.855 | 45.215 | 3534 |
| CMYKOGB7 | native (7 in) | poly3 | 3.326 | 24.370 | 52.838 | 2000 |
| CMYKOGB7 | native | svm | 1.816 | 14.093 | 41.630 | 2000 |
| CMYKOGB7 | native | knn | 2.598 | 8.243 | 41.446 | 2000 |
| CMYKOGB7 | native | gaussian_process | 1.257 | 8.657 | 47.261 | 2000 |
| IFRA_Age64a | CMYK (within-run) | poly3 | 1.075 | 3.313 | 23.533 | 1485 |
| IFRA_Age64a | CMYK | svm | 0.965 | 3.234 | 16.258 | 1485 |
| IFRA_Age64a | CMYK | knn | 1.886 | 4.194 | 6.975 | 1485 |
| IFRA_Age64a | CMYK | gaussian_process | 0.752 | 1.695 | 5.687 | 1485 |
| IFRA_PressJ158 | CMYK (within-run) | poly3 | 3.170 | 10.395 | 33.863 | 1485 |
| IFRA_PressJ158 | CMYK | svm | 2.280 | 5.738 | 11.219 | 1485 |
| IFRA_PressJ158 | CMYK | knn | 3.214 | 6.801 | 12.592 | 1485 |
| IFRA_PressJ158 | CMYK | gaussian_process | 2.194 | 5.425 | 9.982 | 1485 |

Coverage vs. SPEC.md item 6: all requested dataset/variant combinations are present (PC10 CMY+CMYK,
PC11 CMY, FOGRA51 CMY, KCMYG-5, CMYKOGV-7, CMYKOGB-7, both IFRA within-run). I did not compute
PC11-CMYK or FOGRA51-CMYK (the spec's dataset list only asks for CMY on those two; I noticed the
authors' results folder *does* contain those extra variants, but reproducing variants the spec didn't
ask me to cover isn't part of this review, so I left them out of `blind_results.csv`). I did not
extend IFRA to the other 11 press runs present in the authors' results tree (`Arena_111`, `GratN_90`,
etc.) — the raw data for those was not supplied to me in `~/blind_verify/data`, and the spec names
only the two runs (`Age_64a_wb.txt`, `PressJ_158_wb.txt`) I was given data for.

**Headline finding, general shape**: `gaussian_process` is the strongest model everywhere by a wide
margin on median ΔE00, `poly3` is competitive on the CMY (n=3) sets but degrades badly on the 7-ink
sets (median 3.3–5.6, worse than svm/knn there), and `knn`/`svm` sit in between. For CMY n≤4 targets
(PC10/PC11/FOGRA51) GP reaches median ΔE00 ≈ 0.04–0.06 — essentially at measurement-noise level. For
the 7-ink sets even GP's median stays low (0.23–1.26) but p95/max blow out to 8–47, i.e. a handful of
gamut corners are hard to predict for every model tried, including GP.

## 6. Comparison against the authors' published `summary.csv` / `within_run.csv`

I read these files only after `blind_results.csv` was fully written (all 36 rows, cheap models then
GP). Comparison table (mine vs theirs; Δmedian = mine − theirs):

| dataset/variant | model | mine (med/p95/max) | theirs (med/p95/max) | Δmedian | n mine/theirs |
|---|---|---|---|---:|---|
| PC10 CMY | poly3 | 0.280/1.074/7.316 | 0.279/0.966/7.695 | +0.001 | 818/818 |
| PC10 CMY | svm | 0.767/2.046/6.172 | 0.750/1.928/5.442 | +0.017 | 818/818 |
| PC10 CMY | knn | 1.948/4.245/10.344 | 2.006/4.251/8.918 | −0.058 | 818/818 |
| PC10 CMY | gp | 0.047/0.157/2.093 | 0.044/0.162/2.140 | +0.003 | 818/818 |
| PC10 CMYK | poly3 | 0.943/8.193/29.350 | 0.944/8.154/32.059 | −0.001 | 1617/1617 |
| PC10 CMYK | svm | 0.975/5.493/23.100 | 1.033/5.661/20.253 | −0.058 | 1617/1617 |
| PC10 CMYK | knn | 2.252/5.823/13.871 | 2.222/5.799/13.515 | +0.030 | 1617/1617 |
| PC10 CMYK | gp | 0.057/0.471/12.724 | 0.056/0.454/7.099 | +0.001 | 1617/1617 |
| PC11 CMY | poly3 | 0.245/0.827/7.543 | 0.244/0.890/7.107 | +0.001 | 818/818 |
| PC11 CMY | svm | 0.723/1.803/4.987 | 0.714/1.830/4.863 | +0.009 | 818/818 |
| PC11 CMY | knn | 1.881/4.061/10.255 | 1.937/4.127/8.852 | −0.056 | 818/818 |
| PC11 CMY | gp | 0.044/0.140/1.755 | 0.044/0.140/1.773 | 0.000 | 818/818 |
| FOGRA51 CMY | poly3 | 0.369/1.097/8.408 | 0.368/1.163/8.008 | +0.001 | 818/818 |
| FOGRA51 CMY | svm | 0.789/1.890/5.016 | 0.767/1.864/6.194 | +0.022 | 818/818 |
| FOGRA51 CMY | knn | 1.968/4.289/9.905 | 1.884/4.299/9.512 | +0.084 | 818/818 |
| FOGRA51 CMY | gp | 0.058/0.161/0.485 | 0.056/0.176/0.482 | +0.002 | 818/818 |
| KCMYG5 | poly3 | 1.462/8.437/53.986 | 1.480/7.975/50.158 | −0.018 | 2214/2214 |
| KCMYG5 | svm | 1.199/6.694/45.048 | 1.192/6.350/50.622 | +0.007 | 2214/2214 |
| KCMYG5 | knn | 1.814/5.195/41.049 | 1.814/4.877/55.180 | 0.000 | 2214/2214 |
| KCMYG5 | gp | 0.851/2.326/32.424 | 0.851/2.274/42.462 | 0.000 | 2214/2214 |
| CMYKOGV7 | poly3 | 5.625/25.834/56.342 | 5.310/25.118/54.355 | +0.315 | 3534/3302 |
| CMYKOGV7 | svm | 2.008/11.411/27.246 | 2.186/11.166/32.512 | −0.178 | 3534/3302 |
| CMYKOGV7 | knn | 3.451/9.670/26.854 | 3.578/9.520/29.401 | −0.127 | 3534/3302 |
| CMYKOGV7 | gp | 0.226/4.855/45.215 | 0.236/4.324/52.179 | −0.010 | 3534/3302 |
| CMYKOGB7 | poly3 | 3.326/24.370/52.838 | 3.276/24.876/52.696 | +0.050 | 2000/2000 |
| CMYKOGB7 | svm | 1.816/14.093/41.630 | 1.792/13.653/41.002 | +0.024 | 2000/2000 |
| CMYKOGB7 | knn | 2.598/8.243/41.446 | 2.610/8.104/39.455 | −0.012 | 2000/2000 |
| CMYKOGB7 | gp | 1.257/8.657/47.261 | 1.280/8.036/47.473 | −0.023 | 2000/2000 |
| IFRA_Age64a | poly3 | 1.075/3.313/23.533 | 1.051/3.191/26.268 | +0.024 | 1485/1485 |
| IFRA_Age64a | svm | 0.965/3.234/16.258 | 0.959/3.208/22.988 | +0.006 | 1485/1485 |
| IFRA_Age64a | knn | 1.886/4.194/6.975 | 1.824/4.211/8.365 | +0.062 | 1485/1485 |
| IFRA_Age64a | gp | 0.752/1.695/5.687 | 0.755/1.656/6.484 | −0.003 | 1485/1485 |
| IFRA_PressJ158 | poly3 | 3.170/10.395/33.863 | 3.152/9.998/36.131 | +0.018 | 1485/1485 |
| IFRA_PressJ158 | svm | 2.280/5.738/11.219 | 2.294/5.971/15.067 | −0.014 | 1485/1485 |
| IFRA_PressJ158 | knn | 3.214/6.801/12.592 | 3.175/6.740/14.165 | +0.039 | 1485/1485 |
| IFRA_PressJ158 | gp | 2.194/5.425/9.982 | 2.141/5.195/10.397 | +0.053 | 1485/1485 |

**Overall shape of agreement**: `gaussian_process` and `poly3` agree with the authors to within
0.001–0.02 ΔE00 median on every single dataset (several exact to 3 decimals: PC11-CMY GP, KCMYG5 GP,
KCMYG5 knn incidentally too). `svm`/`knn` show the largest, most systematic gaps (0.01–0.09 median) on
exactly the four-or-fewer-input CMYK datasets (PC10/PC11/FOGRA51). p95/max disagree more than medians
everywhere, which is expected — tail statistics are dominated by one or two hardest points and are
much more sensitive to exactly which points land in which fold.

### (a) CMYKOGV7: n=3534 (mine) vs n=3302 (theirs) — a deduplication choice, not an error

§2 already showed CMYKOGV7 has 169 duplicate-recipe groups (401 rows) that are all **bit-identical**
repeats. I initially used every raw row (n=3534, spec's literal "every row is held out exactly
once, pooled over the whole dataset"). The authors' n=3302 is *exactly* 3534 − 232, and 401 − 169 =
232 — i.e. their number is exactly what you get by collapsing each of my 169 duplicate groups down to
one representative row and dropping the other 232 copies. I confirmed this directly: deduplicating
CMYKOGV7 to one row per unique recipe gives **n=3302 exactly**, and rerunning my identical pipeline
on that deduplicated set gives:

| model | all rows (n=3534) | deduplicated (n=3302, my rerun) | authors (n=3302) |
|---|---|---|---|
| poly3 | 5.625/25.834/56.342 | 5.419/25.144/58.402 | 5.310/25.118/54.355 |
| knn | 3.451/9.670/26.854 | 3.626/9.796/26.854 | 3.578/9.520/29.401 |
| svm | 2.008/11.411/27.246 | 2.204/12.003/26.225 | 2.186/11.166/32.512 |
| gaussian_process | 0.226/4.855/45.215 | 0.246/4.630/55.480 | 0.236/4.324/52.179 |

Matching their row count closes most of the median gap: poly3's gap shrinks from 0.315 to 0.109 (65%
explained by dedup alone), svm's from 0.178 to 0.018 (90% explained), knn's from 0.127 to 0.048 (flips
sign, shrinks to 62% of original size). GP's median gap is already tiny either way (0.010 with all
rows, 0.010 deduplicated — unchanged) and its p95/max gaps shrink somewhat (p95: 0.531→0.306, max:
6.964→3.301) but stay sizeable — consistent with GP's known fold-to-fold restart sensitivity on this
dataset (see the tail-statistic point in part (b)) rather than anything the dedup choice explains. The
small residual gaps that remain for poly3/knn/svm after matching n are the same order of magnitude as
the fold-implementation noise seen everywhere else in this table (see part (b)) — i.e. once you match
on "how many rows go into the statistics," CMYKOGV7's median behaves like every other dataset.

**My verdict: their choice (deduplicate exact repeats before pooling CV statistics) is the more
defensible one, and I'd change my own pipeline to match if this were a real project.** The 169
repeated patches are not independent measurements — they are the literal same design point
represented 2–5× in the file (up to and including identical XYZ/Lab to the decimal), almost certainly
verification-wedge patches re-included when the target was extended from 1624 to 3534 rows (see the
`NUMBER_OF_SETS` mismatch in §1/§4). Counting them multiple times in the pooled median/p95/max
statistics silently overweights whatever those particular 169 patches happen to be (plausibly the
easier, more central patches typically used for repeatability checks — cf. the CMYKOGB7 white-patch
QC repeat in §2) relative to the 3302 patches that only appear once. My grouped-fold CV already
prevents the *leakage* version of this problem (a duplicate can never straddle train/test), but
grouping alone doesn't fix the double-counting in the final pooled statistic — that requires
deduplicating identical-measurement rows before the fold split, which the authors evidently did and I
initially did not. This is a real, if second-order (≤0.3 ΔE00 on the worst-affected dataset), finding
about how "pool over the whole dataset" should be interpreted when a dataset itself contains verbatim
repeats; it's specific to CMYKOGV7 because that's the only dataset where repeats are both bit-identical
*and* a non-trivial fraction of the rows (11%). I'd keep the distinction from §2 though: PC10/PC11/
FOGRA51's 29 identical-measurement duplicate groups are a much smaller fraction (3.6% of rows) so this
effect is present but minor there, and IFRA's duplicate groups have genuinely differing measurements
so should *not* be deduplicated (they're real repeated-measurement noise, not redundant rows).

### (b) svm/knn gaps of 0.01–0.09 on the n≤4 sets vs poly3/gp agreeing to ~0.001–0.01 — mechanism

This is not noise or a hyperparameter mismatch — I traced it to fold construction. SPEC.md is explicit:
*"Rows sharing an identical ink recipe must fall in the same fold (no leakage through repeated
measurements)."* I implemented this (group rows by rounded ink recipe, shuffle unique groups with
`RandomState(42)`, split into 5 near-equal chunks). PC10/PC11/FOGRA51 each have 29 duplicate-recipe
groups (58 rows, 3.6% of 1617, or 818 for the CMY subset) whose two copies carry **bit-identical**
measured Lab — the ISO 12642-2 target's built-in repeatability patches.

I tested the alternative explicitly: plain `sklearn.model_selection.KFold(n_splits=5, shuffle=True,
random_state=42)` with **no** recipe grouping at all, on the same data, same scalers, same model
hyperparameters. Result (ungrouped / plain KFold, my code, my side):

| dataset/variant | model | ungrouped (mine) | authors (published) | grouped (mine, in blind_results.csv) |
|---|---|---|---|---|
| PC10 CMY | poly3 | 0.279/0.966/7.694 | 0.279/0.966/7.695 | 0.280/1.074/7.316 |
| PC10 CMY | knn | 2.007/4.251/8.921 | 2.006/4.251/8.918 | 1.948/4.245/10.344 |
| PC10 CMY | svm | 0.749/1.926/5.394 | 0.750/1.928/5.442 | 0.767/2.046/6.172 |
| PC10 CMYK | poly3 | 0.944/8.153/32.060 | 0.944/8.154/32.059 | 0.943/8.193/29.350 |
| PC10 CMYK | knn | 2.222/5.799/13.515 | 2.222/5.799/13.515 | 2.252/5.823/13.871 |
| PC10 CMYK | svm | 1.030/5.624/19.845 | 1.033/5.661/20.253 | 0.975/5.493/23.100 |
| PC11 CMY | poly3 | 0.244/0.890/7.106 | 0.244/0.890/7.107 | 0.245/0.827/7.543 |
| PC11 CMY | knn | 1.937/4.126/8.850 | 1.937/4.127/8.852 | 1.881/4.061/10.255 |
| PC11 CMY | svm | 0.712/1.846/4.899 | 0.714/1.830/4.863 | 0.723/1.803/4.987 |
| FOGRA51 CMY | poly3 | 0.368/1.163/8.007 | 0.368/1.163/8.008 | 0.369/1.097/8.408 |
| FOGRA51 CMY | knn | 1.884/4.298/9.512 | 1.884/4.299/9.512 | 1.968/4.289/9.905 |
| FOGRA51 CMY | svm | 0.769/1.879/6.190 | 0.767/1.864/6.194 | 0.789/1.890/5.016 |

The **ungrouped** run matches the authors' published numbers to 2–3 decimals on essentially every
cell (several exact to 3 decimals: PC10-CMYK poly3, PC10-CMYK knn, PC11-CMY poly3/knn, FOGRA51-CMY
poly3/knn), far tighter than my grouped run does. That is about as strong a fingerprint as this kind
of black-box comparison can produce: **the authors' pipeline, for at least these three CMYK/CMY
datasets, is using a plain `KFold(shuffle=True, random_state=42)` without grouping identical-recipe
rows into the same fold** — i.e. it does not implement the "no leakage through repeated measurements"
requirement SPEC.md states.

Mechanism for *why* this shows up in knn/svm but barely touches poly3/gp: with ungrouped folds, some
of the 29 duplicate-recipe pairs get split so that one copy is in training and its bit-identical twin
is the test point. For `knn` (k=5, uniform weights) that training point is then one of the nearest
neighbours of its own test twin — for a truly identical or near-identical recipe the leaked neighbour
sits at distance ≈0 in the (scaled) input space and pulls the k=5 average of the prediction almost
exactly onto the true target, i.e. a near-zero-error freebie for that specific test row. `svm`'s RBF
kernel produces the analogous effect through the kernel weight on that one very-close support vector.
`poly3` (a single global cubic OLS fit) and `gp` (also fit to minimize a global objective, and in this
case with a data-noise term via `WhiteKernel`) are much less perturbed by whether one specific
duplicate is in-fold or out-of-fold, because neither model can locally "snap" onto a single nearby
point the way a k=5 average or an RBF kernel value can — their bias/variance is dominated by the
global functional form, not by local neighbours. This also explains why the effect is largest exactly
on the datasets with **bit-identical** duplicate measurements (PC10/PC11/FOGRA51, and to a lesser
extent it would apply to CMYKOGV7 if that pipeline path also skips grouping) and much smaller on IFRA,
whose 28 duplicate groups have *real* measurement noise (spread up to 0.033–0.073 reflectance units)
so a leaked "duplicate" isn't a perfect free answer, just a very good prior — consistent with IFRA's
svm/knn gaps (0.006–0.062) being noticeably smaller than PC10/PC11/FOGRA51's when the affected
fraction of rows is similar in both.

**My verdict**: my grouped-fold numbers are the ones that comply with SPEC.md as written, and I stand
behind them as the answer to "what does this protocol, followed correctly, produce." But the evidence
strongly indicates the reference pipeline being verified does not group folds by ink recipe for these
datasets, which is a genuine (if quantitatively modest — 0.01 to 0.09 ΔE00 in the pooled median)
methodological gap relative to their own stated protocol, not a difference in model, scaler, or metric
implementation. It's the kind of bug that's easy to miss because its effect is small in the median and
invisible in aggregate accuracy claims, but it is real optimistic bias, and it is systematic (it always
pushes error down at the leaked points, never up), which is exactly the kind of thing an independent
verification pass is for. I'd flag this as the one actionable finding from this whole review that the
authors should fix and rerun: add the grouped-fold logic to whatever CV code produces `summary.csv`
for datasets that contain any duplicate-recipe rows (PC10, PC11, FOGRA51, CMYKOGV7 at minimum — KCMYG5
and CMYKOGB7's own duplicate counts are 0 and 1 respectively so they're effectively unaffected either
way).

## 7. Anything else in the data worth a second look

- **CMYKOGV7's stale `NUMBER_OF_SETS`** (§1/§4) is worth fixing at the source regardless of the CV
  question — anyone loading this file by trusting its own declared row count (rather than reading
  until `END_DATA`) would silently drop 68% of the data.
- **The 169 duplicate patches in CMYKOGV7** are worth identifying explicitly (which recipes, e.g. via
  the `PLUS_1/2/3_COLOR` primaries or specific verification-wedge corners) — if they cluster in easy
  regions of the gamut (as the CMYKOGB7 white-patch repeat suggests is the house habit for this data
  provider), the double-counting bias in §6(a) is directionally predictable, not just a magnitude
  correction.
- **Both IFRA runs are the same 1485-recipe target, row-aligned** (§2) — the current protocol only
  asks for within-run CV, but the data as supplied directly supports a paired cross-run generalization
  test (train on one press run's ink→colour mapping, predict the other) with zero extra data wrangling
  needed. That's a natural, currently-unexploited use of exactly the two files provided.
- **7-ink poly3 instability**: `poly3`'s p95/max on CMYKOGV7 and CMYKOGB7 (25.8/56.3 and 24.4/52.8)
  are markedly worse than `svm`/`knn`/`gp` on the same data, and worse than poly3 itself on every
  4-or-fewer-input dataset. A 3rd-order polynomial in 7 inputs has 120 basis terms (`C(7+3,3)`) fit on
  ~1600–2800 training rows per fold — well-posed in principle, but the gamut coverage for a 7-ink
  target is much sparser per unit hypervolume than for CMY(K), so cubic extrapolation into
  under-sampled corners of a 7-D ink space is a plausible, physically-motivated explanation for these
  outliers rather than a bug; I'd want to see which specific patches drive the max before concluding
  more, but the pattern is consistent and reproducible on both independent 7-ink datasets.

## Files produced

- `~/blind_verify/blind_results.csv` — 36 rows (9 dataset/variant combinations × {poly3, svm, knn,
  gaussian_process}), written and finalized before any comparison.
- `~/blind_verify/BLIND_REPORT.md` — this file.
- `~/blind_verify/work/` — all analysis code (`parse.py`, `colorimetry.py`, `datasets.py`,
  `variants.py`, `cv.py`, `checks.py`, `run_cheap.py`, `run_gp.py`) and supporting logs/intermediate
  outputs (`checks_report.json`, `run_cheap.log`, `run_gp.log`, `gp_preds/`, the ungrouped-CV and
  CMYKOGV7-dedup verification scripts used in §6).
