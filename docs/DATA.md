# Datasets

All sets are printer characterization data: ink values and the measured color (XYZ/Lab),
evaluated with CIEDE2000 on denormalized XYZ, D50/2°. Results are reported per dataset (the
sets come from different sources and printing conditions, not one controlled sweep).

The data are third-party and are not redistributed in this repository. How to obtain and
convert the public sets is in `journal/REPRODUCE.md`, section 2.

| Dataset | n inks | Channels | Patches | Color data | Loader path | Used for |
|---|---|---|---|---|---|---|
| APTEC PC10 (public) | 3 / 4 | CMY(K) | 1617 raw → **1588 effective**; 818 K=0 → **795 effective** (exact-dedup at load) | Lab → XYZ | `data/cleaned/APTEC_PC10_CardBoard_2023_v1.csv` | n=3 (CMY), n=4 (CMYK) |
| APTEC PC11 (public) | 3 / 4 | CMY(K) | 1617 → **1588**; 818 K=0 → **795** | Lab → XYZ | `data/cleaned/APTEC_PC11_CCNB_2023_v1.csv` | n=3, n=4 |
| FOGRA51 (public) | 3 / 4 | CMY(K) | 1617 → **1588**; 818 K=0 → **795** | Lab → XYZ | `data/cleaned/FOGRA51.csv` | n=3, n=4 |
| Newsprint, 2005 (restricted) | 4 | CMYK + spectral | 13 runs × 1485 | spectral → XYZ | `journal/data/processed/ifra/wb/*.csv` (from `journal.pipeline.ingest_ifra`) | multi-press generalization |
| KCMYG (restricted) | 5 | C M Y K G + spectral | 2214 | spectral → XYZ | `journal/data/processed/ncolor/KCMYG-5.csv` (from `journal.pipeline.ingest_ncolor`) | n=5 |
| APTEC CMYKOGV (public) | 7 | C M Y K O G V | 3534 raw → **3302 effective** (exact-dedup at load; the file header's 1624 is stale) | XYZ + Lab | `journal/data/processed/ncolor/CMYKOGV-7.csv` (from `journal.pipeline.ingest_ncolor`) | n=7 |
| CMYKOGB (restricted) | 7 | C M Y K O G B + spectral | 2000 | spectral → XYZ | `journal/data/processed/ncolor/CMYKOGB-7.csv` (from `journal.pipeline.ingest_ncolor`) | n=7 |

Ink-count ladder: 3 → 4 → 5 → 7 (two independent 7-ink systems, OGV and OGB).

For the restricted sets, the raw file names that `ingest_ifra.py` and `ingest_ncolor.py` expect
are in those scripts. Their results files hold row positions, folds and ΔE00 only, no measurements.

**Out of scope:** the black-backing measurements of the newsprint data. They pose a separate
substrate-correction problem, and their files lack ink recipes and use a different chart layout.
Only the white-backing runs are used.

Notes: CMY experiments use only K=0 rows (818 raw); CMYK uses all 1617 with K as an input (an input
column is never dropped while its rows are kept). Spectral sets convert to XYZ via
`journal.pipeline.ifra.spectral_to_xyz`. The coated sets' XYZ is computed from the measured Lab
with the ICC D50 white (X, Y, Z = 0.96422, 1, 0.82521) by `journal.pipeline.ingest_coated`.

**Deduplication policy.** Byte-identical duplicate rows (same inks *and* same XYZ+Lab to full float
precision) are dropped at load on every **coated** set (PC10, PC11, FOGRA51 and APTEC CMYKOGV).
They are an upstream averaging/duplication artifact, not repeated measurement; left in, they leak
across CV folds and are double-counted in the pooled median
(`docs/research/cv-leakage-2026-08-12.md`). Effective counts: 818→795 (CMY), 1617→1588 (CMYK),
3534→3302 (CMYKOGV-7).

**The newsprint data are deliberately exempt**: their duplicate recipes carry genuinely *differing*
measurements (press repeatability, about 0.6 to 0.8 ΔE00), which is signal about the press rather
than an artifact. All 13 runs keep their full 1485 rows. KCMYG-5 and CMYKOGB-7 contain no
byte-identical duplicates, so the policy is a no-op for them; CMYKOGB-7's single duplicate-recipe
group is its 20 paper-white repeats, which differ in value. The n>4 sets also use grouped CV, so any
repeated recipe stays within one fold.
