# Datasets — quick reference

All printer characterization data: ink values → measured colour (XYZ/Lab), evaluated with CIEDE2000.
Report per-dataset (mixed sources/conditions — not one controlled sweep). ΔE00 on denormalized XYZ, D50/2°.

| Dataset | n inks | Channels | Patches | Colour data | Location | Used for |
|---|---|---|---|---|---|---|
| APTEC PC10 | 3 / 4 | CMY(K) | 1617 raw → **1588 effective**; 818 K=0 → **795 effective** (exact-dedup at load) | XYZ + Lab | `data/cleaned/APTEC_PC10_*.csv` | n=3 (CMY), n=4 (CMYK) |
| APTEC PC11 | 3 / 4 | CMY(K) | 1617 → **1588**; 818 K=0 → **795** | XYZ + Lab | `data/cleaned/APTEC_PC11_*.csv` | n=3, n=4 |
| FOGRA51 | 3 / 4 | CMY(K) | 1617 → **1588**; 818 K=0 → **795** | XYZ + Lab | `data/cleaned/FOGRA51.csv` | n=3, n=4 |
| IFRA newsprint (wb) | 4 | CMYK + spectral | 13 runs × 1485 | spectral→XYZ | `journal/data/raw/Ifra-wb.zip` | multi-press generalization |
| KCMYG | 5 | C M Y K G + spectral | 2214 | spectral→XYZ | `journal/data/raw/ncolor/KCMYG_5clr_spectral.txt` | n=5 (ladder) |
| APTEC CMYKOGV | 7 | C M Y K O G V | 3534 raw (3302 effective after exact-dedup at load; header's 1624 is stale) | XYZ + Lab | `journal/data/raw/ncolor/APTEC_CMYKOGV_7clr_xyzlab.txt` | n=7 (ladder) |
| Apex CMYKOGB | 7 | C M Y K O G B + spectral | 2000 | spectral→XYZ | `journal/data/raw/ncolor/Apex_CMYKOGB_7clr_spectral.txt` | n=7 (ladder) |

**Ink-count ladder for the headline:** 3 → 4 → 5 → 7 (two independent 7-ink systems: OGV vs OGB).

**Excluded (out of scope, per Phil):** IFRA **black-backing** newsprint — a separate "substrate correction"
problem, and its files lacked ink recipes / used a different chart layout. Raw zip moved to
`../_archive_out_of_scope/Ifra-bb.zip` (outside the repos).

Notes: CMY experiments use only K=0 rows (818 raw); CMYK uses all 1617 with K as an input (never drop a
column while keeping its rows). Spectral sets convert to XYZ via `journal.pipeline.ifra.spectral_to_xyz`.

**Deduplication policy (uniform, 12 Aug 2026).** Byte-identical duplicate rows — same inks *and* same
XYZ+Lab to full float precision — are dropped at load on every **coated** set (PC10/PC11/FOGRA51 and
APTEC CMYKOGV). They are an upstream averaging/duplication artifact, not repeated measurement, and
leaving them in both leaked across CV folds and double-counted in the pooled median
(`docs/research/cv-leakage-2026-08-12.md`). Effective counts: 818→795 (CMY), 1617→1588 (CMYK),
3534→3302 (CMYKOGV-7).

**IFRA is deliberately exempt**: its duplicate recipes carry genuinely *differing* measurements (real
press repeatability, ~0.6–0.8 ΔE00), which is signal about the press rather than an artifact. All 13 wb
runs keep their full 1485 rows. KCMYG-5 and Apex CMYKOGB-7 contain no byte-identical duplicates at all
(verified: 0 and 0), so the policy is a no-op for them — CMYKOGB-7's single duplicate-recipe group is
Apex's 20 genuine paper-white repeats, which differ in value. The n>4 sets additionally use grouped CV
so any such repeated recipe co-travels.
