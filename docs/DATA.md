# Datasets — quick reference

All printer characterization data: ink values → measured colour (XYZ/Lab), evaluated with CIEDE2000.
Report per-dataset (mixed sources/conditions — not one controlled sweep). ΔE00 on denormalized XYZ, D50/2°.

| Dataset | n inks | Channels | Patches | Colour data | Location | Used for |
|---|---|---|---|---|---|---|
| APTEC PC10 | 3 / 4 | CMY(K) | 1617 (818 K=0) | XYZ + Lab | `data/cleaned/APTEC_PC10_*.csv` | n=3 (CMY), n=4 (CMYK) |
| APTEC PC11 | 3 / 4 | CMY(K) | 1617 (818 K=0) | XYZ + Lab | `data/cleaned/APTEC_PC11_*.csv` | n=3, n=4 |
| FOGRA51 | 3 / 4 | CMY(K) | 1617 (818 K=0) | XYZ + Lab | `data/cleaned/FOGRA51.csv` | n=3, n=4 |
| IFRA newsprint (wb) | 4 | CMYK + spectral | 13 runs × 1485 | spectral→XYZ | `journal/data/raw/Ifra-wb.zip` | multi-press generalization |
| KCMYG | 5 | C M Y K G + spectral | 2214 | spectral→XYZ | `journal/data/raw/ncolor/KCMYG_5clr_spectral.txt` | n=5 (ladder) |
| APTEC CMYKOGV | 7 | C M Y K O G V | 3534 raw (3302 effective after exact-dedup at load; header's 1624 is stale) | XYZ + Lab | `journal/data/raw/ncolor/APTEC_CMYKOGV_7clr_xyzlab.txt` | n=7 (ladder) |
| Apex CMYKOGB | 7 | C M Y K O G B + spectral | 2000 | spectral→XYZ | `journal/data/raw/ncolor/Apex_CMYKOGB_7clr_spectral.txt` | n=7 (ladder) |

**Ink-count ladder for the headline:** 3 → 4 → 5 → 7 (two independent 7-ink systems: OGV vs OGB).

**Excluded (out of scope, per Phil):** IFRA **black-backing** newsprint — a separate "substrate correction"
problem, and its files lacked ink recipes / used a different chart layout. Raw zip moved to
`../_archive_out_of_scope/Ifra-bb.zip` (outside the repos).

Notes: CMY experiments use only K=0 rows (818); CMYK uses all 1617 with K as an input (never drop a column
while keeping its rows). Spectral sets convert to XYZ via `journal.pipeline.ifra.spectral_to_xyz`.
