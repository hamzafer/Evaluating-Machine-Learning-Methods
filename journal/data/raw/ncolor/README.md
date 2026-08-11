# n>4 colorant datasets (the paper's headline n>4 evidence)

Three multi-ink datasets giving a clean **n = 5 / 7** ladder to pair with the
existing n = 3 (CMY) and n = 4 (CMYK) APTEC/FOGRA results. All need ingestion
via `journal/pipeline` (n-channel-generic). Two are spectral -> convert to XYZ/Lab
with `journal.pipeline.ifra.spectral_to_xyz` (D50, 2deg), reusing the IFRA path.

| File | Inks (n) | Channels | Patches | Measurement | Source |
|---|---|---|---|---|---|
| `APTEC_CMYKOGV_7clr_xyzlab.txt` | 7 | C M Y K O G **V** | 3534 (header says 1624 — stale, see below) | **XYZ+Lab already present** (D50/2, M1) | ICC registry (APTEC), 2025 |
| `KCMYG_5clr_spectral.txt` | 5 | C M Y K **G** (cols 5CLR_1..5) | 2214 | spectral nm380..nm730 (36 bands), D50/2 | Phil (email 11 Aug 2026) |
| `Apex_CMYKOGB_7clr_spectral.txt` | 7 | C M Y K O G **B** (cols 7CLR_1..7) | 2000 | spectral nm380..nm730 (36 bands), D50/2 | Phil (email 11 Aug 2026) |

Notes:
- The two spectral files are X-Rite ProfileMaker/MeasureTool format (LGO* header
  keys); ink names in the LGOMCCHANNEL0n lines; channel order = the nCLR_k column order.
  `spectral_cols` regex already matches nm380 naming.
- APTEC set is CMYKOG**V** (Violet); Apex set is CMYKOG**B** (Blue) - two different
  7-ink systems, a nice robustness point for the paper.
- Phil offered to convert spectral->XYZ; NOT needed - our pipeline does it, and the
  derived-vs-native cross-check (as used for IFRA) validates the conversion.
- Original filenames from Phil: 4645_KCMYG_21.txt, "Apex Averaged 2.txt",
  APTEC_CMYKOGV_Coated_LinearCTV_2025_M1.txt.

Ingestion findings (`journal.pipeline.ingest_ncolor`, Aug 2026):
- **Ink scale**: all three files carry ink values on the **0-100 scale**
  (full-tone patches read 100.00); processed CSVs keep 0-100. The ingester
  auto-detects and would rescale a 0-1 file, but none of these needed it.
- **APTEC patch count**: `NUMBER_OF_SETS` says 1624 (the figure originally
  quoted in the table above) but the data block holds **3534** rows —
  contiguous SAMPLE_ID 1..3534, a single chart (rows 1-1624 vs 1625-3534
  share only 145 ink combos, so it is not a duplicated 1624-patch target),
  3302 unique ink combos, duplicate combos carrying byte-identical XYZ/Lab
  (averaged export). The header count is stale metadata; all 3534 rows are
  ingested. Native XYZ vs native Lab agree at median |dLab| = 0.005
  (D50/2, XYZ 0-100 confirmed).
- Apex file: 2000 rows, one duplicated ink combo — the unprinted patch,
  measured 20 times (max spectral spread 0.031); genuine repeats, kept.
- Paper-white Y (unprinted patches): KCMYG 80.35, CMYKOGV 84.20 (M1,
  coated), CMYKOGB 77.81 — all plausible.
