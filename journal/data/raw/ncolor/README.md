# n>4 colorant datasets (the paper's headline n>4 evidence)

Three multi-ink datasets giving a clean **n = 5 / 7** ladder to pair with the
existing n = 3 (CMY) and n = 4 (CMYK) APTEC/FOGRA results. All need ingestion
via `journal/pipeline` (n-channel-generic). Two are spectral -> convert to XYZ/Lab
with `journal.pipeline.ifra.spectral_to_xyz` (D50, 2deg), reusing the IFRA path.

| File | Inks (n) | Channels | Patches | Measurement | Source |
|---|---|---|---|---|---|
| `APTEC_CMYKOGV_7clr_xyzlab.txt` | 7 | C M Y K O G **V** | 1624 | **XYZ+Lab already present** (D50/2, M1) | ICC registry (APTEC), 2025 |
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
