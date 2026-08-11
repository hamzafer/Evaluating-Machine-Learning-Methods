# Links & resources Phil shared (journal paper)

Keep this current — the durable index of everything Phil has pointed us to.

## Datasets
- **7-ink CMYKOGV (APTEC, coated)** — ICC registry:
  https://registry.color.org/cmyk-registry/aptec_cmykogvcoated
  (downloaded to `journal/data/raw/ncolor/APTEC_CMYKOGV_7clr_xyzlab.txt`; XYZ+Lab, 1624 patches)
- **5-ink KCMYG** — Phil email 11 Aug 2026 → `journal/data/raw/ncolor/KCMYG_5clr_spectral.txt` (spectral, 2214 patches)
- **7-ink CMYKOGB (Apex)** — Phil email 11 Aug 2026 → `journal/data/raw/ncolor/Apex_CMYKOGB_7clr_spectral.txt` (spectral, 2000 patches)
- **IFRA newsprint (wb/bb)** — `journal/data/raw/Ifra-{wb,bb}.zip` (wb only used; bb out of scope)

## Tools / benchmarks
- **colourbill** (Will's characterization-dataset analysis/comparison tool — external benchmark):
  https://chardata.colourbill.com/
- **colourbill profile tool** (compare ICC profiles): https://chardata.colourbill.com/profiletool/

## Journal / submission
- **Special issue page:** https://www.mdpi.com/journal/technologies/special_issues/0O2229T6RE
- **Author instructions / LaTeX template:** https://www.mdpi.com/journal/technologies/instructions
- **Journal home:** https://www.mdpi.com/journal/technologies  (Technologies, ISSN 2227-7080)
- **AIC 2025 proceedings:** https://www.aic2025.org/

## Key references (resolved)
- **Deshpande, Green & Pointer, "Characterisation of the n-colour printing process using the spot colour overprint model," Optics Express 22(26):31786-31800 (2014):** https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-26-31786&id=306859
  — the model-based n-colour prior art to position our ML approach against (Related Work + Discussion).
  NOTE: the earlier "Kiran's Optics Express paper" and "the PhD benchmark" are the SAME researcher — **Kiran Deshpande** (Phil Green's collaborator/former PhD student). Scholar: https://scholar.google.com/citations?user=e19-J04AAAAJ&hl=en
- Related n-colour work by the same authors (separation via inverse printer models; gamut evaluation).
- ISO substrate-correction standard (wb<->bb backing conversion) — Phil mentioned; look up the exact number for a one-line future-work cite.

## Phil's LLM-as-equation prompt (11 Aug meeting)
> "Generate an equation that transforms any coordinate in data set A into a coordinate
> in data set B. The equation should be as simple as possible, and avoid exponents
> greater than 3. The success criterion is minimisation of average and maximum
> differences between CIELAB values in data set B and those estimated by the equation,
> as defined by the CIEDE2000 equation."
