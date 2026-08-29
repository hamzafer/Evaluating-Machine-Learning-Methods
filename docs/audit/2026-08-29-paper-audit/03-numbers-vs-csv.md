# Paper audit (29 Aug 2026) — every number vs the results files

Method: parsed all nine LaTeX tables and compared cell-by-cell to `journal/results/*/summary.csv`, `gp_verification.csv`, `ifra/*.csv`, `llm/equation_summary.csv`, `llm/PC10-CMY_summary.csv`, `colourbill/*.csv`, `run_log.tsv`. Recomputed every derived ratio/percentage in the prose. Re-scored the archived Fable web equation through the repo's parser and ΔE00 path. Re-ran the degree sweep, transform sweep, y^-4/3 control and residual-RMS computation through the pipeline (polynomials only) to check numbers that exist only in prose.

**Headline: 724 table cells match the CSVs exactly. Nothing is fabricated. One cell is off by a rounding digit; four prose statements overstate or mis-scope what the files show; a block of mechanism numbers (Sec 4.6) is reproducible but has no committed CSV and the markdown it cites is stale.**

## (A) Exact mismatches

1. **tab:ifra, SVM cross-run: paper 3.874, file 3.8735 → 3.873** (`ifra/cross_run.csv`, midpoint of entries 78/79 = 3.832, 3.915). Half-up vs banker's rounding; one digit. GP cross-run 3.8435 → 3.844 as printed.
2. **Sec 4.7 "84% of patches use at most four nonzero inks"**: 85.1% of 3,534 as-received, **84.1% of 3,302 analysed**. Correct as written since the sentence follows "3,302 analysed samples".
3. **Sec 4.8 "Claude Fable 5 … spent its entire completion allowance on reasoning tokens and returned empty content at both 1,600 and 78,000"**: true at 78,000 (77,999 reasoning). At 1,600 the archive (`journal/llm/raw/equation/anthropic__claude-fable-5.attempt1-truncated.json`) shows **238 reasoning tokens**, finish_reason=length, and non-empty truncated prose ("I'm modeling the CMY to XYZ conversion…"). The 1,600 case is "truncated mid-derivation, no equation", not "empty content". Overstated.
4. **IFRA "recovers 20–31%"**: (B−C)/(B−A) = GP 23.4%, SVM 30.9%, poly3 **31.5%**, MLP 20.1%. Should read 20–32% or "roughly 20–30%".
5. **Reproducibility: clean-room "reproduced our GP medians to within 0.003 on every coated dataset (exactly on PC11-CMY) and poly3 to within 0.006"**: true per `journal/verification/blind-2026-08-12/BLIND_REPORT.md` §6, but against the **pre-deduplication, pre-plan-10 numbers** (818/1,617 rows), not Tables 3–4. GP deltas +0.003/+0.001/0.000/+0.002; poly3 ≤0.001 on coated (the 0.006 is SVM on IFRA). A reviewer diffing the blind CSV against Table 3 will not find 0.041/0.046. Needs rewording ("an earlier results set") or a re-run of the blind comparison on the current numbers.

## (B) Numbers with no source in the results files

All reproduce on re-run (my values in brackets), but nothing in `journal/results/` records them; `docs/research/cube-root-fitting.md`, which the paper cites, is **stale**:
- **tab:degree "Gap (cbrt)" +0.013/+0.029/+0.029/+0.036**: doc says —/+0.023/+0.027/+0.036. Re-run: **+0.013/+0.029/+0.029/+0.036 → paper right, doc stale.** XYZ gaps +0.121→+0.226 ✓ re-run. Degree-5 XYZ 0.506 ✓ (doc has "—"). Degree-2 9.953/2.656 ✓.
- **tab:transforms PC11-CMYK row**: absent from the doc; re-run reproduces all five values.
- **y^-4/3 control "worse on 8 of 9", PC11-CMYK 0.869→1.400, FOGRA51-CMYK 0.816→1.315**: doc says "3 of 4" and gives only PC10-CMYK 1.551. Re-run with per-row weight = channel-mean of y^-4/3 gives worse on **9 of 9** with 1.704/1.517/1.326. Direction confirmed; exact values depend on an unrecorded weighting detail. **Not reproducible as stated; the script must be committed.**
- **Residual RMS % (PC10-CMY, FOGRA51-CMY, CMYKOGV-7)**: re-run reproduces **all** exactly; OGV values not in the doc.
- **CIELAB-direct 0.8304 vs 0.8320**: only in the doc (0.1476 also in DECISION_LOG). Not re-run.
- **"training mean ΔE00 improves only from 7.86 to 7.32" (Nelder-Mead, OGV)**: in no log, CSV, DECISION_LOG or doc. **Unsourced.**
- "cube root outright winner on only 2, sqrt and y^0.25 3 each": follows from tab:transforms ✓.
- "earlier configuration put GP within-run near 18.8": `docs/plans/10-gp-consistency.md` says ~18.8; recorded runs 16.6–20.1. Acceptable as "near".
- Harness attempts 187/74 tools ✓ (`harness_channel_note.md`).

## (C) Confirmed correct

- 724 table cells exact (tab:n3, n4, n7, de00loss, fitspace, transforms identity/cbrt, degree rows 3–4, ifra bar one cell, llm). Bold markers correct. Sort orders verified (n3/n4 by PC10 median, n7 by KCMYG-5 median).
- Fable web row 0.082/0.241/0.752, 2,542 terms, degree 27, per-variable exponent 9: re-scored 0.0818/0.2412/0.7524 ✓; poly3-ls-local 0.234/0.917/3.052 ✓.
- All LLM token/cost figures ✓ vs raw JSON and CSV.
- Derived ratios: 6.49, 21.63, 1.70, 1.12, 1.16, 1.81, 13.1×, ×3.5/×1.2/×1.5/×1.3; GP vs best poly at n≤4 **1.95–2.42** ("2.0–2.4" ✓); vs poly3_cbrt **3.26–8.15** ("3.3–8.2" ✓); "6× and 12–17×" (5.2–6.5, 11.8–16.8) ✓; OGV factors 33.7/13.0/5.19 ✓; 89% (88.5) ✓; 38–90% ✓; cbrt improves median 8/9 for poly3, poly4, GP ✓; GP max 8/9 with PC11-CMY +0.002 ✓; poly3 max 9/9 ✓; all Powell/NM percentages ✓; IFRA 2.6–4.3× ✓, 0.674–2.141 ✓, best on 12 of 13 ✓, noise floor 0.383–1.934 ✓, CMYKOGB 0.348 ✓; grouped-vs-kfold ≤0.003/≤0.008 ✓, IFRA ≤0.02 ✓; SVM 3rd/GB 4th on all six n≤4 ✓; linear family ranges ✓; duplicates 23/29/232 ✓; 30 of 818 (3.7%), 0.000 vs 4.437 ✓; 58/76 of 96, 0.108 ✓; seed stds ✓; 7 of 9 exact, ≤0.016, 182 IFRA bit-exact ✓; 0.1 s vs 321.4 s ✓; Powell 41–62 min ✓; LML −4909/+5346 ✓; colourbill all values ✓; poly4_cbrt means ✓; GPT-4o/mini pilot ✓; 400 in-context ✓.

## Recommendations
1. Commit `degree_sweep.csv`, `transform_sweep.csv`, `residual_rms.csv` and the y^-4/3 weighting-control script so tab:degree, tab:transforms and the mechanism paragraph have CSV provenance like every other table (the paper's own rule). Refresh `docs/research/cube-root-fitting.md`.
2. Fix SVM cross-run 3.873, "20–32%", the Fable 1,600-token wording, and the clean-room claim's scope.
3. Source or delete the "7.86 to 7.32" Nelder-Mead training figure.
4. Noticed: `journal/llm/protocol.py` builds the pilot split with `drop_duplicates(subset=input_cols)`, so the pilot holdout was drawn from a recipe-deduplicated pool, not "the as-received 818-row pool" as Sec 4.8 says.
