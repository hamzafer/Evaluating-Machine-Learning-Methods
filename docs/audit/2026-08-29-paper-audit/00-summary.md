# Paper audit, 29 Aug 2026 — summary and verdict

Target: `MDPI-Phil-Journal/main.tex` (27 pp in MDPI submit layout, ~16,100 words), deadline 30 Aug 2026.
Four independent audits, each in its own file: `01-references.md`, `02-figures-tables-prose.md`, `03-numbers-vs-csv.md`, `04-literature.md`. This file answers the eight questions asked and ranks the paper.

## The eight questions

### 1. Are any references hallucinated?
**No.** All 23 active bib entries resolve to real works with matching authors, titles, venues, volumes and pages (Crossref, publisher pages, Zenodo, the actual AIC 2025 proceedings PDF). One entry is materially wrong: `pang2024standardization` has the wrong author (Brenda Pang, not Bob), a mangled title, and a URL that 404s. Minor: PC10/PC11 substrate labels in Table 1 disagree with the ICC registry pages (both say CCNB).

### 2. Claims not backed by results?
**Tables: fully backed.** 724 table cells match the results CSVs exactly; sort orders and bold markers are correct. **Prose: five overstatements**, none fabricated:
- IFRA "recovers 20–31%" is 20–32%.
- SVM cross-run 3.874 should be 3.873 (rounding).
- Fable 5 at 1,600 tokens did not return "empty content"; it returned truncated prose (238 reasoning tokens).
- The clean-room reproduction "to within 0.003" was against the pre-dedup results set, not the numbers in Tables 3–4.
- The Nelder-Mead training figure "7.86 → 7.32" has no source anywhere.
**Provenance gap:** Tables 10 (degree sweep), 11 (transform sweep), the y^-4/3 control and the residual-RMS mechanism numbers have no CSV; they cite a markdown doc that is stale (its gaps differ from the paper's; the paper's values re-run correctly). The paper's own rule is "every number from a results CSV". Fix: commit `degree_sweep.csv`, `transform_sweep.csv`, the weighting-control script.

### 3. Mistakes in the paper?
- "Two of the datasets used here (FOGRA51)" (one dataset).
- "the ICC characterization brief" cited twice as defining the baseline; no such public document exists.
- "Will (APTEC)" in acknowledgments; bib says William Li (colourbill), notes say ex-Kodak.
- Cheung 2004 is cited for "modestly better accuracy for the learned model" (paper says polynomial ≈ NN and recommends polynomial) and "commonly train in CIELAB" (they fit to XYZ).
- Bala 2003 is cited for "regression conventionally posed with a perceptually uniform target"; Bala's regression example targets XYZ. Chapter supports CIELAB as the conventional *output space* and CIELAB *evaluation*, not a regression-target rule.
- Four-decimal numbers (0.1476, 0.8304) against the paper's own 2–3-decimal rule; Nelder-Mead hyphenation inconsistent; `\linebreak` hack in `\texttt`; leftover comment block referencing repo paths; "Claude Opus 4.6" model name to verify; "96 cells" described as the "full model matrix" (it is 16 of 18 models).

### 4. Do tables and figures match the text?
Tables yes. **Figures: five of seven have problems.**
- `fig_n3_model_comparison` and `fig_n3_vs_n4` in the paper repo are stale copies with raw code names as labels; the regenerated versions render `--` literally.
- `fig_ncolor_ladder` has an in-figure title asserting the reading the paper says "does not survive".
- `fig_vs_colourbill` shows the XYZ-fitted GP the text calls an artifact; the cbrt GP series is missing; the "max 2.41" in text matches no bar.
- `fig_de00_loss` caption promises median/P95/max for NM and Powell; figure shows max, Powell only.
- `fig_ifra_generalization` has an internal "Plan 10" footnote baked into the image.
All figure *numbers* match the tables.

### 5. Uncorroborated references, and literature to fill the gap
Uncorroborated citation-to-claim: Bala 2003 (l.79), Cheung 2004 (l.77, l.79), Pang 2024 (l.83). Verified replacements with DOIs are in `01-references.md` and `04-literature.md`: Hung 1993, Vrhel & Trussell 1999, Balasubramanian 1999, Gerhardt & Hardeberg 2008, Westland/Ripamonti/Cheung 2012, Hong/Luo/Rhodes 2001.

### 6. Is the paper following the literature, or is it slop?
**The experimental core is not slop.** Numbers are real, reproducible, cross-checked by an external tool and a clean-room reimplementation; the fitting-space correction is a genuine, honestly reported finding. **The positioning is weak** and two claims are wrong: "to the authors' knowledge a first" for ML at n>4 (Shi et al. 2018, 10 inks; Chen & Urban 2021, 6 materials) and for LLM-as-predictor (Vacareanu 2024, LLM-SR 2025). Nothing after 2004 in the paper's own field is cited; Babaei & Hersch 2016 (n-ink), He et al. 2023 (prior XYZ-vs-CIELAB penalty measurement, Pointer co-author), Nussbaum & Hardeberg 2012 (press variation dominates on coldset, NTNU), Su 2021 and Zhan 2025 (direct CMYK→Lab ML competitors) are all missing. **The prose has a slop problem**: the paper narrates its own drafting history ~25 times ("earlier draft", "originally reported", "not a discovery", "the honest reading"), repeats the mechanism sentence four times, and includes an unverifiable paragraph about an agent deploying a serverless function.

### 7. Is it worth submitting? Ranking
**Science: 7/10.** Solid, careful, reproducible, with one novel measured result (the size of the fitting-space penalty across n = 3..7) and a useful negative result (press variation dominates). The n>4 answer is real but narrower than the framing.
**Manuscript as it stands: 4/10.** Roughly double the agreed length, stale figures, a related-work section a colour-science reviewer will reject on sight, two false novelty claims, and a drafting-history voice that reads as AI-generated.
**Verdict: worth submitting, not in this state.** With the deadline tomorrow the realistic path is (a) fix the false "first" claims and add the ~8 must-have citations, (b) delete the drafting narration and the agent paragraph, (c) regenerate the five figures, (d) fix the five prose numbers, (e) cut to ~12–14 pp. That is a day of focused editing, not new experiments. If the deadline cannot move, submit after (a)–(d) and accept length; if it can (MDPI special issues usually extend), do (e) too.

### 8. Are the research questions answered?
- **(a) Can ML match/beat classical at n≤4?** Yes, answered: GP best-median on all six coated conditions, 2.0–2.4× better than the best polynomial. Corrected and cross-validated. Caveat correctly stated: same-press, same-substrate result.
- **(b) Can AI handle n>4?** Yes, answered, and more honestly than the original framing: GP wins on all three n>4 sets but the corrected polynomial is within 1.1–1.7×. The paper's own re-scoping ("and so can polynomial regression, given the space and degree") is the right conclusion and the most defensible contribution.
- Supporting questions: multi-printer generalization (answered: press dominates), direct ΔE00 loss (answered: subsumed by fitting space), LLM (exploratory, n=1, correctly scoped).
The questions are answered. What is missing is the literature context that shows where those answers sit.

## Priority list for tomorrow
1. Remove/narrow both "first" claims; add Babaei & Hersch 2016, Shi 2018, Chen & Urban 2021, He 2023, Nussbaum & Hardeberg 2012, LLM-SR, Vacareanu 2024, Finlayson 2015 (root-polynomial distinction).
2. Fix Pang 2024 entry; reword the Cheung 2004 and Bala 2003 sentences; cite or delete "ICC characterization brief"; name Will properly.
3. Regenerate the five figures (drop titles/footnotes, fix labels, add GP-cbrt to colourbill, fix de00 caption).
4. Five prose-number fixes (20–32%, 3.873, Fable 1,600 wording, clean-room scope, delete 7.86→7.32).
5. Delete drafting-history narration and the serverless-function paragraph; keep one AI-use statement.
6. Commit degree/transform sweep CSVs in the code repo; refresh `cube-root-fitting.md`.
7. Cut length.
