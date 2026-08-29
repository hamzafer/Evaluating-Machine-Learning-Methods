# Paper audit (29 Aug 2026) — figures, tables vs text, prose

Scope: `MDPI-Phil-Journal/main.tex` + `figures/*.png`. Read-only audit; no edits made.
Companion reports: `01-references.md`, `03-numbers-vs-csv.md`, `04-literature.md`, `00-summary.md`.

## Build facts
- main.pdf newer than main.tex; clean build (0 overfull, 0 undefined refs/citations). Only template warnings.
- **27 pages, ~16,100 words** in MDPI `submit` layout. Target 10–12 pp. Roughly double the agreed length.
- `\pending{}` has zero uses. Comment block l.27–34 references "Section 4.6" and repo file paths; delete before submission.

## Figures

### High severity
1. **fig_n3_model_comparison.png and fig_n3_vs_n4.png in the paper repo are STALE** (12 Aug, md5 differs from code repo). Paper copies show raw code names `poly3_de00_nm`/`poly3_de00_powell` as y-labels. Regenerated code-repo versions (13 Aug, f2442df) render `Nelder--Mead` with a literal `--`. Fix label strings in `fig_n3_model_comparison.py:51` and `fig_n3_vs_n4.py:181`, regenerate, copy over. Numbers on both figures are current (0.046→0.056 ×1.2, 0.268→0.942 ×3.5 match Tables 3–4; ranking by mean median holds).
2. **fig_ncolor_ladder.png in-figure title contradicts the paper**: "Polynomial regression degrades as inks grow; the Gaussian process holds". Section 4.7 says that reading "does not survive". Regenerate without the title (MDPI figures should not carry titles) or add the poly4_cbrt series. Also British "n-colour" and em dashes vs US body.
3. **fig_vs_colourbill.png shows the XYZ-fitted GP** (0.15/0.13/0.11/1.17, max 49.76) labelled "our best model (GP)", but text l.571 says the cbrt GP (0.083/0.079/0.091/0.245) is the relevant series. Figure predates the fairness run. Also l.569 "max 2.41 … against our own 2.412" for PC10-CMYK matches no bar in the figure (poly 2.62, GP 7.23); state which model's predictions were exported.
4. **fig_de00_loss caption (l.331) does not describe the figure.** Caption: "Median/P95/max … Nelder-Mead- and Powell-refined". Figure: max only, Powell only, no NM, no P95. Figure title editorialises and uses `--`.
5. **fig_ifra_generalization has an internal footnote baked in**: "GP within-run was anomalous (~18.8) under the earlier kernel init; resolved by the unified GP config (Plan 10)…". Regenerate without it. Caption l.456 cites "source commit 4433a98" (exists) — drop from a journal caption. Values match Table 7.

### Low severity
6. fig_llm_vs_classical: consistent with text; editorial title; duplicates Table 3 ranking. Candidate for deletion in a trim.
7. Ladder values match Table 8. Resolution fine. 0.56\linewidth makes bar charts small; consider 0.75.

## Internal contradictions / unsupported statements
- l.83 "Two of the datasets used here (FOGRA51)" — one dataset, two conditions.
- l.500 and l.583 "the ICC characterization brief" defines the baseline as degree-3 on XYZ — no citation, no such document named. Cite or delete (twice).
- l.612 "Will (APTEC)" vs bib `chardata2026` author "Li, William"; project notes say Will is ex-Kodak. Full name + confirmed affiliation needed.
- l.288 (Sec 4.3, before any n>4 result) forward-references "eight of the nine datasets".
- l.92 vs l.163: 795+793=1,588 and 29 removed per CMYK condition are consistent (23 K=0 dupes + 6 K>0 dupes) but the arithmetic is stated nowhere.
- Term counts checked: 20/35 (l.65), 120 (l.512), 36/120/330/792 (Table 10) all correct.
- Ratios checked and correct: 21.6, 6.5, 1.7, 1.1, 1.2, 34, 13, 5.2, 20, 1.8, 13× (LLM), 6× and 12–17×, 2.0–2.4× (PC11-CMY is 1.95), 3.3–8.2×, 38–90%, Powell/NM %s, IFRA 2.6–4.3× and 20–31%.
- Abstract "best-median on 8 of 9" vs l.288/504 "all 9 (in XYZ on FOGRA51-CMYK)" — both true, abstract reads as a loss.
- l.181: "96 cells" = 16 models × 6 datasets, but "full model matrix" now has 18 models. Say "the 16 models then in the registry".
- Table 1: KCMYG "2026" and "Apex (2026)" are not sources. IFRA row lacks the promised analysed-count footnote.
- l.183/l.612 "Claude Opus 4.6" — verify the model name; rest of list is a different generation.

## Typos / LaTeX
- Nelder-Mead vs Nelder--Mead: 5 vs 4 occurrences. Pick one.
- l.288 `\linebreak[1]` hack inside `\texttt`; use `\path{}` or drop the path.
- l.339 four-decimal numbers (0.1476, 0.8304, 0.8320) violate the paper's own 2–3-decimal rule.
- Spelling consistent US in body; British only in bib titles/tool names (OK).
- Abbreviations table: "SVM" used for a regressor; consider "SVR".

## AI-slop / self-narration (counts)
"earlier draft" 5×, "originally (reported)" 6×, "conference version" 6×, "uncorrected" 9×, "handicapped" 3×, "therefore" 15×, "not a discovery" 2×, "honest" 1×. The "we were wrong, now we're right" refrain appears in abstract, intro (2×), related work, 4.2, 4.6 (3×), 4.7 (2×), discussion, conclusion. "The mechanism is approximability, not perceptual alignment" appears 4×. "not X, but Y" ~20×.

Worst 15 with rewrites:
1. l.69 "…in the first draft of this analysis was handicapped, on two axes at once." → "…was fitted on XYZ and capped at third order; Section 4.6 quantifies the cost of each choice."
2. l.79 "…and in an earlier draft of the present one, overstates…" → drop the drafting clause.
3. l.277 "…as the conference version and an earlier draft of this paper did, attributes to the polynomial basis what belongs to the polynomial fit." → "Most of that asymmetry is a property of the fitting space, not of the polynomial basis:"
4. l.337 "That is a modeling choice, not a neutral default, and it is the wrong one." → "That choice is examined here."
5. l.341 "the correction is not a way of making the classical method look better, it is a condition for comparing any two methods fairly." → "the correction must be applied to every model before comparison."
6. l.427 + l.595 "not a discovery" / "allows the comparison to be believed" → one sentence, once.
7. l.502/587/597 "beats the Gaussian process as this paper originally reported it" → "beats the XYZ-fitted Gaussian process".
8. l.504 "What has changed is the size of the claim, not its direction." → delete.
9. l.521 "Scope, stated first." → "Scope."; l.550 "…generalizes well beyond color." → delete.
10. l.554 "the cost of an LLM … is dominated by whether it stops, not by whether it is right." → "completion cost was dominated by whether the model terminated." (n=1)
11. l.585 "the honest reading has also narrowed" → "the reading is narrower".
12. l.585 "are real numbers but they measure the baseline's handicap as much as the model's merit" → "reflect the baseline's fitting space as much as the model".
13. l.587 "Yes. And so, it turns out, can classical polynomial regression" → "Yes; so can polynomial regression, given a perceptual fitting space and degree 4."
14. l.516 "It is the strongest practical argument in the paper…" → "The transform reduces worst-case error for every model class tested."
15. l.71 + l.185 + l.595: AIC ΔE00 correction explained in full 3×. Keep l.185, one clause in intro, none in conclusion.

## Process/agent narration inappropriate for a color-science journal
- l.550 serverless-function paragraph ("187 tools … deployed a serverless function to a cloud provider") is unverifiable, names no product, is about an agent harness. Cut to one sentence or remove.
- l.183 "Use of generative AI" paragraph and l.612 acknowledgment duplicate each other; MDPI wants one statement.
- l.181 clean-room reimplementation "with no sight of our code" — by whom? Unattributed.
- l.449 LML "−4909 vs +5346" and l.512 "7.86 to 7.32" are debugging diagnostics; supplementary material.

## Priority order
1. Length (27 pp / 16k words vs 10–12 pp).
2. Stale/mislabelled figures (n3 comparison, n3-vs-n4, ladder title, colourbill missing GP-cbrt, de00 caption, IFRA footnote).
3. Delete drafting-history narration (~25 sentences) and the agent-harness paragraph.
4. Unsupported "ICC characterization brief" (2×), "Two of the datasets", "Will (APTEC)", Opus 4.6 name check, 4-decimal numbers.
5. Nelder-Mead hyphenation, \linebreak hack, comment block cleanup.
