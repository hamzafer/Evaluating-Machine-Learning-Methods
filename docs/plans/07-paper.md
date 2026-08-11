# 07 — Paper: Fill, Position, Trim, Submit

> REQUIRED SUB-SKILL: subagent-driven-development for the mechanical parts; content is co-authored with Phil.

**Goal:** Take the ported MDPI draft (`../MDPI-Phil-Journal/main.tex`, 14pp, honest placeholders) to a
submission-ready 10–12pp manuscript by 30 Aug 2026.

**Architecture:** The paper lives in the separate repo `../MDPI-Phil-Journal` (official MDPI `technologies`
template, builds via `latexmk -pdf main.tex`; figures synced into its `figures/`). Content comes from the
code repo's results CSVs; figures from `journal/figures/`. LaTeX agent keeps the PDF current as results land.

## Global Constraints (Phil's rules)
- Full methods incl. equations; median/max/95th (never mean/std as headline); 2–3 decimals; figures from CSVs.
- Correction: brief mention (previous work had an error, corrected here). Already drafted.
- Length 10–12pp. First draft plainer is fine; MDPI formatting polish later. No `Co-Authored-By` trailer.

---

### Task 1: Fill the placeholders as results land
- [ ] n>4 ladder (Plan 06): replace the §"Scaling to n>4" placeholder with the real n=5/7 tables + the ladder figure.
- [ ] Multi-LLM (Plan 08): replace the LLM placeholder with the 4-model comparison; keep the CV-vs-holdout caveat.
- [ ] LLM-equation (Plan 09): add the equation-generator result (predictor vs equation contrast). Decide with Hamza
  whether both LLM flavours stay in the final paper (deferred decision).
- [ ] GP consistency (Plan 10): reconcile all GP numbers; if the IFRA anomaly resolved, update that section + footnote.
- [ ] Discussion: add the reflexive "the corrected pipeline was itself AI-built" point (LLM mode C, discussion-only).

### Task 2: Positioning & references
- [ ] Related-work/Discussion subsection positioning our ML results vs the model-based n-colour prior art:
  **Deshpande, Green & Pointer, Optics Express 22(26):31786-31800 (2014)** (spot-colour overprint / inverse printer models),
  + related n-colour separation work by the same authors. Add colourbill comparison (Plan 11). (Ref resolved — no longer gated on Phil.)
- [ ] Cite: ISO substrate-correction standard (as out-of-scope/future work re newsprint bb), colour-science, CIEDE2000.
- [ ] Answer Phil's questions (a) n<=4 and (b) n>4 explicitly in the conclusion, now with real n>4 data.

### Task 3: Trim + format + submit
- [ ] Trim 14 → ~12pp (tighten prose, merge/appendix over-dense tables). Verify against Phil's rules + MDPI checklist
  (abstract <=200 words, author contributions, funding [APC 1800 CHF status], data-availability → GitHub).
- [ ] Full build clean; PDF to Phil for co-work; revise; submit via MDPI by 30 Aug.
- [ ] Deliverables to Phil THIS week: current draft PDF + a short results summary email.

### Acceptance
- 10–12pp MDPI-formatted paper, all placeholders filled, positioned against prior work, submitted by deadline.
