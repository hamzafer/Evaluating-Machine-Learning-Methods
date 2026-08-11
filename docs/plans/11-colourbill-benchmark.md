# 11 — colourbill External Benchmark Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development (light — mostly manual + one script).

**Goal:** Position our results against an established external tool — Will's **colourbill**
(https://chardata.colourbill.com/ , profile tool https://chardata.colourbill.com/profiletool/) —
so reviewers see how our models compare to a standard characterization/comparison tool on the same data.

**Architecture:** colourbill is an online tool (not a library). This plan is investigative + comparative:
learn what it reports, run the same registry datasets through it, extract its accuracy numbers, and put
them beside ours in one table/figure. Some of it is manual (web tool) — capture exactly what's reproducible.

**Tech Stack:** web (WebFetch/manual), plus a small script to tabulate against our CSVs.

## Global Constraints
- Only claim comparisons that are genuinely like-for-like (same dataset, same/declared metric). State caveats.
- If colourbill's methodology or metric differs from ΔE00-on-denorm-XYZ, say so; do not force a false equivalence.

---

### Task 1: Characterize the tool (investigation)
- [ ] WebFetch/inspect https://chardata.colourbill.com/ and the profiletool: what does it take (which datasets/formats),
  what does it output (accuracy metric? ΔE flavour? model type?), and can it run on the registry sets we use (APTEC etc.)?
- [ ] Write `docs/research/colourbill.md` documenting capabilities, inputs/outputs, metric, and what is reproducibly comparable.
- [ ] Decide the comparison scope (which of our datasets overlap what colourbill can do). Commit the research doc.

### Task 2: Run + tabulate
- [ ] Run the overlapping dataset(s) through colourbill; record its accuracy numbers (screenshots/exports as provenance in
  `journal/results/colourbill/`).
- [ ] Script `journal/figures/fig_vs_colourbill.py` (or a table) placing colourbill's numbers beside our best models on the
  same dataset. dataviz skill; caption states the metric/methodology caveat.
- [ ] Commit provenance + figure/table.

### Acceptance
- A defensible external comparison exists (or a documented reason it isn't like-for-like). Reviewer question
  "how does this compare to existing tools?" has an answer.
