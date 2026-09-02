# Status — 2 Sep 2026: SUBMITTED, editor assigned

**The journal paper was submitted to MDPI Technologies on 30 Aug 2026 at 23:58:47** (73 seconds
before the special-issue deadline). Manuscript ID **technologies-4564760**, special issue
"AI-Driven Color Models for Imaging, Formulation, Appearance Measurement and Computer Vision".
Status: pending review. Title as submitted: **"AI and Machine Learning Methods for n-Colorant
Printer Characterization"** — authors Muhammad Hamza Zafar (corresponding) and **Phil Green**.

What went in: the 23-page build carrying Phil's complete tracked-changes review (296 changes,
20 margin comments — extracted from `journal/final-review-phil/main-PG.docx`, applied by four
parallel agents, then verified record-by-record by two independent agents; server copies
MD5-verified against the local package). Headline changes from the review: subtitle dropped,
corrections narrative removed, Figure 1 and the External Validation section deleted, the IFRA
name replaced by "newsprint" throughout (data-permission concern), "registry" -> "model set",
JND framing removed.

APC: NTNU IOAP — central invoicing to institution, author eligible; fallback 1,260 CHF.
Nothing owed unless accepted.

## Post-submission log

| date | event | action |
|---|---|---|
| 2 Sep 2026, 03:07 | **Assistant Editor assigned: Tyler Yu** (tyler.yu@mdpi.com). Paper passed intake and is in the reviewer-finding stage. | None. Tyler is the point of contact for any question about the manuscript. MDPI's English/figure-editing offer is an upsell; ignored. |
| 2 Sep 2026, 03:07 | **Preprints.org invitation** (automated, sent to every submission; "recommended" carries no quality signal). Optional; no effect on peer review either way. | **Deferred pending Phil.** A preprint is permanent and public. Concerns: Phil's data-permission worry about naming the newsprint source, Apex permission still open on the revision TODO, and a v1 that stays online if reviewers force large changes. Ask Phil before clicking. |

| 2 Sep 2026 | **Phil's answers on the revision items** (chat): prior-work table + implementation details still wanted; extra refs only if they add something; CMYKOGB (Apex) may be used in research but the source must not be named and the data must not be shared; chapter ref confirmed as Wiley 2023 *Fundamentals and Applications of Colour Engineering* ch. 3 pp. 53-70; CharData cite gets v1.18.0. | Applied in the paper repo (commit cf56575): "Apex" removed from the data availability statement, bib fixed. **Open in this repo:** "Apex" still named in docs/README/ingester/raw filename, and the 13 processed newsprint CSVs are tracked in the public repo although the paper says the newsprint data is not redistributed. Both need Hamza's call. |

Expected next: reviewer reports, typically 2–4 weeks from editor assignment. Only trust emails from
`@mdpi.com` addresses (MDPI's own phishing warning).

Next: revision-round work is listed in `../MDPI-Phil-Journal/REVISION-TODO.md` (prior-work
comparison table, implementation-details appendix, Phil's 7 extra references, Apex permission,
Colour Engineering chapter pages). The section below is the pre-submission status of 23 Aug.

---

# Status — 23 Aug 2026

Deadline **30 Aug**, seven days out. Code repo clean, 41 tests green, paper repo has 2 unpushed
commits. This file supersedes the 13 Aug version.

## Are the experiments finished?

**Essentially yes, with one fairness run in flight.** Everything the paper needs is measured, gated
and committed. The one open item exists because of a finding made today (below): having discovered
the polynomial baseline was fitted in the wrong space, the same correction must be offered to its
competitors before any comparison between them is fair. That run is `poly4`, `poly4_cbrt` and
`gaussian_process_cbrt` across the nine datasets.

| plan | state |
|---|---|
| 01-05 (n=3/4, GP verification, IFRA, direct ΔE00) | ✅ done, gated |
| 06 (n>4 ladder: 5-ink and two 7-ink sets) | ✅ done, gated |
| 09 (LLM as equation generator) | ✅ done, gated |
| 10 (unified GP config, IFRA anomaly) | ✅ done, gated |
| 11 (colourbill external benchmark) | ✅ done, gated |
| 08 (200-sample LLM prediction table) | ⛔ dropped — superseded models, and the equation experiment answers Phil's question better |
| fitting-space follow-up | 🏃 fairness run in progress |

## The two results that matter most

### 1. The polynomial baseline was fitted in the wrong space
Fitting the same degree-3 polynomial to `cbrt(XYZ)` (equivalently: in CIELAB) rather than to XYZ
improves the **maximum error on 9 of 9 datasets** and the **median on 8 of 9**. On the 7-ink set,
5.386 → 0.830.

Then degree, on CMYKOGV-7:

| degree | XYZ | CIELAB |
|---|---|---|
| 3 (current cap) | 5.386 | 0.830 |
| **4** | 2.080 | **0.272** |
| Gaussian process | | **0.249** |

**A degree-4 polynomial in CIELAB ties the GP at seven inks**, without overfitting (train/test gap
+0.027, unchanged from degree 3, on 3,302 rows).

Consequence: the paper's n>4 headline was measured against a baseline handicapped on two axes at
once, the fitting space and Phil's 3rd-order cap. The claim must narrow. Detail, controls and the
corrected mechanism are in `docs/research/cube-root-fitting.md`.

### 2. The LLM experiment
Phil's prompt, 150 training pairs, scored on 100 held-out patches with our own ΔE00 code.

| model | condition | median ΔE00 |
|---|---|---|
| Claude Fable 5 | web, **with code execution** | **0.082** |
| our least-squares cubic | baseline | 0.234 |
| GPT-5.6 Sol | API, reasoning | 3.070 |
| Claude Opus 5 | API, no reasoning | 14.291 |
| DeepSeek V4 Pro | API, no reasoning | 23.764 |
| Haiku 4.5 | API, reasoning | 28.781 |
| Claude Opus 5 / Fable 5 | API, reasoning (8k, 24k, 78k tokens) | no answer produced |

Four readings:
- Unaided, no LLM is competitive: the best reasoning-only answer is 13× worse than fitting a cubic.
- Given a code interpreter, an LLM beat our baseline — but that is an automation result, not a
  colour-science one.
- Frontier Anthropic models did not terminate on this task at any budget up to 78,000 tokens, while
  GPT-5.6 Sol converged in 2,747. A real behavioural difference, and the most novel LLM observation
  we have.
- Every accurate answer broke "as simple as possible" (expanded degree 9 and 27).

Weaknesses to state in the paper: n=1 per model, conditions differ per row, 3-ink chart only, so it
does not address Phil's n>4 question.

## What needs Hamza

1. **Talk to Phil** about the 3rd-order cap and the narrowed n>4 claim. This is the one item with an
   external dependency and it changes the paper's argument.
2. Two data questions for the same conversation: the PC10 file's header declares itself
   `APTEC_PC11_CCNB_2023_v1`, and KCMYG's provenance is uncited.
3. **Paper trim**, 17pp against 10-12pp. Prose only; table surgery was measured and saves nothing.
4. Push the paper repo (2 commits).
5. Admin from Phil: APC funding (1800 CHF) and author order.

## Provenance and audit
- `journal/results/run_log.tsv` — every fit ever run, with accuracy, wall time, machine, package
  versions and git commit.
- `journal/results/logs/raw/` — verbatim console output of every run.
- `journal/verification/blind-2026-08-12/` — the independent clean-room reimplementation, its code
  and its report.
- `docs/DECISION_LOG.md` — every decision, gate verdict and incident.
- `docs/TECH_DEBT.md` — deferred items, chiefly the proprietary data in this public repo's history.
