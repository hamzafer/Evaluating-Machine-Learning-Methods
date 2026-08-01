# 00 — Execution Order & Status

> Strategy/context lives in `journal_roadmap.md`. This file is the operational index:
> what runs, in what order, and where each plan stands. Update the Status column
> whenever a plan changes state.

**Deadline: 30 Aug 2026 (MDPI Technologies special issue).** Today's four-week window:

| Week | Focus |
|---|---|
| W1 (1–8 Aug) | Plans 01, 02, 03 + Phil call (correction wording, chase CMYKOGV) |
| W2 (9–15 Aug) | Plans 04, 05; 06 if data arrives |
| W3 (16–23 Aug) | Plan 07 (writing takes over; experiments only to fill reviewer-visible gaps) |
| W4 (24–30 Aug) | Buffer: Phil review round, MDPI formatting, submit |

| Plan | What | Status | Blocked on |
|---|---|---|---|
| [01](01-gp-verification.md) | Verify GP headline result (grouped CV + noise floor) | TODO | — |
| [02](02-ifra-ingestion.md) | IFRA newsprint → pipeline datasets | TODO | — |
| [03](03-ifra-generalization.md) | Cross-press-run generalization experiments | TODO | 02 |
| [04](04-llm-predictor.md) | LLM as direct color predictor | TODO | API key (Hamza provides when ready) |
| [05](05-direct-de00-loss.md) | Direct ΔE00 minimization + classical optimizers | TODO | — |
| [06](06-cmykogv.md) | n=7 CMYKOGV experiments | **BLOCKED** | Data from Will via Phil — parked last |
| [07](07-paper.md) | Paper draft, figures, correction paragraph | TODO | Phil call for §correction wording only |

**Cut list** (Phil: skippable if time is limited — not planned): linearization
preprocessing, colorimetric density input domain, genetic algorithms
(Nelder-Mead/Powell are covered inside plan 05).

Done before this plan set existed (commits `a0c937a`, `afec865`):
v1 evaluation corrected + published state tagged (`aic2025-published`);
journal pipeline built (5-fold CV, per-fold scalers, ΔE00 on denormalized XYZ,
Lab-roundtrip tripwire); first results for 6 variants × 14 models in `journal/results/`.
