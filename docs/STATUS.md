# Status — morning of 13 Aug 2026

Written at the end of an overnight session. Code repo HEAD `d64a8ab`, paper repo `b23491d`,
working tree clean, 41 tests green, `journal/results/run_log.tsv` at 567 recorded fits.
**5 commits unpushed** (`git push origin main`).

## Where the paper stands

Every experiment except the LLM prediction table is complete, gated by a review agent, and
independently verified. Plans 01-06, 10, 11 and 09 are done; 08 is partial (see below).

### The headline: ink count vs method

Median ΔE00, final post-dedup numbers, per dataset (never pooled — different printing systems):

| n | dataset | Gaussian process | poly3 |
|---|---|---|---|
| 3 | PC10-CMY | **0.046** | 0.268 |
| 4 | PC10-CMYK | **0.056** | 0.942 |
| 5 | KCMYG | **0.867** | 1.457 |
| 7 | CMYKOGV | **0.249** | 5.386 |
| 7 | CMYKOGB | **1.280** | 3.332 |

Polynomial regression degrades by roughly 12-20x from 3 to 7 inks; the GP stays under 1.3
everywhere. That answers Phil's question (b) — yes, ML handles n>4 — and it answers it on two
independent 7-ink systems.

### Newsprint (IFRA), and the anomaly that turned out to be an optimizer trap
GP within-run is now **0.674-2.141** across all 13 press runs (median-of-medians 0.899) and ranks
first of fourteen models on 12 of them. It previously looked like the *worst* model at ~19 ΔE00.
Root cause was a local-optimum trap seeded by a near-zero WhiteKernel init, not a GP limitation;
the fix is one documented kernel configuration used for every dataset in the paper.

### External validation
The colourbill/CharData tool ships four of our exact datasets. It independently reproduces our
CMYKOGV deduplication (3534 -> 3302) and confirms our CIEDE2000 arithmetic to displayed precision,
and our cross-validated error beats its in-sample degree-4 fit on the n=4 charts.

### Reproducibility (the strongest part of the methods section)
- **An independent blind reimplementation** (separate machine, raw files plus a prose spec, no sight
  of our code) reproduces GP to within 0.003 ΔE00 on every coated dataset and poly3 to within 0.006.
  Its report, its results and all of its code are committed at `journal/verification/blind-2026-08-12/`.
- **Cross-platform**: 58 of 96 model x dataset cells bit-exact between macOS/arm64 and Linux/x86_64
  under an identical pinned environment; 76 within 0.01. Every large divergence is an MLP. All 182
  IFRA GP results are bit-exact across platforms.
- **Seed sensitivity**: max std of the median across 5 CV seeds is 0.0012 (GP) to 0.0906 (deep MLP).

### The defect the blind reviewer found, and the fix
The n<=4 datasets ran ungrouped, so byte-identical duplicate rows could straddle a train/test split
— contradicting the protocol the paper stated. Measured directly: a decision tree scored **exactly
0.0000** on those rows while scoring 4.4 on the rest. Fixed by one uniform policy (byte-identical
duplicates dropped at load, everywhere), which changed analysed counts to **795 / 1588** and moved
every n<=4 number by <=0.19. No conclusion changed; GP still wins all six datasets. Full evidence in
`docs/research/cv-leakage-2026-08-12.md`.

## LLM track

### Plan 09 (equation generator) — done, and the finding is sharp
Phil's verbatim prompt, 150 in-prompt training pairs, scored on 100 held-out patches with our own
ΔE00 code:

| source | median ΔE00 | terms | total degree | max per-variable exponent |
|---|---|---|---|---|
| our least-squares cubic | **0.234** | 57 | 3 | 3 |
| gpt-5.6-sol | 3.070 | 192 | **9** | 3 |
| deepseek-v4-pro | 23.764 | 39 | 3 | 3 |
| claude-fable-5 (API) | no answer | — | — | — |

Two things worth saying in the paper:
1. **No LLM-written equation came close to simply fitting a cubic** to the same rows — the best was
   ~13x worse on the median and 3x worse on the max, so Phil's criterion (minimise average *and*
   maximum) is met by none of them.
2. **They satisfy the simplicity constraint by gaming it.** Asked for "exponents no greater than 3,
   as simple as possible", gpt-5.6-sol multiplied three cubics (per-variable exponent 3, total degree
   9, 192 terms). The Fable-via-harness run nested a cubic inside a cube (per-variable exponent 9).
   The more accurate the answer, the more thoroughly the constraint was circumvented.

### The harness experiment, and why its result is quarantined
Because the API attempts for Claude failed, the identical prompt went through the Claude Code CLI on
the subscription. It produced the most accurate equation of the study (median **0.215**, beating our
own cubic) — but the archived session shows **5 Bash and 4 Write calls**: `--allowed-tools ""` did
not disable tools, so it wrote code and fitted the coefficients numerically. That measures an agent
with a code interpreter, not a model writing an equation. Recorded as a separate finding in
`journal/results/llm/harness_channel_note.md`, not as Claude's answer. It is good evidence that an
LLM benchmark without a stated channel is uninterpretable.

### Plan 08 (prediction table) — not run
Credit ran out ($0.42 spent of $0.66; **$0.24 left**). The old GPT-4o rows are retired as superseded.

## What needs Hamza

1. **`git push origin main`** — 5 commits.
2. **Two paid calls, both classifier-blocked pending your go:**
   - Claude Fable 5 equation via API with a real reasoning budget (~$0.25). Claude's genuine
     equation-writing ability is currently **untested**, and Phil considers testing Claude mandatory.
   - DeepSeek re-run with reasoning enabled (~$0.02). Its committed row was produced with reasoning
     off and 371 completion tokens, so its coefficients are written from memory, not fitted — the CSV
     says so plainly, but it is not a fair test of that model. Config is already changed and ready.
3. **~$3 of OpenRouter credit** for the 200-sample prediction table across the three models.
4. **Paper trim**: 17pp against the 10-12pp target. Table surgery was tried and measured — it saves
   nothing (floats reflow), so the pages have to come from prose. Methods is 3.5pp and is the safest
   1.5pp; the rest needs your editorial calls. Page map is in the plan-07 notes.
5. **Two questions for Phil**: the PC10 file's header declares itself `APTEC_PC11_CCNB_2023_v1`
   (data is genuinely PC10's, so it looks like a copy-paste artifact at source, but our dataset table
   calls PC10 "cardboard" on the strength of the filename); and KCMYG's provenance is still
   unconfirmed.

## Known debt
`docs/TECH_DEBT.md` — chiefly that Phil's unpublished datasets are recoverable from this public
repo's git history (untracked at the tip, but present in a pushed commit), deferred to after
publication by your call.
