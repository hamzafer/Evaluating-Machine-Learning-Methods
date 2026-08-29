# Reader's guide to the paper

Written for the lead author to read alongside the PDF. Plain language, every claim paired with the
evidence behind it and the question a reviewer would ask.

---

## The argument in six sentences

1. Printers need a model mapping ink percentages to measured colour; the classical tool is
   polynomial regression, and the question is whether machine learning does better.
2. Across three coated CMY(K) datasets, a Gaussian process predicts colour about 5-10x more
   accurately than a third-order polynomial.
3. That advantage persists as ink count grows to 5 and 7 colorants, which is the regime where
   lookup tables and classical models are known to struggle.
4. It also holds on a different printing process (newsprint), though press-to-press variation, not
   model error, dominates once you try to transfer between presses.
5. Optimising the CIEDE2000 error directly, instead of squared error on XYZ, cuts worst-case error
   substantially without helping the median.
6. Large language models, asked to produce a portable equation for the same task, do not come close
   to simply fitting a polynomial unless they are given a code interpreter, in which case they are
   doing numerical fitting rather than reasoning.

---

## Claim by claim: what backs it, and what could be attacked

### Claim 1 — the Gaussian process is the most accurate method at n<=4
**Evidence:** 5-fold cross-validation on 795 (CMY) and 1,588 (CMYK) patches per dataset, every
sample predicted exactly once. PC10: GP 0.046 vs poly3 0.268 at n=3, GP 0.056 vs poly3 0.942 at n=4.

**Numbers you must know:** GP is 0.041-0.070 across all six coated conditions. That is below typical
print measurement repeatability, so the model is more consistent than the printer.

**The corrected size of the advantage (know this, the old "5-10x" was wrong):** against the *best*
polynomial at any degree and space, the GP is **2.0-2.4x** better at n<=4, not 5-10x. Against the
corrected degree-3 CIELAB baseline it is 3.3-8.2x. The 5-10x figure only holds against the
uncorrected degree-3-in-XYZ baseline, which is exactly the comparison this paper argues is unfair.

**Likely challenge:** "Is it overfitting?" No. Grouped cross-validation was verified against plain
k-fold and agrees within 0.003, and an independent clean-room reimplementation reproduced the GP
medians to within 0.003 on every coated dataset.

### Claim 2 — the advantage persists at 5 and 7 inks
**Evidence:** KCMYG (5 inks, 2,214 patches), APTEC CMYKOGV (7 inks, 3,302 effective), Apex CMYKOGB
(7 inks, 2,000). Reported per dataset, never pooled, because they are different printing systems.

**Likely challenge:** "The three datasets are not comparable, so your ladder is not a controlled
sweep." Correct, and the paper says so explicitly. The claim is about the within-model trend across
independent systems, not about levels between them.

### Claim 3 — the polynomial baseline was handicapped, and correcting it narrows the gap
**This is the paper's most original content and the part you must understand best.**

The polynomial was fitted to XYZ. Fitting it instead to the cube root of XYZ, which is
mathematically identical to fitting in CIELAB, improves the maximum error on all nine datasets and
the median on eight. At 7 inks the degree-3 polynomial goes from 5.386 to 0.830.

Adding degree as a second lever: at degree 4 in CIELAB it reaches 0.272, against the GP's 0.249.

**Why it works:** the ink-to-colour response is simply closer to a low-order polynomial after a
compressive transform. It is *not* because least squares in that space "aligns with CIEDE2000" —
that explanation was tested with a y^(-4/3) weighting control and refuted.

**Why cube root specifically is not the point:** square root is better on four of nine medians. The
paper says so rather than over-claiming.

**Is it novel?** No, and the paper must not pretend otherwise. Fitting characterization models in a
perceptually uniform space is long-standing practice. The contribution is honesty: our baseline, and
the literature default we inherited, was fitted in the wrong space, and correcting it changes the
size of the ML advantage.

**Likely challenge, and the one that would hurt most if unanswered:** "Your headline compares a
well-tuned GP against a badly-configured polynomial." The paper now answers this pre-emptively by
reporting the corrected baseline, and by offering the same correction to the GP (which also improves,
to 0.160 at 7 inks).

**Where it leaves the comparison:** GP still wins on 8 of 9 datasets, but at 7 inks the margin is
0.160 vs 0.272, a factor of 1.7, not the factor of 21 the uncorrected comparison suggested. On
KCMYG-5 and CMYKOGB-7 the corrected polynomial beats the GP's original numbers.

### Claim 4 — newsprint, and what generalises between presses
**Evidence:** 13 IFRA white-backing press runs, 1,485 patches each. Within a single run, GP 0.674 to
2.141 (median-of-medians 0.899) and it ranks first of fourteen models on 12 of the 13 runs. Training
on one press and testing on another gives 3.844, i.e. **press-to-press variation dominates model
error by roughly 4x**.

**The story worth telling:** a model characterises a press, not a process. That is a practically
useful negative result.

**Known caveat in the paper:** in the leave-one-out condition the GP fits a 2,000-row subsample while
its competitors fit all 17,820 pooled rows, so that column is a lower bound for the GP and is
labelled as such.

### Claim 5 — optimising ΔE00 directly
**Evidence:** Powell optimisation of a cubic against CIEDE2000 instead of squared error. Medians
barely move; maximum error falls 16-51%. Nelder-Mead is not usable at high ink counts: it exhausts
its evaluation budget having barely moved off the least-squares start, so it silently returns
approximately the least-squares fit.

**Why it matters:** if worst-case colour accuracy is what you care about, the objective function
matters more than the model family.

### Claim 6 — the LLM experiment
**Frame it as exploratory, not a benchmark.** One run per model, conditions differ between rows, and
only the 3-ink chart.

| condition | median ΔE00 |
|---|---|
| Claude Fable 5, web, **with code interpreter** | 0.082 |
| our least-squares cubic, same 150 training rows | 0.234 |
| GPT-5.6 Sol, API, reasoning | 3.070 |
| Claude Opus 5, API, reasoning disabled | 14.291 |
| DeepSeek V4 Pro, API, reasoning disabled | 23.764 |
| Haiku 4.5, API, reasoning | 28.781 |

**Three things to be able to say:**
- Unaided, the best LLM equation is 13x worse than fitting a cubic to the same 150 rows.
- With a code interpreter one beat our baseline, but that measures automation, not colour knowledge.
- With reasoning enabled, several models exhausted their entire completion budget without producing
  an equation. **Archived evidence:** Fable 5 at 1,600 and at 78,000 tokens, DeepSeek V4 Pro at 2,400
  and 8,000. Opus 5 also did this at 8,000 and 24,000 tokens (costing $0.18 and $0.63 for nothing),
  but its raw response was later overwritten by the successful reasoning-disabled run, so that pair
  is evidenced only in `docs/DECISION_LOG.md`, not in the archive. Cite the archived cases.
  GPT-5.6 Sol converged in 2,747. The transcripts show careful work (Neugebauer models, dot-gain
  curves, least-squares normal equations) that simply never terminates.
- Every accurate LLM equation violated the "as simple as possible" instruction: expanded degrees of
  9 and 27, thousands of terms.

**Likely challenge:** "n=1, and non-deterministic models." Concede it immediately. The section is a
probe, and the non-termination observation is the durable part because it reproduced across three
budgets and two models.

---

## The things a careful reviewer will go after

1. **The degree cap.** The paper caps polynomials at 3rd order on the grounds that higher degrees
   overfit. At 7 inks that cap is doing a lot of work, and we have evidence degree 4 does *not*
   overfit on a 3,302-patch chart (train/test gap +0.027, unchanged from degree 3). If the cap stays,
   the paper needs to justify it explicitly. **Discuss with Phil before submitting.**
2. **Cross-platform reproducibility.** Same code, same pinned versions, different CPU: medians agree
   to about 0.03 for closed-form and kernel methods but drift up to 0.42 for neural nets. The paper
   states results are pinned to one platform. That is unusual to admit and it is a strength.
3. **Duplicate rows.** Byte-identical duplicate measurements are removed before splitting, uniformly,
   because a decision tree scored exactly 0.000 on such rows before the fix. Analysed counts are 795
   and 1,588 rather than the raw 818 and 1,617.
4. **The AIC 2025 correction.** The conference version computed ΔE00 on normalised XYZ, understating
   errors roughly fivefold. The paper says so plainly in one paragraph. Do not bury it.

---

## What you personally need to be ready to answer

- Why is a Gaussian process better here? Because it interpolates smoothly through densely sampled
  measurements without assuming a global functional form, and printer response is smooth but not
  polynomial.
- Why report median, P95 and maximum rather than mean and standard deviation? Colour errors are not
  normally distributed and a few large outliers dominate a mean.
- What is the practical recommendation? For a single press with a good measurement set, a Gaussian
  process. If a portable closed-form model is required, a degree-4 polynomial fitted in CIELAB gets
  within a factor of two on 7-ink data.
- What is genuinely new? Three things: the n=5 and n=7 evidence, the finding that the classical
  baseline's failure was substantially a fitting-space and degree artifact, and the LLM
  non-termination observation.
