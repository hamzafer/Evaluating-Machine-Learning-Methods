# First results: CMY / CMYK regression across PC10, PC11, FOGRA51

> STALE NUMBERS (12 Aug 2026): every n<=4 figure in this document predates two changes —
> the unified GP config (plan 10) and the uniform exact-dedup policy that fixed the CV
> duplicate-leakage defect (`docs/research/cv-leakage-2026-08-12.md`). Sample counts are now
> **795** (CMY) and **1588** (CMYK), not 818/1617, and every median here has moved. The
> narrative and mechanisms remain valid; for current values read
> `journal/results/*/summary.csv` or `journal/results/run_log.tsv`.

> UPDATE (11 Aug 2026): IFRA black-backing (bb) was subsequently DESCOPED ENTIRELY (out of scope — a separate 'substrate correction' problem), not merely quarantined pending a data fix. wb-only going forward.


Source: `journal/results/{PC10,PC11,FOGRA51}-{CMY,CMYK}/summary.csv`. Five-fold CV,
per-sample ΔE00 pooled across all folds (every sample predicted exactly once), computed
on denormalized XYZ. CMY = 818 samples with K=0 (3 inputs); CMYK = all 1,617 samples (4
inputs). Reference scale: ΔE00 < 1 is imperceptible, 2–3.5 is noticeable, > 5 is a
clearly different color; print measurement repeatability is typically ~0.2–0.5 ΔE00.

## 1. Overall ranking

The ranking is remarkably consistent across all six dataset/channel-count combinations.
For every one of the six tables the order of the top four is identical:

**gaussian_process < poly3 < svm < gradient_boost**

and mlp_deep consistently beats mlp_shallow, which consistently beats the
random_forest/knn pair, which consistently beats decision_tree, which consistently beats
the block of five linear-family models (lasso, ridge, elastic, pcr, plsr).

The one rank flip is random_forest vs. knn, and it flips with a pattern tied to input
count rather than dataset: random_forest is the better of the two at n=3 (e.g. PC10-CMY
1.716 vs. 2.012; PC11-CMY 1.694 vs. 1.938; FOGRA51-CMY 1.804 vs. 1.867) and knn becomes
the better of the two at n=4 (PC10-CMYK 2.233 vs. 2.274; PC11-CMYK 2.135 vs. 2.244;
FOGRA51-CMYK 2.151 vs. 2.567). This is a small, second-order effect (the two are within
0.1–0.4 ΔE00 of each other everywhere) but it is consistent across all three datasets, so
it is real rather than noise.

Within the bottom linear cluster, the five models (lasso/ridge/elastic/pcr/plsr) differ
from each other by only 0.01–0.05 median ΔE00 in every table — the same order of
magnitude as the sklearn-version sensitivity noted in the repo's own rules. That
ordering should not be over-interpreted or reported as if one regularizer "beats"
another; the honest statement is that all five are statistically indistinguishable and
fail the same way (see §4). Note also that `pcr` and `ridge` are identical to three
decimals in every one of the six tables (e.g. PC10-CMY: both 6.624 / 24.802 / 43.82 /
9.026). This is not a coincidence: `journal/pipeline/models.py` implements `pcr` as
`PCA()` (default: all components retained, no whitening) followed by `Ridge(alpha=0.5)`,
and an un-truncated, unwhitened PCA is just an orthogonal rotation of the input space —
ridge regression is invariant to orthogonal rotations of X, so `pcr` as currently
configured is mathematically equivalent to `ridge` and contributes no independent
information. This should be fixed (either truncate components for a real PCR baseline,
or drop it as redundant) before the two are reported as separate methods.

So: yes, the ranking is dataset-consistent. GP and low-order polynomial regression
dominate, classical black-box regressors (SVM, boosting, MLPs) form a solid middle
tier, tree-based methods and knn trail them, and pure linear models are uniformly poor
regardless of regularization flavor.

## 2. The Gaussian Process result

GP gives a median ΔE00 of 0.054–0.072 across all six tables — below print measurement
repeatability (~0.2–0.5) and far below the imperceptibility threshold of 1. Its p95 is
also small (0.16–0.47) but its max is not trivial (0.81–8.37), so a small number of
samples are predicted much worse than the median suggests (see §3).

This result should be treated as provisional. The evaluation code
(`journal/pipeline/evaluate.py`) currently does a plain shuffled `KFold` with no
awareness of duplicate input recipes. A check of the underlying data shows this matters:
across all three datasets, 58 of 1,617 CMYK rows (3.6%) share an identical CMYK input
tuple with at least one other row, and 46 of the 818 K=0 rows (5.6%) share an identical
CMY tuple with another row. Under plain KFold, a duplicate pair can be split so that one
copy sits in the training fold and its twin sits in the test fold. A GP with a smooth
kernel (and, more severely, memorizing models like knn/decision tree) can then score
near-zero error on that twin by effectively looking up the training point rather than
generalizing — inflating the pooled median.

A duplicate-aware fix is already present but not yet applied: `evaluate.py` in the
working tree (uncommitted) adds a `make_groups()` helper and a `GroupKFold` path to
`cross_validate()`, verified by a unit test
(`journal/pipeline/tests/test_evaluate.py::test_grouped_cv_keeps_duplicates_out_of_train`)
that reproduces exactly this memorization failure mode with 1-NN and confirms grouped CV
blocks it. However, `journal/pipeline/run.py` still calls `cross_validate(X, Y,
m_reg[m_name])` without passing `groups=`, and the `summary.csv` files were written
before the `evaluate.py` change (11:15 vs. 12:23 by file mtime). In short: **the numbers
in these six summary files were generated under plain (non-grouped) KFold**, and the
duplicate-aware CV path exists but has not yet been wired into `run.py` or used to
regenerate results.

Given that duplicates are a small minority of samples (3.6–5.6%), grouped CV is unlikely
to move the GP median off the "imperceptible" range entirely, but it could plausibly move
it from ~0.06 to something higher, and it will more directly affect the models most prone
to memorization (knn, decision_tree, and to a lesser extent random_forest). The GP
result — and the exact ranking among GP/poly3/svm — should be re-run with grouped CV
before being presented as a finding, and the polynomial/SVM/GP numbers here should be
labeled provisional until that re-run is done.

## 3. Behavior from n=3 (CMY) to n=4 (CMYK), by family

Adding the K channel changes every model's median only modestly (roughly ×1.05 to
×1.7), but it changes the tail (p95, max) very differently by family:

- **Polynomial (poly3)** shows the sharpest tail degradation of any model. Median grows
  ~2.2–3.6× (PC10: 0.279→0.944; PC11: 0.244→0.868; FOGRA51: 0.368→0.822), but p95 grows
  ~5.7–8.7× (PC10: 0.966→8.154; PC11: 0.89→7.724; FOGRA51: 1.163→6.643) and max grows
  ~3.5–4.5× (PC10: 7.695→32.059; PC11: 7.107→32.223; FOGRA51: 8.008→28.377). A 3rd-order
  polynomial has 20 terms in 3 inputs but 35 in 4 inputs, and the extra cross-terms
  involving K appear to blow up on some subset of samples rather than uniformly
  degrading fit — worth checking whether these worst cases cluster in high-K
  (near-black/rich-black) patches, which is exactly the region CLAUDE.md flags as
  needing careful handling once K is included.
- **Gaussian process** stays in the imperceptible-median range at n=4, but its tail also
  grows substantially in relative terms: p95 grows ~1.5–2.7× (PC10: 0.189→0.473; PC11:
  0.16→0.435; FOGRA51: 0.191→0.292) and max grows ~3.8–4× (PC10: 2.061→8.37; PC11:
  1.797→6.796; FOGRA51: 0.807→3.248). So GP is not immune to the added dimensionality —
  it is simply starting from such a low base that the degraded tail is still small in
  absolute terms.
- **SVM, gradient boosting, both MLPs** all show a similar pattern to poly3 but softer:
  medians up ~1.3–1.7×, p95 up ~2–3×, max up ~1.2–3.6×. mlp_shallow's max is essentially
  unchanged in two of three datasets (PC10: 21.15→29.006 is up, but FOGRA51:
  18.983→28.828), so this family's tail growth is present but not as extreme as poly3.
- **Tree-based methods (random_forest, decision_tree)** are comparatively insensitive to
  the extra input. decision_tree's p95 and max barely move (PC10: p95 9.828→9.679, max
  25.211→24.345 — essentially flat) and random_forest's p95 grows only ~1.2× with max
  actually *improving slightly* in one case (PC10: 9.264→9.182). Axis-aligned splitting
  on one more feature doesn't destabilize these models the way an added polynomial
  cross-term or kernel dimension does.
- **Linear models** already have a saturated tail at n=3 (p95 ~23–25, max ~42–45) and it
  barely moves at n=4 (p95 stays ~24–26, max stays ~42–43). Adding K doesn't make the
  linear fit's worst case meaningfully worse because it's already about as wrong as it
  can be on the hardest samples (see §4) — there's no headroom left to lose.

Net picture for the paper: methods that fit smooth, high-capacity functions in the
augmented input space (poly3 most of all, then MLP/SVM/GBM) pay a real, quantifiable
tail-behavior cost for going from 3 to 4 inputs, even when their median error looks fine.
GP pays a smaller version of the same cost. Tree ensembles are the most tail-robust to
the extra dimension. This is a genuinely useful finding for the "can ML handle more
inputs" question and should be reported with p95/max, not median alone — reporting only
the median would make poly3 look almost as robust as GP, which the tail numbers show is
false.

## 4. Why linear models fail

All five linear-family models land at median 6.28–6.72 ΔE00 for CMY and 8.47–8.94 for
CMYK — solidly in "clearly different colors" territory, roughly 15–20× repeatability. The
mapping from ink coverage to XYZ is fundamentally not linear: subtractive color mixing
in print follows something closer to a Beer–Lambert/Kubelka–Munk-style exponential
attenuation of reflected light, compounded by dot gain and ink trapping, so density and
tristimulus values saturate nonlinearly as ink coverage approaches 0% and 100%. A
first-order model can only fit the average slope through this curve; it will
systematically over- or under-shoot in the shadows and highlights where the curve bends
away from a straight line.

The shape of the error distribution confirms this is a *concentrated* failure, not a
uniform one: median (~6.6–8.9) is far below p95 (~23–26), which is itself far below max
(~42–45, close to the largest ΔE00 values seen anywhere in these tables). A model that
was uniformly biased would show p95 and max closer to a small multiple of the median;
instead the linear models are roughly fine-ish on the bulk of mid-tone samples and
catastrophically wrong on a tail — almost certainly the darkest/most saturated patches,
where the true CMY(K)→XYZ curve is most sharply nonlinear and where ΔE00's hue/chroma
terms are also most sensitive to small absolute errors at low L*. This is consistent
with all five linear variants (different regularization, same underlying linear form)
failing identically: regularization strength changes the fit only at the margins, it
cannot add curvature the model doesn't have.

## 5. Anomalies to investigate before publication

1. **pcr ≡ ridge.** As described in §1, `pcr` in `journal/pipeline/models.py` is
   `PCA()` (all components, unwhitened) + `Ridge(alpha=0.5)`, which is mathematically
   identical to plain `Ridge(alpha=0.5)` on the original features. The identical
   three-decimal results in every table are not a coincidence but a consequence of this
   implementation choice. Fix before reporting PCR as an independent baseline.
2. **GP/poly3/svm results are pre-grouped-CV.** As detailed in §2, all six summary files
   were generated with plain shuffled KFold, while a duplicate-aware `GroupKFold` fix
   already exists in the working tree but is not wired into `run.py` and was not used to
   produce these numbers. 3.6–5.6% of samples in each dataset have an exact-duplicate
   input twin elsewhere in the dataset. Re-run all six configurations with
   `groups=make_groups(X)` before treating the GP (and secondarily knn/decision_tree)
   numbers as final — memorization-prone models are the ones most likely to move.
3. **Max ΔE00 clustering around 42–45 for every linear model, in every dataset.** This
   ceiling recurs too consistently (PC10-CMY 43.82–43.961, PC11-CMY 45.184–45.338,
   FOGRA51-CMY 41.892–42.062, and similarly ~42–43 for all three CMYK tables) to be six
   independent coincidences. It's worth confirming whether these worst-case samples are
   the same physical patches across datasets/models (e.g., a specific near-black or
   heavy-ink swatch that's genuinely hard, or an edge case in the ΔE00 formula near
   L*≈0 where hue/chroma terms are numerically unstable) before citing these maxima as
   meaningful "worst case" numbers rather than an artifact of one or two extreme points.
4. **poly3's tail blow-up under CMYK** (p95 growing 5.7–8.7× vs. a median growth of only
   2.2–3.6×, see §3) is large enough that it's worth checking whether specific K-heavy
   samples are driving it, both to understand the mechanism and to make sure it isn't a
   numerical conditioning issue in the degree-3 feature expansion (35 terms on a
   0–1-scaled 4-dimensional input) rather than a genuine modeling limitation.
5. **random_forest vs. knn rank flip tied to n** (§1) is small in magnitude (≤0.4 ΔE00)
   but consistent across all three datasets, so it's plausibly real rather than noise;
   still worth a sanity check (e.g. same-seed re-run) given how close the two are.

None of points 1–5 change the qualitative story (GP and low-order polynomial regression
lead, linear models fail badly, tree ensembles are tail-robust) but points 1 and 2 in
particular affect numbers that would otherwise go straight into the paper's headline
table, and should be resolved first.

---

# 2026-08-01: Direct ΔE00 loss, LLM predictor, IFRA newsprint generalization

Source for this section: `journal/results/{PC10,PC11,FOGRA51}-CMY/summary.csv`
(poly3 vs. poly3_de00_nm/powell), `journal/results/llm/PC10-CMY_summary.csv`,
`journal/results/ifra/{within_run,cross_run,leave_one_out}.csv`, and a direct
re-fit/diagnostic of the Gaussian Process model (`journal/pipeline/models.py`)
run against `journal/pipeline/datasets.py` to confirm the within-run anomaly's
root cause. All numbers below were read from the actual CSVs or freshly
computed against the pipeline's own code, not estimated.

## 6. Direct ΔE00-loss optimization vs. least-squares poly3

`journal/pipeline/de00_poly.py`'s `DE00Polynomial` fits the same degree-3
polynomial form as `poly3`, but instead of ordinary least squares on XYZ, it
minimizes CIEDE2000 directly via a derivative-free optimizer — Nelder-Mead
(`poly3_de00_nm`, maxiter=2000) or Powell (`poly3_de00_powell`, maxiter=200)
— starting from the LSQ solution. All three CMY variants:

| dataset  | model              | median | p95   | max    |
|----------|--------------------|--------|-------|--------|
| PC10-CMY | poly3 (LSQ)        | 0.279  | 0.966 | 7.695  |
|          | poly3_de00_nm      | 0.264  | 0.982 | 6.939  |
|          | poly3_de00_powell  | 0.274  | 0.886 | 5.513  |
| PC11-CMY | poly3 (LSQ)        | 0.244  | 0.890 | 7.107  |
|          | poly3_de00_nm      | 0.235  | 0.847 | 6.162  |
|          | poly3_de00_powell  | 0.255  | 0.772 | 5.735  |
| FOGRA51-CMY | poly3 (LSQ)     | 0.368  | 1.163 | 8.008  |
|          | poly3_de00_nm      | 0.372  | 1.170 | 5.783  |
|          | poly3_de00_powell  | 0.375  | 1.043 | 4.135  |

Powell's worst-case reduction vs. LSQ poly3: **28.4% (PC10), 19.3% (PC11),
48.4% (FOGRA51)** — a genuine, dataset-dependent but consistently large cut
in the tail. The median, meanwhile, is flat to marginally *worse* under
direct ΔE00 optimization (PC10 −1.8%, PC11 +4.5%, FOGRA51 +1.9% relative to
LSQ). Nelder-Mead shows the same direction of effect but weaker and less
reliable: it improves p95 on PC11 (0.890→0.847) but *worsens* it on PC10
(0.966→0.982) and FOGRA51 (1.163→1.170), while its max reduction (9.8%,
13.3%, 27.8%) never beats Powell's on the same dataset.

**Why the tail moves and the median doesn't.** Ordinary least squares
minimizes squared error in raw XYZ space, which is a metric-agnostic,
roughly uniform objective across the input domain. CIEDE2000 is not
uniform: it re-weights lightness, chroma and hue differently depending on
where in Lab space a pair of colors sits, and (as already established in §3
above) that re-weighting diverges most sharply from a flat XYZ metric in
exactly the region a fixed-degree polynomial already struggles with — dark,
saturated, high-ink-coverage patches. For the bulk of mid-tone samples,
where ΔE00 and squared-XYZ-error broadly agree on what "close" means, LSQ is
already doing about as well as any refit can do — there's no daylight
between the two objectives there, so the median can't improve much (and can
occasionally get slightly worse, since the optimizer is now explicitly
trading a little bit of typical-case fit for a lot of worst-case fit). In
the tail, where the two metrics disagree substantially, directly optimizing
the metric the paper actually reports — rather than a proxy — lets the
degree-3 polynomial's fixed, limited coefficient budget get reallocated
toward the specific patches that are perceptually expensive rather than
merely numerically large in XYZ. That reallocation is a real win worth
reporting (worst-case ΔE00 is what determines whether a printer profile
ever produces a genuinely bad color, not the median), but it is a tail
effect, not a central-tendency effect, and both should be reported plainly
as such rather than as "direct-loss training improves accuracy."

One secondary point worth a line in the paper: Powell out-performs
Nelder-Mead here despite a 10x smaller iteration budget (200 vs. 2000). A
degree-3 polynomial over 3 inputs has 20 coefficients per output channel ×
3 channels = 60 free parameters; Nelder-Mead's simplex is well known to
degrade in reliability well before 60 dimensions, while Powell's
coordinate-wise line-search scales more gracefully. That the cheaper
optimizer gives the more consistent (never-worse-than-LSQ) p95/max result is
a methods point, not a coincidence, and should inform which optimizer is
used if this approach is extended to CMYK (35 coefficients/channel) or
n>4.

## 7. LLM as color predictor: gpt-4o is mid-field, gpt-4o-mini is not competitive

`journal/llm/run_llm.py` queries OpenAI chat models with a text prompt
containing 400 in-context (recipe → XYZ) training examples and asks for a
JSON XYZ prediction on 100 held-out CMY recipes from PC10 (`build_split` in
`journal/llm/protocol.py`, seed=42, exact-duplicate recipes dropped before
splitting). Results (`journal/results/llm/PC10-CMY_summary.csv`):

| model        | median | p95    | max    | n parsed/total |
|--------------|--------|--------|--------|-----------------|
| gpt-4o       | 3.034  | 10.642 | 14.621 | 100/100 |
| gpt-4o-mini  | 9.445  | 29.928 | 44.792 | 100/100 |

Both models parsed cleanly on every one of the 100 queries (no penalty
applied; `worst_case` and `parsed_only` rows are identical), so these are
genuine prediction errors, not parse-failure artifacts.

Placed against the full PC10-CMY leaderboard (5-fold CV, n=818, from §1):
gpt-4o's median of 3.034 lands **between random_forest (1.716)/knn (2.012)
and decision_tree (4.369)** — worse than every regression method tested
except decision_tree and the five linear models, but clearly better than
plain linear regression (6.6+) and much better than gpt-4o-mini.
gpt-4o-mini's median (9.445) is worse than *every* classical method on this
dataset, including the linear-model floor (6.6–6.64) — it is, on this
sample, the single worst predictor tested against PC10-CMY so far. The
honest one-line summary: **a strong general-purpose LLM, given nothing but
400 text examples and no gradient-based fitting, lands solidly in the
middle of a 14-method regression leaderboard — not competitive with GP,
poly3, SVM, gradient boosting or either MLP, but decisively ahead of raw
linear regression and roughly on par with decision trees.** A smaller/cheaper
model (gpt-4o-mini) is not a viable predictor at all here.

**The CV-vs-holdout caveat, stated plainly.** The classical/GP numbers in
this doc are pooled 5-fold CV results over the *entire* 818-sample CMY
dataset — every sample predicted exactly once, by a model that never saw it
during fitting, aggregated over 5 independent train/test splits. The LLM
numbers are a *single* random 100-sample holdout, "fit" only by pasting 400
example rows into a static prompt (no refitting, no folds, no repeated
draws). This is a much noisier estimate: with only 100 test points, the p95
figure is anchored on roughly 5 samples, and a different seed's 100-row
draw could shift the median and especially the tail meaningfully. The
comparison above (gpt-4o sitting between random_forest and decision_tree) is
a reasonable, useful signal — but it should be reported as "a single-draw
estimate, consistent with mid-field performance," not with the same
statistical confidence as the CV numbers, and ideally re-run over multiple
seeds/folds before being placed in the same results table as the
cross-validated methods.

## 8. IFRA newsprint generalization: press variation dominates, pooling helps

`journal/pipeline/run_ifra.py` runs three experiments on the 13 valid wb
press runs (1,485 samples each; bb is currently quarantined — see §9):

- **A. Within-run** — ordinary 5-fold CV inside a single run (train and test
  on the same press condition).
- **B. Cross-run** — fit on one full run (1,485), test on a *different* full
  run (1,485); all 13×12=156 ordered pairs, subset of 4 models
  (gaussian_process, poly3, svm, mlp_deep).
- **C. Leave-one-run-out (LOO)** — fit on the other 12 runs pooled
  (~17,820 samples), test on the 13th; same 4-model subset.

Excluding Gaussian Process (within-run anomalous under the pre-Plan-10 GP
config — see below, since RESOLVED), the other three subset models give a
consistent story (median-of-medians across all runs/pairs):

| experiment  | poly3 | svm   | mlp_deep | pooled (poly3/svm/mlp_deep) |
|-------------|-------|-------|----------|-------------------------------|
| within-run  | 1.42  | 1.07  | 1.61     | **≈1.4** |
| cross-run   | 4.58  | 4.36  | 4.59  (mean of pair-medians) | **≈4.0** (median-of-medians) |
| LOO         | 3.38  | 3.09  | 3.61  (mean of medians)      | **≈3.0–3.1** (median-of-medians) |

**Interpretation.** Within-run error (~1.4 ΔE00) is what these models can
achieve when asked to predict colors from the *same* press condition they
were trained on — essentially the achievable floor for this substrate,
limited mostly by real press repeatability noise (§9: ~0.6–0.8 ΔE00) plus
whatever the model itself can't capture. Cross-run error (~4.0 ΔE00) is
roughly **2.8× higher** — asking a model fit on one press run to predict an
entirely different run's colors cold is dominated by genuine physical
differences between press conditions (substrate batch, dot gain, ink
behavior, press wear), not by model quality; every one of the four models
tested lands in the same 4.0–4.6 range regardless of method, which is
itself evidence that this is a domain-shift ceiling rather than a
fitting problem. Pooling training data across runs (LOO: train on 12 runs
at once instead of 1) recovers a meaningful chunk of that gap — cross-run
≈4.0 → LOO ≈3.0, roughly a **25% relative reduction**. Seeing many
different presses' realizations of the same nominal recipe grid lets a
model start to average over press-to-press idiosyncrasy instead of
committing to one press's specific behavior, which is directly useful
evidence for the multi-printer generalization question this dataset was
acquired to answer: pooling helps, measurably, but it does not fully close
the gap back to within-run accuracy (LOO's ~3.0 is still roughly 2× the
within-run ~1.4 floor) — cross-press generalization has a real cost of its
own, beyond ordinary regression error, that more pooled training data
narrows but does not eliminate.

**The Gaussian Process within-run anomaly (RESOLVED in plan 10 — see the
addendum at the end of this subsection; the analysis below is the historical
diagnosis that led to the fix).** Under the *original* config, GP's within-run
median was **16.64–20.07 ΔE00 across all 13 wb runs (mean 18.75)** — worse by
an order of magnitude than the other three models on the identical splits, and,
unlike everywhere else in this project, GP was the *worst* model rather than the
best. Under the final unified config (`WhiteKernel(1e-3, bounds 1e-9…1e5)`,
`n_restarts_optimizer=15`) the same 13 runs give **0.674–2.141** (median-of-medians
0.899) and GP ranks first of fourteen models on 12 of the 13 runs. The tables and
median-of-medians figures in this subsection are therefore HISTORICAL; the current
numbers live in `journal/results/ifra/within_run.csv` and
`journal/figures/fig_ifra_generalization.png`. The collapse was systemic
(present in all 13 runs, not one outlier), which is what made it diagnosable:

1. Refitting `journal/pipeline/models.py`'s exact GP config
   (`ConstantKernel()*RBF()+WhiteKernel(1e-5)`, `normalize_y=True`) on one
   within-run fold (`IFRA-wb-Age_64a_wb-CMYK`, first `KFold(5, seed=42)`
   split, replicating the pipeline's own per-fold `[0,1]` `MinMaxScaler` on
   both X and Y) gives a fitted kernel of
   `0.989**2 * RBF(length_scale=1e-05) + WhiteKernel(noise_level=0.00217)`
   — **the RBF length_scale has collapsed to sklearn's own lower
   optimization bound.** The same diagnostic on PC10-CMY (a healthy lab
   dataset, identical code path) gives `3.38**2 * RBF(length_scale=0.885) +
   WhiteKernel(1e-05)` — a sane length_scale on the order of the full
   `[0,1]` input range, which is exactly the regime that produces the
   0.05 ΔE00 median already reported in §2.
2. Consequence, verified sample-by-sample: for the Age_64a_wb fold, 4 of
   the first 5 held-out test predictions are **identical to 8 decimal
   places** to the training set's mean XYZ (`[23.189, 23.117, 16.955]`) —
   the model is not predicting the queried recipe at all, it is returning
   the training set's average color regardless of which of the 297 unseen
   recipes it is asked about. Only query points that happen to sit almost
   exactly on top of a training point (inside the now-microscopic kernel
   bump) get a real, recipe-specific prediction.
3. **Root cause:** `WhiteKernel(1e-5)` sets a noise floor far below the
   real point-to-point scatter in this data. §9 below independently
   measures wb's genuine press repeatability at median ≈0.63–0.8 ΔE00
   between two measurements of the *identical* recipe on the *identical*
   press run — small in absolute terms, but far larger than the training
   data's assumed near-zero noise. Faced with nearby-but-not-identical
   training inputs producing meaningfully different outputs, under a noise
   budget that's initialized and constrained to be tiny, the only way the
   marginal-likelihood optimizer can reconcile this is to shrink the RBF
   length_scale until "nearby" points are no longer considered similar at
   all. Once collapsed, any query that isn't essentially an exact
   duplicate of a training point falls outside the kernel's reach and the
   posterior mean reverts to the (`normalize_y=True`-inverse-transformed)
   prior mean — approximately the training set's average color — which is
   badly wrong for almost every specific, non-average newsprint patch.
   This is "prior-mean reversion, off-recipe."
4. **Why cross-run and LOO don't show it, using the identical kernel
   config:** cross-run's mean pair-median for GP is 4.35 — in line with
   poly3 (4.58)/svm (4.36), not anomalous — and LOO's is 3.10, the *best*
   of the four subset models. Both of those experiments test on a
   *different* run's colors, where genuine press-to-press systematic
   difference (§8's ~4.0–4.6 ΔE00 domain-shift ceiling) already dominates
   the error budget for every model tested, GP included. Whatever the GP's
   internal length-scale is doing, a mean-reverted prediction and a
   genuinely-extrapolated one land in a similar ballpark once the "right
   answer" is already that far from any training point for structural
   (domain-shift) reasons rather than noise reasons — so the pathology is
   masked, not absent. Within-run is the one experiment that puts GP back
   in the "dense, smooth, same-domain" regime where it's supposed to
   dominate (and does, spectacularly, on the lab datasets), which is
   exactly why the noise-floor mismatch is maximally exposed there instead.

This is a config/data interaction, not evidence that Gaussian Processes
are unsuitable for this problem.

**RESOLVED (11 Aug 2026, Plan 10).** The Plan-10 diagnosis refined the
root cause: the noise-floor mismatch is real, but it acts at
*initialization* — the `WhiteKernel(1e-5)` init seeds a local-optimum
basin (RBF length-scale collapse, log-marginal-likelihood ~10,000 nats
worse than the healthy optimum) that a single L-BFGS start cannot leave.
The default *bounds* were never the constraint: even the collapsed fit
had already raised its noise level to ~0.002, matching the
duplicate-measured newsprint repeatability (§9). Optimizer restarts
escape the trap reliably, and the same collapse occurred (without
restarts) on coated KCMYG-5, so it was never newsprint-specific. The fix
is the unified, dataset-agnostic GP config now in `models.py`:
`WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e5))` +
`n_restarts_optimizer=15` (+ the unchanged subsample-to-2000). Under it
the within-run GP medians drop from 16.6–20.1 to ~0.7–2.1 ΔE00 — back in
family (and mostly best-in-family), sitting at each run's own
repeatability floor — while the coated/n>4 summaries improve or tie
(PC10-CMY 0.054 → 0.044) and cross-run/LOO stay at their domain-shift
levels. Why 15 restarts, not 10: the re-run's never-worse gate caught
exactly one regression at 10 — the noisiest pooled-LOO fit (marca_133
held out) landed in the collapsed basin because the widened noise bounds
stretch the restart-init draws over 14 decades; 15 restarts (same seed,
so the first 10 draws are unchanged and the optimum is equal-or-better
in likelihood everywhere) restores it to 3.247, identical to the
pre-change registry. All GP rows in `journal/results/` were regenerated
under this final config; full before/after and the LOO provenance note
(the old LOO rows predate the subsample cap) in
`.superpowers/sdd/10-gp-consistency/task-2-3-report.md`.

## 9. The wb duplicate-patch repeatability finding: a genuine newsprint noise floor

Every one of the 13 wb press-run files contains the same 28 repeated
patches — a fixed QC subset of the ECI2002/CGATS chart design, present in
identical form across all 13 runs — meaning the identical nominal CMYK
recipe is measured twice within a single run's own 1,485-row file, at two
different physical chart positions. Computing ΔE00 between each such pair
(364 pairs total, pooled across all 13 runs, using the same
`make_groups`/pairwise machinery as `journal/pipeline/verify_gp.py`):
**median 0.634, p95 2.812, max 8.260, mean 0.998**, with per-run medians
spanning 0.383 (Mbd_103a) to 1.934 (marca_133) — most runs cluster around
0.5–1.0. Critically, **0 of the 364 pairs are byte-identical**, in sharp
contrast to PC10/PC11/FOGRA51, where `journal/pipeline/verify_gp.py`
already documents that every duplicate-recipe pair *is* byte-identical (an
upstream averaging artifact that makes the "noise floor" computation
invalid for those three lab datasets — the paper must cite literature
repeatability, ~0.1–0.3 ΔE00, for those instead). The wb numbers, by
contrast, are genuine, non-trivial, non-artifactual repeatability: this is
what re-measuring the same recipe on the same newsprint press run actually
costs in ΔE00, roughly **2–4× the lab repeatability figure (~0.2–0.5)**
this document's own reference scale cites for controlled proofing
conditions — consistent with newsprint's rougher substrate and generally
higher process variability.

This number is directly useable in the paper as the honest noise floor for
IFRA within-run results: poly3/svm/mlp_deep's within-run medians of
~1.1–1.6 ΔE00 (§8) sit at roughly 1.5–2.5× this genuine repeatability floor
— a solid, unremarkable result for a well-fit regression model, not "at the
noise floor" the way the (unverified, later partly-invalidated) lab-dataset
GP claim was originally framed in §2. It also independently corroborates
the bb quarantine decision (commit `7719a02`): the identical computation
applied to bb's "duplicate" patches gives a median of 25–28 ΔE00 — proof
those aren't really duplicate measurements under bb's current chart-layout
join, confirming that join is wrong (see teaching briefing, open item (a)),
while wb's ~0.6–0.8 confirms wb's join is sound.
