# First results: CMY / CMYK regression across PC10, PC11, FOGRA51

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
