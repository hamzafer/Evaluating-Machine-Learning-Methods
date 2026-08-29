# The polynomial baseline was fitted in the wrong space (23 Aug 2026)

**This document was rewritten after verification.** The first version reported the right numbers
with the wrong explanation and an overblown novelty claim. Both were caught by the gate agent and
are corrected below; the measurements themselves reproduced exactly.

## What was actually found

Fitting the degree-3 polynomial to `cbrt(XYZ)` instead of `XYZ`, with identical folds, identical
term counts and the identical metric, improves the **maximum error on all nine datasets (38-90%)**
and the **median on eight of nine**.

| dataset | poly3 (XYZ) median | poly3_cbrt median | poly3 max | poly3_cbrt max |
|---|---|---|---|---|
| PC10-CMY | 0.268 | **0.148** | 6.795 | **2.048** |
| PC10-CMYK | 0.942 | **0.219** | 33.978 | **3.506** |
| PC11-CMY | 0.238 | **0.140** | 7.838 | **1.653** |
| PC11-CMYK | 0.869 | **0.216** | 33.070 | **3.511** |
| FOGRA51-CMY | **0.369** | 0.432 | 8.251 | **1.958** |
| FOGRA51-CMYK | 0.816 | **0.437** | 29.287 | **5.786** |
| KCMYG-5 | 1.457 | **1.024** | 65.742 | **29.230** |
| CMYKOGV-7 | 5.386 | **0.830** | 59.833 | **27.979** |
| CMYKOGB-7 | 3.332 | **1.611** | 48.974 | **30.274** |

Reproduced exactly (4 dp) by two independent implementations sharing no code path.

## Three corrections to the original write-up

### 1. It is equivalent to fitting in CIELAB, and that is not novel
CIELAB is an invertible linear map of `(f(X), f(Y), f(Z))` where `f` is essentially a cube root, and
ordinary least squares is equivariant under invertible linear maps of the target. So
**`poly3_cbrt` is the same estimator as "fit the polynomial in CIELAB and convert back"**. Verified:
PC10-CMY 0.1476 both ways; CMYKOGV-7 0.8304 vs 0.8320 (differing only in the third decimal, where
CIELAB's linear segment near black bites on Z ~ 0.45).

Fitting characterization models in a perceptually uniform space is long-standing practice. The
honest framing is therefore **"our baseline was handicapped"**, a baseline correction requiring
prior-art citation, not a discovery. The LLM-provenance story stays out of the paper.

### 2. The CIEDE2000-alignment explanation is wrong
The original claim (least squares in cube-root space aligns with the metric) fails its control:
weighting a plain XYZ fit by `y^(-4/3)`, the first-order equivalent of that alignment, makes the
median **worse on 8 of the 9 datasets** (PC10-CMYK 0.942 -> 1.551, PC11-CMYK 0.869 -> 1.400,
FOGRA51-CMYK 0.816 -> 1.315; the one exception is CMYKOGV-7, 5.386 -> 5.040, still far worse than
the 0.830 the cube-root fit gives). Convention, refreshed 29 Aug 2026: each output channel is
fitted separately with its own `y_c^(-4/3)` weights — channel-mean weights give different numbers
(1.704/1.517/1.326 and a 9-of-9 count). Full run: `journal/results/weighting_control.csv`, script
`journal/pipeline/sweeps.py`. Positivity is not the mechanism either (flooring identity
predictions changes PC10-CMY not at all).

The real mechanism is **approximability**: the ink -> XYZ response is much closer to a low-order
polynomial after a compressive transform. Degree-3 residual RMS as a percentage of target SD on
PC10-CMY falls from 0.99/1.05/1.58 (X/Y/Z) to 0.65/0.62/0.75 under cube root
(`journal/results/residual_rms.csv`).

### 3. Cube root is not special
Any variance-stabilising transform helps, and square root is often better:

| dataset | identity | sqrt | cbrt | y^0.25 | log1p |
|---|---|---|---|---|---|
| PC10-CMY | 0.268 | **0.127** | 0.148 | 0.166 | 0.234 |
| PC11-CMY | 0.238 | **0.124** | 0.140 | 0.154 | 0.206 |
| FOGRA51-CMY | **0.369** | 0.376 | 0.432 | 0.467 | 0.562 |
| PC10-CMYK | 0.942 | 0.252 | **0.219** | 0.250 | 0.385 |
| PC11-CMYK | 0.869 | 0.243 | **0.216** | 0.251 | 0.379 |
| FOGRA51-CMYK | 0.816 | **0.386** | 0.437 | 0.490 | 0.679 |
| KCMYG-5 | 1.457 | 1.045 | 1.024 | **1.023** | 1.112 |
| CMYKOGV-7 | 5.386 | 1.253 | 0.830 | **0.811** | 1.126 |
| CMYKOGB-7 | 3.332 | 1.821 | 1.611 | **1.560** | 1.592 |

## The finding that matters most: degree is the other lever

On CMYKOGV-7 (7 inks, 3302 rows), 5-fold grouped CV, median ΔE00, measured in this repo:

| degree | fitted in XYZ | fitted in cube-root space | terms/channel | train/test gap (cbrt) |
|---|---|---|---|---|
| 2 | 9.953 | 2.656 | 36 | +0.013 |
| 3 | 5.386 | 0.830 | 120 | +0.029 |
| **4** | **2.080** | **0.272** | 330 | +0.029 |
| 5 | 0.506 | 0.126 | 792 | +0.036 |

(Refreshed 29 Aug 2026 from `journal/results/CMYKOGV-7/degree_sweep.csv` — the earlier gap values
+0.023/+0.027 and the two missing cells were from a pre-dedup run; the XYZ-space gaps are
+0.038/+0.121/+0.226/+0.133.)

**The Gaussian process scores 0.249 on this dataset.** A degree-4 polynomial fitted in CIELAB scores
0.272. That is a tie, and it is not overfitting: the train/test gap stays at +0.029, essentially
unchanged from degree 3, at 2.7 rows per parameter.

## Consequence for the paper

The current n>4 headline is that third-order polynomial regression collapses as ink count grows
(0.268 -> 5.386) while the Gaussian process holds. That comparison was against a baseline
handicapped on **two** axes at once: fitted in XYZ rather than a perceptual space, and capped at
3rd order.

Corrected, the claim becomes narrower and more defensible:

> A degree-3 polynomial fitted in XYZ degrades sharply with ink count. Most of that degradation is
> attributable to the fitting space and the degree cap rather than to low-order polynomials as such:
> refitted in CIELAB at degree 4, the same method reaches 0.272 against the Gaussian process's 0.249
> on the 7-ink set.

This weakens "ML beats classical at n>4" and replaces it with something more useful to a
practitioner: **what the classical method needs in order to compete.** The Gaussian process retains
real advantages (no degree choice, no space choice, and it is still 5-10x better at n<=4), and those
should now carry the argument.

Phil's 3rd-order cap needs a stated justification wherever the n>4 comparison rests on it, because
at n=7 the cap is doing much of the work. The evidence here is that degree 4 does not overfit on a
3302-row chart.

## Open items
- **FOGRA51-CMY** is the one median regression (+17%, while its max improves 76%). Explained: it is
  the only coated set where the cube root makes the polynomial fit *worse* on every channel
  (residual RMS 0.95->1.30, 1.05->1.28, 1.56->1.65), so the transform reallocates bias from dark to
  light. Cube root wins its darkest lightness quartile (0.510 vs 0.597) and loses the two lightest.
  Square root shows no regression on that dataset at all (0.376 vs 0.369) while still cutting the
  max from 8.251 to 1.995. The white point is provably irrelevant: `cbrt(X/Xn)` differs from
  `cbrt(X)` by a per-channel constant and OLS is equivariant to target scaling.
- **Do not apply the transform per channel.** Mixed spaces are worse than either pure space, because
  cross-channel residual correlation is what cancels in a* and b*.
- The phrase "same 20 terms per channel" is only true at n=3; degree-3 gives 35 terms at n=4, 56 at
  n=5 and 120 at n=7. The two models always have identical counts *as each other*, so the comparison
  is fair, but the wording must not claim 20 everywhere.
