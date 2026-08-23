# Fitting in cube-root XYZ space (23 Aug 2026)

## Where the idea came from
During the plan-09 equation experiment, Claude Fable 5 (web channel, with a code interpreter)
produced the study's most accurate equation by modelling the **cube root** of each tristimulus
value as a polynomial, rather than modelling XYZ directly. Its stated reason: CIELAB is defined
through a cube root of XYZ, so a least-squares fit in cube-root space is aligned with CIEDE2000,
which is computed in CIELAB.

The equation itself is a weak artifact (2,542 expanded terms, 120 coefficients fitted to 150
points, one run, one chart). **The idea behind it is not**, so it was tested properly here: same
protocol, same folds, same metric, same term count as our existing `poly3`, across every dataset.

## Implementation
`journal/pipeline/models.py::CubeRootPolynomial`, registered as `poly3_cbrt`. Identical to `poly3`
(degree-3 `PolynomialFeatures` + `LinearRegression`, 20 terms per channel) except that it fits
`cbrt(XYZ)` and cubes the prediction. It uses the existing `set_scaler` hook so the root is taken on
**physical** XYZ rather than MinMax-scaled values.

## Result: 5-fold CV, same folds as every other model

| dataset | poly3 median | poly3_cbrt median | change | poly3 max | poly3_cbrt max | change |
|---|---|---|---|---|---|---|
| PC10-CMY | 0.268 | **0.148** | -45% | 6.795 | **2.048** | -70% |
| PC10-CMYK | 0.942 | **0.219** | -77% | 33.978 | **3.506** | -90% |
| PC11-CMY | 0.238 | **0.140** | -41% | 7.838 | **1.653** | -79% |
| PC11-CMYK | 0.869 | **0.216** | -75% | 33.070 | **3.511** | -89% |
| FOGRA51-CMY | 0.369 | 0.432 | **+17%** | 8.251 | **1.958** | -76% |
| FOGRA51-CMYK | 0.816 | **0.437** | -46% | 29.287 | **5.786** | -80% |
| KCMYG-5 | 1.457 | **1.024** | -30% | 65.742 | **29.230** | -56% |
| CMYKOGV-7 | 5.386 | **0.830** | -85% | 59.833 | **27.979** | -53% |
| CMYKOGB-7 | 3.332 | **1.611** | -52% | 48.974 | **30.274** | -38% |

**Median improves on 8 of 9 datasets. Maximum improves on 9 of 9, by 38-90%.** Same number of
coefficients, same folds, no extra tuning.

## Why this matters to the paper's argument
The n>4 headline was that third-order polynomial regression degrades sharply as ink count grows
(0.268 at n=3 to 5.386 at n=7) while the Gaussian process holds. That degradation is **largely an
artifact of fitting in the wrong space**: in cube-root space the same polynomial gets 0.830 at n=7,
recovering most of the gap to the GP (0.249).

The honest revised claim is stronger and more useful than the original: *the classical method's
apparent failure at high ink counts is substantially a modelling-space choice, not an intrinsic
limitation of low-order polynomials.* The GP still wins, but by a far smaller margin, and a
practitioner who wants a simple portable polynomial has a much better option than the literature
default.

The one regression (FOGRA51-CMY median, +17%) needs investigating before this is written up.

## Verification
Reproduced by an independent from-scratch implementation sharing no code with
`CubeRootPolynomial` (no `set_scaler` hook, no pipeline registry): PC10-CMY 0.148/2.048 and
CMYKOGV-7 0.830/27.979, matching to three decimals.
