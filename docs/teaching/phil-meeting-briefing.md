# Briefing for the Phil meeting

Written so you can say this out loud without notes. Read it once, then just talk — don't read it verbatim at Phil.

---

## 1. The 90-second version

"Two things happened since Taipei. First, I found and fixed an evaluation bug in the AIC pipeline: we were computing color error on the normalized [0,1] version of XYZ instead of converting back to real values first, which crushed every error by about 5x — the conference paper's poly3 median of 1.16 is actually 5.54. I confirmed this three independent ways, told you, and we agreed it doesn't need an erratum — the journal paper is a fresh analysis, and the old numbers stay tagged as 'published' for reference.

Second, I built a new pipeline for the journal work — one code path that handles 3, 4, or however many ink channels, proper 5-fold cross-validation, and the corrected color-difference metric baked in from the start with a tripwire that catches the old bug if it ever creeps back. First results: Gaussian Process dominates everywhere, error around 0.05-0.07 ΔE00, which is close to what a printer's own repeat-measurement noise looks like. Polynomial regression is the best classical method at 3 inks but its error roughly triples going to 4 inks with K, while GP barely moves — that's early evidence for exactly the question you asked: ML holds up as ink count grows, classical methods don't. One caveat I want to flag before you ask: I haven't yet run the duplicate-safe cross-validation that rules out GP just memorizing repeated print patches — that's this week's first task, not a settled result yet."

---

## 2. Owning each piece

### 2.1 Why normalization broke ΔE00 — the white-point division

CIELAB isn't computed from raw XYZ directly. The formula divides your XYZ by a **reference white point** first — for D50 that's roughly `[96.422, 100.0, 82.521]` — and only then takes the cube root. That division is where everything went wrong.

The AIC pipeline (`main/cmy2xyz/polynomial_regression.py` etc., before the fix) did this:

```python
scaler = MinMaxScaler()
output_xyz_norm = scaler.fit_transform(output_data)   # squashes XYZ into [0,1]
...
xyz_pred = model.predict(cmy_test)                    # model outputs [0,1]-scale numbers
lab_pred = xyz2lab(xyz_pred)                          # <-- fed straight in, never un-scaled
```

`xyz2lab` has no idea the numbers it received are normalized — it just divides by the D50 white point as if they were real XYZ. Paper white in the real data is XYZ_Y ≈ 100 (L* = 95, i.e. "pretty much white"). Normalized, that same point becomes Y ≈ 0.95 to 1.0. Divide 0.95-ish by a white point of 100 and you get roughly 0.0095 — a number that belongs to something very dark. Run that through the L* formula and paper white reports as L* ≈ 9 instead of 95. Every color in the dataset gets pulled toward "near black" by the same mechanism, and because ΔE00 is roughly proportional to how far apart two points are in this compressed space, all the differences shrink by close to the same factor — about 5x at the median, which is exactly the shift we measured (poly3: 1.159 → 5.539 published vs. corrected).

The fix is one line per script: `xyz2lab(scaler.inverse_transform(xyz_pred))` — un-normalize back to the real 0-100 XYZ scale before converting to Lab. That's commit `a0c937a`. The new pipeline (`journal/pipeline/color.py`) doesn't even give you the chance to get this wrong: it has a standing tripwire, `assert_lab_roundtrip`, that on every dataset load converts the CSV's own XYZ to Lab and checks it reproduces the CSV's own measured Lab column. If a scale mistake like this ever comes back, the pipeline refuses to run instead of silently reporting numbers 5x too good.

**One-liner for Phil:** "We divided already-normalized numbers by a white point meant for real ones — so paper white looked almost black to the math, and every error shrank by the same ratio."

### 2.2 Why K-dropping made the task ill-posed

The AIC paper described training on 818 rows where K=0 (a genuinely 3-input CMY→XYZ problem). The code actually trained on all 1,617 rows and just deleted the K column from the input table — including the 799 rows where K > 0.

Concretely: two rows can have identical C, M, Y and K=0 vs K=0.3. With K in the input, that's two different, perfectly valid recipes producing two different colors (K adds density/darkness). Drop the K column and keep both rows, and you've created a training example where the exact same input (C, M, Y) is labeled with two different, contradictory target colors. That's a one-to-many mapping — not a function. No regression method, however good, can fit both targets to the same input at once; the best any model can do is average or split the difference, which shows up as irreducible error that has nothing to do with the model's real capability.

The new pipeline's dataset spec makes the two experiments explicit and separate: `*-CMY` filters to K=0 only (818 rows, a genuine 3-input problem) and `*-CMYK` uses all 1,617 rows with K as an actual 4th input feature (so identical inputs really are identical). The rule, stated in the dataset code as a comment: "never drop an input column while keeping its rows."

**One-liner for Phil:** "The paper said CMY-only, but the code kept the K>0 rows and just threw away the K column — so the same recipe was sometimes labeled two different colors. We fixed it by either filtering to K=0 rows, or keeping K as an input — never both dropping and keeping."

### 2.3 Why 5-fold (grouped) cross-validation matters here

Standard CV: split the data into 5 chunks, train on 4, test on 1, rotate, so every row gets predicted exactly once by a model that never saw it during training. That's what the new pipeline does by default (`KFold(n_splits=5, shuffle=True, random_state=42)` in `journal/pipeline/evaluate.py`), and it's already better practice than the AIC paper's single 90/10 split — five times more test coverage, no dependence on one lucky/unlucky split.

But these printer datasets have exact duplicates: the same ink recipe was measured more than once (e.g. PC10 has 58 rows out of 1,617 sharing an identical C/M/Y/K recipe with another row — different measurement, same input). With plain random KFold, nothing stops one copy of a duplicate landing in the training fold while its identical twin lands in the test fold. A model that's prone to memorizing nearby points (1-nearest-neighbor is the extreme case, but a low-noise Gaussian Process can behave similarly) can score near-zero error on that test point just by regurgitating what it saw for its twin — that's leakage, not generalization.

The fix is `GroupKFold`: assign every row a group ID based on its exact input recipe (`make_groups` in `journal/pipeline/evaluate.py`), so duplicates always travel together — either both in train or both in test, never split. There's a unit test proving the mechanism works (`journal/pipeline/tests/test_evaluate.py`): on a synthetic dataset where every row is duplicated once, plain KFold + 1-NN gets a suspiciously perfect median error (<0.01), while grouped CV forces a realistic, higher number.

**Where this stands right now:** the grouped-CV machinery is built and unit-tested, but the six `summary.csv` result files you'll see (GP median 0.05-0.07 etc.) were run with plain KFold, not grouped. Running grouped CV across all datasets and models, plus computing a "noise floor" from the duplicate pairs themselves (how far apart are two measurements of the literal same recipe — expected roughly 0.1-0.5 ΔE00 from instrument repeatability), is the very first thing on this week's plan (`docs/plans/01-gp-verification.md`). Say this proactively — it's more credible coming from you than being caught by the question.

**One-liner for Phil:** "Some recipes were measured twice, so plain random splitting can let a model see the answer's twin during training. Grouped CV keeps duplicates together so that can't happen — we've built and tested it, and running it on the full result set is this week's first job."

### 2.4 Why Gaussian Process wins on this kind of mapping

GP regression isn't fitting a fixed-shape curve the way polynomial regression does. It's a kernel method: predict a new point's color as a weighted blend of the *nearby* training points, where "nearby" and "how much weight" are learned from the data via a kernel (here `ConstantKernel() * RBF() + WhiteKernel(1e-5)` — an RBF kernel is essentially a smooth bump function of distance, so points close together in CMY(K) space are assumed to have similar XYZ, with a small noise term for measurement jitter). It has no fixed functional form and no hard cap on how wiggly it can be — it adapts its own smoothness (length-scale) to the data.

Printer color mixing is a physically smooth, continuous process: nudge one ink channel a little, the resulting color moves a little, with no sharp jumps. These datasets are also *dense* relative to their input dimensionality — 818-1,617 samples covering a 3-4 dimensional input space is a lot of nearby neighbors for any given test point. Put a locally-adaptive smooth interpolator on a dense sample of a smooth function, and you get something close to genuine interpolation — which is exactly the regime where GP is strongest and why its errors (median ≈0.05-0.07 ΔE00 across every dataset/channel-count combination) sit near where you'd expect measurement noise to be, rather than model error.

The caveat is baked into this explanation: "close to interpolation" is one short step away from "just memorizing," which is precisely why the duplicate-leakage check in §2.3 matters before this becomes a paper claim rather than a promising number.

**One-liner for Phil:** "GP doesn't assume a fixed shape — it blends nearby training points using a learned smoothness. Printer color mixing is smooth and we have a lot of densely-sampled points, so it's close to genuine interpolation — which is GP's best case."

### 2.5 Why polynomial regression degrades as ink count grows

A degree-3 polynomial over n input channels has a fixed vocabulary of basis terms: every monomial up to degree 3 (x, x², x³, and all the cross-terms like xy, xyz, x²y, ...). The *number* of those terms grows combinatorially with n (roughly `C(n+3, 3)`), but the *shape* of what the model can represent doesn't get any more flexible in a meaningful sense — it's still a single fixed global surface, just with more knobs.

Going from CMY (n=3) to CMYK (n=4) adds a channel (K) that has an outsized, strongly nonlinear effect: it darkens everything and interacts with C/M/Y in ways that are hard to capture with a low, capped polynomial degree. So the model has to spread its fixed-degree budget across more interaction terms while the real function it's approximating got more complicated — the net result is systematic underfit, concentrated in the ink combinations the fixed polynomial handles worst. That shows up exactly in the numbers: poly3 median goes from 0.279 (PC10-CMY) to 0.944 (PC10-CMYK) — roughly 3.4x worse — and similarly for PC11 (0.244→0.868, ~3.6x) and FOGRA51 (0.368→0.822, ~2.2x), with max error also jumping into the high 20s-30s (worst-case catastrophic misses on specific ink combinations). GP, over the same n=3→n=4 jump, barely moves: 0.054→0.063, 0.054→0.066, 0.065→0.072 — about a 15-20% increase, not a 2-4x one.

That contrast — classical method degrades sharply with dimensionality, ML method stays flat — is the first concrete evidence for research question (b) in the roadmap ("can AI handle n>4 where classical methods struggle").

**One-liner for Phil:** "A capped-degree polynomial has a fixed, global shape with a fixed vocabulary of terms. Adding K adds a strongly nonlinear channel that eats into that fixed budget, so it systematically underfits — and the numbers show it: poly3's error roughly triples going from 3 to 4 inks, while GP barely moves."

---

## 3. Likely Phil questions, with strong answers

**"How do you know GP isn't overfitting / just memorizing?"**
Honest answer, not a deflection: right now, partially. The published-looking numbers (median 0.05-0.07) were computed with plain 5-fold CV, and we know the data has duplicate recipes (e.g. 58 of PC10's 1,617 rows share an exact input with another row), so there's a real leakage risk for any model that can behave like a near-interpolator. We've already built and unit-tested a `GroupKFold` variant that keeps duplicate recipes together across train/test, plus a "noise floor" measure — the ΔE00 between two measurements of the literal same recipe, which tells us the best any honest model could possibly do. Running that across all datasets/models is the first task this week. The decision rule is already written down: if GP's grouped-CV median stays within about 2x of the plain-CV median, the headline holds ("GP performs near the measurement noise floor"); if it's much worse, we report the grouped number instead. Either way, the paper reports grouped CV, not the plain numbers you're seeing today.

**"Why is SVM suddenly bad? / Why did SVM look best in the AIC paper?"**
Two separate things happened to SVM, and they cut in opposite directions. In the AIC paper, SVM's published "best" configuration used epsilon=0.1 — the ε-insensitive tube SVR uses to decide "close enough, zero penalty." On [0,1]-normalized targets, an epsilon of 0.1 means the model pays zero training cost for being off by 10% of the entire target range — it's an almost-lazy config, but that never showed up in the paper because the crushed, 5x-too-small metric hid it. Once we reproduced the AIC config exactly and only fixed the metric, SVM's real error came out to a median of 10.18 — worse than everything else, because the tube really was that loose. Separately, in the *new* pipeline we also fixed the hyperparameter itself, not just the metric — epsilon=0.01 instead of 0.1 — and with that sane setting, SVM is a perfectly respectable mid-pack method (median 0.72-1.03 across datasets), consistently third or fourth behind GP and poly3-at-n=3. So: the AIC number was an illusion created by a hidden degenerate hyperparameter; the new number is what SVM actually does once both the metric and the hyperparameter are sane.

**"What changed versus the AIC numbers, concretely?"**
Three things, and they're independent — you can attribute the change to each of them separately if asked. (1) Metric fix: ΔE00 now computed on real, denormalized XYZ, not [0,1]-scaled numbers — this alone moved every legacy-pipeline result up roughly 5x at the median (poly3: 1.159 → 5.539). (2) Data-integrity fix: the CMY experiment now genuinely uses only K=0 rows (818), instead of secretly training on all 1,617 rows with an ill-posed dropped column. (3) New pipeline, new questions: `journal/pipeline/` is a fresh, n-channel-generic codebase (handles CMY and CMYK through one code path) with 5-fold CV instead of one 90/10 split, per-fold-fit scalers (no test-set information leaking into the scaler), and two sane hyperparameter fixes to the model registry (SVR epsilon, Lasso/ElasticNet alpha — the AIC values were degenerate, collapsing to near-constant predictors). The legacy AIC code and its published numbers are kept exactly as published, tagged `aic2025-published` in git, purely as a historical reference — we're not rebuilding on it or claiming an erratum.

**"Why report median/P95/max instead of mean/std?"**
Color error isn't normally distributed — it has a long right tail (a handful of ink combinations the model handles badly can blow up the mean and std while barely touching most of the data). Median tells you the typical customer experience; P95 and max tell you how bad the worst cases get. That's your own stated preference from the April meeting, and it's now baked into the reporting code (`summarize()` in `journal/pipeline/evaluate.py`), not something we compute ad hoc per figure.

**"Why cap polynomial regression at degree 3?"**
Higher degrees fit the training data almost perfectly but the extra wiggle-room is mostly overfitting noise, not signal — also your instruction from April. It also makes the CMY→CMYK comparison in §2.5 a fair one: it's the same fixed-capacity method being asked to do more with the same budget, not a method that's allowed to expand its own capacity as dimensionality grows (which is the whole point of the contrast with ML methods).

**"Are these results anomalous / did you sanity-check them?"**
The XYZ→Lab roundtrip tripwire (§2.1) runs on every dataset load, so any future scale mistake fails loudly instead of silently. Beyond that, the specific worry to flag proactively is the GP-overfitting question above — that's an open verification, not a closed one, and I'd rather you hear that from me now than find it in the draft later.

---

## 4. Glossary

- **ΔE00 / CIEDE2000**: the standard "how different do two colors look" number, computed in CIELAB space. Roughly: <1 is imperceptible, 1-2 is barely visible under close inspection, >5 is obviously a different color.
- **CIELAB (Lab)**: a color space designed so that Euclidean-ish distance between two points roughly tracks perceived difference — unlike raw XYZ, where equal numeric distances don't mean equal perceived difference.
- **White point (D50)**: the XYZ of "reference white" for a given viewing condition; Lab conversion divides every color by it before taking the cube root, so getting this step's scale wrong distorts everything, especially near-white/near-black colors.
- **XYZ (denormalized / 0-100 scale)**: the "real," physically meaningful tristimulus values as measured by the instrument, as opposed to a model's [0,1]-normalized internal representation.
- **MinMaxScaler / normalization**: squashes a variable into [0,1] using the training data's min/max, purely so an optimizer converges nicely. Must be inverted (`inverse_transform`) before doing anything downstream that assumes real-world units — that inversion step is exactly what the AIC bug skipped.
- **K-fold cross-validation**: split data into K parts, train on K-1, test on the remainder, rotate through all K; every row gets tested exactly once by a model that never trained on it.
- **GroupKFold**: like K-fold, but rows sharing a "group ID" (here: identical ink recipe) are always kept together on the same side of the split, so duplicates can't leak between train and test.
- **Ill-posed mapping**: when the same input is associated with more than one correct output — mathematically unlearnable as a function, and it inflates every model's apparent error for reasons that have nothing to do with model quality.
- **Gaussian Process (GP) regression**: a kernel-based method that predicts a point as a learned, weighted blend of nearby training points, with an adaptable notion of "nearby" — no fixed functional form, in contrast to polynomial regression.
- **RBF kernel**: a similarity function that decays smoothly with distance; the core building block of the GP used here.
- **ε-insensitive tube (SVR)**: Support Vector Regression's "free pass" zone — predictions within epsilon of the true value cost nothing during training. Too large an epsilon (AIC's 0.1) means the model can be lazily wrong within a wide band and still score zero training loss.
- **Polynomial regression (degree 3, capped)**: fits a fixed-degree polynomial (all monomials up to that degree) via ordinary least squares; capped at 3 per repo/Phil convention to avoid overfitting.
- **Median / P95 / max**: the three numbers this project always reports for error distributions instead of mean/std, because color error is long-tailed, not normal.
- **Tripwire (`assert_lab_roundtrip`)**: a standing check in the new pipeline that converts each dataset's own XYZ to Lab and confirms it reproduces that dataset's own measured Lab column — catches scale mistakes like the AIC bug automatically.
- **n-channel-generic**: pipeline design where the number of ink channels (3 for CMY, 4 for CMYK, more for CMYKOGV) is a parameter, not a hardcoded assumption — one code path serves all channel counts.
