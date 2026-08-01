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

---

## 5. New results since first briefing (2026-08-01)

Four new things happened since the version of this briefing above, and Phil should hear about all four. None of them overturn the story you already have — GP and low-order polynomial regression lead, linear models fail badly — but they add real texture to it, plus two decisions that need Phil's input.

### 5.1 Optimizing the actual metric (ΔE00) instead of a proxy (squared XYZ error) — the tail gets better, the typical case doesn't

So far, every polynomial fit in this project has worked the same way: minimize squared error on XYZ (ordinary least squares), then measure the result with ΔE00. That's a mismatch — the thing being minimized during fitting isn't the thing being reported at the end. So we tried fitting the same degree-3 polynomial by directly minimizing ΔE00 itself, using two derivative-free optimizers (Nelder-Mead and Powell), starting from the least-squares solution.

The result is clean and a little surprising: the worst-case error drops substantially — 28% better on PC10, 19% on PC11, 48% on FOGRA51 — but the median barely moves, and on two of the three datasets it's actually very slightly worse. That's not a bug, it's exactly what you'd expect once you think about where ΔE00 and squared-XYZ-error actually disagree. In the mid-tones, where most samples live, the two metrics agree closely enough that there's essentially no daylight for a different loss function to exploit — least squares was already doing about as well as anything can there. Where they disagree sharply is in the tail — the dark, heavily-inked patches where ΔE00's perceptual weighting bends away from a flat XYZ distance. Optimizing the metric we actually care about lets the model's limited, fixed-degree budget get spent specifically on those expensive tail cases, at the cost of a hair of typical-case performance. So: this is a genuine, reportable improvement, but it's a worst-case story, not an average-case one — say exactly that, don't oversell it as "ΔE00-loss training improves accuracy" across the board.

One methods note worth mentioning if asked: Powell consistently beat Nelder-Mead here, despite using a tenth of the iteration budget (200 vs 2000). The polynomial has 60 free coefficients at this input size, and Nelder-Mead's simplex approach is known to lose reliability well before that many dimensions — Powell's coordinate-wise search scales better. Worth keeping in mind if we push this to CMYK or n>4, where the coefficient count gets much bigger.

**One-liner for Phil:** "Fitting the polynomial to directly minimize ΔE00 instead of squared XYZ error cuts the worst-case error by 20-48%, but barely touches the median — because the two loss functions only really disagree on the hard, dark, heavily-inked patches, and that's exactly where the improvement shows up."

### 5.2 LLM-as-predictor: gpt-4o is a solid mid-field method, not a contender for the top

We ran the LLM experiment properly this time — gpt-4o and gpt-4o-mini, given 400 example (recipe → XYZ) rows as text in the prompt and asked to predict XYZ for 100 held-out recipes it never saw, on PC10's CMY-only data. gpt-4o comes in at a median error of about 3.0 ΔE00; gpt-4o-mini comes in at about 9.4.

Put in context against the other 14 methods (which all get 5-fold cross-validation over the full dataset): gpt-4o's 3.0 sits between random forest/k-nearest-neighbors (roughly 1.7-2.0) and decision tree (roughly 4.4) — solidly mid-pack. It beats every plain linear regression variant outright (those sit at 6.6+) but doesn't come close to GP, polynomial, SVM, gradient boosting, or either neural net variant, all of which are comfortably under 1.2. gpt-4o-mini, the cheaper/smaller model, is worse than literally everything else tested on this dataset, including the linear regression floor — it's not a usable predictor here.

The honest caveat, and I want to lead with this rather than let you find it: this isn't an apples-to-apples comparison with the CV numbers. Every other method's number comes from pooling five separate train/test splits over all 818 samples — extensive, repeated evidence. The LLM number comes from one single run: one random draw of 400 in-context examples, one random draw of 100 test recipes, no refitting, no folds. It's a reasonable, useful signal that gpt-4o lands in the middle of the pack, but it's a single data point statistically, not a cross-validated result, and I'd want to re-run it over a few different seeds before it goes in the paper with the same confidence as the other 14 methods.

**One-liner for Phil:** "A good general LLM, given nothing but 400 text examples and no real training, lands in the middle of our 14-method leaderboard — beats plain linear regression, loses to everything smarter than a decision tree. The cheap model isn't competitive at all. But it's one run, not five-fold CV, so treat it as a promising signal rather than a final number."

### 5.3 Newsprint generalization: press-to-press variation is the real story, and pooling data helps

This is the big one for the multi-printer generalization question. We ran three experiments on the 13 usable newsprint press runs: (A) predict within the same press run (ordinary cross-validation), (B) train on one press run, test cold on a different one, and (C) train on twelve press runs pooled together, test on the thirteenth held out.

Leaving Gaussian Process out for a moment (next section explains why), the other methods tell a very consistent story: within-run error is about 1.4 ΔE00 — that's what the model can do when it's predicting colors from the same press condition it learned from. Cross-run error jumps to about 4.0 ΔE00 — nearly three times worse — when you ask the same kind of model to predict an entirely different press run's colors cold. And this jump happens for every single model we tried, regardless of method, which is itself the tell: this isn't a modeling failure, it's a real physical difference between press runs — different newsprint batch, different dot gain, different press wear, different everything. No CMYK-to-color function learned from press run A can be expected to know press run B's actual physics.

Here's the encouraging part: when we pool twelve press runs together for training and test on the thirteenth, error drops from about 4.0 back down to about 3.0 — roughly a quarter of the gap recovered. Seeing the same nominal ink recipe reproduced across many different presses lets the model start averaging over press-to-press quirks instead of committing to one press's specific behavior. That's real, useful evidence that pooling more printer data helps generalization — but it doesn't get you all the way back to the 1.4 you get from staying within one press run. There's a genuine floor to how well you can predict an unseen press's colors, and more data narrows it without eliminating it.

**One-liner for Phil:** "Predicting within one press run: about 1.4 ΔE00. Predicting a completely different, unseen press run: about 4.0 — the physical differences between presses dominate, not the model. Pool twelve press runs together before predicting the thirteenth, and that comes back down to about 3.0. So pooling helps meaningfully, but press-to-press variation has a real floor that more data narrows rather than erases."

### 5.4 The GP-newsprint anomaly — a config problem, not a real GP failure

Here's a number that would look terrible if reported without investigation: for within-run prediction, Gaussian Process — our best performer everywhere else in this project, by a wide margin — comes in at a median of about 18.8 ΔE00. That's not just bad, it's the *worst* of everything we tried on that experiment, the exact opposite of GP's role everywhere else. It happens on all 13 press runs consistently, so it's not a one-off fluke — I dug into it rather than either reporting it as-is or quietly hiding it.

Here's what's actually happening. The GP's kernel has a setting — think of it as "how much random measurement noise do I expect in the training data" — that's set very close to zero, because on our lab-conditioned printer datasets (PC10, PC11, FOGRA51), that assumption is basically correct: those measurements really are close to noise-free. Newsprint is different — real press repeatability is around 0.6-0.8 ΔE00 (see next section), not near-zero. When you fit a model that assumes almost no noise to data that actually has real point-to-point scatter, something has to give: the model shrinks its notion of "how far apart do two ink recipes need to be before I stop treating them as similar" down to almost nothing, trying to explain that scatter some other way. I confirmed this directly — refit the exact same model on newsprint data and checked its internal settings: that "similarity radius" had collapsed to the smallest value the software allows. Once that happens, the model can't generalize between recipes at all anymore — for almost any ink recipe you ask it about, unless it's nearly identical to something it saw in training, it just gives you back roughly the average color across its whole training set. I checked this literally, sample by sample: four out of the first five test predictions I looked at were the training set's average color to eight decimal places, regardless of what recipe was actually being asked about.

Why doesn't this show up in the other two newsprint experiments (cross-run, leave-one-out)? Because those experiments test on a genuinely different press run, where the real, large press-to-press difference (the ~4.0 ΔE00 from the last section) already swamps everything — a model reverting to an average color and a model that's honestly trying to extrapolate to unfamiliar territory end up making similarly-sized mistakes either way, so the pathology is masked rather than fixed. Within-run is the one place this could have been caught, because it's the one experiment where GP is supposed to be predicting nearby, familiar territory — exactly its best case everywhere else in this project — which is exactly why the mismatch stands out so starkly there.

Bottom line: this is a configuration issue specific to newsprint's noisier data, not evidence that GP doesn't work for this problem. The fix — telling the model to expect more realistic measurement noise for newsprint — is straightforward but not yet done. That's open item (b) below.

**One-liner for Phil:** "GP looks catastrophically bad on one newsprint experiment, but I traced it: it's configured to assume almost no measurement noise, which is right for our lab datasets but wrong for newsprint's noisier presses. Once that assumption is wrong, the model stops trusting nearby recipes and just gives back an average color instead. It's a settings problem, not GP failing — and it only shows up in the one experiment structured exactly like GP's usual best case, which is why we hadn't seen it before."

### 5.5 The newsprint repeatability number: about 0.6-0.8 ΔE00, and it's real this time

Every one of the 13 newsprint press-run charts has the same 28 patches repeated on it — a built-in quality-control feature of the chart design — meaning the identical ink recipe gets measured twice within a single press run, at two different spots on the sheet. We measured the color difference between each such pair: median about 0.63 ΔE00 pooled across all 13 runs (individual runs range from about 0.4 to 1.9), and — importantly — none of the 364 pairs are exact duplicates. That last point matters: when we ran the same check on our lab datasets (PC10/PC11/FOGRA51) a while back, every "duplicate" pair there turned out to be byte-for-byte identical, which told us that was an artifact of how those files were built upstream, not a real repeatability measurement — so we can't use those as a noise floor. Newsprint's duplicates are genuinely different measurements of the same recipe, so this number is real: about 2-4 times the lab repeatability figure we've been citing (0.2-0.5 ΔE00), which makes sense given newsprint's rougher, less controlled printing process.

This gives us an honest yardstick: our models' within-run newsprint error (about 1.1-1.6 ΔE00, per section 5.3) is roughly 1.5-2.5 times this real noise floor — a solid, sensible result, not a spectacular one, and a much more defensible claim than "at the noise floor" would have been. It also confirms something else we found this week: applying this same check to the black-backing ("bb") newsprint runs gave a "repeatability" of 25-28 ΔE00 for supposedly identical recipes — obviously not real repeatability, which told us the bb chart's CMYK-to-measurement mapping is currently wired up wrong. That's open item (a) below.

**One-liner for Phil:** "Measuring the same ink recipe twice on the same newsprint press run differs by about 0.6-0.8 ΔE00 on average — and unlike our lab datasets, this is a real repeatability number, not an artifact. Our models land at about 1.5-2.5 times that floor within a single press run, which is a solid, honest result."

### 5.6 Two open items for Phil

**(a) The bb (black-backing) chart layout.** We have 13 usable "wb" (white-backing) newsprint press runs and, if we can sort this out, 25 more "bb" runs — nearly triple the data. Right now the bb runs are quarantined: when we checked their supposedly-duplicate patches (previous section's method) we got a repeatability of 25-28 ΔE00, meaning the patches we think are duplicates are actually completely different colors — the mapping from ink recipe to spectral measurement for bb files is using the wrong chart layout. We need the correct bb chart layout/patch-ID mapping from you (or whoever supplied the original files) to rescue those 25 runs. Until then, all newsprint results in this briefing are wb-only.

**(b) Whether to widen the GP noise floor for newsprint.** Per section 5.4, the fix for the GP anomaly is to tell the model to expect realistic newsprint measurement noise (around 0.6-0.8 ΔE00) instead of the near-zero setting that works for our lab datasets. This is a one-line config change and a re-run, not a research question — but I want your sign-off before we do it, partly because it means newsprint and lab datasets will use different GP configurations going forward (currently everything shares one setting), and partly because I'd like your view on whether that's worth stating explicitly in the paper's methods section as "we tuned the noise assumption per data source" versus finding one setting that works reasonably for both.

### 5.7 Likely Phil questions on this new material

**"Is the ΔE00-direct-loss improvement worth the extra complexity?"**
For the worst-case number, yes — 20-48% reduction in max error is a real, reportable win, and it costs nothing extra at prediction time (it only changes how the polynomial is fit, not its form). For the typical case, no — median barely moves, and I'd say so plainly rather than imply direct-loss training is a general accuracy win. It's specifically a tail-risk reduction technique.

**"Should we trust the LLM number enough to put it in the paper?"**
As a single-run estimate showing gpt-4o lands mid-pack, yes, with the caveat stated explicitly. As a number precise enough to rank against our cross-validated methods to two decimal places, not yet — that needs a few more seeds/folds first, which is quick to do and on the list.

**"Is the newsprint generalization result strong enough to claim 'ML generalizes across printers'?"**
Not unconditionally — the honest claim is narrower and still useful: models trained on one press run transfer poorly to another (large, physically-driven gap), but pooling more press runs closes part of that gap. That's a real, defensible finding about the value of multi-source training data; it's not a claim that any model fully solves cross-printer generalization.

**"Why should I believe the GP anomaly is a config issue and not you rationalizing a bad result?"**
Because it's verifiable, not asserted: I refit the exact model and looked at its internal settings directly (its "similarity radius" collapsed to the smallest allowed value), and I checked its actual predictions sample-by-sample (they're identical to the training average, to eight decimal places, for most test points). That's a mechanism, not a guess, and it's specific to the one experiment (within-run) that exposes it — the other two newsprint experiments using the identical model config don't show the problem, which is exactly what the mechanism predicts.

**"Why didn't we know about the newsprint noise floor before now?"**
We only just got clean, ingested newsprint data with verified chart structure this week — computing this required identifying which patches are genuine repeats within a chart, which came out of the same investigation that caught the bb chart-layout problem.
