# Duplicate-row leakage in the n<=4 cross-validation (found 12 Aug 2026)

Found by the independent blind reviewer (`journal/verification/blind-2026-08-12/`), then
measured directly by the coordinator. **FIXED 12 Aug 2026** — see "Fixed" at the bottom for the
commits, the before/after table, and the mechanism behind every number that moved.

## The defect
`PC10/PC11/FOGRA51` register with `grouped=False`, so they use `KFold(shuffle=True,
random_state=42)`. Each of those datasets contains duplicate ink recipes whose measured values are
**byte-identical**: 23 groups (46 rows) in the CMY subset, 29 groups (58 rows) in CMYK. Nothing keeps
the two copies of a recipe on the same side of a split, so a test row's identical twin can sit in the
training fold. The paper's protocol section states the opposite ("folds are grouped by distinct input
recipe ... preventing train/test leakage through repeated measurements").

## Direct measurement (PC10-CMY, our actual folds, seed 42)
30 of 818 test rows (3.7%) had their twin in the training fold. Median ΔE00 on those rows vs the rest:

| model | leaked rows | clean rows | ratio | pooled median | median excl. leaked |
|---|---|---|---|---|---|
| decision_tree | **0.0000** | 4.4367 | 0.00 | 4.3685 | 4.4367 |
| random_forest | 0.5536 | 1.7494 | 0.32 | 1.7162 | 1.7494 |
| gaussian_process | 0.0290 | 0.0451 | 0.64 | 0.0443 | 0.0451 |
| knn | 1.5153 | 2.0592 | 0.74 | 2.0120 | 2.0592 |
| poly3 | 0.3411 | 0.2752 | 1.24 | 0.2787 | 0.2752 |

The decision tree scoring exactly 0.0000 on those rows is the mechanism made visible: it recites
memorised training values. Local/interpolating methods (tree, forest, GP, k-NN) get a partial
freebie; the global cubic does not — which is why poly3 agreed with the blind reviewer to 0.001 while
the local methods did not.

## Severity
- **Conclusions: unaffected.** Every affected pooled median moves by <=0.07 ΔE00; the ranking and the
  ladder story are unchanged (GP still wins by an order of magnitude; poly3 still degrades with n).
- **Method: must be fixed or the protocol claim rewritten.** A reviewer who checks will find the
  decision-tree column, and "identical rows on both sides of the split" reads badly regardless of
  magnitude.

## Agreed fix (Hamza, 12 Aug)
Apply `dedup_exact=True` to all six coated specs — one uniform policy across the paper ("byte-identical
duplicate rows are dropped at load"), matching what CMYKOGV-7 already does, keeping the seeded shuffled
k-fold. After dedup no duplicate recipes remain in those sets, so grouping is unnecessary by
construction, and the double-counting the blind reviewer warned about is removed as well.
Consequences: n becomes 795 (CMY) / 1588 (CMYK); all n=3/n=4 numbers shift by <=0.07; two figures and
the affected tables regenerate.

---

# Fixed (12 Aug 2026)

Code + tests: **403c7d5**. Results + figures + docs: see the follow-up commit in this pair.
All runs on the laptop (arm64, `.venv/bin/python`); 6 datasets x 16 models; slow chunk 1219 s wall.
Effective rows are now **795** (CMY) and **1588** (CMYK), asserted in
`journal/pipeline/tests/test_datasets.py` as a *property* — no duplicate input recipe survives the
load — rather than as a flag check.

**The one prediction that was wrong: the <=0.07 bound.** It was calibrated on PC10-CMY's GP and poly3
and does not hold for models with broad error distributions. 22 of 90 comparable cells move by more
than 0.07; the largest is decision_tree on PC11-CMY at **+0.191**. Every one is explained by the
mechanism below, and **no conclusion moves**: GP wins on all six datasets before and after, Spearman
rank correlation over the 16 models is 0.970-1.000 per dataset, and every rank change larger than one
place is *inside* the linear family, whose five members sit within 0.02 dE00 of each other and are not
distinguishable in the first place.

## Why the linear family got uniformly BETTER (all 30 cells negative)

Not leakage — a global linear fit has no capacity to memorise a duplicated row. The duplicated recipes
are **not a random subsample**: they are light, low-ink patches. On PC10-CMY the dropped rows have
median total ink 40% against 125% for the set as a whole, and median XYZ_Y (lightness proxy) 66.4
against 24.7.

That subregion is **above**-median error for global models and **below**-median error for local ones,
so removing it moves the pooled median in opposite directions for the two families. Measured on the
*old* data and *old* folds, error on the 23/29 rows dedup drops vs the rows it keeps:

| model | err @ dropped | err @ kept | ratio | percentile of dropped in full distribution |
|---|---|---|---|---|
| ridge | 8.703 | 6.580 | 1.32 | 71% |
| plsr | 8.632 | 6.558 | 1.32 | 71% |
| lasso | 8.702 | 6.531 | 1.33 | 72% |
| poly3 | 0.339 | 0.277 | 1.22 | 63% |
| gaussian_process | 0.018 | 0.045 | 0.40 | 12% |
| decision_tree | **0.000** | 4.423 | 0.00 | 0% |
| random_forest | 0.725 | 1.737 | 0.42 | 11% |
| knn | 1.549 | 2.045 | 0.76 | 33% |

(PC10-CMY; PC11-CMY and PC10-CMYK give the same picture — linear ratios 1.26-1.33, decision_tree
0.000 on all three.)

So the whole sign pattern follows from one two-part mechanism:
- **Global models** (linear family, poly3): the dropped rows are high-error, so deleting them pulls the
  pooled median **down**. Systematic, deterministic, and nothing to do with train/test contamination.
  Largest effect where the linear error distribution is broadest (median ~6.6, p95 ~25), which is why
  the linear family shows the biggest negative deltas.
- **Local/interpolating models** (decision_tree, random_forest, knn, GP): the dropped rows are
  low-error *because of the leakage* — their twin was in training. Two effects then push the same way,
  both making the model look **worse**: the pool loses its easiest rows, and the surviving copy no
  longer gets a freebie. Hence decision_tree, random_forest and GP are positive in all six cells.

Decomposition on PC10-CMY, separating pool composition (re-median over surviving rows with the folds
held **fixed**) from the fold reshuffle:

| model | before | pool effect only (old folds) | after | total delta | of which pool |
|---|---|---|---|---|---|
| ridge | 6.624 | 6.580 | 6.545 | -0.079 | -0.044 |
| lasso | 6.601 | 6.531 | 6.478 | -0.123 | -0.070 |
| plsr | 6.643 | 6.558 | 6.520 | -0.123 | -0.085 |
| poly3 | 0.279 | 0.277 | 0.268 | -0.011 | -0.002 |
| decision_tree | 4.369 | 4.423 | 4.410 | +0.041 | +0.055 |

Roughly half to two-thirds is pool composition; the rest is that `KFold(shuffle, seed=42)` on 795 rows
is a completely different partition than on 818.

## The excursions are inside each model's own split noise

Median over CV seeds 0-9, PC11-CMY (the dataset with the largest single move):

| model | undeduped min..max (spread) | deduped min..max (spread) |
|---|---|---|
| decision_tree | 4.008..4.220 (0.212) | 4.045..4.342 (0.297) |
| mlp_deep | 0.977..1.104 (0.127) | 0.976..1.126 (0.150) |
| mlp_shallow | 1.167..1.251 (0.084) | 1.088..1.304 (0.216) |
| ridge | 6.195..6.301 (0.106) | 6.151..6.238 (0.087) |
| plsr | 6.188..6.287 (0.099) | 6.152..6.232 (0.080) |

decision_tree's +0.191 sits inside its own 0.21-0.30 seed-to-seed spread; the linear family's ~0.15
moves sit inside its 0.08-0.11 spread. The MLP cells that moved ~0.10 in *both* directions across
datasets are likewise noise, not signal. GP, the headline model, moves +0.000..+0.002 — it barely
notices, because only 3.7% of rows were affected and its clean-row error already dominated.

## Before/after medians (dE00), all 16 models x 6 datasets

`—` = the model had no committed value for that dataset; the two dE00-loss poly3 variants had never
been run on the CMYK sets, so this pass also fills those six gaps.

### PC10-CMY (n 818 -> 795)

| model | before | after | delta |
|---|---|---|---|
| gaussian_process | 0.044 | 0.046 | +0.002 |
| poly3_de00_nm | 0.264 | 0.261 | -0.003 |
| poly3_de00_powell | 0.274 | 0.261 | -0.013 |
| poly3 | 0.279 | 0.268 | -0.011 |
| svm | 0.754 | 0.730 | -0.024 |
| gradient_boost | 0.881 | 0.901 | +0.020 |
| mlp_deep | 1.102 | 1.134 | +0.032 |
| mlp_shallow | 1.195 | 1.265 | +0.070 |
| random_forest | 1.716 | 1.731 | +0.015 |
| knn | 2.012 | 1.998 | -0.014 |
| decision_tree | 4.369 | 4.410 | +0.041 |
| lasso | 6.601 | 6.478 | -0.123 |
| elastic | 6.611 | 6.532 | -0.079 |
| ridge | 6.624 | 6.545 | -0.079 |
| pcr | 6.624 | 6.545 | -0.079 |
| plsr | 6.643 | 6.520 | -0.123 |

### PC10-CMYK (n 1617 -> 1588)

| model | before | after | delta |
|---|---|---|---|
| gaussian_process | 0.056 | 0.056 | +0.000 |
| poly3_de00_nm | — | 0.894 | (new) |
| poly3_de00_powell | — | 0.867 | (new) |
| poly3 | 0.944 | 0.942 | -0.002 |
| svm | 1.029 | 1.017 | -0.012 |
| gradient_boost | 1.528 | 1.541 | +0.013 |
| mlp_deep | 1.766 | 1.729 | -0.037 |
| mlp_shallow | 1.986 | 2.002 | +0.016 |
| random_forest | 2.274 | 2.376 | +0.102 |
| knn | 2.233 | 2.229 | -0.004 |
| decision_tree | 4.605 | 4.611 | +0.006 |
| lasso | 8.908 | 8.886 | -0.022 |
| elastic | 8.937 | 8.876 | -0.061 |
| ridge | 8.929 | 8.876 | -0.053 |
| pcr | 8.929 | 8.876 | -0.053 |
| plsr | 8.940 | 8.887 | -0.053 |

### PC11-CMY (n 818 -> 795)

| model | before | after | delta |
|---|---|---|---|
| gaussian_process | 0.044 | 0.046 | +0.002 |
| poly3_de00_nm | 0.235 | 0.231 | -0.004 |
| poly3_de00_powell | 0.255 | 0.249 | -0.006 |
| poly3 | 0.244 | 0.238 | -0.006 |
| svm | 0.717 | 0.693 | -0.024 |
| gradient_boost | 0.791 | 0.801 | +0.010 |
| mlp_deep | 1.054 | 1.132 | +0.078 |
| mlp_shallow | 1.176 | 1.137 | -0.039 |
| random_forest | 1.694 | 1.710 | +0.016 |
| knn | 1.938 | 1.923 | -0.015 |
| decision_tree | 4.136 | 4.327 | +0.191 |
| lasso | 6.292 | 6.151 | -0.141 |
| elastic | 6.284 | 6.143 | -0.141 |
| ridge | 6.309 | 6.148 | -0.161 |
| pcr | 6.309 | 6.148 | -0.161 |
| plsr | 6.307 | 6.161 | -0.146 |

### PC11-CMYK (n 1617 -> 1588)

| model | before | after | delta |
|---|---|---|---|
| gaussian_process | 0.056 | 0.057 | +0.001 |
| poly3_de00_nm | — | 0.843 | (new) |
| poly3_de00_powell | — | 0.789 | (new) |
| poly3 | 0.868 | 0.869 | +0.001 |
| svm | 0.993 | 0.994 | +0.001 |
| gradient_boost | 1.389 | 1.410 | +0.021 |
| mlp_deep | 1.612 | 1.633 | +0.021 |
| mlp_shallow | 1.847 | 1.879 | +0.032 |
| random_forest | 2.244 | 2.349 | +0.105 |
| knn | 2.135 | 2.132 | -0.003 |
| decision_tree | 4.352 | 4.413 | +0.061 |
| lasso | 8.467 | 8.394 | -0.073 |
| elastic | 8.500 | 8.446 | -0.054 |
| ridge | 8.471 | 8.443 | -0.028 |
| pcr | 8.471 | 8.443 | -0.028 |
| plsr | 8.630 | 8.476 | -0.154 |

### FOGRA51-CMY (n 818 -> 795)

| model | before | after | delta |
|---|---|---|---|
| gaussian_process | 0.056 | 0.057 | +0.001 |
| poly3_de00_nm | 0.372 | 0.360 | -0.012 |
| poly3_de00_powell | 0.375 | 0.372 | -0.003 |
| poly3 | 0.368 | 0.369 | +0.001 |
| svm | 0.766 | 0.796 | +0.030 |
| gradient_boost | 0.819 | 0.834 | +0.015 |
| mlp_deep | 1.176 | 1.083 | -0.093 |
| mlp_shallow | 1.408 | 1.304 | -0.104 |
| random_forest | 1.804 | 1.887 | +0.083 |
| knn | 1.867 | 1.905 | +0.038 |
| decision_tree | 4.320 | 4.412 | +0.092 |
| lasso | 6.658 | 6.642 | -0.016 |
| elastic | 6.692 | 6.640 | -0.052 |
| ridge | 6.719 | 6.658 | -0.061 |
| pcr | 6.719 | 6.658 | -0.061 |
| plsr | 6.715 | 6.650 | -0.065 |

### FOGRA51-CMYK (n 1617 -> 1588)

| model | before | after | delta |
|---|---|---|---|
| gaussian_process | 0.067 | 0.069 | +0.002 |
| poly3_de00_nm | — | 0.768 | (new) |
| poly3_de00_powell | — | 0.785 | (new) |
| poly3 | 0.822 | 0.816 | -0.006 |
| svm | 1.006 | 0.996 | -0.010 |
| gradient_boost | 1.539 | 1.508 | -0.031 |
| mlp_deep | 1.639 | 1.658 | +0.019 |
| mlp_shallow | 1.955 | 1.957 | +0.002 |
| random_forest | 2.567 | 2.635 | +0.068 |
| knn | 2.151 | 2.216 | +0.065 |
| decision_tree | 4.464 | 4.493 | +0.029 |
| lasso | 8.771 | 8.699 | -0.072 |
| elastic | 8.790 | 8.780 | -0.010 |
| ridge | 8.820 | 8.788 | -0.032 |
| pcr | 8.820 | 8.788 | -0.032 |
| plsr | 8.699 | 8.596 | -0.103 |

## Side effects worth knowing

- `gp_verification.csv`: the coated sets no longer emit a `noise_floor` row at all, because no
  duplicate recipe survives. Those rows read 0.000 and were never a valid noise estimate (the pairs
  were byte-identical); they are now gone rather than merely annotated. IFRA keeps its noise_floor
  rows, which are meaningful. `verify_gp.py` gained `--datasets` (merge-update, like `run.py`) so the
  coated rows could be recomputed without disturbing the IFRA and n>4 rows — verified: all 78
  non-coated rows are byte-identical to the previous CSV. Its `kfold` medians reproduce `run.py`'s
  summary values exactly on all 12 coated cells.
- What the coated `grouped` rows in that CSV *mean* has changed, and the paper should not over-read
  them. After dedup every coated row is its own group, so GroupKFold does no grouping — it just yields
  a different (deterministic, unshuffled) partition than the seeded shuffled KFold. The two differ by
  ~0.000-0.008 dE00, which is ordinary partition variation an order of magnitude below the seed-to-seed
  spread, not leakage. They remain a genuine leakage check only on IFRA.
- `fig_llm_vs_classical.py` no longer hardcodes n=818 — it reads n from the summary CSV.
- `fig_vs_colourbill.py`: the step-2 "same model class under OUR protocol" bridge now uses the
  deduplicated rows (1588), since leaving it un-deduplicated would reintroduce the double-counting into
  a published number. The step-1 colourbill reproduction check still uses colourbill's own rows (1617)
  because that is the point of a reproduction check, and colourbill is a compiled external tool we
  cannot re-run. That 29-row mismatch between the two series is documented in the script.
- **IFRA and the n>4 ladder are untouched** by design: no IFRA or KCMYG-5/CMYKOGV-7/CMYKOGB-7 result
  was re-run or rewritten. IFRA's duplicate recipes genuinely differ (~0.6-0.8 dE00 press
  repeatability) and deduplicating them would destroy signal; KCMYG-5 and CMYKOGB-7 contain no
  byte-identical duplicates, so the policy is a no-op there; CMYKOGV-7 already had it on.

## What the paper must now say

The protocol sentence that claimed "folds are grouped by distinct input recipe" was false and must not
be restored. The accurate statement is: *byte-identical duplicate rows are removed at load on every
coated dataset (795 CMY / 1588 CMYK / 3302 for CMYKOGV-7), after which no repeated input recipe
remains and a seeded shuffled 5-fold KFold cannot leak; the IFRA newsprint sets retain their repeated
recipes, whose measurements genuinely differ, and are reported per press run.*
