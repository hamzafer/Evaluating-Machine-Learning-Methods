# Duplicate-row leakage in the n<=4 cross-validation (found 12 Aug 2026)

Found by the independent blind reviewer (`journal/verification/blind-2026-08-12/`), then
measured directly by the coordinator. **Not yet fixed** — this note is the evidence base.

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
