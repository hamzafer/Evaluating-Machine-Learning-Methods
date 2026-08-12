# Blind verification task — printer colour characterization

You are an INDEPENDENT verifier. Another team has produced results for a journal paper.
Your job: implement this specification FROM SCRATCH and report what YOU get. You have not
seen their code and MUST NOT look at it (see Rules). Disagreement with them is a valid and
valuable outcome — do not try to match anyone's numbers.

## The task being verified
For a printing system with n inks, predict measured colour from ink percentages:
input = n ink values (0-100 scale), output = CIE XYZ tristimulus (0-100 scale),
error metric = CIEDE2000 colour difference between predicted and measured colour in CIELAB
under illuminant D50 and the CIE 1931 2-degree standard observer.

## Data (in ./data, all raw as supplied by the sources)
| file | inks | notes |
|---|---|---|
| APTEC_PC10_CardBoard_2023_v1.txt | CMYK | CGATS/ISO28178 text. Contains CMYK + CIELAB (no XYZ). |
| APTEC_PC11_CCNB_2023_v1.txt | CMYK | same format |
| FOGRA51.txt | CMYK | same format |
| KCMYG_5clr_spectral.txt | 5 | X-Rite ProfileMaker export; ink columns + 36 spectral bands 380-730nm at 10nm |
| APTEC_CMYKOGV_7clr_xyzlab.txt | 7 | ink columns + native XYZ and Lab |
| Apex_CMYKOGB_7clr_spectral.txt | 7 | ink columns + 36 spectral bands |
| Age_64a_wb.txt, PressJ_158_wb.txt | CMYK | IFRA 2005 newsprint press runs; CMYK + 36 spectral bands |

Where only spectral reflectance is given, integrate it to XYZ yourself under D50/2-degree.
Where only Lab is given, derive XYZ yourself (D50 white point). Decide and DOCUMENT the ink
scale of each file (0-1 vs 0-100) and the exact number of measurement rows you find.

## Protocol to implement
1. Two input variants for the CMYK datasets: (a) **CMY**: use only rows where K = 0, with C, M, Y
   as inputs; (b) **CMYK**: all rows, with K as a fourth input. Never drop a column while keeping
   its rows.
2. 5-fold cross-validation, seed 42. Rows sharing an identical ink recipe must fall in the same
   fold (no leakage through repeated measurements). Every row is held out exactly once, and
   statistics pool over the whole dataset.
3. Inside each fold: fit a MinMax scaler on the TRAINING fold's inputs and, separately, on the
   training fold's XYZ targets; fit the model in normalized space; inverse-transform predictions
   back to physical XYZ before computing any error; clip negative tristimulus values to zero.
4. Report median, 95th percentile and maximum of the per-sample CIEDE2000, 3 decimals.
5. Models to fit (use scikit-learn defaults except as stated):
   - `poly3`: 3rd-order polynomial features + ordinary least squares
   - `svm`: RBF-kernel SVR, C=10, gamma='scale', epsilon=0.01, one regressor per output
   - `knn`: k=5, uniform weights
   - `gaussian_process`: kernel = ConstantKernel * RBF + WhiteKernel(noise_level=1e-3,
     noise_level_bounds=(1e-9, 1e5)), normalize_y=True, n_restarts_optimizer=15, seed 42.
     If a training fold exceeds 2000 rows, fit on a seed-42 random subsample of 2000 rows.
6. Datasets to cover: PC10 (CMY and CMYK), PC11 (CMY), FOGRA51 (CMY), KCMYG-5, CMYKOGV-7,
   CMYKOGB-7, and 5-fold within-run CV on each of the two IFRA runs.

## Additional checks (report findings, do not assume)
- Any duplicate rows? Distinguish "identical ink recipe with identical measured values" from
  "identical recipe, different measurements" and say how many of each, per file.
- For files with BOTH spectral and native XYZ/Lab: does your integration agree with the native
  values? Quantify.
- Does each file's own XYZ<->Lab pair agree under D50/2-degree? Quantify.
- Read each file's header metadata and say whether it is self-consistent with its filename and
  content. Report anything odd.

## Rules (blindness)
- Work ONLY in `~/blind_verify`. Use `~/blind_verify/.venv/bin/python` (pinned: numpy 2.3.3,
  scikit-learn 1.7.2, scipy 1.16.2, colour-science 0.4.6).
- **You must NOT read the other team's implementation.** Do not open, list, grep or copy
  anything under `~/Desktop/HAMZA/Evaluating-Machine-Learning-Methods/journal/pipeline`,
  `journal/llm`, `journal/figures`, `docs/`, or `.superpowers/`. Do not read their git history.
  If you read any of it, the review is void — say so explicitly in your report.
- Write your own numbers to `~/blind_verify/blind_results.csv` FIRST and state you have done so.
- ONLY AFTER that file exists may you compare against their published result CSVs, which are
  data-only and located at
  `~/Desktop/HAMZA/Evaluating-Machine-Learning-Methods/journal/results/<dataset>/summary.csv`
  and `.../journal/results/ifra/within_run.csv`. Comparing is the point; borrowing code is not.
- Where you disagree with them, investigate on YOUR side and say what you think is right and why.
  Where you agree, say how closely (to how many decimals).
