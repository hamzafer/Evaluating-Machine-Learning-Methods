# Post-submission checklist

**Paper:** "AI and Machine Learning Methods for n-Colorant Printer Characterization"
**Journal:** MDPI Technologies, manuscript technologies-4564760, submitted 30 Aug 2026
**Editor contact:** Tyler Yu, tyler.yu@mdpi.com (assigned 2 Sep 2026)
**Status:** pending review. Reviewer reports expected roughly 2–4 weeks after 2 Sep.

This is the single to-do list for the period between submission and the first
reviewer reports. Paper-writing items live in `../MDPI-Phil-Journal/REVISION-TODO.md`;
this file covers everything else and points there for the rest.

---

## 1. Decisions to make (need you, or you + Phil)

### 1a. Preprint on Preprints.org — ask Phil
MDPI sent an automated invitation to post the submitted PDF as a preprint
(2 Sep). Optional, no effect on peer review, permanent once posted.

- **Why maybe yes:** time-stamped DOI, readable and citable now.
- **Why maybe no:** the posted version is permanent even if reviewers force
  big changes; the CMYKOGB provider must not be named (see 1b), so the posted
  PDF must be the *current* main.tex build, not the submitted one, which still
  says "Apex" in the data availability statement.
- **Action:** one-line question to Phil. If yes, rebuild main.pdf first, then
  click "Upload to Preprints.org" in the MDPI submission system.

### 1b. Newsprint CSVs are in the public code repo — decide, then tell Phil
The 13 processed IFRA white-backing CSVs under `journal/data/processed/ifra/wb/`
are tracked and pushed to GitHub. Each holds full CMYK + Lab + XYZ per patch,
i.e. the dataset itself. The paper's data availability statement says the
newsprint data is licensed and *not* redistributed there.

- **Why it matters:** contradicts the published statement; may breach the
  licence Phil holds.
- **Fix options:**
  1. `git rm` the files and commit (they stay in history, weak fix).
  2. Rewrite history to purge them (`git filter-repo`), force-push, and ask
     GitHub support to clear cached views. Proper fix, destructive, needs a
     backup clone first.
- **Decision (2 Sep):** fix quietly first, mention to Phil afterwards. Not
  raised in the 2 Sep reply. The code still runs because the raw zip and the
  ingester are local-only.

---

## 2. Work to do in this repo (code repo)

### 2a. Scrub "Apex" from the public repo
Phil's condition for the CMYKOGB data: research use only, **source not named,
data not shared**. The raw file is gitignored and only `summary.csv` aggregates
are tracked, but the name appears in:

| where | what |
|---|---|
| `journal/pipeline/ingest_ncolor.py` | docstring + filename constant `Apex_CMYKOGB_7clr_spectral.txt` |
| `journal/pipeline/datasets.py` | comment ("Apex's 20 genuine paper-white repeats") |
| `journal/pipeline/tests/test_ncolor.py` | comment |
| `journal/data/raw/ncolor/README.md` | table row, original filename "Apex Averaged 2.txt" |
| `journal/verification/blind-2026-08-12/` | SPEC.md, BLIND_REPORT.md, work/datasets.py |
| `docs/DATA.md`, `docs/LINKS.md`, `docs/PAPER_GUIDE.md`, `docs/TECH_DEBT.md`, `docs/plans/06-ncolor-ladder.md`, `docs/plans/journal_roadmap.md`, `docs/audit/2026-08-29-paper-audit/02-figures-tables-prose.md`, `docs/STATUS.md` | prose mentions |

- **Plan:** rename the local raw file to `CMYKOGB_7clr_spectral.txt`, update the
  ingester constant and tests, replace "Apex" with "CMYKOGB-7 (confidential
  commercial)" everywhere, re-run the 41 tests, commit, push.
- Name stays in git history. Acceptable: the *data* was never in history.

### 2b. Save the editor's details
Already in `docs/STATUS.md`. Nothing more to do unless Phil asks.

---

## 3. Work to do in the paper repo (`../MDPI-Phil-Journal`)

Already done on 2 Sep (commit cf56575):
- "Apex" removed from the data availability statement; CMYKOGB described as
  confidential commercial data that can be neither identified nor shared.
- Phil's chapter reference corrected: *Fundamentals of Device Characterization*,
  in *Fundamentals and Applications of Colour Engineering* (Wiley 2023, ed. Green),
  ch. 3, pp. 53–70, doi:10.1002/9781119827214.ch3. Bib key is now
  `green2023fundamentals`.
- CharData citation carries version 1.18.0. Note it is not cited in the current
  text (the External Validation section was cut); re-cite it or drop the entry.

Still open, in priority order (details in `REVISION-TODO.md` there):
1. **Prior-work comparison table.** Phil's main ask. Numbers read out of the
   papers, never from memory. Rows = prior papers, columns = median / mean /
   max ΔE by colorant count.
2. **Implementation-details appendix.** Compact table: model, objective, key
   equations, hyperparameters, pointer to the code repo.
3. **Extra references.** Only those that add something to the table above.
4. Leftovers: Phil's reference for "outliers dominate the mean" (comment c323);
   log-XYZ / colorimetric density in the transform sweep (c318); check figures
   for dataset names Phil removed from prose; PC10/PC11 substrate wording.

---

## 4. When the reviews arrive

- Read the decision letter, list every reviewer point in
  `REVISION-TODO.md` with a one-line planned response.
- MDPI revision windows are short (often 10 days). Items 3.1 and 3.2 are the
  slow ones, so doing them *before* the reviews come back buys that time.
- Only trust emails from `@mdpi.com` addresses.

---

## Suggested order

1. Message Phil: preprint yes/no (1a). Sent 2 Sep. Newsprint CSVs (1b): fix first, tell him after.
2. Scrub "Apex" from this repo (2a). Half an hour.
3. Start the prior-work table (3.1). This is the long pole.
