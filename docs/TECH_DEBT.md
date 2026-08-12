# Tech debt / deferred cleanups

Deliberately postponed. Each item says what it is, why it was deferred, and what fixing it costs.

## 1. Proprietary datasets are in the public repo's git history
**Status:** deferred to after publication (Hamza's call, 12 Aug 2026).

`hamzafer/Evaluating-Machine-Learning-Methods` is a **public** repo. Commit `df8caf8`
(plan-06 ingestion) force-added files that `.gitignore` declares local-only:

- `journal/data/raw/ncolor/{KCMYG_5clr_spectral,Apex_CMYKOGB_7clr_spectral,APTEC_CMYKOGV_7clr_xyzlab}.txt`
- `journal/data/processed/ncolor/{KCMYG-5,CMYKOGV-7,CMYKOGB-7}.csv`

`3c8a7ca` untracked them, so they are **absent from the current tree**, but the push on
12 Aug included `df8caf8`, so the blobs remain reachable through history. Of the three raw
sets, APTEC CMYKOGV is publicly redistributable (ICC registry; the colourbill tool ships it);
**KCMYG and Apex CMYKOGB came privately from Phil** and KCMYG's provenance is still to be
confirmed with him.

Also already public from earlier pushes (same policy, predates this work): `data/cleaned/*`
(PC10/PC11/FOGRA51) and the 13 `journal/data/processed/ifra/wb/*.csv`.

**Fix options when we return to it:**
1. `gh repo edit … --visibility private` — instant, effective, covers the earlier pushes too.
2. `git filter-repo` to strip the paths + force-push. Rewrites ~20 commit SHAs, which breaks
   the ~48 SHA citations in `docs/`, `.superpowers/sdd/progress.md`, `journal/results/logs/README.md`
   and the paper repo's commit messages; needs every clone re-cloned; and GitHub retains
   unreferenced blobs until a GC only Support can trigger — so it does not stand alone.
**Recommended sequence:** private now-or-at-any-time, history rewrite (if wanted) at
publication, when the SHA provenance trail no longer needs to resolve.

## 2. GP leave-one-out numbers are not model-comparable
**Status:** needs one sentence in the paper's methods (do during the plan-07 trim).

In the IFRA leave-one-out protocol the GP fits a seed-42 subsample of 2,000 rows
(`FitSubsampled`) while poly3/SVM/MLP fit all 17,820 pooled rows. The LOO table therefore
understates GP relative to the others — conservative in our favour's opposite direction, but
it must be stated rather than left for a reviewer to notice.

## 3. No behavioural guard distinguishing `n_restarts_optimizer=15` from `10`
**Status:** accepted, documented.

`test_gp_config.py` pins 15 by assertion, and the collapse fixture fails decisively for the
genuinely broken configs, but no test separates 15 from 10 — the only case that needs 15 is
marca_133's pooled LOO fit, which is far too slow for a unit test. Recorded in the plan-10
gate findings.
