# Run history — how to go back in time

Summary CSVs (`journal/results/*/summary.csv`, `ifra/*.csv`) hold only the **latest**
value per (dataset, model): they are overwritten whenever a config changes. This directory
plus `journal/results/run_log.tsv` preserve the history they discard.

## `../run_log.tsv` — the audit trail (append-only)
One tab-separated line per model fit. Columns:

`ts_utc, script, protocol, dataset, model, median, p95, max, mean, n, seconds,
git_commit, git_dirty, host, os_arch, python, sklearn, numpy, scipy, notes`

Written automatically by `journal.pipeline.run` and `journal.pipeline.run_ifra` via
`journal/pipeline/runlog.py` — every fit records its accuracy (ΔE00 median/P95/max/mean),
its wall time, the machine it ran on, the package versions, and the git commit + dirty flag
of the tree it ran from. **Never rewritten, only appended**, so a number that later changes
still has its earlier value on record.

Useful queries:

```bash
# every GP result ever recorded for PC10-CMY, oldest first
awk -F'\t' '$4=="PC10-CMY" && $5=="gaussian_process"' journal/results/run_log.tsv

# slowest fits
sort -t$'\t' -k11 -g -r journal/results/run_log.tsv | head

# laptop vs remote for one dataset
awk -F'\t' '$4=="CMYKOGV-7"{print $14, $5, $6, $11}' journal/results/run_log.tsv
```

## `raw/` — verbatim stdout of the actual runs
The console logs the numbers came from, kept as-is (including convergence warnings):

| file | what |
|---|---|
| `stream{1,2,3}.log` | laptop plan-10 GP re-run streams (11–12 Aug) |
| `run_ogv_{powell,nm,gp}.log`, `run_ogb_powell.log` | plan-06 slow-model chunks; source of the 3699.7 s Powell / 10.3 s NM timings |
| `verify_gp_rerun.log` | `verify_gp.py` regenerated under the final GP config |
| `remote_gp_specs.log`, `remote_gp_ifra.log` | colourlab final-config GP mirror (cross-platform check) |
| `remote_gp_seed.log`, `remote_repro{1,2,3,4}.log` | colourlab replication matrix + seed sweeps |

## Provenance caveat on the backfill
The 469 rows present at first commit were **reconstructed** from the `raw/` logs, which print
accuracies and (for `run.py`) per-fit seconds but no per-line timestamp. Those rows carry
`ts_utc` = the log file's mtime and say so in `notes`; their `git_commit` is blank because it
cannot be recovered after the fact. Rows written from now on are exact.
