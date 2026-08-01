# Task 2 report — LLM API runner + scoring (docs/plans/04-llm-predictor.md)

## Provider change

Plan specified Claude models; this run used an **OpenAI key** supplied by the
user instead, per explicit instruction. Ran `gpt-4o` (strong) and
`gpt-4o-mini` (small) via the `openai` Python SDK (v2.50.0), chat
completions, `temperature=0`, `max_tokens=200`. The `claude-api` skill was
NOT loaded (wrong provider).

## What was built

- `journal/llm/run_llm.py` — CLI runner:
  - Builds the split via `journal.llm.protocol.build_split` (PC10-CMY,
    `n_train=400` in-context examples, `n_test=100` queries, seed 42 —
    identical task definition to the classical models).
  - One API call per test query per model, `temperature=0`, `max_tokens=200`.
  - Retries once on `openai.APIError` with a 2s sleep before recording a
    failure.
  - Saves each raw response body (no auth headers, no request headers) to
    `journal/llm/raw/<model>/PC10-CMY/<sample_id>.json`.
  - Parses predictions with `journal.llm.protocol.parse_xyz` and scores with
    `journal.pipeline.color.delta_e00` against true XYZ.
  - `summarize()` (median/p95/max/mean/n via numpy) reimplemented locally in
    the runner — `journal.pipeline.evaluate` imports sklearn, which is not
    installed in the venv used for this run.
  - Prints a cost estimate (rough ~8k prompt tokens/query assumption) before
    calling the API; requires `--yes` to proceed (cost gate).
  - Early-abort guard: if `n_fail / n_total > 20%` after at least 20 samples
    for a given model, stops that model's run and reports.
  - Writes `journal/results/llm/PC10-CMY_summary.csv` with two rows per
    model: `parsed_only` (score only successfully parsed predictions) and
    `worst_case` (failed parses substituted with a documented penalty
    constant of ΔE00 = 30.0, so parse failures can't silently flatter a
    model). Columns: `model_id,variant,median,p95,max,mean,n_parsed,n_total,penalty`.

## Runs performed

1. Dry run, `--max-test-samples 3`, both models — verified JSON parsing,
   raw-response format (confirmed no auth headers / key material present),
   and rough cost math before committing to the full run.
2. Full run, `--n-train 400 --n-test 100`, both models, 200 total API calls.
   Completed with exit code 0, no early abort triggered for either model.

## Results (full 100-sample run)

| model | n_parsed/n_total | median | p95 | max |
|---|---|---|---|---|
| gpt-4o | 100/100 | 3.034 | 10.642 | 14.621 |
| gpt-4o-mini | 100/100 | 9.445 | 29.928 | 44.792 |

Both models parsed 100% of responses, so `parsed_only` and `worst_case`
summary rows are identical for this run (no penalty substitutions applied).
No parse-failure abort triggered for either model (0% failures, threshold
was >20%).

For context, the classical baselines on the same PC10-CMY dataset
(`journal/results/PC10-CMY/summary.csv`, note: those are 5-fold pooled CV
over 818 rows, not a 100-row holdout, so not a strictly apples-to-apples
comparison — see Task 3 for the proper merge) range from gaussian_process
median 0.054 (best) to plsr median 6.643 (worst). Placed on that scale:
gpt-4o's median (3.034) falls mid-field, beating the linear-model cluster
(ridge/pcr/elastic/lasso/plsr, medians ~6.6) and decision_tree (4.369), but
behind everything else (gp, poly3, svm, gradient_boost, mlp, random_forest,
knn — all median < 2.1). gpt-4o-mini's median (9.445) is worse than every
classical model in the table.

## Cost

- Printed pre-run estimate (rough 8k-token/query assumption): **$2.18** for
  100 samples across both models.
- Actual cost, computed from the `usage` fields in the saved raw responses
  (prompt tokens ran ~12,084/query, higher than the rough estimate because
  400 in-context examples is a long prompt):
  - gpt-4o: 100 calls, 1,208,400 prompt tokens + 2,789 completion tokens ≈ **$3.05**
  - gpt-4o-mini: 100 calls, 1,208,400 prompt tokens + 2,292 completion tokens ≈ **$0.18**
  - **Total actual: ≈ $3.23** (within the pre-approved ~$3-5 budget).

## Key hygiene

- Key was loaded only as `OPENAI_API_KEY=$(cat <path>)` on the run command
  line, never read into any file, log, or this report.
- Verified (`grep -rl "sk-"` and `grep -c "Authorization"`) that no raw
  response file or log contains key material or auth headers.

## Files touched

- `journal/llm/run_llm.py` (new)
- `journal/llm/raw/gpt-4o/PC10-CMY/*.json` (100 files, new)
- `journal/llm/raw/gpt-4o-mini/PC10-CMY/*.json` (100 files, new)
- `journal/results/llm/PC10-CMY_summary.csv` (new)

## Concerns / follow-ups for Task 3

- The classical `journal/results/PC10-CMY/summary.csv` numbers come from
  5-fold pooled CV over all 818 K=0 rows; this LLM run scores a single
  100-row holdout with 400 rows held out as in-context examples (and the
  remaining ~318 rows unused). Task 3's comparison table should note this
  methodological difference rather than merge the numbers as if directly
  comparable.
- Rough cost-estimate constant (8k prompt tokens/query) undershot the actual
  prompt size (~12k tokens) for this prompt design (400 examples). Worth
  updating the estimate constant if this runner is reused with different
  `n_train` values.
