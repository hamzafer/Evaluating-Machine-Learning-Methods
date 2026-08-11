# 08 — Multi-LLM Colour Predictor Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.
> Load the `claude-api` skill only if calling Anthropic directly; here we route via OpenRouter.

**Goal:** Turn the single-model GPT experiment (Plan 04) into a proper multi-LLM comparison —
**Claude Fable, Claude Opus, GPT (latest), DeepSeek (latest)** — predicting colour from ink
values, scored identically to the classical models. Phil made Claude a reviewer requirement.

**Architecture:** The existing `journal/llm/` harness is provider-agnostic. Add an OpenRouter
client so one API + one key drives all models; keep the leakage-guarded split, JSON parsing, raw-
response archiving, and ΔE00 scoring from Plan 04. Purpose is to **quantify**, not to beat classical.

**Tech Stack:** `journal/llm/`, OpenRouter (OpenAI-compatible API), the `openai` SDK pointed at
`https://openrouter.ai/api/v1`. **Gate: needs `OPENROUTER_API_KEY` from Hamza before any paid call.**

## Global Constraints
- Same data discipline as Plan 04: PC10-CMY (818 K=0), seeded split, test patches never in the in-context examples.
- Temperature 0; deterministic; every raw response saved to `journal/llm/raw/<model>/...` (no key material).
- Cost cap: print an estimate before running; require `--yes`. Report n_parsed per model; parse-failure worst-case row.
- Model IDs are OpenRouter slugs (resolve exact current slugs at execution — e.g. anthropic/claude-*, openai/*, deepseek/*).

---

### Task 1: OpenRouter client (no key needed to write; needs key to run)
**Files:** Create `journal/llm/openrouter.py`; Test `journal/llm/tests/test_openrouter.py`.
**Produces:** `predict_xyz(model_slug, prompt) -> str` (raw content), retry-once, temp 0, max_tokens ~200.

- [ ] Step 1: Failing test with a mocked HTTP client — asserts the request targets the OpenRouter base URL,
  sends temperature 0, and returns `choices[0].message.content`; retries once on error. (No real call.)
- [ ] Step 2: Run → fails.
- [ ] Step 3: Implement using the `openai` SDK with `base_url=https://openrouter.ai/api/v1` and
  `api_key=os.environ['OPENROUTER_API_KEY']`. Key ONLY from env, never written/logged.
- [ ] Step 4: Run tests → pass.
- [ ] Step 5: Commit (no trailer).

### Task 2: Multi-model runner  *(gate: OPENROUTER_API_KEY)*
**Files:** Create `journal/llm/run_multi_llm.py`; Output `journal/llm/raw/<model>/PC10-CMY/*.json`,
`journal/results/llm/PC10-CMY_multi_summary.csv`.

- [ ] Step 1: Reuse `journal.llm.protocol.build_split/build_prompt/parse_xyz` and
  `journal.pipeline.color.delta_e00`. For each model slug: 400 in-context examples, 100 test queries,
  save raw responses, parse, score. Worst-case penalty row = dataset p99 (or documented constant), reported separately.
- [ ] Step 2: Print cost estimate from OpenRouter per-model pricing; require `--yes`. Dry-run 3 samples/model first.
- [ ] Step 3: Full run for all four models. Abort a model early if >20% parses fail; report.
- [ ] Step 4: Write `PC10-CMY_multi_summary.csv` (model_id, median, p95, max, mean, n_parsed, n_total). Commit code+CSV+raw.

### Task 3: Comparison table + figure
- [ ] Merge multi-LLM summary with classical `journal/results/PC10-CMY/summary.csv` → `journal/results/llm/comparison.csv`.
- [ ] Update/extend `fig_llm_vs_classical` to show all four LLMs vs classical (dataviz skill; methodology caveat in caption:
  classical = 5-fold CV/818 rows, LLM = 100-sample holdout/400 in-context). Commit.

### Acceptance
- Four LLMs scored with identical ΔE00 machinery; Claude included (reviewer requirement satisfied).
- Honest framing: where each LLM lands vs classical, with the CV-vs-holdout caveat stated.
