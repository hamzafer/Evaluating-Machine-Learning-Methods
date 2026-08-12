# LLM-as-equation-generator — the equations the models actually returned

Plan 09. Run date 12 Aug 2026. Dataset **PC10-CMY**, seeded split
`build_split('PC10-CMY', n_train=400, n_test=100, seed=42)` — the same split Plan 04's
GPT-4o live-prediction run used, so the 100 held-out patches are identical. The models
were shown the **first 150 rows** of that train block as `(c, m, y) -> (X, Y, Z)` pairs
(a budget choice: the full 400 would have tripled the input cost).

Prompt: `journal/llm/raw/equation/prompt.txt` (verbatim, includes Phil Green's wording
from `docs/LINKS.md` unchanged). Raw responses: `journal/llm/raw/equation/<model>.json`
(full body incl. usage) and `.txt` (content only). Re-score everything without spending
anything:

```
PYTHONPATH=. .venv/bin/python journal/llm/run_equation.py --from-archive
```

Scoring is the repo's own ΔE00 path — `journal.pipeline.color.delta_e00` on
denormalized XYZ (0–100, D50) — identical to every classical model in
`journal/results/PC10-CMY/summary.csv`.

## Summary (full table in `equation_summary.csv`)

| model | parsed | terms | max total degree | max single-var exponent | median ΔE00 | p95 | max | mean | charged |
|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-sol` | yes | 192 | 9 | 3 | **3.070** | 7.312 | 10.318 | 3.628 | $0.1131 |
| `deepseek/deepseek-v4-pro` | yes | 39 | 3 | 3 | **23.764** | 42.444 | 49.739 | 24.412 | $0.0013 |
| `anthropic/claude-fable-5` | no answer | — | — | — | — | — | — | — | $0.1518 (+$0.1368 dead attempt) |
| `poly3-ls-local` (not an LLM) | — | 57 | 3 | 3 | **0.234** | 0.917 | 3.052 | 0.370 | $0 |

`poly3-ls-local` is the honest comparator: a least-squares 3rd-order polynomial fitted
on **the same 150 training rows** and scored on **the same 100 test rows**. (The repo's
headline `poly3` row — median 0.268 / p95 0.982 / max 6.795 — is 5-fold CV over all 795
rows, a different protocol; the 150-row fit landing at 0.234 confirms 150 rows is
already plenty for a cubic.)

**Finding:** the best LLM-written equation is **13x worse than the least-squares cubic
it was competing with** on the same data (median 3.07 vs 0.234 ΔE00), and its maximum
error is 10.3 ΔE00 against 3.05. Phil's success criterion (minimise average *and*
maximum difference) is not met by either model. An LLM asked to *write* the transform
is not competitive with fitting the transform; that is a different question from the
Plan 04 result, where GPT-4o used as a live predictor reached median 3.03 on these same
100 patches — i.e. gpt-5.6-sol's *portable equation* is about as good as GPT-4o's
*live per-patch prediction*, and both are an order of magnitude behind least squares.

## `openai/gpt-5.6-sol` — median 3.070 ΔE00

A Beer–Lambert-flavoured **product** of per-ink cubics plus two interaction terms. Note
what this does to Phil's constraint: every single-variable exponent is ≤ 3 (his literal
wording is satisfied), but multiplying three cubics gives an expanded total degree of
**9** and 192 terms — the opposite of "as simple as possible". Verified independently:
a hand transcription of these three lines into numpy agrees with the sympy-parsed
version to 2e-14, and it predicts (85.2, 87.4, 73.9) at c=m=y=0, a sensible coated
white.

```
X = 85.2*(1-1.3*(c/100)+0.572*(c/100)**2-0.102*(c/100)**3)*(1-0.9834*(m/100)+0.5556*(m/100)**2-0.2222*(m/100)**3)*(1-0.33*(y/100)+0.18*(y/100)**2-0.05*(y/100)**3)-5.0*(c/100)*(y/100)+5.5*(c/100)*(m/100)*(y/100)
Y = 87.4*(1-0.95*(c/100)+0.25*(c/100)**2-0.05*(c/100)**3)*(1-1.25*(m/100)+0.36*(m/100)**2+0.07*(m/100)**3)*(1-0.2*(y/100)+0.04*(y/100)**2-0.01*(y/100)**3)-2.0*(c/100)*(y/100)+3.2*(c/100)*(m/100)*(y/100)
Z = 73.9*(1-0.4*(c/100)+0.12*(c/100)**2-0.02*(c/100)**3)*(1-1.1*(m/100)+0.35*(m/100)**2-0.03*(m/100)**3)*(1-1.7*(y/100)+1.1*(y/100)**2-0.32*(y/100)**3)+3.0*(c/100)*(y/100)-0.6*(c/100)*(m/100)*(y/100)
```

## `deepseek/deepseek-v4-pro` — median 23.764 ΔE00

A textbook-shaped full cubic with coefficients that were clearly **written down rather
than fitted**: the three channels share nearly proportional coefficients, so the
equation cannot produce chroma. It predicts (52.0, 52.0, 46.4) — a mid grey — for a
patch measured at (57.96, 63.76, 6.53), a saturated yellow. That, not a parsing
problem, is where median 23.8 comes from. Caveat: this answer was produced with
**reasoning disabled**, forced by the budget (see anomalies below); the reasoning
transcript of its earlier, unaffordable attempts shows it working through the data
seriously.

```
X = 81.83 - 0.654*c - 0.598*m - 0.382*y + 0.00212*c**2 + 0.00345*m**2 + 0.00198*y**2 + 0.00187*c*m + 0.00093*c*y + 0.00112*m*y - 0.000008*c**3 - 0.000012*m**3 - 0.000006*y**3
Y = 84.95 - 0.718*c - 0.662*m - 0.415*y + 0.00245*c**2 + 0.00382*m**2 + 0.00215*y**2 + 0.00203*c*m + 0.00101*c*y + 0.00124*m*y - 0.000009*c**3 - 0.000013*m**3 - 0.000007*y**3
Z = 73.12 - 0.582*c - 0.531*m - 0.345*y + 0.00189*c**2 + 0.00305*m**2 + 0.00176*y**2 + 0.00165*c*m + 0.00082*c*y + 0.00098*m*y - 0.000007*c**3 - 0.000010*m**3 - 0.000005*y**3
```

## `anthropic/claude-fable-5` — no equation, $0.29 spent

Recorded outcome, not a hidden failure. Three paid/attempted calls, four archived
responses:

| attempt | routing | setting | outcome | charged |
|---|---|---|---|---|
| 1 (`.attempt1-truncated.json`) | Amazon Bedrock | default (thinking on) | `finish_reason=length`, all 1600 completion tokens spent thinking, `content=null` | $0.1368 |
| 2 | Amazon Bedrock | `reasoning.enabled=false` | HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled" | $0 |
| 3 | Amazon Bedrock | `reasoning.max_tokens=1024` | **refusal**: "triggered restrictions on violative cyber content" — a false positive on a prompt containing nothing but colour measurements | $0 |
| 4 (`.json`) | Anthropic (pinned) | `reasoning.max_tokens=1024`, `max_tokens=1900` | `finish_reason=length` again; the thinking budget was not honoured as a hard cap, `content=null` | $0.1518 |

The engineering finding is a cost one: at $10/M in and $50/M out, a frontier
reasoning model that cannot be told to stop thinking costs ~$0.15 **per attempt that
returns nothing**. A fourth attempt with `max_tokens=3000` would have cost ~$0.21 and
breached the $0.15 hard credit floor, so it was not made. Its reasoning transcript
(archived) shows it abandoning a least-squares fit as "too tedious by hand" and
switching to a Beer–Lambert product model — the same idea gpt-5.6-sol shipped.

## Anomalies (all investigated)

1. **Reasoning tokens, not answers, are the budget risk.** deepseek-v4-pro at
   `effort=low` spent 2400 then 8000 completion tokens thinking and returned
   `content=null` both times ($0.0033 + $0.0092); fable did the same twice at $0.14–0.15
   a go. Only `reasoning.enabled=false` (deepseek) fixed it; fable rejects that flag.
   `max_tokens` must be treated as a *reasoning* budget for these models.
2. **`GET /credits` lags one to two calls behind.** After the first three-model run it
   reported $0.5339 available while the per-call `usage.cost` values summed to $0.259
   spent (true available $0.400, confirmed on a later poll). Every runner therefore
   prints both, and the per-call `usage.cost` is treated as authoritative.
3. **Bedrock content filter false-positive** on a prompt of pure colour data (see the
   fable table). Fixed by pinning `provider: {"order": ["anthropic"]}`. Cost $0.
4. **Prompt-token estimates.** The usual ~4 chars/token heuristic understates a
   digit-dense prompt by ~2.3x: the 8438-char prompt billed 4902 tokens (1.72
   chars/token). The runner's estimator was corrected mid-run.
5. **gpt-5.6-sol's temperature.** Sent as 0 and accepted. For fable, OpenRouter may
   override temperature when Anthropic extended thinking is active; noted in the CSV
   because it makes that model's (absent) answer non-deterministic in principle.

## Not run: the Phase 2 prediction pilot

`journal/results/llm/predict_summary.csv` was **not** produced. Credit after Phase 1 was
$0.244, below the $0.25 threshold set for starting a 3-model, 20-sample pilot, and the
$0.15 floor left only ~$0.09 usable — not enough for even two models at a consistent n.
The client already carries the prompt-caching path (`chat(..., cache_prefix=...)`) that
would make such a pilot cost roughly $0.09–0.12 for 20 samples x 2 models if credit is
topped up. Plan 04's GPT-4o numbers (median 3.034 / p95 10.642 / max 14.621, n=100)
remain the only live-prediction measurements and are untouched.
