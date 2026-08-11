# 09 — LLM-as-Equation-Generator Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** Run Phil's brief — have an LLM emit a *portable* ink→colour equation (a communicable
polynomial, exponents <=3) rather than act as a live predictor — then score that equation's
avg/max ΔE00 with the same pipeline. This is the "solution isn't a permanent ChatGPT dependency" idea.

**Architecture:** Give the LLM training patches + Phil's prompt; parse the returned equation into an
evaluable form (sympy or a constrained coefficient vector); apply it to the held-out test patches;
score ΔE00. Reuses the OpenRouter client (Plan 08) and the ΔE00 scorer.

**Tech Stack:** `journal/llm/`, sympy (parse/evaluate the emitted equation), OpenRouter.

## Global Constraints
- Phil's exact prompt is the starting point (docs/LINKS.md); adapt minimally. Enforce/verify exponents <=3.
- Same split discipline (leakage-guarded); score the EQUATION on held-out patches, not the LLM live.
- Report avg AND max ΔE00 (Phil's success criterion) plus median/p95 for comparability.
- Run across the same models as Plan 08 (Claude Fable/Opus, GPT, DeepSeek) so predictor-vs-equation is comparable.

---

### Task 1: Equation parser/evaluator
**Files:** Create `journal/llm/equation.py`; Test `journal/llm/tests/test_equation.py`.
**Produces:** `parse_equation(text, input_names) -> callable(X)->XYZ`; `max_exponent(expr) -> int`.

- [ ] Step 1: Failing tests — parse a known cubic (e.g. "X = 2 + 3*C + 0.5*C^2*M ...") into a callable that
  reproduces expected values on sample inputs; `max_exponent` returns 3; reject/flag an equation with exponent 4.
- [ ] Step 2: Run → fails.
- [ ] Step 3: Implement with sympy: extract the three output expressions (X,Y,Z) as functions of the ink names;
  build a numpy-vectorized callable; compute max total degree.
- [ ] Step 4: Run tests → pass. [ ] Step 5: Commit (no trailer).

### Task 2: Prompt + run + score  *(gate: OPENROUTER_API_KEY)*
**Files:** Create `journal/llm/run_equation.py`; Output `journal/results/llm/equation_summary.csv`, raw responses.

- [ ] Step 1: Build the prompt from Phil's text + N training patches (ink -> XYZ). Ask for three equations
  (X,Y,Z) as functions of the inks, exponents <=3, plain text.
- [ ] Step 2: For each model: call via OpenRouter, save raw, `parse_equation`, verify max_exponent<=3
  (flag violations), evaluate on held-out test patches, `delta_e00` vs measured.
- [ ] Step 3: Write summary (model_id, avg, median, p95, max, max_exponent, n_test). Compare against poly3's numbers
  on the same split (poly3 is the human-written cubic baseline — the natural comparator).
- [ ] Step 4: Commit code + CSV + raw + the actual emitted equations (as text, provenance).

### Acceptance
- Each model's emitted equation is captured, validated (<=3 exponents), and scored; table vs poly3.
- Honest finding: does an LLM-written cubic approach the least-squares cubic? (Expected: worse, but quantified.)
