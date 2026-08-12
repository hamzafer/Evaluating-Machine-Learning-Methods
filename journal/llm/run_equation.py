"""Plan 09 — LLM-as-equation-generator: run Phil's brief and score the equation.

Phil's ask (11 Aug 2026 meeting, docs/LINKS.md) is NOT "use an LLM as a live
predictor" but "have an LLM write down a *portable* equation", so the deliverable
is a communicable polynomial that anyone can implement without an API key. This
script:

  1. builds the SAME seeded split as Plan 04 (`build_split('PC10-CMY', 400, 100,
     seed=42)`), so the 100 held-out test patches are the very rows the GPT-4o
     live-prediction run was scored on;
  2. shows the model the first `--n-train` rows of that train block as
     (ink -> XYZ) pairs (150 -- a deliberate budget choice, see the BUDGET note
     below) wrapped in Phil's verbatim prompt;
  3. asks for three equations X/Y/Z in the ink variables `c, m, y`, in a fenced
     block, so the answer is machine-parseable;
  4. archives the response verbatim (`journal/llm/raw/equation/<model>.txt` plus
     the full JSON body with usage in `<model>.json`);
  5. parses it with `journal.llm.equation` (tokenizer whitelist + sympy; the
     model's text is NEVER eval'd) and evaluates it on the held-out patches;
  6. scores it with `journal.pipeline.color.delta_e00` -- the identical ΔE00 path
     used by every classical model, on denormalized XYZ (0-100, D50);
  7. fits a least-squares 3rd-order polynomial on exactly the same 150 training
     rows and scores it on the same test rows: the human-written cubic the LLM's
     cubic has to be compared against (`poly3-ls-local` row of the CSV). The
     repo's headline poly3 number comes from 5-fold CV over all 795 rows and is
     NOT the right comparator for a 150-row fit.

BUDGET: total OpenRouter credit for this run was $0.66 with no server-side spend
cap, so the runner is the safety limit: `/credits` is polled before and after
every model, the balance is printed, and the run stops if available credit falls
below --floor (default $0.15). `--n-train 150` (rather than the full 400) keeps
the prompt near 4.9k tokens, which matters at claude-fable-5's $10/M input.

Usage:
    PYTHONPATH=<repo root> .venv/bin/python journal/llm/run_equation.py --yes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from journal.llm import openrouter as orr
from journal.llm.equation import EquationParseError, parse_equation
from journal.llm.protocol import build_split
from journal.pipeline.color import delta_e00

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "journal" / "llm" / "raw" / "equation"
RESULTS_DIR = REPO_ROOT / "journal" / "results" / "llm"

# Pinned model ids (OpenRouter slugs) + per-model call settings. Anthropic
# models have extended thinking OFF unless asked, so `reasoning` stays None
# there; the others get effort=low to stop reasoning tokens eating the budget.
# USD per 1M tokens from the OpenRouter catalogue on the run date.
MODELS = [
    # `reasoning` history (see the ANOMALIES note at the bottom of this
    # docstring block in the task report): at effort=low BOTH deepseek-v4-pro
    # and claude-fable-5 spent every allowed completion token on reasoning and
    # returned content=None with finish_reason=length -- $0.15 of dead calls.
    # They are re-run with reasoning explicitly disabled; gpt-5.6-sol answered
    # fine at effort=low and keeps that setting so its recorded run is the one
    # actually scored.
    {"id": "deepseek/deepseek-v4-pro", "max_tokens": 3000,
     "reasoning": {"enabled": False}, "price_in": 1.168, "price_out": 2.336,
     "note": "reasoning DISABLED (budget): with reasoning on it spent 2400 and "
             "then 8000 completion tokens thinking and returned no answer"},
    {"id": "openai/gpt-5.6-sol", "max_tokens": 3000,
     "reasoning": {"effort": "low"}, "price_in": 5.0, "price_out": 30.0,
     "note": "reasoning effort=low"},
    {"id": "anthropic/claude-fable-5", "max_tokens": 1900,
     "reasoning": {"max_tokens": 1024}, "price_in": 10.0, "price_out": 50.0,
     "provider": {"order": ["anthropic"], "allow_fallbacks": False},
     "note": "routed to the first-party Anthropic endpoint (the Amazon Bedrock "
             "route refused this colour-data prompt as 'violative cyber content', "
             "a false positive, at zero cost); "
             "thinking budget pinned to the 1024-token Anthropic minimum "
             "(budget): reasoning cannot be disabled on this endpoint (HTTP 400) "
             "and its default-on thinking had spent all 1600 completion tokens "
             "without answering; OpenRouter may override temperature to 1 when "
             "Anthropic extended thinking is active"},
]

# Phil Green's wording, verbatim from docs/LINKS.md (11 Aug 2026 meeting).
PHIL_PROMPT = (
    "Generate an equation that transforms any coordinate in data set A into a "
    "coordinate in data set B. The equation should be as simple as possible, and "
    "avoid exponents greater than 3. The success criterion is minimisation of "
    "average and maximum differences between CIELAB values in data set B and "
    "those estimated by the equation, as defined by the CIEDE2000 equation."
)

ANSWER_FORMAT = """\
Return your answer as exactly one fenced code block tagged `equations`, holding \
exactly three lines and nothing else:

```equations
X = <expression in c, m, y>
Y = <expression in c, m, y>
Z = <expression in c, m, y>
```

Syntax rules for those three lines (they are parsed by a program, not read by a \
human): use only plain decimal numbers, the lowercase variables c, m and y, the \
operators + - * / and ** for powers (write c**2, not c^2 or c2), and round \
brackets. No other variable or function names, no piecewise or conditional \
definitions, no units, no comments, no ellipses -- every coefficient must be \
written out. Keep any explanation outside the code block to at most three \
sentences."""


def build_equation_prompt(train_df, input_cols) -> str:
    lines = [
        "Data set A is a set of ink coordinates for a printing press: three inks, "
        "c (cyan), m (magenta) and y (yellow), each a percentage from 0 to 100.",
        "Data set B is the measured colour of the corresponding printed patch, "
        "given as CIE XYZ tristimulus values on the 0-100 scale, illuminant D50, "
        "2 degree observer.",
        "",
        PHIL_PROMPT,
        "",
        "The CIELAB values are obtained from XYZ by the standard CIE XYZ-to-CIELAB "
        "transform with the D50 white point, so give the equation as three "
        "expressions predicting X, Y and Z from c, m and y; the CIEDE2000 "
        "differences will be computed from those.",
        "",
        f"Here are the {len(train_df)} (A -> B) pairs to fit:",
    ]
    for _, r in train_df.iterrows():
        ink = ", ".join(f"{c.split('_')[1].lower()}={r[c]:g}" for c in input_cols)
        lines.append(f"{ink} -> X={r.XYZ_X:.2f}, Y={r.XYZ_Y:.2f}, Z={r.XYZ_Z:.2f}")
    lines += ["", ANSWER_FORMAT]
    return "\n".join(lines)


def summarize(de: np.ndarray) -> dict:
    de = np.asarray(de, dtype=float)
    return {"median": float(np.median(de)), "p95": float(np.percentile(de, 95)),
            "max": float(np.max(de)), "mean": float(np.mean(de)), "n": int(de.size)}


def score_xyz(pred: np.ndarray, true: np.ndarray) -> tuple[dict, int]:
    """ΔE00 through the repo's own path. Physically impossible negative XYZ is
    clipped to 0 before the Lab conversion (which is undefined there); the number
    of clipped values is reported so the reader can see it happened."""
    n_clipped = int((pred < 0).sum())
    de = delta_e00(np.clip(pred, 0.0, None), true)
    return summarize(de), n_clipped


def poly3_least_squares(X_tr, Y_tr, X_te):
    """Human-written cubic baseline fitted on the same rows the LLM saw."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False),
                          LinearRegression())
    model.fit(X_tr, Y_tr)
    return model.predict(X_te)


def fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="PC10-CMY")
    ap.add_argument("--n-train", type=int, default=150,
                    help="in-prompt training pairs (from the seeded 400-row train block)")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=None, help="subset of pinned slugs")
    ap.add_argument("--floor", type=float, default=0.15,
                    help="stop before a model if available credit is below this (USD)")
    ap.add_argument("--from-archive", action="store_true",
                    help="re-score the archived raw responses in "
                         "journal/llm/raw/equation/ without calling the API "
                         "(zero cost; this is the verification path)")
    ap.add_argument("--yes", action="store_true", help="cost gate")
    args = ap.parse_args()

    models = MODELS if not args.models else [m for m in MODELS if m["id"] in args.models]
    if not models:
        sys.exit("no known model selected")

    train_df, test_df, spec = build_split(args.dataset, 400, args.n_test, seed=42)
    prompt_train = train_df.iloc[:args.n_train]
    prompt = build_equation_prompt(prompt_train, spec.input_cols)
    # 1.75 chars/token, measured: the 8438-char 150-pair prompt billed 4902
    # prompt tokens on the first real call. The usual ~4 chars/token rule of
    # thumb understates a prompt that is almost entirely digits by ~2.3x.
    est_prompt_tok = len(prompt) / 1.75

    print(f"Dataset {args.dataset}: {len(prompt_train)} in-prompt training pairs, "
          f"{len(test_df)} held-out test patches. Inputs {spec.input_cols}")
    print(f"Prompt: {len(prompt)} chars, ~{est_prompt_tok:.0f} tokens")
    print("Worst-case cost estimate (all max_tokens spent as output):")
    est_total = 0.0
    for m in models:
        e = est_prompt_tok / 1e6 * m["price_in"] + m["max_tokens"] / 1e6 * m["price_out"]
        est_total += e
        print(f"  {m['id']}: ${e:.4f}  (in ~{est_prompt_tok:.0f} tok, "
              f"out <= {m['max_tokens']} tok)")
    print(f"  TOTAL worst case: ${est_total:.4f}")

    if not (args.yes or args.from_archive):
        sys.exit("refusing to call the API without --yes (cost gate)")

    c0 = None
    if not args.from_archive:
        c0 = orr.get_credits()
        print(f"\nCredit before run: ${c0['available']:.4f} available "
              f"(total {c0['total_credits']:.2f}, used {c0['total_usage']:.4f})")
        if c0["available"] < args.floor:
            sys.exit(f"available credit ${c0['available']:.4f} already below floor "
                     f"${args.floor:.2f} -- not calling anything")
    else:
        print("\n--from-archive: re-scoring saved responses, no API calls, $0.00")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    X_test = test_df.loc[:, list(spec.input_cols)].to_numpy(dtype=float)
    Y_test = test_df.loc[:, list(spec.target_cols)].to_numpy(dtype=float)
    if not args.from_archive:
        (RAW_DIR / "prompt.txt").write_text(prompt)  # provenance: the exact prompt
    else:
        archived = (RAW_DIR / "prompt.txt").read_text()
        if archived != prompt:
            sys.exit("--from-archive: the prompt this code builds no longer matches "
                     "journal/llm/raw/equation/prompt.txt -- the archived responses "
                     "answered a different prompt, refusing to re-score.")

    rows, equations = [], []
    for m in models:
        slug = m["id"]
        safe = slug.replace("/", "__")
        err = None
        if args.from_archive:
            print(f"\n=== {slug} === (archived response)")
            path = RAW_DIR / f"{safe}.json"
            if not path.exists():
                print(f"  no archived response at {path}; skipping")
                continue
            resp = json.loads(path.read_text())
            latency = float("nan")
        else:
            cred = orr.get_credits()
            print(f"\n=== {slug} === credit available ${cred['available']:.4f}")
            if cred["available"] < args.floor:
                print(f"STOP: ${cred['available']:.4f} < floor ${args.floor:.2f}; "
                      f"skipping {slug} and everything after it.")
                break

            t0 = time.time()
            try:
                resp = orr.chat(slug, prompt, max_tokens=m["max_tokens"],
                                temperature=0.0, reasoning=m["reasoning"],
                                provider=m.get("provider"))
            except orr.OpenRouterError as e:
                resp, err = None, str(e)
            latency = time.time() - t0

        text = orr.content_of(resp) if resp else None
        finish = None
        if resp:
            try:
                finish = resp["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError):
                finish = None
        usage = orr.usage_of(resp) if resp else {}
        if resp is not None and not args.from_archive:
            (RAW_DIR / f"{safe}.json").write_text(json.dumps(resp, indent=2))
        if text and not args.from_archive:
            (RAW_DIR / f"{safe}.txt").write_text(text)
        print(f"  latency {latency:.1f}s  usage {usage}  err={err}")
        latency_out = None if args.from_archive else round(latency, 1)

        notes, parsed, n_terms, mx_deg, mx_var = [], False, None, None, None
        if m.get("note"):
            notes.append(m["note"])
        stats, n_clipped = None, None
        if err:
            notes.append(f"API error: {err[:120]}")
        elif not text:
            refusal = None
            try:
                refusal = resp["choices"][0]["message"].get("refusal")
            except (KeyError, IndexError, TypeError):
                pass
            if refusal:
                notes.append(f"MODEL REFUSED (finish_reason={finish}): "
                             f"{refusal[:160]}")
            else:
                notes.append(f"empty content in response (finish_reason={finish}; "
                             f"max_tokens spent on reasoning tokens)")
        else:
            if finish == "length":
                notes.append("response truncated at max_tokens (finish_reason=length)")
            try:
                eq = parse_equation(text, tuple(c.split('_')[1].lower()
                                                for c in spec.input_cols))
                parsed = True
                notes += eq.notes
                n_terms, mx_deg, mx_var = eq.n_terms, eq.max_total_degree, eq.max_var_exponent
                if eq.nonpolynomial:
                    notes.append("EXPONENT CAP: non-polynomial term, degree undefined")
                elif mx_var <= 3 and mx_deg > 3:
                    notes.append(
                        f"EXPONENT CAP, partial: every single-variable exponent is "
                        f"<=3 (Phil's literal constraint met) but the expression is a "
                        f"PRODUCT of per-ink polynomials, so its expanded total "
                        f"degree is {mx_deg}")
                elif eq.violates_exponent_cap:
                    notes.append(f"EXPONENT CAP VIOLATED: max single-variable "
                                 f"exponent {mx_var}, total degree {mx_deg}")
                pred = eq(X_test)
                if not np.all(np.isfinite(pred)):
                    notes.append(f"{int((~np.isfinite(pred)).sum())} non-finite "
                                 f"predictions -> treated as 0 XYZ")
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                stats, n_clipped = score_xyz(pred, Y_test)
                if n_clipped:
                    notes.append(f"{n_clipped} negative XYZ components clipped to 0")
                equations.append((slug, text, eq))
                print(f"  ΔE00 median {stats['median']:.3f}  p95 {stats['p95']:.3f}  "
                      f"max {stats['max']:.3f}  mean {stats['mean']:.3f}")
            except EquationParseError as e:
                notes.append(f"parse failure: {e}")
                equations.append((slug, text, None))
                print(f"  PARSE FAILURE: {e}")

        rows.append({
            "model_id": slug, "parsed": parsed,
            "n_terms": n_terms, "max_total_degree": mx_deg, "max_var_exponent": mx_var,
            "median_de00": stats["median"] if stats else None,
            "p95_de00": stats["p95"] if stats else None,
            "max_de00": stats["max"] if stats else None,
            "mean_de00": stats["mean"] if stats else None,
            "n_test": len(test_df), "n_train_in_prompt": len(prompt_train),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cost_usd": usage.get("cost_usd"), "latency_s": latency_out,
            "finish_reason": finish, "provider": (resp or {}).get("provider"),
            "notes": "; ".join(notes),
        })

        if not args.from_archive:
            after = orr.get_credits()
            print(f"  credit after {slug}: ${after['available']:.4f} "
                  f"(spent {cred['available'] - after['available']:.4f})")

    # Local least-squares cubic on the identical rows: the honest comparator.
    X_tr = prompt_train.loc[:, list(spec.input_cols)].to_numpy(dtype=float)
    Y_tr = prompt_train.loc[:, list(spec.target_cols)].to_numpy(dtype=float)
    ls_stats, ls_clipped = score_xyz(poly3_least_squares(X_tr, Y_tr, X_test), Y_test)
    rows.append({
        "model_id": "poly3-ls-local", "parsed": True, "n_terms": 19 * 3,
        "max_total_degree": 3, "max_var_exponent": 3,
        "median_de00": ls_stats["median"], "p95_de00": ls_stats["p95"],
        "max_de00": ls_stats["max"], "mean_de00": ls_stats["mean"],
        "n_test": len(test_df), "n_train_in_prompt": len(prompt_train),
        "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
        "cost_usd": 0.0, "latency_s": None, "finish_reason": None, "provider": "local",
        "notes": "NOT an LLM: sklearn least-squares 3rd-order polynomial fitted on "
                 "the same in-prompt training rows and scored on the same test rows "
                 f"({ls_clipped} negative XYZ clipped). Reference cubic for the LLM "
                 "equations; the repo headline poly3 uses 5-fold CV over all rows.",
    })
    print(f"\npoly3-ls-local (same rows): median {ls_stats['median']:.3f} "
          f"p95 {ls_stats['p95']:.3f} max {ls_stats['max']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    out = RESULTS_DIR / "equation_summary.csv"
    # Carry forward rows for models NOT re-run in this invocation, so running a
    # single model does not silently drop the others (each model is one paid
    # call; we never pay twice just to rewrite a CSV).
    if out.exists():
        import csv as _csv
        fresh = {r["model_id"] for r in rows}
        with open(out) as f:
            for old in _csv.DictReader(f):
                if old["model_id"] not in fresh:
                    rows.insert(len(rows) - 1, {c: old.get(c) or None for c in cols})
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                if v is None:
                    vals.append("")
                elif isinstance(v, float) and c.endswith("de00"):
                    vals.append(f"{v:.3f}")
                elif isinstance(v, str) and ("," in v or '"' in v):
                    vals.append('"' + v.replace('"', "'") + '"')
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"Wrote {out}")

    print("\nPer-call charged cost (OpenRouter usage.cost, authoritative -- the "
          "/credits balance lags a call or two behind):")
    billed = sum(r["cost_usd"] or 0.0 for r in rows)
    for r in rows:
        print(f"  {r['model_id']}: ${r['cost_usd'] or 0.0:.4f}")
    print(f"  billed this table: ${billed:.4f}")
    if not args.from_archive:
        c1 = orr.get_credits()
        print(f"Credit after run: ${c1['available']:.4f} available; /credits delta "
              f"${c0['available'] - c1['available']:.4f} "
              f"(worst-case estimate was ${est_total:.4f})")


if __name__ == "__main__":
    main()
