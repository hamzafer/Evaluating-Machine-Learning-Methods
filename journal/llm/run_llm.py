"""LLM-as-color-predictor runner (Task 2 of docs/plans/04-llm-predictor.md).

Calls OpenAI chat models with in-context CMY(K)->XYZ examples built by
journal.llm.protocol, parses the JSON XYZ prediction out of the response,
and scores it with the identical DeltaE00 machinery used for the classical
models (journal.pipeline.color.delta_e00).

Provider note: the plan originally specified Claude models, but this run
uses an OpenAI key supplied by the user, so it calls gpt-4o (strong) and
gpt-4o-mini (small) via the `openai` SDK instead.

Usage (key is NEVER written to disk or logged -- load only as an env var):

    OPENAI_API_KEY=$(cat /path/to/openai.key) PYTHONPATH=<repo root> \
        <venv>/bin/python journal/llm/run_llm.py \
        --dataset PC10-CMY --models gpt-4o gpt-4o-mini \
        --n-train 400 --n-test 100 --yes

Outputs:
    journal/llm/raw/<model>/<dataset>/<sample_id>.json   -- one raw response
        body per query (response body only, no auth headers, no key material)
    journal/results/llm/<dataset>_summary.csv            -- scored summary,
        two rows per model: 'parsed_only' (score only successfully parsed
        predictions) and 'worst_case' (failed parses count as ΔE00 =
        PENALTY, a documented worst-case constant, so parse failures can't
        silently flatter a model).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from journal.llm.protocol import build_split, build_prompt, parse_xyz
from journal.pipeline.color import delta_e00

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "journal" / "llm" / "raw"
RESULTS_DIR = REPO_ROOT / "journal" / "results" / "llm"

# Documented worst-case ΔE00 penalty substituted for un-parseable responses
# in the 'worst_case' summary variant (see plan Task 2 / docstring above).
PENALTY = 30.0

# USD per 1M tokens (input / output), as of the run date.
PRICES = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}

# Rough per-query token estimate for the cost printout (400 in-context
# examples make prompts long; outputs are a short JSON object).
EST_PROMPT_TOKENS = 8000
EST_OUTPUT_TOKENS = 60

RETRY_SLEEP_S = 2.0
BETWEEN_CALL_SLEEP_S = 0.2
ABORT_MIN_SAMPLES = 20
ABORT_FAIL_RATE = 0.20


def summarize(de: np.ndarray) -> dict:
    """Local reimplementation of journal.pipeline.evaluate.summarize (that
    module imports sklearn, unavailable in the llm-venv interpreter)."""
    de = np.asarray(de, dtype=float)
    return {
        "median": float(np.median(de)),
        "p95": float(np.percentile(de, 95)),
        "max": float(np.max(de)),
        "mean": float(np.mean(de)),
        "n": int(de.size),
    }


def sample_id_for(row) -> str:
    if "SAMPLE_ID" in row.index:
        return str(row["SAMPLE_ID"])
    return str(row.name)


def estimate_cost(models, n_test) -> tuple[str, float]:
    lines = ["Estimated cost (rough, ~{}k prompt tokens/query):".format(EST_PROMPT_TOKENS // 1000)]
    total = 0.0
    for m in models:
        p = PRICES[m]
        cost = n_test * (
            EST_PROMPT_TOKENS / 1e6 * p["in"] + EST_OUTPUT_TOKENS / 1e6 * p["out"]
        )
        lines.append(
            f"  {m}: {n_test} calls x ~{EST_PROMPT_TOKENS} in / ~{EST_OUTPUT_TOKENS} out tok "
            f"-> ${cost:.2f}"
        )
        total += cost
    lines.append(f"  TOTAL estimate: ${total:.2f}")
    return "\n".join(lines), total


def call_model(client, model: str, prompt: str, retries: int = 1):
    """Call chat completions; retry once on APIError with a sleep.
    Returns (raw_response_dict_or_None, error_str_or_None)."""
    from openai import APIError

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            return resp.model_dump(), None
        except APIError as e:
            if attempt >= retries:
                return None, str(e)
            attempt += 1
            time.sleep(RETRY_SLEEP_S)


def run_for_model(client, model, dataset, train_df, test_df, spec, n_test):
    out_dir = RAW_DIR / model / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    de_parsed = []
    de_penalty = []
    n_total = 0
    n_parsed = 0
    n_fail = 0
    aborted = False

    rows = test_df.iloc[:n_test]
    for _, row in rows.iterrows():
        n_total += 1
        sid = sample_id_for(row)
        prompt = build_prompt(train_df, row, spec.input_cols)
        raw, err = call_model(client, model, prompt)

        record = {"sample_id": sid, "model": model, "dataset": dataset}
        text = None
        if raw is not None:
            record["response"] = raw
            try:
                text = raw["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = None
        else:
            record["error"] = err

        with open(out_dir / f"{sid}.json", "w") as f:
            json.dump(record, f, indent=2)

        true_xyz = row[list(spec.target_cols)].to_numpy(dtype=float)
        parsed = parse_xyz(text) if text else None
        if parsed is not None:
            de = float(delta_e00(np.array(parsed, dtype=float), true_xyz))
            de_parsed.append(de)
            de_penalty.append(de)
            n_parsed += 1
        else:
            de_penalty.append(PENALTY)
            n_fail += 1

        print(f"  [{model}] {n_total}/{n_test} sample={sid} "
              f"parsed={'yes' if parsed is not None else 'NO'}")

        if n_total >= ABORT_MIN_SAMPLES and n_fail / n_total > ABORT_FAIL_RATE:
            print(f"[{model}] ABORTING EARLY: {n_fail}/{n_total} parse failures "
                  f"(> {ABORT_FAIL_RATE:.0%}) -- stopping this model's run.")
            aborted = True
            break

        time.sleep(BETWEEN_CALL_SLEEP_S)

    return {
        "model": model,
        "de_parsed": np.array(de_parsed, dtype=float),
        "de_penalty": np.array(de_penalty, dtype=float),
        "n_total": n_total,
        "n_parsed": n_parsed,
        "aborted": aborted,
    }


def write_summary(dataset: str, results: list) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{dataset}_summary.csv"
    header = ["model_id", "variant", "median", "p95", "max", "mean",
              "n_parsed", "n_total", "penalty"]
    lines = [",".join(header)]
    for r in results:
        model = r["model"]
        n_parsed, n_total = r["n_parsed"], r["n_total"]
        if r["de_parsed"].size:
            s = summarize(r["de_parsed"])
            lines.append(",".join(str(x) for x in [
                model, "parsed_only", round(s["median"], 3), round(s["p95"], 3),
                round(s["max"], 3), round(s["mean"], 3), n_parsed, n_total, ""]))
        else:
            lines.append(f"{model},parsed_only,,,,,{n_parsed},{n_total},")

        s_wc = summarize(r["de_penalty"])
        lines.append(",".join(str(x) for x in [
            model, "worst_case", round(s_wc["median"], 3), round(s_wc["p95"], 3),
            round(s_wc["max"], 3), round(s_wc["mean"], 3), n_parsed, n_total, PENALTY]))

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="PC10-CMY")
    ap.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-4o-mini"])
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--max-test-samples", type=int, default=None,
                     help="override --n-test for a cheap dry run")
    ap.add_argument("--yes", action="store_true",
                     help="proceed past the cost estimate without an interactive prompt")
    args = ap.parse_args()

    n_test = args.max_test_samples if args.max_test_samples is not None else args.n_test

    for m in args.models:
        if m not in PRICES:
            print(f"Unknown model '{m}', no pricing on file. Known: {list(PRICES)}", file=sys.stderr)
            sys.exit(1)

    cost_msg, total_cost = estimate_cost(args.models, n_test)
    print(cost_msg)

    if not args.yes:
        print("Refusing to call the API without --yes (cost gate).", file=sys.stderr)
        sys.exit(1)

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    train_df, test_df, spec = build_split(args.dataset, args.n_train, args.n_test, seed=42)
    print(f"Split: {len(train_df)} train (in-context) / {len(test_df)} test "
          f"(using first {n_test} for this run) for {args.dataset}. "
          f"Input cols: {spec.input_cols}")

    results = []
    for model in args.models:
        print(f"\n=== Running {model} on {args.dataset} ({n_test} queries) ===")
        r = run_for_model(client, model, args.dataset, train_df, test_df, spec, n_test)
        results.append(r)
        s = summarize(r["de_penalty"])
        print(f"[{model}] done: n_parsed={r['n_parsed']}/{r['n_total']} "
              f"parsed_only_median={summarize(r['de_parsed'])['median'] if r['de_parsed'].size else 'n/a'} "
              f"worst_case_median={s['median']:.3f}")

    out_path = write_summary(args.dataset, results)
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
