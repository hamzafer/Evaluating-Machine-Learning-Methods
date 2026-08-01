"""Prompt protocol for LLM-as-color-predictor. Split is seeded and leakage-free."""
import json
import re

import pandas as pd

from journal.pipeline.datasets import registry


def build_split(dataset_name: str, n_train: int, n_test: int, seed: int = 42):
    spec = registry()[dataset_name]
    df = pd.read_csv(spec.csv)
    if spec.filter_k_zero:
        df = df[df['CMYK_K'] == 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=list(spec.input_cols))      # kill recipe twins
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_test], spec


def build_prompt(train_df, query_row, input_cols) -> str:
    lines = [
        "You are a printer color characterization model. Given ink percentages, "
        "predict the measured CIE XYZ tristimulus values (0-100 scale, D50).",
        "Training measurements from this printer:",
    ]
    for _, r in train_df.iterrows():
        ink = ", ".join(f"{c.split('_')[1]}={r[c]:g}" for c in input_cols)
        lines.append(f"{ink} -> X={r.XYZ_X:.2f}, Y={r.XYZ_Y:.2f}, Z={r.XYZ_Z:.2f}")
    ink = ", ".join(f"{c.split('_')[1]}={query_row[c]:g}" for c in input_cols)
    lines.append(
        f"Predict for: {ink}. Respond with ONLY a JSON object "
        '{"X": <number>, "Y": <number>, "Z": <number>}.')
    return "\n".join(lines)


def parse_xyz(text: str):
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return float(d['X']), float(d['Y']), float(d['Z'])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
