"""Generic parser for the CGATS/ISO28178-like tab-delimited text files used in this task.

All files share the shape:
  <header lines, KEY<TAB>VALUE ...>
  ...
  NUMBER_OF_FIELDS  <n>
  BEGIN_DATA_FORMAT
  <col1>\t<col2>\t...\t<colN>
  END_DATA_FORMAT
  ...
  NUMBER_OF_SETS  <n>
  BEGIN_DATA
  <row1>
  ...
  END_DATA
"""
import re
import numpy as np
import pandas as pd


def parse_cgats_like(path):
    """Return (header_lines: list[str], columns: list[str], data: pd.DataFrame (all string/object cols initially, or numeric where possible))."""
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    header_lines = []
    columns = None
    data_rows = []
    n_declared_fields = None
    n_declared_sets = None

    i = 0
    n = len(lines)
    in_data = False
    in_format = False
    while i < n:
        raw = lines[i]
        line = raw.rstrip("\n").rstrip("\r")
        stripped = line.strip()
        if stripped == "BEGIN_DATA_FORMAT":
            in_format = True
            i += 1
            continue
        if stripped == "END_DATA_FORMAT":
            in_format = False
            i += 1
            continue
        if in_format:
            columns = [c.strip() for c in line.split("\t") if c.strip() != ""]
            i += 1
            continue
        if stripped == "BEGIN_DATA":
            in_data = True
            i += 1
            continue
        if stripped == "END_DATA":
            in_data = False
            i += 1
            continue
        if in_data:
            if stripped == "":
                i += 1
                continue
            parts = line.split("\t")
            # strip trailing empty fields caused by trailing tabs
            while parts and parts[-1].strip() == "":
                parts.pop()
            data_rows.append(parts)
            i += 1
            continue
        # header region
        header_lines.append(line)
        m = re.match(r"NUMBER_OF_FIELDS\s+(\d+)", stripped)
        if m:
            n_declared_fields = int(m.group(1))
        m = re.match(r"NUMBER_OF_SETS\s+(\d+)", stripped)
        if m:
            n_declared_sets = int(m.group(1))
        i += 1

    if columns is None:
        raise ValueError(f"{path}: could not find BEGIN_DATA_FORMAT/END_DATA_FORMAT block")

    ncols = len(columns)
    # normalize row lengths (pad short rows with NaN, error on rows too long)
    fixed_rows = []
    ragged = 0
    for r in data_rows:
        if len(r) < ncols:
            ragged += 1
            r = r + [""] * (ncols - len(r))
        elif len(r) > ncols:
            ragged += 1
            r = r[:ncols]
        fixed_rows.append(r)

    df = pd.DataFrame(fixed_rows, columns=columns)
    # try numeric conversion column by column
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        # only replace if essentially all non-empty values converted
        non_empty = df[c].str.strip() != ""
        if non_empty.sum() == 0:
            continue
        bad = conv.isna() & non_empty
        if bad.sum() == 0:
            df[c] = conv

    meta = {
        "n_declared_fields": n_declared_fields,
        "n_declared_sets": n_declared_sets,
        "n_actual_rows": len(df),
        "n_ragged_rows": ragged,
        "columns": columns,
    }
    return header_lines, df, meta


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        hl, df, meta = parse_cgats_like(p)
        print(p, meta)
        print(df.head())
        print(df.dtypes)
        print()
