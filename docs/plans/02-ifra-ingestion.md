# 02 — IFRA Newsprint Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `journal/data/raw/Ifra-{wb,bb}.zip` into per-press-run processed CSVs registered in the pipeline's dataset registry, so IFRA runs work in every experiment exactly like PC10 does.

**Architecture:** A parser module converts CGATS text → DataFrame (CMYK + 36-band spectral), a converter integrates spectral → XYZ (colour-science, D50/2°) and derives Lab; processed CSVs use the same column names as `data/cleaned/*.csv` so `DatasetSpec` needs no changes beyond registration. wb and bb stay separate end-to-end (Phil).

**Tech Stack:** Python, pandas, colour-science (`colour.sd_to_XYZ`, D50 illuminant, CIE 1931 2° observer).

## Global Constraints

- Encoding is latin-1 (roadmap note). Spectral bands 380–730 nm @ 10 nm = 36 values.
- Keep wb/bb separate; wb is the industry norm (headline), bb the larger appendix set.
- The Lab-roundtrip tripwire (`assert_lab_roundtrip`) must run on IFRA loads too. IFRA Lab is *derived* at ingestion (from the same XYZ), so the tripwire only guards against later scale corruption — note this limitation in the processed README.
- Roadmap says "~200 reproductions", zip description says 43 press runs — **the zip contents win**; record the true count in `journal/data/processed/ifra/README.md`.

---

### Task 1: Inspect the zips and lock the format (no code committed yet)

**Files:**
- Create: `journal/data/processed/ifra/README.md` (findings go here)

- [ ] **Step 1: Unzip to scratch and inventory**

```bash
mkdir -p /tmp/ifra && cd /tmp/ifra
unzip -o "$(git rev-parse --show-toplevel)/journal/data/raw/Ifra-wb.zip" -d wb
unzip -o "$(git rev-parse --show-toplevel)/journal/data/raw/Ifra-bb.zip" -d bb
find wb bb -type f | head -50 && find wb -type f | wc -l && find bb -type f | wc -l
```

- [ ] **Step 2: Read one file's header (latin-1) and record the exact field names**

```bash
head -60 $(find /tmp/ifra/wb -type f | head -1) | iconv -f latin1 -t utf-8
```

Record in the README: file count per backing, sample count per file, the exact
`BEGIN_DATA_FORMAT` field list (expect `CMYK_C CMYK_M CMYK_Y CMYK_K` +
`SPECTRAL_NM_380 … SPECTRAL_NM_730` or `nm380…` — copy verbatim), and whether
LAB/XYZ fields are already present (if XYZ present, keep it and skip integration
for those files; still derive Lab consistently).

- [ ] **Step 3: Commit the README**

```bash
git add journal/data/processed/ifra/README.md
git commit -m "journal: IFRA zip inventory — actual run counts and CGATS field map"
```

> **STATUS (11 Aug 2026): bb is OUT OF SCOPE.** During execution bb WAS ingested then quarantined
> (commit 7719a02): its chart layout differs from wb, so the CMYK join was invalid (duplicate-recipe
> ΔE00 came out 25–28 vs wb's sane ~0.7). bb was then descoped entirely ('substrate correction', separate
> problem) and its zip moved out of the repo. Only the 13 wb runs are used. The bb-reader parts below are historical.
>
> **AMENDMENT (post-Task-1 inventory — overrides the code below where they conflict).**
> Ground truth from `journal/data/processed/ifra/README.md`:
> 1. **Two formats, two readers.** wb (13 files) is genuine CGATS as assumed. bb is NOT CGATS: 4-line preamble + plain tab header (`SampleID  XYZ-X … CIE-Lab-b  380nm…730nm`); add `parse_bb(path)` alongside `parse_cgats`.
> 2. **Skip the 5 `lab*`-prefixed bb files** — duplicate exports, not runs. True runs: 13 wb + 25 bb = 38.
> 3. **bb has no CMYK columns.** Build the chart table (SAMPLE_ID → CMYK, byte-identical across all wb files) from any wb file and join it into every bb run on SampleID; assert no unmatched ids.
> 4. **Derive XYZ/Lab from spectral uniformly for ALL runs** (wb and bb). Where bb ships native XYZ/Lab, do not use it as output — instead assert median |ΔE00(derived, native)| < 1 as a free cross-check of our spectral integration, and record the actual value in the ingestion log.
> 5. Spectral band naming differs (`SPECTRAL_NM_380` vs `380nm`) — the `spectral_cols` regex must match both (it does).
> 6. Filenames are inconsistent — never parse metadata from filenames; use file contents + backing directory only.

### Task 2: CGATS parser

**Files:**
- Create: `journal/pipeline/ifra.py`
- Test: `journal/pipeline/tests/test_ifra.py`

**Interfaces:**
- Produces: `parse_cgats(path: Path) -> pd.DataFrame` (columns exactly as in the file's DATA_FORMAT), `spectral_to_xyz(df) -> pd.DataFrame` adding `XYZ_X, XYZ_Y, XYZ_Z` (0–100 scale) and `LAB_L, LAB_A, LAB_B` (derived via `journal.pipeline.color.xyz_to_lab`).

- [ ] **Step 1: Write the failing test with an embedded minimal CGATS fixture**

```python
# journal/pipeline/tests/test_ifra.py
import textwrap
import numpy as np
from journal.pipeline.ifra import parse_cgats, spectral_to_xyz

FIXTURE = textwrap.dedent("""\
    CGATS.17
    BEGIN_DATA_FORMAT
    SAMPLE_ID\tCMYK_C\tCMYK_M\tCMYK_Y\tCMYK_K\t{bands}
    END_DATA_FORMAT
    NUMBER_OF_SETS 1
    BEGIN_DATA
    1\t0\t0\t0\t0\t{flat}
    END_DATA
""").format(
    bands="\t".join(f"SPECTRAL_NM_{nm}" for nm in range(380, 740, 10)),
    flat="\t".join(["0.60"] * 36),   # flat 60% reflectance ≈ light gray
)


def test_parse_and_convert(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text(FIXTURE, encoding="latin-1")
    df = parse_cgats(p)
    assert len(df) == 1 and df.loc[0, "CMYK_K"] == 0
    out = spectral_to_xyz(df)
    # flat 60% reflectance: Y ≈ 60 under any illuminant, L* ≈ 82 (mid-light gray)
    assert abs(out.loc[0, "XYZ_Y"] - 60) < 2
    assert 78 < out.loc[0, "LAB_L"] < 86
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest journal/pipeline/tests/test_ifra.py -v`
Expected: FAIL with `ModuleNotFoundError: journal.pipeline.ifra`

- [ ] **Step 3: Implement**

```python
# journal/pipeline/ifra.py
"""IFRA 2005 newsprint: CGATS parsing + spectral -> XYZ/Lab (D50, 2 deg)."""
import re
from pathlib import Path

import colour
import numpy as np
import pandas as pd

from .color import xyz_to_lab

_CMF = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
_D50 = colour.SDS_ILLUMINANTS['D50']


def parse_cgats(path: Path) -> pd.DataFrame:
    text = Path(path).read_text(encoding='latin-1')
    fmt = re.search(r'BEGIN_DATA_FORMAT\s*\n(.*?)\nEND_DATA_FORMAT', text, re.S)
    cols = fmt.group(1).split()
    data = re.search(r'BEGIN_DATA\s*\n(.*?)\nEND_DATA', text, re.S).group(1)
    rows = [ln.split() for ln in data.strip().splitlines()]
    df = pd.DataFrame(rows, columns=cols)
    return df.apply(pd.to_numeric, errors='ignore')


def spectral_cols(df: pd.DataFrame) -> list:
    pat = re.compile(r'(?:SPECTRAL_)?NM_?(\d{3})$', re.I)
    found = [(int(m.group(1)), c) for c in df.columns if (m := pat.search(c))]
    return [c for _, c in sorted(found)]


def spectral_to_xyz(df: pd.DataFrame) -> pd.DataFrame:
    cols = spectral_cols(df)
    assert len(cols) == 36, f"expected 36 bands, found {len(cols)}"
    wl = np.arange(380, 740, 10)
    out = df.copy()
    xyz = np.array([
        colour.sd_to_XYZ(
            colour.SpectralDistribution(dict(zip(wl, row)), name='r'),
            cmfs=_CMF, illuminant=_D50)
        for row in df[cols].to_numpy(dtype=float)])
    out[['XYZ_X', 'XYZ_Y', 'XYZ_Z']] = xyz
    out[['LAB_L', 'LAB_A', 'LAB_B']] = xyz_to_lab(xyz)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest journal/pipeline/tests/test_ifra.py -v`
Expected: PASS (if the real files' band naming differs from the fixture's,
extend `spectral_cols`'s regex to match the verbatim names recorded in Task 1 —
do not change the test's physics assertions).

- [ ] **Step 5: Commit**

```bash
git add journal/pipeline/ifra.py journal/pipeline/tests/test_ifra.py
git commit -m "journal: CGATS parser + spectral->XYZ/Lab for IFRA newsprint"
```

### Task 3: Batch ingestion + registration

**Files:**
- Create: `journal/pipeline/ingest_ifra.py`
- Modify: `journal/pipeline/datasets.py` (registration loop)
- Output: `journal/data/processed/ifra/{wb,bb}/<run>.csv`

**Interfaces:**
- Consumes: `parse_cgats`, `spectral_to_xyz`.
- Produces: processed CSVs with columns `SAMPLE_ID, CMYK_C, CMYK_M, CMYK_Y, CMYK_K, LAB_L, LAB_A, LAB_B, XYZ_X, XYZ_Y, XYZ_Z` (identical to `data/cleaned/`); registry entries named `IFRA-wb-<run>-CMYK` etc. via a `DatasetSpec` per file.

- [ ] **Step 1: Write the ingestion script**

```python
# journal/pipeline/ingest_ifra.py
"""Unzip + convert all IFRA runs. Run: .venv/bin/python -m journal.pipeline.ingest_ifra"""
import zipfile
from pathlib import Path

from .ifra import parse_cgats, spectral_to_xyz

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'ifra'
KEEP = ['SAMPLE_ID', 'CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K',
        'LAB_L', 'LAB_A', 'LAB_B', 'XYZ_X', 'XYZ_Y', 'XYZ_Z']


def main():
    for backing in ('wb', 'bb'):
        out_dir = OUT / backing
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(RAW / f'Ifra-{backing}.zip') as z:
            tmp = out_dir / '_tmp'
            z.extractall(tmp)
            for f in sorted(tmp.rglob('*')):
                if not f.is_file():
                    continue
                df = spectral_to_xyz(parse_cgats(f))
                if 'SAMPLE_ID' not in df.columns:
                    df.insert(0, 'SAMPLE_ID', range(1, len(df) + 1))
                df[KEEP].to_csv(out_dir / f'{f.stem}.csv', index=False)
                print(f'{backing}/{f.stem}: {len(df)} samples')
            import shutil; shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m journal.pipeline.ingest_ifra`
Expected: one CSV per run (13 wb + 30 bb if the Task-1 inventory confirms those
counts), each ~1,485 samples. Spot-check one file: `XYZ_Y` of the unprinted
patch (CMYK all 0) should be ~50–70 for newsprint (it is not white paper).

- [ ] **Step 3: Register in datasets.py**

```python
# append to journal/pipeline/datasets.py registry()
    ifra_root = REPO_ROOT / 'journal' / 'data' / 'processed' / 'ifra'
    for backing in ('wb', 'bb'):
        for csv in sorted((ifra_root / backing).glob('*.csv')):
            name = f'IFRA-{backing}-{csv.stem}-CMYK'
            specs[name] = DatasetSpec(
                name=name, csv=csv,
                input_cols=('CMYK_C', 'CMYK_M', 'CMYK_Y', 'CMYK_K'),
                filter_k_zero=False)
```

- [ ] **Step 4: Verify registration + tripwire**

Run: `.venv/bin/python -c "from journal.pipeline.datasets import registry; r={k:v for k,v in registry().items() if 'IFRA' in k}; print(len(r)); X,Y=next(iter(r.values())).load(); print(X.shape, Y.shape)"`
Expected: run count printed; a load succeeds (tripwire passes by construction).

- [ ] **Step 5: Commit**

```bash
git add journal/pipeline/ingest_ifra.py journal/pipeline/datasets.py journal/data/processed/ifra/
git commit -m "journal: ingest IFRA newsprint (wb/bb separate) and register datasets"
```
