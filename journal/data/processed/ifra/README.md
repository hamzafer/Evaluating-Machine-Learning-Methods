# IFRA newsprint zip inventory (Task 1 of `docs/plans/02-ifra-ingestion.md`)

Findings from directly unzipping and inspecting `journal/data/raw/Ifra-wb.zip` and
`journal/data/raw/Ifra-bb.zip` to `/tmp/ifra/{wb,bb}`. Nothing here is inferred from
documentation — every claim below was checked against the actual file bytes
(`grep`, `awk`, `python3`, byte-level diffs). Encoding is **latin-1** for both
archives (confirmed: files contain byte `0xB0`, the latin-1 degree sign, which is
not valid UTF-8 — `°` appears in `MEASUREMENT_SOURCE`/`Measuring conditions`
fields). Line endings are CRLF.

## Headline result: wb and bb are NOT the same file format

This is the single most important finding. `Ifra-wb.zip` and `Ifra-bb.zip` were
exported by different tools and are structurally different, not just
differently-populated instances of the same CGATS layout:

- **wb** files are genuine CGATS/ECI2002 text: `ECI2002` marker line,
  `BEGIN_DATA_FORMAT`/`END_DATA_FORMAT`, `NUMBER_OF_SETS`, `NUMBER_OF_FIELDS`
  keywords, tab-delimited. Fields = `SAMPLE_ID`, `SAMPLE_NAME`, CMYK, 36 spectral
  bands. **No LAB or XYZ columns.**
- **bb** files have **no CGATS keywords at all** — no `BEGIN_DATA_FORMAT`, no
  `NUMBER_OF_SETS`. They're a simple 4-line key/value preamble
  (`Originator:`/`Date:`/`Measuring device:`/`Measuring conditions:`), one blank
  line, one tab-delimited header row, one blank line, then data rows. Fields =
  `SampleID` + XYZ + Lab + 36 spectral bands (or, for 5 files, Lab + spectral
  only — see below). **No CMYK columns at all in any bb file.**

The parser (Task 2) needs two genuinely different readers, not one CGATS reader
with an optional-field toggle.

## File counts

| Backing | Files found | Expected (roadmap) | Match? |
|---|---|---|---|
| wb | **13** | 13 | Yes |
| bb | **30** | 30 | Yes, as raw file count — **but see caveat below** |

**Caveat:** of the 30 `bb/*.txt` files, only **25 are independent press-run
measurement files**. The other 5 (`labdag._120_2._bb.txt`, `labmalm_105_bb.txt`,
`labManH_94_bb.txt`, `labMarca_133_2_bb.txt`, `labVkrant_152_bb.txt`) are
companion duplicate exports of the `CIE-Lab-L/a/b` columns already present in 5
sibling files (`dag._120_2._bb.txt`, `malm_105_bb.txt`, `ManH_94_bb.txt`,
`Marca_133_2_bb.txt`, `Vkrant_152_bb.txt`). They have no header, no `SampleID`
column, just 3 raw tab-separated numeric columns, 1485 rows, no CGATS/preamble.
Byte-for-byte value comparison (allowing for trailing-zero formatting
differences, e.g. `34.3` vs `34.30`) confirms the `lab*` file for a given press
is numerically identical to the Lab columns of its non-`lab` sibling — same
1485 rows, same values, just re-exported with 3 columns instead of the full
XYZ+Lab+spectral set. They are not a repeat measurement or a different press
condition; they are the same data restated.

**So: true distinct bb press runs = 25, not 30.** Combined with wb: **13 + 25 =
38 distinct press runs**, not 43. The zip's file count (13 wb + 30 bb = 43)
matches the roadmap's stated "43 press runs," but 5 of those 43 files carry no
new information. Task 2's ingestion should either skip the 5 `lab*` files
entirely (recommended — they add zero new data) or explicitly document them as
intentional duplicates if kept.

Full file listing:

```
wb/ (13 files)
Age_64a_wb.txt  Arena_111_wb.txt  Dnvk_131_wb.txt  GratN_90_wb.txt
marca_133_wb.txt  Mbd_103a_wb.txt  NordK_92a_wb.txt  PressJ_158_wb.txt
RZPOI_125_2_wb.txt  SudK_99_2_wb.txt  Sudz_100_wb.txt  TagesA_147_2_wb.txt
TagesA_147_wb.txt

bb/ (30 files; 25 real runs + 5 lab*-duplicate companions marked below)
BerlTid_81_bb.txt  coop_140_2_bb.txt  COOP_140a_bb.txt  dag._120_2._bb.txt
Ecom_9_2_bb.txt  Ecom_9_bb.txt  EveExp_157_2_bb.txt  EveExp_157_bb.txt
Graf_90_2_bb.txt  HDM_174_bb.txt  Heilbs_91_2_bb.txt  heilbs_91_bb.txt
labdag._120_2._bb.txt *DUP*  labmalm_105_bb.txt *DUP*  labManH_94_bb.txt *DUP*
labMarca_133_2_bb.txt *DUP*  labVkrant_152_bb.txt *DUP*
Ludk_101a_bb.txt  malm_105_bb.txt  ManH_94_bb.txt  Marca_133_2_bb.txt
Nbk_88_2_bb.txt  NLZ_141_2_bb.txt  NLZ_141_bb.txt  Pdemo_53_2_bb.txt
pressJ_158_2_bb.txt  sudk__99_bb.txt  Sudz_100_2_bb.txt  Tb_118a_bb.txt
Vkrant_152_bb.txt
```

## Samples per file

All 13 wb files: `NUMBER_OF_SETS 1485`, and the actual `BEGIN_DATA`…`END_DATA`
block was counted at exactly 1485 rows in every file (no discrepancy between
declared and actual count).

All 25 real bb files (main + the 5 no-XYZ ones): 1485 data rows counted
directly (bb has no `NUMBER_OF_SETS` keyword to cross-check against, since it
isn't CGATS). The 5 `lab*` duplicate files also have exactly 1485 rows each
(matching their sibling, as expected for a duplicate).

**1485 samples per file matches the ECI2002 standard chart size exactly, in
every one of the 38 real files.**

## Does the CMYK combination set match ECI2002 (1,485 samples)?

Yes, and more specifically: it's not just a matching *count* — it's the same
**fixed chart layout**. Every wb file's `SAMPLE_ID` column ranges exactly
1–1485 (1485 unique values, no gaps/dupes). Directly diffing the
`SAMPLE_ID → (CMYK_C, CMYK_M, CMYK_Y, CMYK_K)` mapping across all 13 wb files
confirms it is **byte-identical in every file** — `SAMPLE_ID` is a fixed
patch-ID for a fixed CMYK recipe under the standard ECI2002 chart, not a
per-run arbitrary index. (`SAMPLE_NAME`, e.g. `A1`, `A33`, `B1`, is a separate
chart-position label, also present but not needed once `SAMPLE_ID` is used as
the join key.)

bb files use the same `SampleID` numbering (1–1485, all unique, in every real
bb file checked) but — since bb carries no CMYK columns at all — **the CMYK
values for bb samples must be recovered by joining `SampleID` against the
canonical mapping extracted from any wb file** (verified identical across all
13, so any one of them is a valid source of truth for the mapping table).
Sanity check: `SampleID 1` is `CMYK (0,0,0,0)` (bare substrate) in the wb
mapping; in bb files, `SampleID 1` has the highest Y/L in the chart in every
bb file inspected (`BerlTid_81`: Y=30.43, L*=62.02; `COOP_140a`: Y=32.09,
L*=63.41) — consistent with bare newsprint over a black backing (translucent
substrate, so darker than bare-paper-over-white, but still the brightest patch
on the chart), which is physically the expected reading for the IFRA
white/black-backing opacity-test design. This is corroborating evidence, not a
row-by-row proof, but combined with the exact-match cross-check on the wb side
it's strong enough to treat the join as safe.

**Architectural implication for Task 2:** the bb parser cannot produce a
CMYK-labeled DataFrame from a bb file alone. It needs the wb-derived
`SAMPLE_ID → CMYK` table as a side input. This doesn't break "keep wb/bb
separate" (the measurement runs and Lab/XYZ/spectral data stay fully
separate) — it just means the *CMYK identity of a patch* is a shared,
chart-level constant, not something duplicated per backing in the raw data.

## Exact `BEGIN_DATA_FORMAT` field list (wb) — copied verbatim

Identical (hash-verified) across all 13 wb files:

```
SAMPLE_ID	SAMPLE_NAME	CMYK_C	CMYK_M	CMYK_Y	CMYK_K	SPECTRAL_NM_380	SPECTRAL_NM_390	SPECTRAL_NM_400	SPECTRAL_NM_410	SPECTRAL_NM_420	SPECTRAL_NM_430	SPECTRAL_NM_440	SPECTRAL_NM_450	SPECTRAL_NM_460	SPECTRAL_NM_470	SPECTRAL_NM_480	SPECTRAL_NM_490	SPECTRAL_NM_500	SPECTRAL_NM_510	SPECTRAL_NM_520	SPECTRAL_NM_530	SPECTRAL_NM_540	SPECTRAL_NM_550	SPECTRAL_NM_560	SPECTRAL_NM_570	SPECTRAL_NM_580	SPECTRAL_NM_590	SPECTRAL_NM_600	SPECTRAL_NM_610	SPECTRAL_NM_620	SPECTRAL_NM_630	SPECTRAL_NM_640	SPECTRAL_NM_650	SPECTRAL_NM_660	SPECTRAL_NM_670	SPECTRAL_NM_680	SPECTRAL_NM_690	SPECTRAL_NM_700	SPECTRAL_NM_710	SPECTRAL_NM_720	SPECTRAL_NM_730
```

`NUMBER_OF_FIELDS 42` (matches: 6 non-spectral + 36 spectral). No LAB or XYZ
fields present in wb — as expected, since wb is raw spectral + CMYK and Lab/XYZ
must be derived by the pipeline.

Representative full preamble (from `Age_64a_wb.txt`):

```
ECI2002
ORIGINATOR	""
DESCRIPTOR	"Output Characterisation"
PRINT_CONDITIONS	""
CREATED	"10/17/2005"  # Time: 15:24
INSTRUMENTATION	"SpectroScan"
MEASUREMENT_SOURCE	""Illumination=D50	ObserverAngle=2°	WhiteBase=Abs	Filter=No""
ILLUMINATION_NAME	"D50"
OBSERVER_ANGLE	"2"
KEYWORD	"SAMPLE_NAME"
NUMBER_OF_FIELDS	42
BEGIN_DATA_FORMAT
... (field list above) ...
END_DATA_FORMAT
NUMBER_OF_SETS	1485
BEGIN_DATA
... 1485 rows ...
END_DATA
```

## Exact header row (bb) — copied verbatim, both sub-formats found

bb has **no `BEGIN_DATA_FORMAT` block** (it isn't CGATS) — the field names are
just the tab-delimited header row after the 4-line preamble + blank line. Two
distinct header rows were found (hash-verified within each group):

**Group A — 20 files, has XYZ (the majority format):**

```
SampleID	XYZ-X	XYZ-Y	XYZ-Z	CIE-Lab-L	CIE-Lab-a	CIE-Lab-b	380nm	390nm	400nm	410nm	420nm	430nm	440nm	450nm	460nm	470nm	480nm	490nm	500nm	510nm	520nm	530nm	540nm	550nm	560nm	570nm	580nm	590nm	600nm	610nm	620nm	630nm	640nm	650nm	660nm	670nm	680nm	690nm	700nm	710nm	720nm	730nm
```

Files: `BerlTid_81_bb.txt`, `COOP_140a_bb.txt`, `Ecom_9_2_bb.txt`,
`Ecom_9_bb.txt`, `EveExp_157_2_bb.txt`, `EveExp_157_bb.txt`, `Graf_90_2_bb.txt`,
`HDM_174_bb.txt`, `Heilbs_91_2_bb.txt`, `heilbs_91_bb.txt`, `Ludk_101a_bb.txt`,
`Nbk_88_2_bb.txt`, `NLZ_141_2_bb.txt`, `NLZ_141_bb.txt`, `Pdemo_53_2_bb.txt`,
`pressJ_158_2_bb.txt`, `sudk__99_bb.txt`, `Sudz_100_2_bb.txt`,
`Tb_118a_bb.txt`, `coop_140_2_bb.txt`.

**Group B — 5 files, Lab + spectral only, NO XYZ (deviant subset):**

```
SampleID	CIE-Lab-L	CIE-Lab-a	CIE-Lab-b	380nm	390nm	400nm	410nm	420nm	430nm	440nm	450nm	460nm	470nm	480nm	490nm	500nm	510nm	520nm	530nm	540nm	550nm	560nm	570nm	580nm	590nm	600nm	610nm	620nm	630nm	640nm	650nm	660nm	670nm	680nm	690nm	700nm	710nm	720nm	730nm
```

Files: `dag._120_2._bb.txt`, `malm_105_bb.txt`, `ManH_94_bb.txt`,
`Marca_133_2_bb.txt`, `Vkrant_152_bb.txt`. Each of these has an accompanying
`lab*` duplicate file (see the file-count caveat above) that restates only
these same Lab columns with no header — i.e. this is the one place XYZ is
genuinely missing and cannot be recovered from the file itself; XYZ for these
5 runs would need to be re-derived from the spectral data (which is present)
rather than trusted from a column, same as every other file in the pipeline.

Representative preamble (from `BerlTid_81_bb.txt`):

```
Originator:	
Date:	10/18/2005
Measuring device:	Spectrolino  10771
Measuring conditions:	[D50,2°,ANSI T,No,Abs]

SampleID	XYZ-X	XYZ-Y	XYZ-Z	CIE-Lab-L	CIE-Lab-a	CIE-Lab-b	380nm	...	730nm

1	 30.81	 30.43	 20.17	 62.02	  5.56	  9.43	0.112935	...
```

## LAB/XYZ presence — summary

| Source | CMYK | XYZ | Lab | Spectral |
|---|---|---|---|---|
| wb (all 13) | yes | no | no | yes (36 bands) |
| bb Group A (20 files) | no | **yes** | **yes** | yes (36 bands) |
| bb Group B (5 files) | no | no | **yes** | yes (36 bands) |
| bb `lab*` duplicates (5 files) | no | no | yes (dup of Group B) | no |

Per the plan's original instruction ("if XYZ present, keep it and skip
integration for those files; still derive Lab consistently") — this applies
cleanly to bb Group A (20 files). For consistency across all bb runs, and
because Group B lacks XYZ outright, the recommended approach for Task 2 is to
**derive XYZ and Lab from the spectral data for every bb file uniformly**
(D50/2°, `colour.sd_to_XYZ`), rather than branching per file on whether a
native XYZ/Lab column exists. The native Lab in Group A can be used as a
plausibility check/tripwire against the derived Lab (they should be close but
won't be bit-identical, since they came from the instrument's own
firmware/illuminant pipeline, not necessarily D50/2° CIE 2000-consistent
processing) — flag any run where derived vs. native Lab diverges beyond normal
instrument tolerance.

## Spectral band naming and count

- wb: `SPECTRAL_NM_380` … `SPECTRAL_NM_730`, i.e. `SPECTRAL_NM_{nm}` for
  `nm` in `380, 390, …, 730`.
- bb: `380nm` … `730nm`, i.e. `{nm}nm` for the same `nm` range.
- Both: **36 bands**, 380–730 nm inclusive, 10 nm step — identical range and
  step, different naming convention only. `(730-380)/10 + 1 = 36`, confirmed
  by literal field count in both formats.

## Other observations worth flagging

- wb `SAMPLE_ID` is not row-order-sorted in the file (e.g. first three rows of
  `Age_64a_wb.txt` are `SAMPLE_ID` 810, 1369, 1393 — see `SAMPLE_NAME` A1, A2,
  A3 for the intended display order). Any ingestion code must not assume
  `SAMPLE_ID` order in the raw file corresponds to any particular
  chart-position order; join/sort on `SAMPLE_ID` explicitly.
- bb `Measuring conditions` string uses `°` (0xB0 in latin-1) exactly like wb's
  `MEASUREMENT_SOURCE` — confirms latin-1 for both archives, not just wb.
- bb file naming has inconsistent case and punctuation quirks that a filename
  parser should tolerate rather than assume a strict pattern: e.g.
  `heilbs_91_bb.txt` vs `Heilbs_91_2_bb.txt` (case), `dag._120_2._bb.txt`
  (trailing/embedded dots), `sudk__99_bb.txt` (double underscore, no `_2`
  suffix despite the double underscore looking like a placeholder for one).
  Several `_2`-suffixed / non-suffixed pairs across wb and bb appear to be
  repeat measurements of the same nominal press (e.g. `TagesA_147_wb.txt` +
  `TagesA_147_2_wb.txt`, `NLZ_141_bb.txt` + `NLZ_141_2_bb.txt`) — this
  inventory does not attempt to resolve press-identity/repeat-measurement
  pairing across the naming quirks; that's a Task 2+ concern if a
  per-physical-press (rather than per-file) grouping is ever needed.
- `NUMBER_OF_FIELDS` in wb (42) is a genuine cross-check value baked into the
  file and matches the literal field count; bb has no equivalent self-declared
  field count, so field-count validation for bb has to be done by the parser
  (count columns in the header row) rather than checked against a
  self-declared total.

## Recorded numbers at a glance

- wb files: 13 (matches roadmap expectation exactly)
- bb files: 30 raw, **25 real distinct runs** + 5 duplicate `lab*` companions
  (roadmap expected 30; the zip's file count matches, but the *information
  content* is 25 runs, not 30 — flagging as the key deviation from a naive
  "43 press runs" reading)
- Total real distinct press runs: 13 + 25 = **38** (zip's raw file count sums
  to 43, matching the "43 press runs" description only if the 5 duplicates are
  counted as if they were runs)
- Samples per file: 1485 everywhere, matching ECI2002 exactly, verified both
  by declared count (wb) and literal row count (wb and bb)
- CMYK combination table: fixed, chart-defined, byte-identical across all 13
  wb files; bb requires this table to be joined in via `SampleID` since bb
  carries no CMYK columns at all
- Spectral bands: 36, 380–730 nm @ 10 nm, present in every real file
- LAB/XYZ: absent in wb (as expected); present in 20/25 real bb files (Group
  A); Lab-only (no XYZ) in 5/25 real bb files (Group B)

## QUARANTINE NOTICE (1 Aug 2026) — bb runs unusable pending layout resolution

The CMYK join from the wb chart table is provably WRONG for bb: patches sharing
a joined recipe agree at median dE00 ~0.8-1.1 within wb runs, but 25-28 within
bb runs (verified on 6 bb files, 28 duplicate pairs each) — i.e. the bb charts
use a different ECI2002 patch layout than wb (visual vs random layout, most
likely). All bb processed CSVs were therefore deleted; only the 13 wb runs are
registered. The bb raw data remains in the zip. To rehabilitate bb we need the
actual bb chart layout (ask Phil — he supplied the data), or a validated
permutation recovery. Do NOT re-join bb against the wb table.

Bonus finding: wb duplicate-recipe pairs give a GENUINE within-run
print-repeatability floor for newsprint: median dE00 ~0.7-1.1 across wb runs —
usable in the paper where the coated-paper datasets offered none (their
duplicates are byte-identical copies).
