import textwrap

from journal.pipeline.ifra import parse_bb, parse_cgats, spectral_to_xyz

# --- wb fixture: genuine CGATS, per task-2-brief.md verbatim ------------------

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


# --- bb fixture: plain-header format (NOT CGATS), per README -----------------
# Preamble (4 key/value lines + blank), then a tab header row
# `SampleID  XYZ-X ... CIE-Lab-b  380nm...730nm`, blank line, one data row.
# Real files use CRLF and latin-1 (degree sign 0xB0 in "Measuring conditions").

_BB_BANDS = "\t".join(f"{nm}nm" for nm in range(380, 740, 10))
_BB_FLAT = "\t".join(["0.60"] * 36)  # same flat 60% reflectance as the wb fixture

BB_FIXTURE = (
    "Originator:\t\r\n"
    "Date:\t10/18/2005\r\n"
    "Measuring device:\tSpectrolino  10771\r\n"
    "Measuring conditions:\t[D50,2\xb0,ANSI T,No,Abs]\r\n"
    "\r\n"
    f"SampleID\tXYZ-X\tXYZ-Y\tXYZ-Z\tCIE-Lab-L\tCIE-Lab-a\tCIE-Lab-b\t{_BB_BANDS}\t\r\n"
    "\r\n"
    f"1\t 54.00\t 60.00\t 45.00\t 82.00\t -1.00\t  3.00\t{_BB_FLAT}\t\r\n"
)


def test_parse_bb_and_convert(tmp_path):
    p = tmp_path / "run_bb.txt"
    p.write_bytes(BB_FIXTURE.encode("latin-1"))
    df = parse_bb(p)
    assert len(df) == 1
    # native XYZ/Lab columns preserved verbatim, distinct from derived names
    assert df.loc[0, "XYZ-X"] == 54.0
    assert df.loc[0, "XYZ-Y"] == 60.0
    assert df.loc[0, "XYZ-Z"] == 45.0
    assert df.loc[0, "CIE-Lab-L"] == 82.0
    assert "CMYK_C" not in df.columns  # bb has no CMYK columns (amendment #3)

    out = spectral_to_xyz(df)
    # native columns untouched by the derived conversion
    assert out.loc[0, "XYZ-X"] == 54.0
    assert out.loc[0, "CIE-Lab-L"] == 82.0
    # derived columns are distinct names, computed from the flat 60% spectrum
    assert abs(out.loc[0, "XYZ_Y"] - 60) < 2
    assert 78 < out.loc[0, "LAB_L"] < 86
    # derived values differ from the (deliberately different) native placeholders
    assert out.loc[0, "XYZ_X"] != out.loc[0, "XYZ-X"]
