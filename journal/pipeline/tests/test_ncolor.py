import textwrap

import pandas as pd

from journal.pipeline.ifra import parse_cgats, spectral_to_xyz
from journal.pipeline.ingest_ncolor import ink_cols, standardize

# --- ProfileMaker/MeasureTool fixture (LGO* header keys, CGATS-like data block).
# Mirrors KCMYG_5clr_spectral.txt / Apex_CMYKOGB_7clr_spectral.txt: tab-separated,
# nCLR_k ink columns on the 0-100 scale, nm380..nm730 spectral reflectance 0-1.

PM_FIXTURE = textwrap.dedent("""\
    LGOROWLENGTH\t54
    LGOMCCHANNEL01\t"InkName = 'Cyan' InkSample = '[0.13,0.27];' "
    LGOMCCHANNEL02\t"InkName = 'Magenta' InkSample = '[0.34,0.30];' "
    CREATED\t"8/20/2009"  # Time: 14:42
    INSTRUMENTATION\t"Eye-One iO"
    ILLUMINATION_NAME\t"D50"
    OBSERVER_ANGLE\t"2"
    KEYWORD\t"SampleID"
    KEYWORD\t"SAMPLE_NAME"
    NUMBER_OF_FIELDS\t43
    BEGIN_DATA_FORMAT
    SampleID\tSAMPLE_NAME\t5CLR_1\t5CLR_2\t5CLR_3\t5CLR_4\t5CLR_5\t{bands}
    END_DATA_FORMAT
    NUMBER_OF_SETS\t1
    BEGIN_DATA
    1\tA1\t10.00\t20.00\t0.00\t0.00\t100.00\t{flat}
    END_DATA
""").format(
    bands="\t".join(f"nm{nm}" for nm in range(380, 740, 10)),
    flat="\t".join(["0.60"] * 36),  # flat 60% reflectance -> Y ~= 60
)


def test_profilemaker_parses_and_converts(tmp_path):
    p = tmp_path / "pm.txt"
    p.write_text(PM_FIXTURE, encoding="latin-1")
    df = parse_cgats(p)
    assert len(df) == 1
    # ink columns found in the file's nCLR_k order
    assert ink_cols(df) == ["5CLR_1", "5CLR_2", "5CLR_3", "5CLR_4", "5CLR_5"]
    out = spectral_to_xyz(df)
    assert abs(out.loc[0, "XYZ_Y"] - 60) < 2
    assert 78 < out.loc[0, "LAB_L"] < 86


def test_standardize_spectral_schema(tmp_path):
    p = tmp_path / "pm.txt"
    p.write_text(PM_FIXTURE, encoding="latin-1")
    std = standardize(parse_cgats(p))
    assert list(std.columns) == [
        "SAMPLE_ID", "INK_1", "INK_2", "INK_3", "INK_4", "INK_5",
        "XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B"]
    # ink order preserved, values already 0-100 stay untouched
    assert std.loc[0, ["INK_1", "INK_2", "INK_3", "INK_4", "INK_5"]].tolist() == \
        [10.0, 20.0, 0.0, 0.0, 100.0]


def test_standardize_rescales_fractional_inks():
    # a file carrying inks on the 0-1 scale must come out on 0-100
    df = pd.DataFrame({
        "SAMPLE_ID": [1], "3CLR_1": [0.5], "3CLR_2": [1.0], "3CLR_3": [0.0],
        "XYZ_X": [50.0], "XYZ_Y": [60.0], "XYZ_Z": [40.0],
        "LAB_L": [81.8], "LAB_A": [-2.9], "LAB_B": [7.9]})
    std = standardize(df)
    assert std.loc[0, ["INK_1", "INK_2", "INK_3"]].tolist() == [50.0, 100.0, 0.0]


# --- APTEC-style file: native XYZ + Lab, no spectral bands -------------------

APTEC_FIXTURE = textwrap.dedent("""\
    CGATS.17
    ORIGINATOR\t"APTEC"
    NUMBER_OF_FIELDS\t15
    BEGIN_DATA_FORMAT
    SAMPLE_ID\tSAMPLE_NAME\t7CLR_1\t7CLR_2\t7CLR_3\t7CLR_4\t7CLR_5\t7CLR_6\t7CLR_7\tXYZ_X\tXYZ_Y\tXYZ_Z\tLAB_L\tLAB_A\tLAB_B
    END_DATA_FORMAT
    NUMBER_OF_SETS\t1
    BEGIN_DATA
    8\t8\t0\t0\t0\t0\t0\t0\t0\t81.9966\t84.2039\t74.4764\t93.539\t1.559\t-4.441
    END_DATA
""")


def test_native_xyz_used_directly(tmp_path):
    p = tmp_path / "aptec.txt"
    p.write_text(APTEC_FIXTURE, encoding="latin-1")
    df = parse_cgats(p)
    std = standardize(df)
    # native XYZ/Lab pass through verbatim -- no spectral conversion happened
    assert std.loc[0, "XYZ_X"] == 81.9966
    assert std.loc[0, "XYZ_Y"] == 84.2039
    assert std.loc[0, "LAB_L"] == 93.539
    assert list(std.columns) == [
        "SAMPLE_ID", "INK_1", "INK_2", "INK_3", "INK_4", "INK_5", "INK_6", "INK_7",
        "XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B"]
