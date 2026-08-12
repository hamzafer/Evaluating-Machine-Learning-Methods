import json
import numpy as np
import pandas as pd
from datasets import load_all
from colorimetry import XYZ100_to_Lab, delta_e00

pd.set_option("display.width", 160)

results = {}


def dedup_report(ds):
    X = np.round(ds.X, 6)
    key = pd.DataFrame(X, columns=ds.ink_cols)
    key["_grp"] = key.apply(lambda r: tuple(r.values), axis=1)
    groups = key.groupby("_grp").indices  # dict recipe -> row indices

    if ds.kind == "lab_only":
        meas = ds.native_Lab
    elif ds.kind == "xyz_lab_native":
        meas = np.hstack([ds.native_XYZ, ds.native_Lab])
    else:
        meas = ds.spectral

    n_dup_groups = 0
    n_dup_rows = 0
    n_identical_meas_groups = 0
    n_identical_meas_rows = 0
    n_diff_meas_groups = 0
    n_diff_meas_rows = 0
    max_meas_spread = 0.0

    for recipe, idx in groups.items():
        if len(idx) < 2:
            continue
        n_dup_groups += 1
        n_dup_rows += len(idx)
        sub = meas[idx]
        spread = np.max(sub, axis=0) - np.min(sub, axis=0)
        max_spread = np.max(np.abs(spread))
        max_meas_spread = max(max_meas_spread, max_spread)
        if np.allclose(sub, sub[0], atol=1e-9):
            n_identical_meas_groups += 1
            n_identical_meas_rows += len(idx)
        else:
            n_diff_meas_groups += 1
            n_diff_meas_rows += len(idx)

    return dict(
        n_rows=len(ds.X),
        n_unique_recipes=len(groups),
        n_duplicate_recipe_groups=n_dup_groups,
        n_rows_in_duplicate_groups=n_dup_rows,
        n_groups_identical_measurement=n_identical_meas_groups,
        n_rows_identical_measurement=n_identical_meas_rows,
        n_groups_differing_measurement=n_diff_meas_groups,
        n_rows_differing_measurement=n_diff_meas_rows,
        max_measurement_spread_within_dup_group=float(max_meas_spread),
    )


def xyz_lab_own_consistency(ds):
    """Only meaningful for datasets with BOTH native XYZ and native Lab in the same file."""
    if ds.kind != "xyz_lab_native":
        return None
    Lab_from_XYZ = XYZ100_to_Lab(ds.native_XYZ)
    dE = delta_e00(ds.native_Lab, Lab_from_XYZ)
    diffs = Lab_from_XYZ - ds.native_Lab
    return dict(
        median_dE00=float(np.median(dE)),
        p95_dE00=float(np.percentile(dE, 95)),
        max_dE00=float(np.max(dE)),
        mean_abs_dL=float(np.mean(np.abs(diffs[:, 0]))),
        mean_abs_da=float(np.mean(np.abs(diffs[:, 1]))),
        mean_abs_db=float(np.mean(np.abs(diffs[:, 2]))),
    )


def spectral_vs_native(ds):
    """Only meaningful if a file had BOTH spectral AND native XYZ/Lab. None of our files do; document that."""
    has_spectral = ds.spectral is not None
    has_native = (ds.native_XYZ is not None) or (ds.native_Lab is not None)
    return dict(has_spectral=has_spectral, has_native_colorimetry=has_native, both_present=has_spectral and has_native)


def header_notes(ds):
    notes = []
    header_text = "\n".join(ds.header)
    if ds.meta["n_declared_sets"] is not None and ds.meta["n_declared_sets"] != ds.meta["n_actual_rows"]:
        notes.append(
            f"NUMBER_OF_SETS declared={ds.meta['n_declared_sets']} but actual data rows={ds.meta['n_actual_rows']}"
        )
    if "DESCRIPTOR" in header_text:
        for line in ds.header:
            if line.strip().startswith("DESCRIPTOR") and ds.name not in line and "PC11" in line and ds.name == "PC10":
                notes.append(f"DESCRIPTOR line mismatches dataset identity: {line.strip()}")
    # generic descriptor-vs-name eyeball note captured elsewhere manually
    if ds.meta["n_ragged_rows"]:
        notes.append(f"{ds.meta['n_ragged_rows']} ragged data rows (field count mismatch) encountered while parsing")
    return notes


def duplicate_sample_ids(ds):
    if "SAMPLE_ID" in ds.df.columns:
        col = "SAMPLE_ID"
    elif "SampleID" in ds.df.columns:
        col = "SampleID"
    else:
        return None
    vc = ds.df[col].value_counts()
    dup = vc[vc > 1]
    return dict(n_duplicate_ids=int(len(dup)), example_ids=dup.index.tolist()[:10])


def ink_scale_note(ds):
    return dict(min=float(ds.X.min()), max=float(ds.X.max()))


all_ds = load_all()
for name, ds in all_ds.items():
    results[name] = dict(
        n_rows=ds.meta["n_actual_rows"],
        declared_sets=ds.meta["n_declared_sets"],
        ink_cols=ds.ink_cols,
        ink_range=ink_scale_note(ds),
        dedup=dedup_report(ds),
        xyz_lab_self_consistency=xyz_lab_own_consistency(ds),
        spectral_vs_native=spectral_vs_native(ds),
        header_notes=header_notes(ds),
        duplicate_sample_ids=duplicate_sample_ids(ds),
    )
    print(f"=== {name} ===")
    print(json.dumps(results[name], indent=2, default=str))
    print()

with open("/home/user1/blind_verify/work/checks_report.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("wrote checks_report.json")
