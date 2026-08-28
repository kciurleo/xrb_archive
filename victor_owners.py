#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match SMARTS Table 1 photometric measurements to FITS exposures,
then inspect FITS headers for OWNER / ProjectID information.

Created on Mon Aug 24 2026
@author: kmc249
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ============================================================
# FILES / DIRECTORIES
# ============================================================

phot_file = "/home/kmc249/Downloads/SMARTS_Table1_final(1).csv"
log_file = "/neta/xrb/A0620-00/product/log_files_A0620-00.csv"

base_dir = Path("/neta/xrb/A0620-00")

one_m_dir = base_dir / "1m"
one_point_three_m_dir = base_dir / "1.3m"


# ============================================================
# READ FILES
# ============================================================

phot = pd.read_csv(phot_file)

log = pd.read_csv(
    log_file,
    low_memory=False
)


# ============================================================
# KEEP ORIGINAL ROW NUMBERS
# ============================================================

phot["phot_original_row"] = np.arange(len(phot))
log["log_original_row"] = np.arange(len(log))


# ============================================================
# CONVERT LOG DATETIME -> MJD
# ============================================================

log["datetime"] = pd.to_datetime(
    log["datetime"],
    format="mixed",
    errors="coerce",
    utc=True
)

log["MJD"] = np.nan

valid_datetime = log["datetime"].notna()

log.loc[valid_datetime, "MJD"] = (
    log.loc[valid_datetime, "datetime"].astype("int64")
    / 1e9
    / 86400
    + 40587
)


# ============================================================
# CLEAN LOG FILTER
# ============================================================

log["filter"] = (
    log["filter"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# ============================================================
# RESHAPE PHOTOMETRY TABLE
# ============================================================

bands = ["B", "V", "I", "H"]

magnitude_columns = [
    f"{band}_total_mag"
    for band in bands
]

id_vars = [
    column
    for column in phot.columns
    if column not in (
        [
            f"{band}_total_mag"
            for band in bands
        ]
        +
        [
            f"{band}_total_mag_err"
            for band in bands
        ]
        +
        [
            f"{band}_nonstellar_mag"
            for band in bands
        ]
    )
]

phot_long = phot.melt(
    id_vars=id_vars,
    value_vars=magnitude_columns,
    var_name="filter",
    value_name="magnitude"
)

phot_long["filter"] = (
    phot_long["filter"]
    .str.replace(
        "_total_mag",
        "",
        regex=False
    )
    .str.upper()
)

phot_long = phot_long[
    phot_long["magnitude"].notna()
].copy()

phot_long["phot_measurement"] = np.arange(
    len(phot_long)
)


# ============================================================
# MATCH TABLE 1 -> LOG
# ============================================================

results = []

for filt in bands:

    print()
    print("-" * 70)
    print(f"MATCHING FILTER: {filt}")
    print("-" * 70)

    p = phot_long[
        phot_long["filter"] == filt
    ].copy()

    l = log[
        (log["filter"] == filt)
        &
        (log["MJD"].notna())
    ].copy()

    print(
        f"Photometry measurements: {len(p):,}"
    )

    print(
        f"Log exposures:          {len(l):,}"
    )

    if len(p) == 0:
        continue

    if len(l) == 0:

        p["filename"] = np.nan
        p["log_original_row"] = np.nan
        p["matched_MJD"] = np.nan
        p["time_diff_sec"] = np.nan
        p["datetime"] = pd.NaT
        p["EXPTIME"] = np.nan
        p["TELESCOP"] = np.nan

        results.append(p)

        continue

    p = p.sort_values("MJD")
    l = l.sort_values("MJD")

    l_match = l[
        [
            "log_original_row",
            "MJD",
            "filename",
            "datetime",
            "EXPTIME",
            "TELESCOP"
        ]
    ].rename(
        columns={
            "MJD": "log_MJD"
        }
    )

    m = pd.merge_asof(
        p,
        l_match,
        left_on="MJD",
        right_on="log_MJD",
        direction="nearest",
        tolerance=0.01
    )

    m["matched_MJD"] = m["log_MJD"]

    m["time_diff_sec"] = (
        (
            m["matched_MJD"]
            -
            m["MJD"]
        ).abs()
        * 86400
    )

    results.append(m)


# ============================================================
# COMBINE MATCHING RESULTS
# ============================================================

matched = (
    pd.concat(
        results,
        ignore_index=True
    )
    .sort_values("phot_measurement")
    .reset_index(drop=True)
)


# ============================================================
# SUCCESSFUL / FAILED
# ============================================================

successful = matched[
    matched["filename"].notna()
].copy()

failed = matched[
    matched["filename"].isna()
].copy()


print()
print("=" * 70)
print("MATCHING SUMMARY")
print("=" * 70)

print(
    f"Original photometry rows : {len(phot):,}"
)

print(
    f"Individual measurements  : {len(phot_long):,}"
)

print(
    f"Successful matches       : {len(successful):,}"
)

print(
    f"Failed matches           : {len(failed):,}"
)


# ============================================================
# BUILD FITS FILE INDEX
# ============================================================

print()
print("=" * 70)
print("BUILDING FITS FILE INDEX")
print("=" * 70)

print("Scanning 1m...")
print("Scanning 1.3m...")


def build_fits_index(directory, telescope_name):
    """
    Recursively find FITS files and return a dictionary:

        filename -> full path

    Also records duplicate filenames.
    """

    index = {}
    duplicates = []

    fits_extensions = {
        ".fits",
        ".fit",
        ".fts",
        ".fits.gz",
        ".fit.gz",
        ".fts.gz"
    }

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        name_lower = path.name.lower()

        if not any(
            name_lower.endswith(ext)
            for ext in fits_extensions
        ):
            continue

        filename = path.name

        if filename in index:
            duplicates.append(
                {
                    "filename": filename,
                    "first_path": str(index[filename]),
                    "duplicate_path": str(path),
                    "telescope": telescope_name
                }
            )
        else:
            index[filename] = path

    return index, duplicates


fits_1m, duplicates_1m = build_fits_index(
    one_m_dir,
    "1m"
)

fits_1p3m, duplicates_1p3m = build_fits_index(
    one_point_three_m_dir,
    "1.3m"
)

print(
    f"1m FITS files found   : {len(fits_1m):,}"
)

print(
    f"1.3m FITS files found : {len(fits_1p3m):,}"
)

duplicates = (
    duplicates_1m
    +
    duplicates_1p3m
)

print(
    f"Duplicate filenames   : {len(duplicates):,}"
)


# ============================================================
# DETERMINE TELESCOPE FROM DATE
# ============================================================

# User specified:
#
#   before January 2003 -> 1m
#   January 2003 onward  -> 1.3m
#
# We use the FITS/log observation datetime.

cutoff_date = pd.Timestamp(
    "2003-01-01",
    tz="UTC"
)


def get_telescope(dt):

    if pd.isna(dt):
        return pd.NA

    if dt < cutoff_date:
        return "1m"

    return "1.3m"


successful["telescope"] = (
    successful["datetime"]
    .apply(get_telescope)
)


# ============================================================
# FIND FITS PATHS
# ============================================================

print()
print("=" * 70)
print("LOCATING FITS FILES")
print("=" * 70)


def find_fits_file(filename, telescope):

    if pd.isna(filename):
        return None

    filename = str(filename)

    if telescope == "1m":
        return fits_1m.get(filename)

    if telescope == "1.3m":
        return fits_1p3m.get(filename)

    return None


successful["fits_path"] = successful.apply(
    lambda row: find_fits_file(
        row["filename"],
        row["telescope"]
    ),
    axis=1
)


successful["fits_found"] = (
    successful["fits_path"].notna()
)


print(
    f"FITS files found     : "
    f"{successful['fits_found'].sum():,}"
)

print(
    f"FITS files not found : "
    f"{(~successful['fits_found']).sum():,}"
)


# ============================================================
# READ FITS HEADERS
# ============================================================

print()
print("=" * 70)
print("READING FITS HEADERS")
print("=" * 70)


# Header values we will store
successful["header_OWNER"] = pd.NA
successful["header_ProjectID"] = pd.NA
successful["header_owner_used"] = pd.NA

successful["header_read"] = False
successful["header_error"] = pd.NA


# Keep track of every FITS file we couldn't read
header_failures = []


for idx, row in successful[
    successful["fits_found"]
].iterrows():

    fits_path = row["fits_path"]

    try:

        # Read primary header only.
        #
        # This is much faster than loading the image data.
        with fits.open(
            fits_path,
            memmap=True
        ) as hdul:

            header = hdul[0].header

        # ----------------------------------------------------
        # OWNER
        # ----------------------------------------------------

        owner = header.get("OWNER", None)

        # ----------------------------------------------------
        # ProjectID
        # ----------------------------------------------------

        project_id = header.get(
            "ProjectID",
            None
        )

        # Some FITS headers may use uppercase PROJECTID.
        if project_id is None:

            project_id = header.get(
                "PROJECTID",
                None
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        successful.at[
            idx,
            "header_OWNER"
        ] = owner

        successful.at[
            idx,
            "header_ProjectID"
        ] = project_id

        successful.at[
            idx,
            "header_read"
        ] = True

        # ----------------------------------------------------
        # Decide which identifier to use
        #
        # Prefer OWNER if present, otherwise ProjectID.
        # ----------------------------------------------------

        if owner is not None and str(owner).strip():

            successful.at[
                idx,
                "header_owner_used"
            ] = str(owner).strip()

        elif project_id is not None and str(project_id).strip():

            successful.at[
                idx,
                "header_owner_used"
            ] = str(project_id).strip()

        else:

            successful.at[
                idx,
                "header_owner_used"
            ] = "MISSING"

    except Exception as e:

        successful.at[
            idx,
            "header_error"
        ] = str(e)

        header_failures.append(
            {
                "phot_measurement":
                    row["phot_measurement"],

                "filename":
                    row["filename"],

                "fits_path":
                    str(fits_path),

                "error":
                    str(e)
            }
        )


# ============================================================
# HEADER SUMMARY
# ============================================================

successful["header_owner_used"] = (
    successful["header_owner_used"]
    .astype("string")
    .str.strip()
)

header_failures = pd.DataFrame(
    header_failures
)


print(
    f"Headers successfully read : "
    f"{successful['header_read'].sum():,}"
)

print(
    f"Header read failures       : "
    f"{len(header_failures):,}"
)

print(
    f"Missing OWNER/ProjectID    : "
    f"{"MISSING" in set(successful['header_owner_used'].dropna())}"
)


# ============================================================
# OWNER / PROJECT COUNTS
# ============================================================

print()
print("=" * 70)
print("OWNER / PROJECT COUNTS")
print("=" * 70)

owner_counts = (
    successful["header_owner_used"]
    .value_counts(dropna=False)
)

print(owner_counts)


# ============================================================
# OWNER COUNTS BY FILTER
# ============================================================

print()
print("=" * 70)
print("OWNER / PROJECT BY FILTER")
print("=" * 70)

owner_by_filter = pd.crosstab(
    successful["header_owner_used"],
    successful["filter"]
)

print(owner_by_filter)


# ============================================================
# OWNER COUNTS BY TELESCOPE
# ============================================================

print()
print("=" * 70)
print("OWNER / PROJECT BY TELESCOPE")
print("=" * 70)

owner_by_telescope = pd.crosstab(
    successful["header_owner_used"],
    successful["telescope"]
)

print(owner_by_telescope)


# ============================================================
# FAILED MATCH COUNTS
# ============================================================

print()
print("=" * 70)
print("FAILED TABLE 1 MATCHES")
print("=" * 70)

print(
    f"Total failed Table 1 measurements: "
    f"{len(failed):,}"
)

print()

if len(failed) > 0:

    print("Failed by filter:")

    print(
        failed["filter"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# FITS FILES NOT FOUND
# ============================================================

fits_not_found = successful[
    ~successful["fits_found"]
].copy()

print()
print("=" * 70)
print("FITS FILES NOT FOUND")
print("=" * 70)

print(
    f"Matched log entries with missing FITS file: "
    f"{len(fits_not_found):,}"
)


# ============================================================
# LOG EXPOSURES THAT WERE NEVER USED
# ============================================================

used_log_rows = (
    successful["log_original_row"]
    .dropna()
    .astype(int)
)

unused_log = log[
    ~log["log_original_row"].isin(
        used_log_rows
    )
].copy()

print()
print("=" * 70)
print("UNUSED LOG EXPOSURES")
print("=" * 70)

print(
    f"Log exposures not selected by Table 1: "
    f"{len(unused_log):,}"
)


# ============================================================
# DUPLICATE USE OF FITS EXPOSURES
# ============================================================

duplicate_log_use = (
    successful
    .groupby(
        "log_original_row"
    )
    .size()
    .reset_index(
        name="n_phot_measurements"
    )
)

duplicate_log_use = duplicate_log_use[
    duplicate_log_use["n_phot_measurements"] > 1
]

print()
print("=" * 70)
print("FITS EXPOSURES USED MORE THAN ONCE")
print("=" * 70)

print(
    f"Log exposures used by >1 Table 1 measurement: "
    f"{len(duplicate_log_use):,}"
)


# ============================================================
# PLOT 1: OWNER / PROJECT COUNTS
# ============================================================

plt.figure(
    figsize=(12, 6)
)

owner_counts_plot = owner_counts.copy()

owner_counts_plot = owner_counts_plot[
    owner_counts_plot.index.notna()
]

owner_counts_plot.sort_values(
    ascending=False
).plot(
    kind="bar"
)

plt.xlabel(
    "OWNER / ProjectID"
)

plt.ylabel(
    "Number of matched measurements"
)

plt.title(
    "SMARTS Measurements by FITS OWNER / ProjectID"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "SMARTS_owner_counts.png",
    dpi=200
)

plt.show()


# ============================================================
# PLOT 2: OWNER / PROJECT BY FILTER
# ============================================================

owner_by_filter_plot = owner_by_filter.copy()

owner_by_filter_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.xlabel(
    "OWNER / ProjectID"
)

plt.ylabel(
    "Number of matched measurements"
)

plt.title(
    "SMARTS Measurements by OWNER / ProjectID and Filter"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Filter"
)

plt.tight_layout()

plt.savefig(
    "SMARTS_owner_by_filter.png",
    dpi=200
)

plt.show()


# ============================================================
# PLOT 3: SUCCESSFUL VS FAILED
# ============================================================

status_counts = pd.Series(
    {
        "Successful match": len(successful),
        "Failed match": len(failed),
        "FITS not found": len(fits_not_found),
        "Header read failure": len(header_failures)
    }
)

plt.figure(
    figsize=(9, 6)
)

status_counts.plot(
    kind="bar",
    color=[
        "seagreen",
        "firebrick",
        "darkorange",
        "purple"
    ]
)

plt.ylabel(
    "Number of measurements"
)

plt.title(
    "SMARTS / FITS Matching Status"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "SMARTS_matching_status.png",
    dpi=200
)

plt.show()


# ============================================================
# SAVE DATA
# ============================================================

matched.to_csv(
    "SMARTS_matches.csv",
    index=False
)

successful.to_csv(
    "SMARTS_successful_matches_with_headers.csv",
    index=False
)

failed.to_csv(
    "SMARTS_failed_matches.csv",
    index=False
)

unused_log.to_csv(
    "SMARTS_unused_log_exposures.csv",
    index=False
)

fits_not_found.to_csv(
    "SMARTS_fits_not_found.csv",
    index=False
)

header_failures.to_csv(
    "SMARTS_header_failures.csv",
    index=False
)

owner_counts.to_csv(
    "SMARTS_owner_counts.csv"
)

owner_by_filter.to_csv(
    "SMARTS_owner_by_filter.csv"
)

owner_by_telescope.to_csv(
    "SMARTS_owner_by_telescope.csv"
)

duplicate_log_use.to_csv(
    "SMARTS_duplicate_log_use.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Table 1 measurements       : {len(phot_long):,}"
)

print(
    f"Matched to log             : {len(successful):,}"
)

print(
    f"Failed Table 1 matches     : {len(failed):,}"
)

print(
    f"FITS files not found       : {len(fits_not_found):,}"
)

print(
    f"Header read failures       : {len(header_failures):,}"
)

print(
    f"Unused log exposures       : {len(unused_log):,}"
)

print(
    f"Repeatedly-used exposures  : {len(duplicate_log_use):,}"
)

print()
print("OWNER / PROJECT COUNTS:")
print(owner_counts)

print()
print("Output files written:")
print("  SMARTS_matches.csv")
print("  SMARTS_successful_matches_with_headers.csv")
print("  SMARTS_failed_matches.csv")
print("  SMARTS_unused_log_exposures.csv")
print("  SMARTS_fits_not_found.csv")
print("  SMARTS_header_failures.csv")
print("  SMARTS_owner_counts.csv")
print("  SMARTS_owner_by_filter.csv")
print("  SMARTS_owner_by_telescope.csv")
print("  SMARTS_duplicate_log_use.csv")
print("  SMARTS_owner_counts.png")
print("  SMARTS_owner_by_filter.png")
print("  SMARTS_matching_status.png")