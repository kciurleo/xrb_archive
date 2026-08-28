#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate the photometric-system corrections for the Aql X-1
light curves.

The ORIGINAL Aql X-1 calibration used:

    B : Gaia BP
    V : Gaia G ("Gaia")
    R : Pan-STARRS r
    I : Pan-STARRS i

The desired CORRECTED calibration uses Pan-STARRS-based
Johnson-Cousins transformations for ALL filters:

    B : Pan-STARRS -> Johnson-Cousins B
    V : Pan-STARRS -> Johnson-Cousins V
    R : Pan-STARRS -> Johnson-Cousins R
    I : Pan-STARRS -> Johnson-Cousins I

Because the same ensemble was used throughout the Aql X-1
light curve, the difference is a single constant magnitude
offset per filter.

This script ONLY reads the standards catalog and prints
the results. No files are saved.
"""

import numpy as np
import pandas as pd


# ============================================================
# INPUT FILE
# ============================================================

standards_file = (
    '/home/kmc249/Downloads/BEST_ens_stds_info.csv'
)

standards = pd.read_csv(standards_file)


def get_mag_from_panstarrs(i, g, r, di, dg, dr, band):
    '''
    given Pan-STARRS i, g, r mags and errors, return synthesized band and associated error
    '''

    coeffs = {
        'I': ('i', -0.366, -0.136, -0.018, 0.017),
        'R': ('r', -0.137,  0.108, -0.029, 0.015),
        'V': ('r',  0.005,  0.462,  0.013, 0.012),
        'B': ('g',  0.212,  0.556,  0.034, 0.032),
    }

    ref_band, a, b, c, scatter = coeffs[band]

    color = g - r

    # Select the magnitude and error of the reference band
    ref_mag = {
        'i': i,
        'g': g,
        'r': r
    }[ref_band]

    ref_err = {
        'i': di,
        'g': dg,
        'r': dr
    }[ref_band]

    mag = ref_mag + a + b*color + c*color**2

    color_term = b + 2*c*color
    color_err = np.sqrt(dg**2 + dr**2)

    mag_err = np.sqrt(
        ref_err**2 +
        color_term**2 * color_err**2 +
        scatter**2
    )

    return mag, mag_err


#calculate johnsons values for all the standards
for band in ['B', 'V', 'R', 'I']:
    standards[f'{band}_panstarrs'], standards[f'd{band}_panstarrs'] = get_mag_from_panstarrs(
        standards['i'],
        standards['g'],
        standards['r'],
        standards['di'],
        standards['dg'],
        standards['dr'],
        band
    )





# ============================================================
# AQL X-1 ENSEMBLE
# ============================================================
#
# Put the EXACT ensemble IDs used in your original Aql X-1
# calibration here.
#
# You can get these from the old pipeline with:
#
#     print(ensemble_cols)
#
# ============================================================

ensemble_ids = [492, 187, 244, 467, 295, 371, 744,258,116,80,506,66,120,729,395,681,215,318,104,290,800]
print('length of ensemble',len(ensemble_ids))

if len(ensemble_ids) == 0:

    raise ValueError(
        "\nNo ensemble IDs have been entered.\n\n"
        "Please fill in ensemble_ids with the exact "
        "comparison stars used for Aql X-1."
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    'num int',

    # Original calibration values
    'BP',
    'Gaia',
    'r',
    'i',

    # Pan-STARRS -> Johnson-Cousins transformations
    'B_panstarrs',
    'V_panstarrs',
    'R_panstarrs',
    'I_panstarrs'
]

missing_columns = [
    col for col in required_columns
    if col not in standards.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns in standards file:\n\n"
        + "\n".join(missing_columns)
    )


# Make ID numeric
standards['num int'] = pd.to_numeric(
    standards['num int'],
    errors='coerce'
)


# ============================================================
# SELECT EXACT ENSEMBLE
# ============================================================

ens = standards[
    standards['num int'].isin(ensemble_ids)
].copy()


found_ids = set(
    ens['num int'].dropna().astype(int)
)

missing_ids = sorted(
    set(ensemble_ids) - found_ids
)


print("\n")
print("=" * 75)
print("Aql X-1 photometric-system correction")
print("=" * 75)

print("\nRequested ensemble:")
print(ensemble_ids)

print(
    f"\nNumber of requested stars: "
    f"{len(ensemble_ids)}"
)

print(
    f"Number found in catalog: "
    f"{len(found_ids)}"
)

if missing_ids:

    print("\nWARNING:")
    print(
        "These ensemble stars were not found in "
        "the standards catalog:"
    )
    print(missing_ids)

print("\nStars actually used:")
print(sorted(found_ids))


# ============================================================
# FUNCTION TO CALCULATE CORRECTION
# ============================================================

def calculate_delta(
    old_column,
    corrected_column,
    old_label,
    corrected_label
):

    old = pd.to_numeric(
        ens[old_column],
        errors='coerce'
    )

    corrected = pd.to_numeric(
        ens[corrected_column],
        errors='coerce'
    )

    valid = (
        np.isfinite(old) &
        np.isfinite(corrected)
    )

    old_mean = np.mean(
        old[valid]
    )

    corrected_mean = np.mean(
        corrected[valid]
    )

    delta = corrected_mean - old_mean

    # Individual-star differences
    individual_delta = (
        corrected[valid].values
        -
        old[valid].values
    )

    delta_std = np.std(
        individual_delta
    )

    delta_median = np.median(
        individual_delta
    )

    print("\n" + "-" * 75)
    print(
        f"{old_label}  ->  {corrected_label}"
    )
    print("-" * 75)

    print(
        f"Original {old_label:18s} mean = "
        f"{old_mean:.6f}"
    )

    print(
        f"Corrected {corrected_label:15s} mean = "
        f"{corrected_mean:.6f}"
    )

    print(
        f"Delta ({corrected_label} - {old_label}) = "
        f"{delta:+.6f} mag"
    )

    print(
        f"Median individual delta              = "
        f"{delta_median:+.6f} mag"
    )

    print(
        f"Scatter of individual deltas         = "
        f"{delta_std:.6f} mag"
    )

    print(
        f"Stars used                            = "
        f"{valid.sum()} / {len(ens)}"
    )

    return {
        'old_mean': old_mean,
        'corrected_mean': corrected_mean,
        'delta': delta,
        'individual_delta_std': delta_std
    }


# ============================================================
# B
#
# ORIGINAL:
#     Gaia BP
#
# CORRECTED:
#     Pan-STARRS -> Johnson-Cousins B
# ============================================================

result_B = calculate_delta(
    old_column='BP',
    corrected_column='B_panstarrs',
    old_label='Gaia BP',
    corrected_label='JC B'
)


# ============================================================
# V
#
# ORIGINAL:
#     Gaia G
#
# CORRECTED:
#     Pan-STARRS -> Johnson-Cousins V
# ============================================================

result_V = calculate_delta(
    old_column='Gaia',
    corrected_column='V_panstarrs',
    old_label='Gaia G',
    corrected_label='JC V'
)


# ============================================================
# R
#
# ORIGINAL:
#     Pan-STARRS r
#
# CORRECTED:
#     Pan-STARRS -> Johnson-Cousins R
# ============================================================

result_R = calculate_delta(
    old_column='r',
    corrected_column='R_panstarrs',
    old_label='Pan-STARRS r',
    corrected_label='JC R'
)


# ============================================================
# I
#
# ORIGINAL:
#     Pan-STARRS i
#
# CORRECTED:
#     Pan-STARRS -> Johnson-Cousins I
# ============================================================

result_I = calculate_delta(
    old_column='i',
    corrected_column='I_panstarrs',
    old_label='Pan-STARRS i',
    corrected_label='JC I'
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FINAL ZERO-POINT CORRECTIONS")
print("=" * 75)

print(
    "\nThese are the values to ADD to the existing "
    "Aql X-1 light curves:"
)

print(
    f"\nB:  {result_B['delta']:+.6f} mag"
)

print(
    f"V:  {result_V['delta']:+.6f} mag"
)

print(
    f"R:  {result_R['delta']:+.6f} mag"
)

print(
    f"I:  {result_I['delta']:+.6f} mag"
)


# ============================================================
# INDIVIDUAL-STAR DIAGNOSTIC
# ============================================================

diagnostic = ens[
    [
        'num int',

        'BP',
        'B_panstarrs',

        'Gaia',
        'V_panstarrs',

        'r',
        'R_panstarrs',

        'i',
        'I_panstarrs'
    ]
].copy()


diagnostic['B_delta'] = (
    diagnostic['B_panstarrs']
    - diagnostic['BP']
)

diagnostic['V_delta'] = (
    diagnostic['V_panstarrs']
    - diagnostic['Gaia']
)

diagnostic['R_delta'] = (
    diagnostic['R_panstarrs']
    - diagnostic['r']
)

diagnostic['I_delta'] = (
    diagnostic['I_panstarrs']
    - diagnostic['i']
)


print("\n")
print("=" * 75)
print("INDIVIDUAL ENSEMBLE-STAR DELTAS")
print("=" * 75)

print(
    diagnostic[
        [
            'num int',
            'B_delta',
            'V_delta',
            'R_delta',
            'I_delta'
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


print("\n")
print("=" * 75)
print("NO FILES WERE SAVED.")
print("=" * 75)