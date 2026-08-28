#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:27:24 2026

@author: kmc249
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time


# ============================================================
# READ IN ALL LIGHT CURVES
# ============================================================

data = {}

data['B SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_B_corrected_lc_4_27.csv'
    )
}

data['V SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_V_corrected_lc_4_27.csv'
    )
}

data['R SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_R_corrected_lc.csv'
    )
}

data['I SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_I_corrected_lc_4_27.csv'
    )
}

data['J SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_J_corrected_lc.csv'
    )
}

data['H SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_H_corrected_lc.csv'
    )
}

data['K SMARTS'] = {
    'df': pd.read_csv(
        '/neta/xrb/AqlX-1/product/just_subtracted_shifted/'
        'AqlX-1_K_corrected_lc.csv'
    )
}


# ============================================================
# STANDARDIZE MAGNITUDE COLUMN NAMES
# ============================================================

for key in data:

    df = data[key]['df']

    if key == 'J SMARTS':
        nameband = 'J'
    elif key == 'K SMARTS':
        nameband = 'K'
    elif key == 'H SMARTS':
        nameband = 'H'
    else:
        nameband = 'R'

    if key == 'V SMARTS':
        df['mag_shifted'] = df[f'{nameband}mag_shifted']
    else:
        df['mag_shifted'] = df[f'{nameband}mag_corr']

    df['mag_corr_err'] = df[f'e_{nameband}mag_corr']
    df['mag_err'] = df[f'e_{nameband}mag']
    df['mag'] = df[f'{nameband}mag']

    df['flag'] = np.nan
    df['error_flag'] = np.nan


# ============================================================
# LCO DATA
# ============================================================

def read_corrected_txt(path):

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=[
            "MJD",
            "mag_corr",
            "mag_corr_err",
            "flag",
            "mag",
            "mag_err",
            "error_flag"
        ]
    )

    df["nice time"] = pd.to_datetime(
        Time(df["MJD"], format="mjd").to_datetime()
    )

    return df


def read_lco(banzai_path, orac_path):

    banzai = read_corrected_txt(banzai_path)
    orac = read_corrected_txt(orac_path)

    return pd.concat([banzai, orac], ignore_index=True)


for band in ['V', 'gp', 'R', 'rp', 'ip']:

    data[f'{band} LCO'] = {
        'df': pd.read_csv(
            f'/home/kmc249/Downloads/{band}_LCO_shifted.csv'
        )
    }


data['V LCO']['df']['mag_shifted'] = (
    data['V LCO']['df']['mag_corr']
)

data['V LCO']['df']['alt_mag_shifted'] = (
    data['V LCO']['df']['alt_mag_corr']
)


# ============================================================
# OUTBURST DATA
# ============================================================

oldfull = pd.read_csv(
    "/home/kmc249/Downloads/full_outbursts.csv"
)

full = pd.read_csv(
    "/home/kmc249/Downloads/table.tsv",
    sep='\t',
    header=0,
    names=['OB', 'Start MJD', 'End MJD']
)

mini = pd.read_csv(
    "/home/kmc249/Downloads/mini_outbursts.csv"
)


# ============================================================
# SELECT OUTBURST INTERVALS
# ============================================================

count_mini = False

full_intervals = (
    list(zip(full["Start MJD"], full["End MJD"])) +
    list(zip(mini["Start MJD"], mini["End MJD"]))
)

if count_mini:
    intervals = full_intervals
else:
    intervals = list(
        zip(full["Start MJD"], full["End MJD"])
    )


# ============================================================
# FIND GLOBAL DATA RANGE
# ============================================================

all_mjd = np.concatenate([
    data[key]["df"]["MJD"].values
    for key in data
    if "MJD" in data[key]["df"].columns
])

data_start = np.min(all_mjd)
data_end = np.max(all_mjd)

intervals = sorted(
    intervals,
    key=lambda x: x[0]
)


# ============================================================
# QUiescent intervals
# ============================================================

quiescent_intervals = []

quiescent_intervals.append(
    (data_start, intervals[0][0])
)

for i in range(len(intervals) - 1):

    end_current = intervals[i][1]
    start_next = intervals[i + 1][0]

    if start_next > end_current:

        quiescent_intervals.append(
            (end_current, start_next)
        )

quiescent_intervals.append(
    (intervals[-1][1], data_end)
)


def get_quiescent(df, intervals):

    mask = np.ones(len(df), dtype=bool)

    for start, end in intervals:

        mask &= ~(
            (df["MJD"] >= start) &
            (df["MJD"] <= end)
        )

    return df[mask].copy()


# ============================================================
# STORE QUIESCENT DATA
# ============================================================

for key in data:

    df = data[key]['df']

    data[key]['quiescence'] = get_quiescent(
        df,
        intervals
    )

    data[key]['global med q mag'] = (
        data[key]['quiescence']['mag_shifted'].median()
    )

    data[key]['global mean q mag'] = (
        data[key]['quiescence']['mag_shifted'].mean()
    )

    print(
        key,
        data[key]['global mean q mag']
    )


# ============================================================
# PAN-STARRS TRANSFORMATION
#
# g,r,i --> B,V,R,I
# ============================================================

def get_mag_from_panstarrs(
    i, g, r,
    di, dg, dr,
    band
):
    '''
    Given Pan-STARRS i, g, r magnitudes and errors,
    return synthesized band magnitude and error.
    '''

    coeffs = {

        'I': (
            'i',
            -0.366,
            -0.136,
            -0.018,
            0.017
        ),

        'R': (
            'r',
            -0.137,
             0.108,
            -0.029,
             0.015
        ),

        'V': (
            'r',
             0.005,
             0.462,
             0.013,
             0.012
        ),

        'B': (
            'g',
             0.212,
             0.556,
             0.034,
             0.032
        )
    }

    ref_band, a, b, c, scatter = coeffs[band]

    color = g - r

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

    mag = (
        ref_mag
        + a
        + b * color
        + c * color**2
    )

    color_term = (
        b
        + 2 * c * color
    )

    color_err = np.sqrt(
        dg**2 + dr**2
    )

    mag_err = np.sqrt(
        ref_err**2
        + color_term**2 * color_err**2
        + scatter**2
    )

    return mag, mag_err


# ============================================================
# GAIA TRANSFORMATION
#
# Gaia G,BP,RP --> V,R,I
# ============================================================

def get_mag_from_gaia(
    gaia, bp, rp,
    dgaia, dbp, drp,
    band
):
    '''
    Given Gaia G, BP, RP magnitudes and errors,
    return synthesized band magnitude and error.
    '''

    coeffs = {

        'V': (
            -0.01760,
            -0.006860,
            -0.1732,
            0.045858
        ),

        'R': (
            -0.003226,
             0.3833,
            -0.1345,
             0.04840
        ),

        'I': (
             0.02085,
             0.7419,
            -0.09631,
             0.04956
        )
    }

    a, b, d, scatter = coeffs[band]

    color = bp - rp

    mag = (
        gaia
        - (
            a
            + b * color
            + d * color**2
        )
    )

    color_term = (
        b
        + 2 * d * color
    )

    color_err = np.sqrt(
        dbp**2 + drp**2
    )

    mag_err = np.sqrt(
        dgaia**2
        + color_term**2 * color_err**2
        + scatter**2
    )

    return mag, mag_err


# ============================================================
# IMPORTANT:
#
# YOUR CURRENT CALIBRATION IS:
#
# B = Gaia BP
# V = Gaia G
# R = Pan-STARRS r
# I = Pan-STARRS i
#
# The transformations below are the ALTERNATIVE
# calibration system.
#
# We do NOT overwrite the current calibration.
# ============================================================


# ============================================================
# EXAMPLE CATALOG DATAFRAME
#
# Replace "calstars" with your actual calibration-star
# dataframe.
#
# It needs columns:
#
# Gaia:
#   Gaia, BP, RP
#   e_Gaia, e_BP, e_RP
#
# Pan-STARRS:
#   g, r, i
#   e_g, e_r, e_i
#
# ============================================================

# calstars = YOUR_CALIBRATION_STAR_DATAFRAME


# ============================================================
# MAKE ALTERNATIVE SYNTHETIC MAGNITUDES
# ============================================================

def make_alternative_magnitudes(calstars):

    calstars = calstars.copy()

    # --------------------------------------------------------
    # Pan-STARRS transformations
    # --------------------------------------------------------

    for band in ['B', 'V', 'R', 'I']:

        calstars[
            f'{band}_PS'
        ], calstars[
            f'e_{band}_PS'
        ] = get_mag_from_panstarrs(

            i=calstars['i'],
            g=calstars['g'],
            r=calstars['r'],

            di=calstars['e_i'],
            dg=calstars['e_g'],
            dr=calstars['e_r'],

            band=band
        )


    # --------------------------------------------------------
    # Gaia transformations
    # --------------------------------------------------------

    for band in ['V', 'R', 'I']:

        calstars[
            f'{band}_Gaia'
        ], calstars[
            f'e_{band}_Gaia'
        ] = get_mag_from_gaia(

            gaia=calstars['Gaia'],
            bp=calstars['BP'],
            rp=calstars['RP'],

            dgaia=calstars['e_Gaia'],
            dbp=calstars['e_BP'],
            drp=calstars['e_RP'],

            band=band
        )


    # --------------------------------------------------------
    # CURRENT calibration system
    # --------------------------------------------------------

    calstars['B_current'] = calstars['BP']
    calstars['V_current'] = calstars['Gaia']
    calstars['R_current'] = calstars['r']
    calstars['I_current'] = calstars['i']

    return calstars


# ============================================================
# COMPARE CATALOG MAGNITUDES
#
# This tells you how large the transformation actually is.
# ============================================================

def plot_catalog_transformations(calstars):

    fig, axes = plt.subplots(
        4, 1,
        figsize=(10, 12),
        sharex=False
    )

    comparisons = {

        'B': (
            'B_current',
            'B_PS',
            'BP'
        ),

        'V': (
            'V_current',
            'V_PS',
            'Gaia'
        ),

        'R': (
            'R_current',
            'R_Gaia',
            'r'
        ),

        'I': (
            'I_current',
            'I_Gaia',
            'i'
        )
    }

    for ax, (band, vals) in zip(
        axes,
        comparisons.items()
    ):

        current, transformed, _ = vals

        delta = (
            calstars[transformed]
            - calstars[current]
        )

        ax.hist(
            delta.dropna(),
            bins=30,
            alpha=0.7
        )

        ax.axvline(
            0,
            color='k',
            linestyle='--'
        )

        ax.axvline(
            delta.median(),
            color='red',
            linestyle='-',
            label=(
                f"median = "
                f"{delta.median():.3f} mag"
            )
        )

        ax.set_xlabel(
            f"{band} transformed - {band} current (mag)"
        )

        ax.set_ylabel("N")
        ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# SIMPLE LIGHT-CURVE PLOT
#
# This plots your CURRENT SMARTS calibration.
# ============================================================

def plot_current_optical():

    bands = [
        'B SMARTS',
        'V SMARTS',
        'R SMARTS',
        'I SMARTS'
    ]

    colors = {
        'B SMARTS': 'blue',
        'V SMARTS': 'green',
        'R SMARTS': 'red',
        'I SMARTS': 'darkorange'
    }

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 12),
        sharex=True
    )

    for ax, key in zip(
        axes,
        bands
    ):

        q = data[key]['quiescence']

        ax.errorbar(
            q['MJD'],
            q['mag_shifted'],
            yerr=q['mag_err'],
            fmt='.',
            ms=4,
            alpha=0.6,
            color=colors[key]
        )

        ax.axhline(
            data[key]['global med q mag'],
            color='k',
            linestyle='--',
            alpha=0.6,
            label=(
                f"median = "
                f"{data[key]['global med q mag']:.3f}"
            )
        )

        ax.set_ylabel(
            f"{key.split()[0]} mag"
        )

        ax.invert_yaxis()
        ax.legend()

    axes[-1].set_xlabel("MJD")

    plt.tight_layout()
    plt.show()


# ============================================================
# RUN CURRENT LIGHT CURVE PLOT
# ============================================================

plot_current_optical()


# ============================================================
# IF YOU HAVE YOUR CALIBRATION-STAR DATAFRAME:
#
# Uncomment these:
#
# calstars = make_alternative_magnitudes(calstars)
# plot_catalog_transformations(calstars)
# ============================================================
