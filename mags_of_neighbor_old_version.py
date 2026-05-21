#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:02:41 2026

@author: kmc249
"""

import numpy as np
import pandas as pd

from scipy.optimize import minimize

import astropy.units as u

from dust_extinction.parameter_averages import CCM89

our_ratios=[np.nan, 17, 21, 31, 28, 35, 52]
our_errors=[np.nan, 6, 3, 9, 2, 2, 4]
their_ratios=np.array([np.nan, 11.8, 17, 22.5, np.nan, np.nan, 41])
their_errors=np.array([np.nan, 1.1, 2.,1.5, np.nan, np.nan,9])

bands = ["V", "R", "I", "J", "H", "K"]

#Our magnitudes
neighbor_mags = np.array([
    18.284,   # V
    19.023,   # R
    18.226,   # I
    16.850,   # J
    16.391,   # H
    15.820    # K
])

#Uncertainties
neighbor_errs = np.array([
    0.026,   # V
    0.035,   # R
    0.023,   # I
    0.230,   # J
    0.100,   # H
    0.216    # K
])


############################################################
# EFFECTIVE WAVELENGTHS (microns)
############################################################

wav = np.array([
    0.551,   # V
    0.658,   # Rc
    0.806,   # Ic
    1.235,   # J
    1.662,   # H
    2.159    # Ks
]) * u.micron


############################################################
# LOAD MAMAJEK TABLE
############################################################

table = pd.read_csv(
    "/home/kmc249/Downloads/EEM_dwarf_UBVIJHK_colors_Teff.txt",
    delim_whitespace=True,
    na_values=["....", "...", "-----", "------"]
)

#numerical

for c in table.columns[1:-1]:
    table[c] = pd.to_numeric(table[c], errors="coerce")

# Keep only rows with required columns
required_cols = [
    "SpT",
    "Teff",
    "Mv",
    "V-Rc",
    "V-Ic",
    "V-Ks",
    "J-H",
    "H-Ks"
]

table = table.dropna(subset=required_cols)

#make it only the G stars
#table=table.loc[table['SpT'].str.startswith('G')]

# Reset indexing
table = table.reset_index(drop=True)


############################################################
# BUILD ABSOLUTE MAGNITUDES
############################################################

# Reconstruct absolute magnitudes in each band
#
# We use:
#
# V-R = Mv - Mr
# V-I = Mv - Mi
# V-K = Mv - Mk
#
# etc.

table["M_R"] = table["Mv"] - table["V-Rc"]

table["M_I"] = table["Mv"] - table["V-Ic"]

table["M_K"] = table["Mv"] - table["V-Ks"]

table["M_H"] = (
    table["M_K"] + table["H-Ks"]
)

table["M_J"] = (
    table["M_H"] + table["J-H"]
)

# Final model magnitude vector
model_mag_cols = [
    "Mv",
    "M_R",
    "M_I",
    "M_J",
    "M_H",
    "M_K"
]


############################################################
# CCM89 EXTINCTION FUNCTION
############################################################

def extinction_vector(Av, Rv):

    ext_model = CCM89(Rv=Rv)

    A_lambda_over_Av = ext_model(wav)

    return Av * A_lambda_over_Av


############################################################
# CHI-SQUARED FOR ONE STELLAR TYPE
############################################################

def fit_star_type(row):

    intrinsic = row[model_mag_cols].values.astype(float)

    def chi2(params):

        Av, Rv, offset = params

        # physically sensible bounds
        if Av < 0:
            return 1e30

        if Rv < 2.0 or Rv > 6.0:
            return 1e30

        A_lambda = extinction_vector(Av, Rv)

        model = intrinsic + A_lambda + offset

        return np.sum(
            ((neighbor_mags - model) / neighbor_errs)**2
        )

    result = minimize(
        chi2,
        x0=[0.5, 3.1, 10.0],
        method="Nelder-Mead"
    )

    return result.fun, result.x


############################################################
# LOOP OVER ALL STELLAR TYPES
############################################################

best = None

for i, row in table.iterrows():

    chi2, params = fit_star_type(row)

    if best is None or chi2 < best["chi2"]:

        best = {
            "chi2": chi2,
            "row": row,
            "params": params
        }


############################################################
# PRINT RESULTS
############################################################

best_row = best["row"]

Av, Rv, offset = best["params"]

print()
print("BEST FIT")
print("========")
print(f"Spectral type : {best_row['SpT']}")
print(f"Teff          : {best_row['Teff']:.0f} K")
print(f"Av            : {Av:.3f}")
print(f"Rv            : {Rv:.3f}")
print(f"offset        : {offset:.3f}")
print(f"chi2          : {best['chi2']:.2f}")
print()


############################################################
# OPTIONAL: PRINT MODEL VS DATA
############################################################

intrinsic = best_row[model_mag_cols].values.astype(float)

A_lambda = extinction_vector(Av, Rv)

model = intrinsic + A_lambda + offset

print("Band   Obs     Model")
print("----------------------")

for b, obs, mod in zip(bands, neighbor_mags, model):

    print(f"{b:>3}   {obs:6.3f}   {mod:6.3f}")
    
############################################################
# FIXED EXTINCTION FROM PAPER
############################################################

Rv_fixed = 3.1
EBV_fixed = 0.5

Av_fixed = Rv_fixed * EBV_fixed

# Compute extinction ONCE
A_lambda_fixed = extinction_vector(Av_fixed, Rv_fixed)

############################################################
# CHI-SQUARED FOR ONE STELLAR TYPE
############################################################

def fit_star_type(row):

    intrinsic = row[model_mag_cols].values.astype(float)

    # model magnitudes with fixed extinction
    reddened_model = intrinsic + A_lambda_fixed

    # only fit the offset
    def chi2(offset):

        model = reddened_model + offset

        return np.sum(
            ((neighbor_mags - model) / neighbor_errs)**2
        )

    result = minimize(
        chi2,
        x0=[10.0],
        method="Nelder-Mead"
    )

    return result.fun, result.x[0]


best = None

for i, row in table.iterrows():

    chi2, offset = fit_star_type(row)

    if best is None or chi2 < best["chi2"]:

        best = {
            "chi2": chi2,
            "row": row,
            "offset": offset
        }
        
best_row = best["row"]
offset = best["offset"]

print()
print("BEST FIT")
print("========")
print(f"Spectral type : {best_row['SpT']}")
print(f"Teff          : {best_row['Teff']:.0f} K")
print(f"Av            : {Av_fixed:.3f}")
print(f"Rv            : {Rv_fixed:.3f}")
print(f"offset        : {offset:.3f}")
print(f"chi2          : {best['chi2']:.2f}")
print()

intrinsic = best_row[model_mag_cols].values.astype(float)

model = intrinsic + A_lambda_fixed + offset