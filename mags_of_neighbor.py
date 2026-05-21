#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:02:41 2026

@author: kmc249
"""

import numpy as np
import pandas as pd
import astropy.units as u
from dust_extinction.parameter_averages import CCM89
import matplotlib.pyplot as plt

our_ratios=[np.nan, 17, 21, 31, 28, 35, 52]
our_errors=[np.nan, 6, 3, 9, 2, 2, 4]
their_ratios=np.array([np.nan, 11.8, 17, 22.5, np.nan, np.nan, 41])
their_errors=np.array([np.nan, 1.1, 2.,1.5, np.nan, np.nan,9])

bands = ["V", "R", "I", "J", "H", "K"]

#Our magnitudes
neighbor_mags = np.array([
    18.846,   # V
    19.023,   # R
    18.494,   # I
    16.850,   # J
    16.391,   # H
    16.038    # K
])

#Uncertainties
neighbor_errs = np.array([
    0.040,   # V
    0.035,   # R
    0.024,   # I
    0.230,   # J
    0.100,   # H
    0.143    # K
])

#and for aql
aql_mags = np.array([
    21.030,
    20.482,
    19.837,
    17.878,
    17.064,
    15.962
])
    
aql_errs=np.array([
    0.108,
    0.133,
    0.073,
    0.240,
    0.106,
    0.143
])


#Central wavelengths, from SMARTS transmission info that I then calculated
wav = np.array([
    0.5476,  #V
    0.6576,  #R
    0.8275,  #I
    1.2467,  #J
    1.6312,  #H
    2.1426   #K
]) * u.micron


#table of stellar type and colors
table = pd.read_csv(
    "/home/kmc249/Downloads/EEM_dwarf_UBVIJHK_colors_Teff.txt",
    delim_whitespace=True,
    na_values=["....", "...", "-----", "------"]
)
table = table.iloc[::-1].reset_index(drop=True)
#make things numerical

for c in table.columns[1:-1]:
    table[c] = pd.to_numeric(table[c], errors="coerce")


# Keep only rows with required columns
required_cols = [
    "SpT",
    "V-Rc",
    "V-Ic",
    "V-Ks",
    "J-H",
    "H-Ks"
]

table = table.dropna(subset=required_cols)

#make it only the G stars
#table=table.loc[table['SpT'].str.startswith('G')]

table = table.reset_index(drop=True)

#which one are we doing?
mags=aql_mags
errs=aql_errs
#%%
#Function for dereddening our magnitudes
def deredden_wav(EBV, Rv=3.1):
    '''

    Parameters
    ----------
    EBV : color excess, to be fit
    Rv : assumed to be 3.1
    wl: effective wavelength
    mags: observed magnitudes

    Returns
    -------
    dereddened magnitude

    '''
    #from relationships:
    Av=Rv*EBV
    
    ext_model = CCM89(Rv=Rv)

    A_lambda = ext_model(wav) * Av

    return mags - A_lambda

#Function for getting the colors out of our magnitudes
def our_colors(EBV):
    #given a certain reddening value, calculate the color vector we need
    #first, deredden
    dered=deredden_wav(EBV)
    
    #assuming an order of ["V", "R", "I", "J", "H", "K"], and assuming we want
    #a color order of "V-Rc","V-Ic","V-Ks","J-H","H-Ks"
    deredcolor=np.array([
        dered[0]-dered[1],
        dered[0]-dered[2],
        dered[0]-dered[5],
        dered[3]-dered[4],
        dered[4]-dered[5]
        ])
    return deredcolor

def sigma_quick(error1, error2):
    #given two errors, do sqrt sum of squares
    return np.sqrt(error1**2+error2**2)

#%%
#Define some values of EBV:
ebv_array=np.linspace(0, 2, 73)

chi2_grid = np.zeros((len(table), len(ebv_array)))

#terrible error propogation
sigma=np.array([
        sigma_quick(errs[0],errs[1]),
        sigma_quick(errs[0],errs[2]),
        sigma_quick(errs[0],errs[5]),
        sigma_quick(errs[3],errs[4]),
        sigma_quick(errs[4],errs[5]),
        ])

#should we use all the colors
use_colors = np.array([True, True, True, True, True])

#For each spectral type:
for id, row in table.iterrows():
    spt=row['SpT']
    
    #get the model colors
    model_colors=row[["V-Rc","V-Ic","V-Ks","J-H","H-Ks"]].values
    
    #then for all our ebv array, calculate our deredened data values
    for j, ebv in enumerate(ebv_array):
        data_colors = our_colors(ebv)
    
        # apply mask here
        dc = data_colors[use_colors]
        mc = model_colors[use_colors]
        sig = sigma[use_colors]
        
        #and get a reduced chi squared value
        chi2 = np.sum((dc - mc)**2 / sig**2)
        dof = len(dc) - 2
        chi2_red = chi2 / dof
    
        chi2_grid[id, j] = chi2_red


best_flat_index = np.argmin(chi2_grid)
best_spT_idx, best_ebv_idx = np.unravel_index(best_flat_index, chi2_grid.shape)

best_spt = table["SpT"].iloc[best_spT_idx]
best_ebv = ebv_array[best_ebv_idx]

best_chi2 = chi2_grid[best_spT_idx, best_ebv_idx]

print("Best fit:")
print("SpT:", best_spt)
print("E(B-V):", best_ebv)
print("chi2_red:", best_chi2)

#%%
#plotting
from matplotlib.colors import LogNorm

spt_labels = table["SpT"].values
best_x = ebv_array[best_ebv_idx]
best_y = best_spT_idx
#target_spt = "G9V" #for aql, K4
target_spt = "K4V"

g5_idx = table.index[table["SpT"] == target_spt][0]
ebv_target = 0.5

plt.figure(figsize=(12, 12))

im = plt.imshow(
    chi2_grid,
    aspect='auto',
    origin='lower',
    extent=[ebv_array[0], ebv_array[-1], 0, len(table)],
    cmap='RdYlGn_r',
    norm=LogNorm()
)
plt.colorbar(im, label=r'$\chi^2_{red}$')

plt.xlabel("E(B-V)")
plt.ylabel("Spectral Type Index")
plt.title("Reduced Chi-Squared Grid")
plt.yticks(
    ticks=np.arange(len(table)),
    labels=spt_labels
)

plt.scatter(
    best_x,
    best_y,
    color='black',
    s=80,
    marker='o',
    label='Best fit'
)

plt.scatter(
    ebv_target,
    g5_idx,
    color='black',
    s=80,
    marker='x',
    label='G5V, EBV=0.5'
)
plt.legend()
plt.show()

#%%
# Recompute best-fit quantities
best_row = table.iloc[best_spT_idx]

model_colors_best = best_row[["V-Rc","V-Ic","V-Ks","J-H","H-Ks"]].values
data_colors_best = our_colors(best_ebv)

color_labels = ["V-R", "V-I", "V-K", "J-H", "H-K"]
x = np.arange(len(color_labels))
plt.figure(figsize=(8,5))

plt.plot(x, model_colors_best, 'o-', label=f"Model ({best_spt})")
plt.errorbar(
    x, data_colors_best,
    yerr=sigma,
    fmt='s--',
    capsize=3,
    label=f"Data (E(B-V)={best_ebv:.2f})")

plt.xticks(x, color_labels)
plt.ylabel("Color (mag)")
plt.title("Best-fit Model vs Dereddened Data Colors")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#%%



###doing really stupid stuff over here
from itertools import product

color_names = ["V-R", "V-I", "V-K", "J-H", "H-K"]

all_masks = list(product([False, True], repeat=5))

# optional: remove the all-False case (invalid)
all_masks = [m for m in all_masks if any(m)]

results = []

for mask in all_masks:
    use_colors = np.array(mask)

    chi2_grid = np.zeros((len(table), len(ebv_array)))

    for i, row in table.iterrows():
        model_colors = row[["V-Rc","V-Ic","V-Ks","J-H","H-Ks"]].values

        for j, ebv in enumerate(ebv_array):
            data_colors = our_colors(ebv)

            dc = data_colors[use_colors]
            mc = model_colors[use_colors]
            sig = sigma[use_colors]

            chi2 = np.sum((dc - mc)**2 / sig**2)

            dof = len(dc) - 2
            chi2_red = chi2 #/ dof

            chi2_grid[i, j] = chi2_red

    best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)

    best_spT_idx, best_ebv_idx = best_idx

    results.append({
        "mask": mask,
        "best_spt": table["SpT"].iloc[best_spT_idx],
        "best_ebv": ebv_array[best_ebv_idx],
        "best_chi2": chi2_grid[best_spT_idx, best_ebv_idx]
    })
    
    
    spt_labels = table["SpT"].values
    best_x = ebv_array[best_ebv_idx]
    best_y = best_spT_idx
    #target_spt = "G9V" #for aql, K4
    target_spt = "K4V"
    
    g5_idx = table.index[table["SpT"] == target_spt][0]
    ebv_target = 0.5
    
    plt.figure(figsize=(12, 12))
    
    im = plt.imshow(
        chi2_grid,
        aspect='auto',
        origin='lower',
        extent=[ebv_array[0], ebv_array[-1], 0, len(table)],
        cmap='RdYlGn_r',
        norm=LogNorm()
    )
    plt.colorbar(im, label=r'$\chi^2_{red}$')
    
    plt.xlabel("E(B-V)")
    plt.ylabel("Spectral Type Index")
    plt.title("Reduced Chi-Squared Grid")
    plt.yticks(
        ticks=np.arange(len(table)),
        labels=spt_labels
    )
    
    plt.scatter(
        best_x,
        best_y,
        color='black',
        s=80,
        marker='o',
        label='Best fit'
    )
    
    plt.scatter(
        ebv_target,
        g5_idx,
        color='black',
        s=80,
        marker='x',
        #label='G9V, EBV=0.5'
    )
    plt.legend()
    plt.show()
    
best_run = min(results, key=lambda x: x["best_chi2"])

print("Best mask:", best_run["mask"])
print("Best SpT:", best_run["best_spt"])
print("Best E(B-V):", best_run["best_ebv"])
print("Best chi2:", best_run["best_chi2"])