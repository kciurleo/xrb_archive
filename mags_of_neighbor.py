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
    15.820    # K
])

#Uncertainties
neighbor_errs = np.array([
    0.040,   # V
    0.035,   # R
    0.024,   # I
    0.230,   # J
    0.100,   # H
    0.216    # K
])

#and for aql
aql_mags = np.array([
    21.030,
    20.482,
    19.837,
    17.878,
    17.064,
    16.215
])
    
aql_errs=np.array([
    0.108,
    0.133,
    0.073,
    0.240,
    0.106,
    0.275
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
doing='aql'
if doing=='neighbor':
    mags=neighbor_mags
    errs=neighbor_errs
    target_spt = "G9V" #for aql, K4
if doing=='aql':
    mags=aql_mags
    errs=aql_errs
    target_spt = "K4V"
    
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
    
        chi2_grid[id, j] = chi2#chi2_red


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

#1sigma thing???
chi2_min = np.min(chi2_grid)
delta_chi2 = chi2_grid - chi2_min
levels = [2.30, 6.17, 11.8]

spt_labels = table["SpT"].values
best_x = ebv_array[best_ebv_idx]
best_y = best_spT_idx


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


plt.contour(
    ebv_array,
    np.arange(len(table)),
    delta_chi2,
    levels=levels,
    colors='k',
    linewidths=1.5
)

plt.clabel(
    plt.contour(
        ebv_array,
        np.arange(len(table)),
        delta_chi2,
        levels=levels,
        colors='k'
    ),
    fmt={
        2.30: '1σ',
        6.17: '2σ',
        11.8: '3σ'
    }
)

plt.colorbar(im, label=r'$\chi^2$')

plt.xlabel("E(B-V)")
plt.ylabel("Spectral Type Index")
plt.title("Chi-Squared Grid")
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
    label=f'{target_spt}, EBV=0.5'
)
plt.legend()
plt.show()

#%%
# Recompute best-fit quantities
best_row = table.iloc[best_spT_idx]
target_row = table.iloc[g5_idx]

model_colors_best = best_row[["V-Rc","V-Ic","V-Ks","J-H","H-Ks"]].values
model_colors_lit = target_row[["V-Rc","V-Ic","V-Ks","J-H","H-Ks"]].values
data_colors_best = our_colors(best_ebv)

color_labels = ["V-R", "V-I", "V-K", "J-H", "H-K"]
x = np.arange(len(color_labels))
plt.figure(figsize=(8,5))

plt.plot(x, model_colors_best, 'o-', label=f"Best-fit Model ({best_spt})")
plt.plot(x, model_colors_lit, 'o-', label=f"Lit. Model ({target_spt})")
plt.errorbar(
    x, data_colors_best,
    yerr=sigma,
    fmt='s--',
    capsize=3,
    label=f"Data (E(B-V)={best_ebv:.2f})")

plt.xticks(x, color_labels)
plt.ylabel("Color (mag)")
plt.title(doing)
plt.legend()
plt.grid(alpha=0.3)

plt.show()


#%%

#Forced reddening
def mags_to_colors(mags):
    return np.array([
        mags[0] - mags[1],  # V-R
        mags[0] - mags[2],  # V-I
        mags[0] - mags[5],  # V-K
        mags[3] - mags[4],  # J-H
        mags[4] - mags[5],  # H-K
    ])

neighbor_colors = mags_to_colors(neighbor_mags)
neighbor_color_errors=np.array([
    sigma_quick(neighbor_errs[0],neighbor_errs[1]),
    sigma_quick(neighbor_errs[0],neighbor_errs[2]),
    sigma_quick(neighbor_errs[0],neighbor_errs[5]),
    sigma_quick(neighbor_errs[3],neighbor_errs[4]),
    sigma_quick(neighbor_errs[4],neighbor_errs[5]),
    ])

g9_row = table.loc[table["SpT"] == "G9V"].iloc[0]
k4_row = table.loc[table["SpT"] == "K4V"].iloc[0]

g9_colors = g9_row[
    ["V-Rc", "V-Ic", "V-Ks", "J-H", "H-Ks"]
].values.astype(float)
k4_colors = k4_row[
    ["V-Rc", "V-Ic", "V-Ks", "J-H", "H-Ks"]
].values.astype(float)

color_residuals = neighbor_colors - g9_colors

print("Color residuals:")
for name, resid in zip(
        ["V-R","V-I","V-K","J-H","H-K"],
        color_residuals):
    print(name, resid)
    
aql_colors = mags_to_colors(aql_mags)

aql_colors_corrected = aql_colors - color_residuals

chi2_spt = np.zeros(len(table))

for i, row in table.iterrows():

    model_colors = row[
        ["V-Rc", "V-Ic", "V-Ks", "J-H", "H-Ks"]
    ].values.astype(float)

    chi2 = np.sum(
        (aql_colors_corrected - model_colors)**2 /
        sigma**2
    )

    chi2_spt[i] = chi2

best_idx = np.argmin(chi2_spt)

print("Best-fit Aql spectral type:")
print(table["SpT"].iloc[best_idx])
print("chi2 =", chi2_spt[best_idx])

plt.figure(figsize=(8,5))

plt.errorbar(
    np.arange(5),
    aql_colors_corrected,
    yerr=sigma,
    fmt='o-',
    label='Aql X-1 Dereddened'
)

best_model = table.iloc[best_idx][
    ["V-Rc","V-Ic","V-Ks","J-H","H-Ks"]
].values.astype(float)

plt.plot(
    np.arange(5),
    best_model,
    's-',
    label=table["SpT"].iloc[best_idx]
)

plt.plot(
    np.arange(5),
    k4_colors,
    's-',
    label="K4V"
)

plt.xticks(
    np.arange(5),
    ["V-R","V-I","V-K","J-H","H-K"]
)
plt.xlabel('Color (mag)')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

color_labels = ["V-R", "V-I", "V-K", "J-H", "H-K"]
x = np.arange(5)

plt.figure(figsize=(9,5))

# Neighbor observed
plt.errorbar(x, neighbor_colors, yerr=neighbor_color_errors, fmt='o-', label='Neighbor')

# G9V reference
plt.plot(x, g9_colors, 's-', label='G9V')

# Neighbor forced dereddened (should overlap G9V)
plt.plot(x, neighbor_colors-color_residuals, 'x--', label='Dereddened Neighbor')

plt.xticks(x, color_labels)
plt.ylabel("Color (mag)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#%%

def table_colors_new(row):
    vr = row["V-Rc"]
    vi = row["V-Ic"]
    vk = row["V-Ks"]
    jh = row["J-H"]
    hk = row["H-Ks"]

    ri = vi - vr
    ij = vk - vi - jh - hk

    return np.array([
        vr,
        ri,
        ij,
        jh,
        hk
    ])

def mags_to_colors_new(mags):
    return np.array([
        mags[0] - mags[1],  # V-R
        mags[1] - mags[2],  # R-I
        mags[2] - mags[3],  # I-J
        mags[3] - mags[4],  # J-H
        mags[4] - mags[5],  # H-K
    ])

sigma_new = np.array([
    sigma_quick(errs[0], errs[1]),  # V-R
    sigma_quick(errs[1], errs[2]),  # R-I
    sigma_quick(errs[2], errs[3]),  # I-J
    sigma_quick(errs[3], errs[4]),  # J-H
    sigma_quick(errs[4], errs[5]),  # H-K
])

neighbor_color_errors_new = np.array([
    sigma_quick(neighbor_errs[0], neighbor_errs[1]),
    sigma_quick(neighbor_errs[1], neighbor_errs[2]),
    sigma_quick(neighbor_errs[2], neighbor_errs[3]),
    sigma_quick(neighbor_errs[3], neighbor_errs[4]),
    sigma_quick(neighbor_errs[4], neighbor_errs[5]),
])

color_labels_new = ["V-R", "R-I", "I-J", "J-H", "H-K"]


neighbor_colors_new = mags_to_colors_new(neighbor_mags)


g9_colors_new = table_colors_new(g9_row)

k4_colors_new = table_colors_new(k4_row)

color_residuals_new = neighbor_colors_new - g9_colors_new

print("Color residuals:")
for name, resid in zip(
        color_labels_new,
        color_residuals):
    print(name, resid)
    
aql_colors_new = mags_to_colors_new(aql_mags)

aql_colors_corrected_new = aql_colors_new - color_residuals_new

chi2_spt_new = np.zeros(len(table))

for i, row in table.iterrows():

    model_colors = table_colors_new(row)

    chi2 = np.sum(
        (aql_colors_corrected_new - model_colors)**2 /
        sigma_new**2
    )

    chi2_spt_new[i] = chi2

best_idx_new = np.argmin(chi2_spt_new)

print("Best-fit Aql spectral type:")
print(table["SpT"].iloc[best_idx_new])
print("chi2 =", chi2_spt[best_idx_new])

plt.figure(figsize=(8,5))

plt.errorbar(
    np.arange(5),
    aql_colors_corrected_new,
    yerr=sigma_new,
    fmt='o-',
    label='Aql X-1 Dereddened'
)

best_model_new = table_colors_new(table.iloc[best_idx_new])

plt.plot(
    np.arange(5),
    best_model_new,
    's-',
    label=table["SpT"].iloc[best_idx]
)

plt.plot(
    np.arange(5),
    k4_colors_new,
    's-',
    label="K4V"
)

plt.xticks(
    np.arange(5),
    color_labels_new
)
plt.xlabel('Color (mag)')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

x = np.arange(5)

plt.figure(figsize=(9,5))

# Neighbor observed
plt.errorbar(x, neighbor_colors_new, yerr=neighbor_color_errors_new, fmt='o-', label='Neighbor')

# G9V reference
plt.plot(x, g9_colors_new, 's-', label='G9V')

# Neighbor forced dereddened (should overlap G9V)
plt.plot(x, neighbor_colors_new-color_residuals_new, 'x--', label='Dereddened Neighbor')

plt.xticks(x, color_labels)
plt.ylabel("Color (mag)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#%%

'''

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
    plt.title(f'Using: {np.array(["V-Rc","V-Ic","V-Ks","J-H","H-Ks"])[use_colors]}')
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
        label=f'{target_spt}, EBV=0.5'
    )
    plt.legend()
    plt.show()
    
best_run = min(results, key=lambda x: x["best_chi2"])

print("Best mask:", best_run["mask"])
print("Best SpT:", best_run["best_spt"])
print("Best E(B-V):", best_run["best_ebv"])
print("Best chi2:", best_run["best_chi2"])
'''