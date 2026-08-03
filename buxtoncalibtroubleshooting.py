#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:28:37 2026

@author: kmc249
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file='/neta/xrb/GX339-4/product/GX339_ref_stars_kciurleo.csv'
refstars=pd.read_csv(file)
one=refstars.iloc[340]
two=refstars.iloc[287]
three=refstars.iloc[336]

iis=[]
diis=[]
gaias=[]
dgaias=[]
bps=[]
rps=[]
gs=[]
rs=[]
bps_err=[]
rps_err=[]
gs_err=[]
rs_err=[]
for row in [one, two, three]:
    gaias.append(row['Gaia'])
    iis.append(row['i'])
    dgaias.append(row['dGaia'])
    diis.append(row['di'])
    bps.append(row['BP'])
    rps.append(row['RP'])
    rs.append(row['r'])
    gs.append(row['g'])
    bps_err.append(row['dBP'])
    rps_err.append(row['dRP'])
    rs_err.append(row['dr'])
    gs_err.append(row['dg'])
    
iis=np.array(iis)
gaias=np.array(gaias)
diis=np.array(diis)
dgaias=np.array(dgaias)
bps=np.array(bps)
rps=np.array(rps)
rs=np.array(rs)
gs=np.array(gs)
bps_err=np.array(bps_err)
rps_err=np.array(rps_err)
rs_err=np.array(rs_err)
gs_err=np.array(gs_err)

#from gaia dr2 table of phot transformations
c = bps - rps
synth_V=gaias - (-0.01760-0.006860*c-0.1732*c**2)

#from panstarrs transformation
d = gs - rs
synth_I=iis-0.366-0.136*d-0.018*d**2

#V synth errors
c_err = np.sqrt(bps_err**2 + rps_err**2)

synth_V_err = np.sqrt(
    dgaias**2 +
    (0.006860 + 0.3464*c)**2 * c_err**2 +
    0.03765**2      # transformation scatter, comes from table
)

#I synth errors
d_err = np.sqrt(gs_err**2 + rs_err**2)

synth_I_err = np.sqrt(
    diis**2 +
    (-0.136 - 0.036*d)**2 * d_err**2 +
    0.017**2        # transformation scatter, comes from paper
)

names=[1,2,3]

buxtonV=np.array([17.54, 17.43, 16.99])
buxtonI=np.array([15.26,15.49,15.62])

V_err=np.array([0.06,0.05,0.05])
I_err=np.array([0.07,0.07,0.06])

colors=['red','green','blue']

#V band
x=np.linspace(16.3, 17.6, 20)
plt.figure(figsize=(8,6))
for g, gerr, v, verr, name, color in zip(gaias, dgaias, buxtonV, V_err, names, colors):
    plt.errorbar(
        g, v,
        yerr=verr,
        xerr=gerr,
        fmt='o',
        color=color,
        label=str(name)
    )

plt.plot(x, x, color='black', linestyle='dashed')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('Gaia')
plt.ylabel('Buxton V')
plt.legend()
plt.title('V band')
plt.show()


#I band
x=np.linspace(15.2, 16.1, 20)
plt.figure(figsize=(8,6))
for g, gerr, v, verr, name, color in zip(iis,diis,  buxtonI, I_err, names, colors):
    plt.errorbar(
        g, v,
        yerr=verr,
        xerr=gerr,
        fmt='o',
        color=color,
        label=str(name)
    )

plt.plot(x, x, color='black', linestyle='dashed')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('PanSTARRS i')
plt.ylabel('Buxton I')
plt.legend()
plt.title('I band')
plt.show()

#colors
our_colors=gaias-iis
her_colors=buxtonV-buxtonI
her_color_errors=np.sqrt(V_err**2+I_err**2)
our_color_errors=np.sqrt(dgaias**2+diis**2)
x=np.linspace(0.43, 2.3, 20)
plt.figure(figsize=(8,6))
for g, gerr, v, verr, name, color in zip(our_colors, our_color_errors,her_colors, her_color_errors, names, colors):
    plt.errorbar(
        g, v,
        yerr=verr,
        xerr=gerr,
        fmt='o',
        color=color,
        label=str(name)
    )

plt.plot(x, x, color='black', linestyle='dashed')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('Our colors')
plt.ylabel('Her colors')
plt.legend()
plt.title('Color')
plt.show()

#Synth Vs
x=np.linspace(16.3, 17.6, 20)
plt.figure(figsize=(8,6))
for g, gerr, v, verr, name, color in zip(synth_V, synth_V_err, buxtonV, V_err, names, colors):
    plt.errorbar(
        g, v,
        yerr=verr,
        xerr=gerr,
        fmt='o',
        color=color,
        label=str(name)
    )

plt.plot(x, x, color='black', linestyle='dashed')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('Gaia V Converted')
plt.ylabel('Buxton V')
plt.legend()
plt.title('V band')
plt.show()

#Synth Is
x=np.linspace(15.2, 16.1, 20)
plt.figure(figsize=(8,6))
for g, gerr, v, verr, name, color in zip(synth_I, synth_I_err, buxtonI, I_err, names, colors):
    plt.errorbar(
        g, v,
        yerr=verr,
        xerr=gerr,
        fmt='o',
        color=color,
        label=str(name)
    )

plt.plot(x, x, color='black', linestyle='dashed')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('PanSTARRS i Converted')
plt.ylabel('Buxton I')
plt.legend()
plt.title('I band')
plt.show()

#colors
synth_colors=synth_V-synth_I
her_colors=buxtonV-buxtonI
her_color_errors=np.sqrt(V_err**2+I_err**2)
synth_color_errors=np.sqrt(synth_V_err**2+synth_I_err**2)
x=np.linspace(0.43, 2.3, 20)
plt.figure(figsize=(8,6))
for g, gerr, v, verr, name, color in zip(synth_colors, synth_color_errors,her_colors, her_color_errors, names, colors):
    plt.errorbar(
        g, v,
        yerr=verr,
        xerr=gerr,
        fmt='o',
        color=color,
        label=str(name)
    )

plt.plot(x, x, color='black', linestyle='dashed')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlabel('Cacluated colors')
plt.ylabel('Her colors')
plt.legend()
plt.title('Color')
plt.show()


#%%
#Big plot
def make_panel(ax, xdata, xerr, ydata, yerr,
               xlabel, ylabel, title,
               xmin, xmax):

    xx = np.linspace(xmin, xmax, 20)

    for x, xe, y, ye, name, color in zip(
            xdata, xerr, ydata, yerr, names, colors):

        ax.errorbar(
            x, y,
            xerr=xe,
            yerr=ye,
            fmt='o',
            color=color,
            label=str(name)
        )

    ax.plot(xx, xx, 'k--')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: V
make_panel(
    axes[0,0],
    gaias, dgaias,
    buxtonV, V_err,
    'Gaia', 'Buxton V',
    'Observed V',
    16.3, 17.6
)

make_panel(
    axes[1,0],
    synth_V, synth_V_err,
    buxtonV, V_err,
    'Cacluated V', 'Buxton V',
    'Cacluated V',
    16.3, 17.6
)

# Row 2: I
make_panel(
    axes[0,1],
    iis, diis,
    buxtonI, I_err,
    'PanSTARRS i', 'Buxton I',
    'Observed I',
    15.2, 16.1
)

make_panel(
    axes[1,1],
    synth_I, synth_I_err,
    buxtonI, I_err,
    'Cacluated I', 'Buxton I',
    'Cacluated I',
    15.2, 16.1
)

# Row 3: Colors
make_panel(
    axes[0,2],
    our_colors, our_color_errors,
    her_colors, her_color_errors,
    'Our color', 'Buxton color',
    'Observed Color',
    0.3, 2.5
)

make_panel(
    axes[1,2],
    synth_colors, synth_color_errors,
    her_colors, her_color_errors,
    'Cacluated color', 'Buxton color',
    'Cacluated Color',
    0.3, 2.5
)

# V panels
for ax in [axes[0,0], axes[1,0]]:
    ax.set_xlim(17.6, 16.3)  # inverted
    ax.set_ylim(17.6, 16.3)

# I panels
for ax in [axes[0,1], axes[1,1]]:
    ax.set_xlim(16.1, 15.2)
    ax.set_ylim(16.1, 15.2)

# Color panels
for ax in [axes[0,2], axes[1,2]]:
    ax.set_xlim(0.3, 2.5 )
    ax.set_ylim(0.3, 2.5)

# one legend for the whole figure
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center',
           ncol=3,
           bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



#%%%

#getting b band comps for pagny
    
iis=np.array(iis)
gaias=np.array(gaias)
diis=np.array(diis)
dgaias=np.array(dgaias)
bps=np.array(bps)
rps=np.array(rps)
rs=np.array(rs)
gs=np.array(gs)
bps_err=np.array(bps_err)
rps_err=np.array(rps_err)
rs_err=np.array(rs_err)
gs_err=np.array(gs_err)

#PanSTARRS transformation:
panstarrsB=gs+0.212+0.556*d+0.034*d**2
#transformation error from paper is 0.032
print(panstarrsB)

lam_B = 4380   # Angstrom
lam_V = 5450
lam_I = 7980


# effective wavelengths (Angstrom)
wavelengths = np.array([4380, 5450, 7980])

# Effective wavelengths (Angstrom)
wavelengths = np.array([4380, 5450, 7980])

plt.figure(figsize=(7,6))

for i, color, name in zip(range(3), colors, names):

    mags = np.array([
        panstarrsB[i],
        buxtonV[i],
        buxtonI[i]
    ])

    plt.plot(
        wavelengths,
        mags,
        '-o',
        color=color,
        lw=2,
        ms=8,
        label=f'Star {name}'
    )

plt.gca().invert_yaxis()      # brighter stars at the top
plt.xticks(wavelengths, ['B', 'V', 'I'])
plt.xlabel('Filter')
plt.ylabel('Magnitude')
plt.title('with buxton v, i and synth b')
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(7,6))

for i, color, name in zip(range(3), colors, names):

    mags = np.array([
        panstarrsB[i],
        synth_V[i],
        synth_I[i]
    ])

    plt.plot(
        wavelengths,
        mags,
        '-o',
        color=color,
        lw=2,
        ms=8,
        label=f'Star {name}'
    )

plt.gca().invert_yaxis()      # brighter stars at the top
plt.xticks(wavelengths, ['B', 'V', 'I'])
plt.xlabel('Filter')
plt.ylabel('Magnitude')
plt.title('with all synth colors')
plt.legend()
plt.tight_layout()
plt.show()