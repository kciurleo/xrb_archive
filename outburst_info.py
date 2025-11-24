#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:02:44 2025

@author: kmc249
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time

###million of tables to read in
#table=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv')

table=pd.read_csv('/home/kmc249/Downloads/psf_fluxes_with_extras.csv', low_memory=False)

table['nice time'] = pd.to_datetime(table['time'])

standards=pd.read_csv('/home/kmc249/Downloads/BEST_ens_stds_info.csv')


table['mjds'] = Time(table['nice time']).mjd
table = table[['mjds', 'aql mag']].replace([np.inf, -np.inf], np.nan).dropna()
table = table.sort_values('mjds').reset_index(drop=True)

lco1=pd.read_csv('/home/kmc249/Downloads/R_usable_banzai_lt25.txt', sep=r'\s+', header=None, comment='#', names=['MJD', 'R_mag', 'uncertainty', 'upperlimitflag'])
lco2=pd.read_csv('/home/kmc249/Downloads/R_usable_orac_lt25.txt', sep=r'\s+', header=None, comment='#', names=['MJD', 'R_mag', 'uncertainty', 'upperlimitflag'])
lco = pd.concat([lco1, lco2], ignore_index=True)


def parabola(x, a, b, c):
    return a*x**2 + b*x + c



'''
plt.figure(figsize=(12,3))
plt.scatter(table['mjds'], table['aql mag'], s=13, color='k')
plt.scatter(lco['MJD'], lco['R_mag'], s=13, color='r')
plt.ylim(20,15)
plt.show()
'''
###mini table stuff
data = {
    "Number": ["1", "*", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"],
    "start_cont_x": [np.nan, np.nan, np.nan, np.nan, 53940, 54227, 54320, 54590, np.nan, np.nan, 55375, 56439, 56830, 57590, 57890],
    "end_cont_x":   [52747, np.nan, 53184, 53510, 53977, 54288, 54397, 54700, 54934, 55283, np.nan, 56525, 56910, 57663, 57971],
    "x_peak": [
        52714.41068, 53082.38323, 53154.3162, 53473.35633, 53950.29111,
        54256.33723, 54363.08255, 54643.32935, 54916.34583, 55262.36593,
        55453.13166, 56467.26106, 56856.20316, 57612.13958, 57937.14793
    ],
    "y_peak": [
        15.55058, 16.599621, 15.964228, 16.284791, 17.406423,
        17.190351, 16.639893, 17.303642, 17.303114, 17.285291,
        16.536119, 15.53116, 16.546151, 15.390391, 16.539681
    ],
    "y_cont": [
        18.36509932, 18.41057968, 18.41057968, 18.49627411, 18.51343353,
        18.5171618, 18.50612705, 18.50489723, 18.50858794, 18.49456548,
        18.52299187, 18.51910386, 18.52208131, 18.46065513, 18.41527361
    ],
    "y_half": [
        16.95783966, 17.50510034, 17.18740384, 17.39053255, 17.95992827,
        17.8537564, 17.57301002, 17.90426962, 17.90585097, 17.88992824,
        17.52955544, 17.02513193, 17.53411616, 16.92552306, 17.4774773
    ],
}

outbursts = pd.DataFrame(data)

outbursts = outbursts[~outbursts['Number'].isin(['*', '8','9'])]
n=4
for idx, row in outbursts.iterrows():
    peak_index = (table['mjds'] - row['x_peak']).abs().idxmin()
    if row['Number']=='3':
        start_index = max(peak_index - 17, 0)
        end_index = peak_index + 5
    elif row['Number']=='10':
        start_index = max(peak_index - 6, 0)
        end_index = peak_index + 17
    elif row['Number']=='2':
        start_index = max(peak_index - 40, 0)
        end_index = peak_index + 35
    elif row['Number']=='13':
        start_index = max(peak_index - 11, 0)
        end_index = peak_index + 15
    else:
        num_points=10
        start_index = max(peak_index - num_points, 0)
        end_index = peak_index + 1+num_points
    subset = table.iloc[start_index:end_index]
    if pd.isna(row['start_cont_x']):
        start=row['x_peak']-80
    else:
        start=row['start_cont_x']-20
    if pd.isna(row['end_cont_x']):
        end=row['x_peak']+80
    else:
        end=row['end_cont_x']+20
        
    popt, pcov = curve_fit(parabola, subset['mjds'], subset['aql mag'])
    x_fit = np.linspace(min(subset['mjds']), max(subset['mjds']), 100)
    y_fit = parabola(x_fit, *popt)
    
    #get new max
    a, b, c = popt
    x_vertex = -b / (2*a)
    y_vertex = a*x_vertex**2 + b*x_vertex + c
    outbursts.at[idx, 'new_x_peak'] = x_vertex
    outbursts.at[idx, 'new_y_peak'] = y_vertex


    mini=table.loc[((table['mjds']>start) &(table['mjds']<end))]
    
    plt.figure(figsize=(12,6))
    plt.scatter(mini['mjds'], mini['aql mag'], s=8, color='gray', alpha=0.4)
    plt.scatter(subset['mjds'], subset['aql mag'], s=20, color='k', label='Fit region')
    plt.plot(x_fit, y_fit, color='blue', label='Parabola fit')
    plt.scatter(x_vertex, y_vertex, color='red',label='New Max')
    plt.ylim(20, 15)
    plt.legend()
    plt.title(f'{row["Number"]}')
    plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/outburstplots/{row["Number"]}_max.png', dpi=200)
    #plt.show()


    ###finding half max locations
    x_peak     = x_vertex#row['new_x_peak']
    y_peak     = y_vertex#row['new_y_peak']
    y_cont     = row['y_cont']
    y_half     = y_peak+np.abs(y_cont-y_peak)/2
    
    for xval in [row['start_cont_x'], row['end_cont_x']]:
        if pd.notna(xval): 
            
            #do first half and second half, which xval is smaller may change
            xmin = min(xval, x_peak)
            xmax = max(xval, x_peak)

            subset = table[(table['mjds'] >= xmin) & (table['mjds'] <= xmax)]
            subset = subset.dropna(subset=['mjds', 'aql mag'])
            
            # n-degree polynomial fit
            coeffs = np.polyfit(subset['mjds'], subset['aql mag'], n)
    
            # Solve P(x) = y_half  by subtracting y_half from constant term
            coeffs_shift = coeffs.copy()
            coeffs_shift[-1] -= y_half
    
            p = np.poly1d(coeffs)
            p_shift = np.poly1d(coeffs_shift)
    
            # roots
            roots = p_shift.r
    
            # valid real roots inside interval
            real_roots = [r.real for r in roots
                          if np.isreal(r) and xmin <= r.real <= xmax]
    
            x_half = real_roots[0] if real_roots else np.nan
            
            plt.figure(figsize=(12,6))

            # Scatter points (all)
            plt.scatter(mini['mjds'], mini['aql mag'], s=8, color='gray', alpha=0.4)

            # Highlight the points used in the fit
            plt.scatter(subset['mjds'], subset['aql mag'], s=20, color='k', label='Fit region')

            # Regression line across the fit region
            x_line = np.linspace(xmin, xmax, 400)
            y_line = p(x_line)
            plt.plot(x_line, y_line, color='blue', label=f'{n}-degree polyfit')
    
            if not np.isnan(x_half):
                plt.axvline(x_half, color='red', linestyle='--', label='Half-max x')
                plt.scatter([x_half],[y_half], color='red', s=40)
            plt.plot(x_line, y_line, color='blue', label='SciPy linear fit')

            # Vertical line at half max
            plt.axvline(x_half, color='red', linestyle='--', label='Half-max x')

            # Mark the half-max point
            plt.scatter([x_half], [y_half], color='red', s=40)

            plt.ylim(20, 15)
            plt.legend()
            if x_half<x_peak:
                plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/outburstplots/{row["Number"]}_hm1.png', dpi=200)
            else:
                plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/outburstplots/{row["Number"]}_hm2.png', dpi=200)
            #plt.show()

            if x_half<x_peak:
                outbursts.at[idx, 'new_x_half_one'] = x_half
            else:
                outbursts.at[idx, 'new_x_half_two'] = x_half
            outbursts.at[idx, 'new_y_half'] = y_half
            
print(outbursts)

keep_cols = [
    'Number',
    'start_cont_x',
    'end_cont_x',
    'y_cont',
    'new_x_peak',
    'new_y_peak',
    'new_x_half_one',
    'new_x_half_two'
]

df = outbursts[keep_cols].copy()

rename_map = {
    'Number'        : 'Number',
    'start_cont_x'  : 'Start (MJD)',
    'end_cont_x'    : 'End (MJD)',
    'y_cont'        : 'Mean Q (mag)',
    'new_x_peak'    : 'Peak Time (MJD)',
    'new_y_peak'    : 'Peak Brightness (mag)',
    'new_x_half_one': 'First HM (MJD)',
    'new_x_half_two': 'Second HM (MJD)',
}

df = df.rename(columns=rename_map)

#do some calculations
df['Duration (MJD)']=df['End (MJD)']-df['Start (MJD)']
df['HM Duration (MJD)']=df['Second HM (MJD)']-df['First HM (MJD)']
df['HM Asymmetrty Factor']=np.abs(df['Second HM (MJD)']-df['Peak Time (MJD)'])/np.abs(df['First HM (MJD)']-df['Peak Time (MJD)'])
df['HM Tailing Factor']=(np.abs(df['Second HM (MJD)']-df['Peak Time (MJD)'])+np.abs(df['First HM (MJD)']-df['Peak Time (MJD)']))/(2*np.abs(df['First HM (MJD)']-df['Peak Time (MJD)']))

exclude_cols = ['Start (MJD)', 'End (MJD)', 'First HM (MJD)', 'Second HM (MJD)']
plot_cols = [c for c in df.columns if c not in exclude_cols and c != df.columns[0]]  # exclude first column for plotting


# Convert all columns to numeric where possible
for col in df.columns[1:]:  # skip first column (labels)
    df[col] = pd.to_numeric(df[col], errors='coerce')

import matplotlib.pyplot as plt
import itertools

labels = df.iloc[:, 0]  # first column as labels
plt.figure(figsize=(20, 15))
n = len(plot_cols)
plot_idx = 1

for i, j in itertools.product(range(n), repeat=2):
    plt.subplot(n, n, plot_idx)
    
    x = df[plot_cols[j]]
    y = df[plot_cols[i]]
    
    # Mask to ignore NaNs
    mask = ~x.isna() & ~y.isna()
    plt.scatter(x[mask], y[mask])
    
    # Add labels to each point
    for k, txt in enumerate(labels[mask]):
        plt.text(x[mask].iloc[k], y[mask].iloc[k], str(txt), fontsize=8)
    # ---- Flip axes if the column name contains "mag" ----
    if "mag" in plot_cols[j].lower():
        plt.gca().invert_xaxis()
    
    # Flip Y-axis
    if "mag" in plot_cols[i].lower():
        plt.gca().invert_yaxis()
    if i == n-1:
        plt.xlabel(plot_cols[j])
    if j == 0:
        plt.ylabel(plot_cols[i])
    
    plot_idx += 1

plt.tight_layout()
plt.savefig('/home/kmc249/test_data/xrb_archive/internal_plots/outburstplots/big_outburst_plot.png', dpi=250)
#plt.show()
