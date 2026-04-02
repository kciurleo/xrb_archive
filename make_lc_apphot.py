#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:40:56 2026

@author: kmc249
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates

#table=pd.read_csv('/Users/katieciurleo/Downloads/yalestuff/psf_fluxes.csv', low_memory=False)
table1=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_08.csv', low_memory=False)
table1['nice time'] = pd.to_datetime(table1['time'])
table2=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv', low_memory=False)
table2['nice time'] = pd.to_datetime(table2['time'])
table2= table2[table2['nice time'].dt.year == 2008]
standards=pd.read_csv('/home/kmc249/Downloads/BEST_ens_stds_info.csv')
#table=table2

hiresphot=pd.read_csv("/home/kmc249/ens_phot_hires.csv")

def f(x, a, c):
    return a*np.log10(x)+c
for table in [table1, table2]:
    fig, axes = plt.subplots(figsize=(8, 8))
    xdata3=[]
    ydata3=[]
    badlist=[]
    for e in table.columns:
        if  e not in ['nice time','time', 'filename', '1320']:
            try:
                row=standards.loc[standards['num int']==int(e)]
            except:
                continue
            if len(row)<1:
                continue
            y=row['r'].iloc[0]
            flux = hiresphot.loc[hiresphot['id'].eq(int(e)), 'flux_fit'].iloc[0]
    
            x = -2.5 * np.log10(np.nanmean(flux))
            xdata3.append(x)
            ydata3.append(y)
            axes.scatter(x, y)
            axes.annotate(
                str(e),
                (x,y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color='red',
                alpha=0.5
            )
    
    axes.set_xlabel('psf mag of standard stars')
    axes.set_ylabel('panstarrs mag')
    slope, intercept, r, p, se =linregress(xdata3, ydata3)
    x3_arr=np.linspace(np.min(xdata3), np.max(xdata3))
    axes.plot(x3_arr, slope*x3_arr+intercept, 'g--', label=f'y={np.round(slope,2)}x+{np.round(intercept, 2)}')
    axes.invert_yaxis()
    axes.invert_xaxis()
    plt.legend()
    #plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_ensemble_to_stds_psf.png', dpi=250)
    plt.show()
    print('badlist:',badlist)
    
    
    fig, axes = plt.subplots(figsize=(8, 8))
    for e in table.columns:
        if  e not in ['nice time','time', 'filename', '1320','a','b','c','d',]:
            try:
                row=standards.loc[standards['num int']==int(e)]
            except:
                continue
            if len(row)<1:
                continue
            x=row['g'].iloc[0]-row['r'].iloc[0]
            flux = hiresphot.loc[hiresphot['id'].eq(int(e)), 'flux_fit'].iloc[0]
    
            y=row['r'].iloc[0]-(slope*(-2.5 * np.log10(np.nanmean(flux)))+intercept)
            axes.scatter(x, y)
            axes.annotate(
                str(e),
                (y,x),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color='red',
                alpha=0.5
            )
    
    axes.set_ylabel('resid (panstarrs r - linear model)')
    axes.set_xlabel('panstarrs g-r (mag)')
    plt.legend()
    #plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_resids_stds_color.png', dpi=250)
    plt.show()
    
    
    
    exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','a','b','c','d','1418','1069','1105', '1320', 'aql mag','ave mag']
    ensemble_cols = [
        c for c in table.columns
        if c not in exclude_cols and c.isdigit()
    ]
    
    ensemble_ids = [int(c) for c in ensemble_cols]
    to_sum = [table[name] for name in table.columns if name not in exclude_cols]
    print(to_sum)
    avg=np.nanmean(to_sum)
    ensemble_r = standards.loc[
        standards['num int'].isin(ensemble_ids), 'r'
    ]
    #panstarrs r mag of ensemble
    ensemble_r_mean=ensemble_r.mean()
    
    table['aql mag'] = np.nan       # pre-create column
    table['ave mag'] = np.nan      # if needed for table2
    plt.figure(figsize=(12,3))
    print('to sum:', [name for name in table.columns if name not in exclude_cols])
    for id, row in table.iterrows():
        # average flux of comparison stars only
        to_sum = [row[name] for name in table.columns if name not in exclude_cols]
        
        avg = np.nanmean(to_sum)
        avgmag=-2.5*np.log10(avg)
        #h1=plt.scatter(row['nice time'], -2.5*np.log10(avg), s=15, color='gray',label='mean ens mag')
        #delta between panstarrs r mag and ensemble average magnitude
        delta=ensemble_r_mean-avgmag
        print(delta)
    
        for name in table.columns:
            if name  in ['aql']:#['nice time','time','filename']:
                flux = row[name]
                # skip non-positive fluxes
                if flux <= 0 or np.isnan(flux):
                    continue
                mags = slope*(-2.5 * np.log10(flux) + delta)
                table.at[id, 'aql mag']=mags
                ave=slope*(-2.5*np.log10(avg))+intercept
                table.at[id, 'ave mag']=ave
                h2=plt.scatter(row['nice time'], mags, marker='.', color='k',label=f'{name}', s=15)
                #h3=plt.scatter(row['nice time'], ave, marker='.', color='grey', s=15)
    #handles = [h2, h3]
    #labels = ['aql', 'ens (offset)']
    #plt.legend(handles=handles, labels=labels)
    plt.ylabel('Pan-STARRS r')
    #plt.ylim(20,16.5)
    plt.gca().invert_yaxis()
    
    
    # --- Primary x-axis: date ---
    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # --- Secondary x-axis (MJD), properly aligned ---
    ax2 = ax1.twiny()  # create a second x-axis that shares the same y
    ax2.set_xlim(ax1.get_xlim())  # align limits
    
    # Convert tick locations to MJD
    tick_locs = ax1.get_xticks()
    tick_dates = mdates.num2date(tick_locs)
    tick_mjds = Time(tick_dates).mjd
    
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])
    
    # Shift second axis downward for clarity (optional)
    ax2.xaxis.set_ticks_position('bottom')
    ax1.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('bottom')
    plt.subplots_adjust(bottom=0.25)
    ax1.xaxis.set_ticks_position('top')
    plt.tight_layout()
    #plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_lc_psf_try1.png', dpi=250)
    plt.show()
    
    eids= [431, 244, 214, 522, 1199, 545, 948, 271, 1065, 1423, 1115, 679, 295, 1081, 1434, 1416, 318, 1140, 397, 1413, 269, 1476, 983, 659, 1146, 670, 376, 784, 1407, 1169, 1458, 1086, 566, 158, 729, 235, 1207, 783, 505, 482, 1337, 812, 1234, 996, 268, 1243, 290, 794, 1160, 1195, 1490, 213, 1053, 759, 1379, 1006, 1280, 704, 1187, 1069, 173, 1039, 744, 514, 153, 1192, 786, 1046, 751, 982, 713, 381, 1320, 1099, 1263, 1167, 1113, 160, 582, 461, 533, 134, 613, 1215, 825, 1460, 1433, 757, 307, 1451, 758, 493, 597, 1477, 1085, 403, 1038, 67, 479, 1341, 1344, 1305, 80, 863, 82, 66, 104, 895, 55, 890, 1340, 163, 1380, 178, 224, 1345, 116, 234, 139, 218, 187, 215, 155, 226, 120, 209, 1382, 309, 1058, 399, 1404, 374, 262, 1007, 1414, 1399, 258, 312, 1385, 1415, 300, 260, 1042, 345, 288, 377, 1388, 1072, 371, 331, 1021, 369, 1023, 404, 395, 1109, 485, 492, 467, 1435, 418, 451, 413, 1094, 503, 445, 1418, 417, 1105, 410, 499, 1104, 1419, 506, 621, 1470, 639, 1161, 1457, 525, 1143, 1440, 1467, 657, 668, 641, 1139, 1471, 534, 695, 1165, 1459, 1141, 558, 530, 1145, 622, 1191, 1132, 1443, 792, 1238, 756, 1502, 1241, 1225, 1504, 1261, 703, 1506, 707, 681, 1489, 1509, 1483, 761, 1208, 1242, 781, 1512, 1211, 1519, 813, 800, 820]
    
    #double plot
    plt.figure(figsize=(4,3))
    plt.hist(table['ave mag'], bins=45)
    plt.xlabel('r mag')
    plt.gca().invert_xaxis()
    plt.show()
    vals = table['ave mag'].values
    vals = vals[np.isfinite(vals)]  # drop NaNs
    
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)      # sample standard deviation
    sdom = std / np.sqrt(len(vals)) # standard deviation of the mean
    
    print(f"mean = {mean:.5f}")
    print(f"std  = {std:.5f}")
    print(f"sdom = {sdom:.5f}")
    
    
    #delta mags
    from collections import defaultdict
    
    delta_by_star = defaultdict(list)
    
    for idx, row in table.iterrows():
    
        # ensemble average magnitude for this night
        fluxes = np.array([row[c] for c in ensemble_cols], dtype=float)
        good = fluxes > 0
    
        if np.sum(good) < 3:
            continue
    
        ens_mag = -2.5 * np.log10(np.nanmean(fluxes[good]))
    
        # per-star deltas
        for c in ensemble_cols:
            flux = row[c]
            if flux <= 0 or np.isnan(flux):
                continue
    
            star_mag = -2.5 * np.log10(flux)
            delta = star_mag - ens_mag
    
            delta_by_star[int(c)].append(delta)
    
    plt.figure(figsize=(5,4))
    
    for sid, deltas in delta_by_star.items():
        if len(deltas) < 10:   # skip poorly sampled stars
            continue
    
        plt.hist(
            deltas,
            bins=40,
            histtype='step',
            alpha=0.3
        )
    
    plt.xlabel(r'$\Delta m$ (star − ensemble)')
    plt.title('Per-star delta ensemble')
    plt.show()
    
    for sid, deltas in delta_by_star.items():
        deltas = np.array(deltas)
    
        if len(deltas) < 10:
            continue
    
        std = np.std(deltas, ddof=1)
        sdom = std / np.sqrt(len(deltas))
    
        print(f"{sid:6d}  {std:8.4f}  {sdom:8.4f}")
    
    mean_delta = []
    r_mag = []
    star_id = []
    r_lookup = standards.set_index('num int')['r']
    for sid, deltas in delta_by_star.items():
        if len(deltas) < 10:
            continue
    
        if sid not in r_lookup:
            continue
        std = np.std(deltas, ddof=1)
        sdom = std / np.sqrt(len(deltas))
        thing=sdom
    
        mean_delta.append(thing)
        r_mag.append(r_lookup.loc[sid])
        star_id.append(sid)
    
    
    plt.figure(figsize=(4,3))
    plt.scatter(r_mag, mean_delta)
    #plt.axhline(0, color='k', ls='--', lw=1)
    
    plt.xlabel('Pan-STARRS r (mag)')
    plt.ylabel(r'sdom of (star − ensemble)')
    plt.gca().invert_xaxis()
    plt.title(r'sdom of $\Delta m$ per star')
    plt.show()
#%%
# Plot both light curves together
on2=False
plt.figure(figsize=(12,3))
for table in [table1, table2]:
    if on2:
        label='AP'
        c='red'
    else:
        label='PSF'
        c='blue'
    for id, row in table.iterrows():
        # average flux of comparison stars only
        to_sum = [row[name] for name in table.columns if name not in exclude_cols]
        
        avg = np.nanmean(to_sum)
        avgmag=-2.5*np.log10(avg)
        #h1=plt.scatter(row['nice time'], -2.5*np.log10(avg), s=15, color='gray',label='mean ens mag')
        #delta between panstarrs r mag and ensemble average magnitude
        delta=ensemble_r_mean-avgmag
        print(delta)
        on2=True
        for name in table.columns:
            if name  in ['aql']:#['nice time','time','filename']:
                flux = row[name]
                # skip non-positive fluxes
                if flux <= 0 or np.isnan(flux):
                    continue
                mags = slope*(-2.5 * np.log10(flux) + delta)
                table.at[id, 'aql mag']=mags
                ave=slope*(-2.5*np.log10(avg))+intercept
                table.at[id, 'ave mag']=ave
                h2=plt.scatter(row['nice time'], mags, marker='.',color=c, s=15)
                #h3=plt.scatter(row['nice time'], ave, marker='.', color='grey', s=15)
    #handles = [h2, h3]
#labels = ['aql', 'ens (offset)']
#plt.legend(handles=handles, labels=labels)
plt.ylabel('Pan-STARRS r')
plt.legend()
#plt.ylim(20,16.5)
#plt.gca().invert_yaxis()
plt.ylim(19.5, 17)

# --- Primary x-axis: date ---
ax1 = plt.gca()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# --- Secondary x-axis (MJD), properly aligned ---
ax2 = ax1.twiny()  # create a second x-axis that shares the same y
ax2.set_xlim(ax1.get_xlim())  # align limits

# Convert tick locations to MJD
tick_locs = ax1.get_xticks()
tick_dates = mdates.num2date(tick_locs)
tick_mjds = Time(tick_dates).mjd

ax2.set_xticks(tick_locs)
ax2.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])

# Shift second axis downward for clarity (optional)
ax2.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('bottom')
plt.subplots_adjust(bottom=0.25)
ax1.xaxis.set_ticks_position('top')
plt.tight_layout()
#plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_lc_psf_try1.png', dpi=250)
plt.show()