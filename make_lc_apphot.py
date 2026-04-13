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
#table1=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_smaller_ap.csv', low_memory=False)
df1 = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_13m_apr_07.csv', low_memory=False)
df2 = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_apr_07.csv', low_memory=False)
wideR = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_wideR_apr_07.csv', low_memory=False)
#table1 = pd.concat([df1, df2], ignore_index=True)
table1=df2

table2=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv', low_memory=False)
table3=pd.read_csv('/home/kmc249/Downloads/psf_fluxes_2011.csv')
table2 = pd.concat([table2, table3], ignore_index=True)
#table2=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_smaller_ap.csv', low_memory=False)
#uncomment for aperture comparison
'''
table1=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_08.csv', low_memory=False)
table2=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_smaller_ap.csv', low_memory=False)
table2['nice time'] = pd.to_datetime(table2['time'])
table2= table2[table2['nice time'].dt.year == 2008]
'''
table1['nice time'] = pd.to_datetime(table1['time'])
table2['nice time'] = pd.to_datetime(table2['time'])
wideR['nice time'] = pd.to_datetime(wideR['time'])
standards=pd.read_csv('/home/kmc249/Downloads/BEST_ens_stds_info.csv')
#table=table2

hiresphot=pd.read_csv("/home/kmc249/ens_phot_hires.csv")
#hiresphot=pd.read_csv("/home/kmc249/best_r_ensemble.csv")
#^^^ should be this one, no?
def f(x, a, c):
    return a*np.log10(x)+c
for table in [table2, table1, wideR]:
    fig, axes = plt.subplots(figsize=(8, 8))
    xdata3=[]
    ydata3=[]
    badlist=[]
    for e in table.columns:
        if  e not in ['nice time','time', 'filename', '1320', '413']:
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
    
    
    
    exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','a','b','c','d','1418','1069','1105', '1320', 'aql mag','ave mag', '413']
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
        thing=std
    
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
for table in [table2, table1]:
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


#%%

#getting aql errors
quiescence=pd.read_csv('/home/kmc249/Downloads/quiescence_mjd_ranges_v5.csv')
quiescence['start_dt'] = pd.to_datetime(
    Time(quiescence['q_start_mjd'].values, format='mjd').to_datetime()
)

quiescence['end_dt'] = pd.to_datetime(
    Time(quiescence['q_end_mjd'].values, format='mjd').to_datetime()
)

# build one combined mask
mask = np.zeros(len(table1), dtype=bool)

for start, end in zip(quiescence['start_dt'], quiescence['end_dt']):
    mask |= (table1['nice time'] >= start) & (table1['nice time'] <= end)

# compute overall mean
mean_quiescent = table1.loc[mask, 'aql mag'].mean()

print(mean_quiescent)
x_vals = []
y_vals = []
cols_used = []
plt.figure(figsize=(8,8))

for col in table1.columns:
    if col not in ['filename', 'time',  'neighbor', '413', '1320', 'nice time', 'aql mag', 'ave mag']:
        if col == 'aql':
            # use quiescence mask to select only rows in quiescence
            flux_safe = table1.loc[mask, col].values
            print('aql flux safe')
            print(flux_safe)
            flux_safe = flux_safe[flux_safe > 0]
        else:
            # for other columns, use positive fluxes
            flux_safe = table1[col].values
            flux_safe = flux_safe[flux_safe > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes
            
        mag_safe=-2.5 * np.log10(flux_safe)
        x = np.std(mag_safe)#/np.sqrt(len(mag_safe))
        y = -2.5 * np.log10(np.nanmean(flux_safe))
        plt.scatter(y, x)
        x_vals.append(x)
        y_vals.append(y)
        cols_used.append(col)
        if col == 'aql': 
            print(x, y)
            a=1

        else:
            a=0.2
        plt.annotate(
            str(col),
            (y, x),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color='red',
            alpha=a

        )


plt.ylabel('std of mag')
plt.xlabel('instrumental mag')
#axes.set_yscale('log')

plt.gca().invert_xaxis()

plt.show()
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)
cols_used = np.array(cols_used)

# find aql index
aql_idx = np.where(cols_used == 'aql')[0][0]
aql_y = y_vals[aql_idx]

# compute distance in y to aql
y_dist = np.abs(y_vals - aql_y)

# get 5 closest stars excluding aql itself
closest_idx = np.argsort(y_dist)
closest_idx = [i for i in closest_idx if cols_used[i] != 'aql'][:4]

# average their x values
constanterror = np.mean(x_vals[closest_idx])
#print(constanterror)
print("Stars closest to aql:", cols_used[closest_idx])
print("Average x (std) of 4 closest stars:", constanterror)


#real constant error from the other way of getting std
# Convert cols_used to integers
cols_used_int = cols_used[closest_idx].astype(int)

# Lookup mean_delta for each star_id in cols_used
mean_delta_for_cols_used = [mean_delta[star_id.index(sid)] for sid in cols_used_int]
constanterror=np.mean(mean_delta_for_cols_used)
print(constanterror)
#%%
#getting errors
r_mag = np.array(r_mag)
mean_delta = np.array(mean_delta)

# fit polynomial
coeffs = np.polyfit(r_mag, mean_delta, 4)
poly = np.poly1d(coeffs)

xfit = np.linspace(min(r_mag), max(r_mag), 500)
yfit = poly(xfit)

# find minimum of polynomial
min_idx = np.argmin(yfit)
xmin = xfit[min_idx]
ymin = yfit[min_idx]

plt.figure(figsize=(5,4))

# scatter
plt.scatter(r_mag, mean_delta, alpha=0.7)

# polynomial fit
plt.plot(xfit, yfit, 'r-', label='4th-order fit')

# horizontal floor (extend to bright end)
plt.axhline(ymin, color='k', ls='--',
            label=f'precision floor = {ymin:.4f}')

plt.xlabel('Pan-STARRS r (mag)')
plt.ylabel(r'$\sigma(\Delta m)$')
plt.gca().invert_xaxis()

plt.title('Photometric precision vs magnitude')
plt.legend()
plt.show()

#%%
#once you run above can just do this
lco1=pd.read_csv('/home/kmc249/Downloads/R_usable_banzai_lt25.txt', sep=r'\s+', header=None, comment='#', names=['MJD', 'R_mag', 'uncertainty', 'upperlimitflag'])
lco2=pd.read_csv('/home/kmc249/Downloads/R_usable_orac_lt25.txt', sep=r'\s+', header=None, comment='#', names=['MJD', 'R_mag', 'uncertainty', 'upperlimitflag'])
lco = pd.concat([lco1, lco2], ignore_index=True)

t = Time(lco['MJD'].values, format='mjd')
lco['nice time'] = t.to_datetime()


plt.figure(figsize=(12,6))

plt.scatter(table1['nice time'], table1['aql mag']-0.4, marker='.',color='red', s=15, label='AP')
plt.scatter(table2['nice time'], table2['aql mag']-0.4, marker='.',color='blue', s=15, label='PSF')
plt.errorbar(lco['nice time'], lco['R_mag'], yerr=lco['uncertainty'], fmt='.',color='black', markersize=5, label='LCO')

plt.ylabel('Pan-STARRS r')
plt.legend()
plt.ylim(20,15.3)
#plt.gca().invert_yaxis()
#plt.ylim(19.5, 17)
#plt.xlim(pd.Timestamp('2013-01-01'),pd.Timestamp('2015-01-01'))

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

#%%
#split
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharey=True)

# sort by time just in case
table1 = table1.sort_values('nice time')

# split into 3 equal chunks
time_chunks = np.array_split(table1['nice time'], 4)

for ax, chunk in zip(axes, time_chunks):
    tmin = chunk.min()
    tmax = chunk.max()
    
    # --- plot everything ---
    ax.scatter(table1['nice time'], table1['aql mag']-0.4,
               marker='.', color='red', s=15, label='AP')
    
    ax.scatter(table2['nice time'], table2['aql mag']-0.4,
               marker='.', color='blue', s=15, label='PSF')
    
    ax.errorbar(lco['nice time'], lco['R_mag'],
                yerr=lco['uncertainty'],
                fmt='.', color='black', markersize=5, label='LCO')
    
    # --- limit THIS panel ---
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(20, 15)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    

# only put legend once
axes[0].legend()

plt.tight_layout()
plt.show()

#%%
merged = table.merge(table2, on='nice time', suffixes=('_ap', '_psf'))
merged['resid'] = merged['aql mag_ap'] - merged['aql mag_psf']

fig, axes = plt.subplots(
    8, 1, figsize=(28, 24),
    gridspec_kw={'height_ratios': [3,1, 3,1, 3,1, 3,1]}
)

# sort everything
table1 = table1.sort_values('nice time')
table2 = table2.sort_values('nice time')
lco    = lco.sort_values('nice time')
merged = merged.sort_values('nice time')

# split time

# get start and end times
t_start = table1['nice time'].min()
t_end   = table1['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=5)  # 4 chunks = 5 edges

# now split table1 into 4 chunks by time range
time_chunks = [table1[(table1['nice time'] >= edges[i]) & (table1['nice time'] < edges[i+1])]
               for i in range(4)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = table1[(table1['nice time'] >= edges[-2]) & (table1['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[2*i]
    ax_res  = axes[2*i + 1]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (table1['nice time'] >= tmin) & (table1['nice time'] <= tmax)
    mask2 = (table2['nice time'] >= tmin) & (table2['nice time'] <= tmax)
    maskl = (lco['nice time'] >= tmin) & (lco['nice time'] <= tmax)
    maskm = (merged['nice time'] >= tmin) & (merged['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(table1.loc[mask1, 'nice time'],
                    table1.loc[mask1,  'aql mag']-0.4, yerr=constanterror,
                    color='red', markersize=3, fmt='none', label='AP')
    
    ax_main.errorbar(table2.loc[mask2, 'nice time'],
                    table2.loc[mask2, 'aql mag']-0.4, yerr=0,
                    color='blue', markersize=3, fmt='.', label='PSF')
    
    ax_main.errorbar(lco.loc[maskl, 'nice time'],
                     lco.loc[maskl, 'R_mag'],
                     yerr=lco.loc[maskl, 'uncertainty'],
                     fmt='.', color='black', markersize=3, label='LCO')
    
    ax_main.set_xlim(tmin, tmax)
    ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    
    # --- RESIDUALS (AP - PSF) ---
    ax_res.scatter(merged.loc[maskm, 'nice time'],
                   merged.loc[maskm, 'resid'],
                   color='gray', s=12, label='AP - PSF')
    
    ax_res.axhline(0, color='k', ls='--', lw=1)
    
    ax_res.set_xlim(tmin, tmax)
    ax_res.set_ylabel('Δmag')
    ax_res.set_ylim(1,-1)
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[2].legend(loc='upper right')
axes[3].legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/kmc249/Downloads/lccomp.png', dpi=300)
plt.show()



#%%
'''
merged = table.merge(table2, on='nice time', suffixes=('_ap', '_psf'))
merged['resid'] = merged['aql mag_ap'] - merged['aql mag_psf']

fig, axes = plt.subplots(
    2, 1, figsize=(16,8),
    gridspec_kw={'height_ratios': [3,1]}
)

# sort everything
table1 = table1.sort_values('nice time')
table2 = table2.sort_values('nice time')
lco    = lco.sort_values('nice time')
merged = merged.sort_values('nice time')

# split time

# get start and end times
t_start = table1['nice time'].min()
t_end   = table1['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=2)  # 4 chunks = 5 edges

# now split table1 into 4 chunks by time range
time_chunks = [table1[(table1['nice time'] >= edges[i]) & (table1['nice time'] < edges[i+1])]
               for i in range(1)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = table1[(table1['nice time'] >= edges[-2]) & (table1['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[2*i]
    ax_res  = axes[2*i + 1]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (table1['nice time'] >= tmin) & (table1['nice time'] <= tmax)
    mask2 = (table2['nice time'] >= tmin) & (table2['nice time'] <= tmax)
    maskl = (lco['nice time'] >= tmin) & (lco['nice time'] <= tmax)
    maskm = (merged['nice time'] >= tmin) & (merged['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(table1.loc[mask1, 'nice time'],
                    table1.loc[mask1,  'aql mag']-0.4, yerr=constanterror,
                    color='red', markersize=3, fmt='none', label='8pix')
    
    ax_main.errorbar(table2.loc[mask2, 'nice time'],
                    table2.loc[mask2, 'aql mag']-0.4, yerr=0.0175,
                    color='g', markersize=3, fmt='.', label='5pix')
    
    ax_main.errorbar(lco.loc[maskl, 'nice time'],
                     lco.loc[maskl, 'R_mag'],
                     yerr=lco.loc[maskl, 'uncertainty'],
                     fmt='.', color='black', markersize=3, label='LCO')
    
    ax_main.set_xlim(tmin, tmax)
    ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    
    # --- RESIDUALS (AP - PSF) ---
    ax_res.scatter(merged.loc[maskm, 'nice time'],
                   merged.loc[maskm, 'resid'],
                   color='gray', s=12, label='8pix - 5pix')
    
    ax_res.axhline(0, color='k', ls='--', lw=1)
    
    ax_res.set_xlim(tmin, tmax)
    ax_res.set_ylabel('Δmag')
    ax_res.set_ylim(1,-1)
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/kmc249/Downloads/apcomp.png', dpi=300)
plt.show()
'''