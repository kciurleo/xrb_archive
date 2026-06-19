#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:48:02 2026

@author: kmc249
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.dates as mdates
from astropy.timeseries import LombScargle
from astropy.time import Time

#just replot the light curve for a sec

colors = {
    'B': 'blue',
    'V': 'green',
    'R': 'red',
    'I': 'chocolate'
}

data={'B':{},'V':{},'I':{}}

plt.figure(figsize=(14, 4))
for band in ['B','V','I']:
    file=f'/neta/xrb/GROJ1655-40/product/first_pass_lightcurves/GROJ1655-40_1.3m_{band}_first_pass_lc.csv'
    
    df=pd.read_csv(file, low_memory=False)
    
    
    df['nice time'] = pd.to_datetime(
        df['time'],
        format='mixed',
        utc=True,
        errors='coerce'
    ).dt.tz_localize(None)
    
    # force real datetime64 dtype
    df['nice time'] = pd.Series(
        df['nice time'].values,
        dtype='datetime64[ns]'
    )
    
    #exclude pre-2006
    df=df.loc[df['nice time']>pd.to_datetime('2006-01-01')]
    
    valid = (
        df['target mag'].notna() &
        df['error'].notna() &
        df['nice time'].notna()
    )

    if valid.sum() == 0:
        continue

    plt.errorbar(
        df.loc[valid, 'nice time'].to_numpy(),
        df.loc[valid, 'target mag'].to_numpy(),
        yerr=df.loc[valid, 'error'].to_numpy(),
        fmt='o',
        ms=3,
        lw=0.5,
        color=colors.get(band, 'black'),
        alpha=0.7,
        label=f'1.3m {band}'
    )
    data[band]['df']=df

plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.ylabel("Magnitude")

plt.legend(ncol=4, fontsize=8)
plt.tight_layout()

plt.show()


#%%
for key in data.keys():
    table=data[key]['df']
    #periodogramming things
    baseline=table['nice time'].max()-table['nice time'].min()
    base_days=baseline.total_seconds() / 3600 /24
    print(base_days)
    
    #folded??
    P = 2.621928  # period in days
    times = Time(table['nice time']).mjd
    t0 = times.min()
    phase = ((times - t0) / P) % 1
    table['their phase']=phase
    
    ##periodograms
    min_frequency = 1/2.622
    max_frequency = 1/2.6218
    
    deltaf=P/base_days/4
    print('DELTA F:', deltaf)
    print(np.abs(min_frequency-max_frequency)/10000)
    
    frequency = np.linspace(min_frequency, max_frequency, 1000)#np.arange(min_frequency, max_frequency, deltaf)
    
    fall, pall = LombScargle(times, table['target mag']-np.nanmean(table['target mag'])).autopower(maximum_frequency=2)
    power = LombScargle(times, table['target mag']-np.nanmean(table['target mag'])).power(frequency)
    
    # Convert frequency to period in hours
    period_hours = 24 / frequency
    sorted_idx = np.argsort(period_hours)
    period_hours_sorted = period_hours[sorted_idx]
    power_sorted = power[sorted_idx]
    
    # Plot periodogram in period units
    plt.figure(figsize=(8,4))
    plt.plot(period_hours_sorted, power_sorted)
    plt.xlabel('Period (hours)')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram')
    plt.axvline(x=P*24,alpha=0.5, color='red')
    plt.show(block=False)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(frequency, power)
    plt.show(block=False)
    
    # samme thing for fall pall
    pall_hours = 24 / fall
    sorted_idxall = np.argsort(pall_hours)
    period_hours_sortedall = pall_hours[sorted_idxall]
    power_sortedall = pall[sorted_idxall]
    
    # Plot periodogram in period units
    
    fig, ax = plt.subplots()
    ax.plot(fall, pall)
    plt.show(block=False)
    
    best_frequency = frequency[np.argmax(power)]
    P2 = 1 / best_frequency
    best_period_hours = P2 * 24
    print(best_period_hours)
    
    phase2 = ((times - t0) / P2) % 1
    table['our phase']=phase2
    
    # Number of bins
    nbins = 16
    bins = np.linspace(0, 1, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Assign each phase to a bin
    table['our phase bin'] = pd.cut(table['our phase'], bins=bins, include_lowest=True, labels=bin_centers)
    table['their phase bin'] = pd.cut(table['their phase'], bins=bins, include_lowest=True, labels=bin_centers)
    
    # Compute mean and std per bin
    binned = table.groupby('our phase bin')['target mag'].agg(['mean','std']).reset_index()
    binned_them = table.groupby('their phase bin')['target mag'].agg(['mean','std']).reset_index()
    
    yerr_up_us = []
    yerr_down_us = []
    
    for center in binned['our phase bin']:
        # Get all points in this phase bin
        points = table.loc[table['our phase bin'] == center, 'target mag'].values
        if len(points) == 0:
            yerr_up_us.append(0)
            yerr_down_us.append(0)
            continue
        
        mean_bin = np.mean(points)
        
        # Points above/below mean
        above = points[points > mean_bin]
        below = points[points < mean_bin]
        
        # 68% confidence ~ 1 sigma
        sigma_up = np.percentile(above, 68.3) - mean_bin if len(above) > 0 else 0
        sigma_down = mean_bin - np.percentile(below, 31.7) if len(below) > 0 else 0
        
        yerr_up_us.append(sigma_up)
        yerr_down_us.append(sigma_down)
    
    # Make 2×N array for asymmetric error bars
    yerr_asym_us = np.array([yerr_down_us, yerr_up_us])
    
    yerr_up_them = []
    yerr_down_them = []
    
    for center in binned['our phase bin']:
        # Get all points in this phase bin
        points = table.loc[table['our phase bin'] == center, 'target mag'].values
        if len(points) == 0:
            yerr_up_them.append(0)
            yerr_down_them.append(0)
            continue
        
        mean_bin = np.mean(points)
        
        # Points above/below mean
        above = points[points > mean_bin]
        below = points[points < mean_bin]
        
        # 68% confidence ~ 1 sigma
        sigma_up = np.percentile(above, 68.3) - mean_bin if len(above) > 0 else 0
        sigma_down = mean_bin - np.percentile(below, 31.7) if len(below) > 0 else 0
        
        yerr_up_them.append(sigma_up)
        yerr_down_them.append(sigma_down)
    
    # Make 2×N array for asymmetric error bars
    yerr_asym_them = np.array([yerr_down_them, yerr_up_them])
    
    # Plot with asymmetric error bars
    plt.figure(figsize=(8,4))
    plt.scatter(phase2, table['target mag'], s=15, color='gray', label='Data')
    plt.scatter(phase2 + 1, table['target mag'], s=15, color='gray', alpha=0.5)
    plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=yerr_asym_us,
                 fmt='o', color='red', label='Binned Avg')
    plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=yerr_asym_us,
                 fmt='o', color='red', alpha=0.5)
    plt.xlabel('Orbital Phase')
    plt.ylabel(key)
    plt.gca().invert_yaxis()
    plt.title(f'{'target mag'} Our Period: {best_period_hours/24} days')
    plt.legend()
    plt.tight_layout()
    
    plt.show(block=False)
    
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.scatter(phase, table['target mag'], s=15, color='gray', label='Data')
    plt.scatter(phase + 1, table['target mag'], s=15, color='gray', alpha=0.5)
    
    plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=yerr_asym_them,
                 fmt='o', color='red', label='Binned Avg')
    plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=yerr_asym_them,
                 fmt='o', color='red', alpha=0.5)
    
    plt.xlabel('Orbital Phase')
    plt.ylabel(key)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'{'target mag'} Their Period: {P} days')
    plt.tight_layout()
    plt.show()