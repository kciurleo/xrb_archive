#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:21:41 2026

@author: kmc249
"""
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates
from astropy.io import fits
import datetime as dt
from scipy.optimize import curve_fit
import numpy as np
from astropy.timeseries import LombScargle

#read in all the dfs

data = {}

data['V SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_V_corrected_lc_4_27.csv')
}

data['R SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv')
}

data['I SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv')
}

data['J SMARTS'] = {
    'df': pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_J_corrected_lc.csv')
}


for key in data:
    df = data[key]['df']
    if key=='J SMARTS':
        nameband='J'
    else:
        nameband='R'
    if key=='V SMARTS':
        df['mag_shifted']=df[f'{nameband}mag_shifted']
    else:
        df['mag_shifted']=df[f'{nameband}mag_corr']
    df['mag_corr_err']=df[f'e_{nameband}mag_corr']
    df['mag_err']=df[f'e_{nameband}mag']
    df['mag']=df[f'{nameband}mag']
    
    df['flag']=np.nan
    df['error_flag']=np.nan
    df['nice time']=pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())

#assume we've gotten rid of the bad guys already.

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
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

def read_lco(banzai_path, orac_path):
    banzai = read_corrected_txt(banzai_path)
    orac   = read_corrected_txt(orac_path)
    return pd.concat([banzai, orac], ignore_index=True)


for band in ['V','gp','R','rp','ip']:
    data[f'{band} LCO']={'df':pd.read_csv(f'/home/kmc249/Downloads/{band}_LCO_shifted.csv')}

data['V LCO']['df']['mag_shifted']=data['V LCO']['df']['mag_corr']
data['V LCO']['df']['alt_mag_shifted']=data['V LCO']['df']['alt_mag_corr']

#only get stuff in quiescence

#Outburst list
full = pd.read_csv("/home/kmc249/Downloads/full_outbursts.csv")
mini = pd.read_csv("/home/kmc249/Downloads/mini_outbursts.csv")

#Mask out quiescence
intervals = list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))

def get_quiescent(df, intervals):
    mask = np.ones(len(df), dtype=bool)
    
    for start, end in intervals:
        mask &= ~((df["MJD"] >= start) & (df["MJD"] <= end))
    
    return df[mask].copy()

for key in data:
    df = data[key]['df']
    data[key]['quiescence'] = get_quiescent(df, intervals)
    data[key]['global med q mag'] = data[key]['quiescence']['mag_shifted'].median()
    data[key]['global mean q mag'] = data[key]['quiescence']['mag_shifted'].mean()
    print(key, data[key]['global mean q mag'])


#%%
for key in data:
    print('running ', key)
    #if key!='J SMARTS':
    #    continue
    table=data[key]['quiescence']
    maglabel=key
    magstring='mag_shifted'
    #upper=15.6
    #lower=19
    #table=table.loc[(table[magstring]>upper) & (table[magstring]<lower)]
    
    
    #periodogramming things
    baseline=table['nice time'].max()-table['nice time'].min()
    base_days=baseline.total_seconds() / 3600 /24
    print(base_days)
    
    #folded??
    P = 0.789498  # period in days
    times = Time(table['nice time']).mjd
    t0 = times.min()
    phase = ((times - t0) / P) % 1
    table['their phase']=phase
    
    ##periodograms
    min_frequency = 24/19.5
    max_frequency = 24/18.5
    
    deltaf=P/base_days/4
    print('DELTA F:', deltaf)
    print(np.abs(min_frequency-max_frequency)/10000)
    
    frequency = np.arange(min_frequency, max_frequency, deltaf)
    
    fall, pall = LombScargle(times, table[magstring]-np.nanmean(table[magstring])).autopower(maximum_frequency=2)
    power = LombScargle(times, table[magstring]-np.nanmean(table[magstring])).power(frequency)
    
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
    plt.savefig(f'/home/kmc249/folded_curves/{key}_periodogram.png')
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
    '''
    plt.figure(figsize=(8,4))
    plt.plot(period_hours_sortedall, power_sortedall)
    plt.xlabel('Period (hours)')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodogram (all freq, capped)')
    plt.axvline(x=P*24,alpha=0.5, color='red')
    plt.xlim(12, 45)
    plt.show(block=False)
    '''
    
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
    binned = table.groupby('our phase bin')[magstring].agg(['mean','std']).reset_index()
    binned_them = table.groupby('their phase bin')[magstring].agg(['mean','std']).reset_index()
    
    yerr_up_us = []
    yerr_down_us = []
    
    for center in binned['our phase bin']:
        # Get all points in this phase bin
        points = table.loc[table['our phase bin'] == center, magstring].values
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
        points = table.loc[table['our phase bin'] == center, magstring].values
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
    plt.scatter(phase2, table[magstring], s=15, color='gray', label='Data')
    plt.scatter(phase2 + 1, table[magstring], s=15, color='gray', alpha=0.5)
    plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=yerr_asym_us,
                 fmt='o', color='red', label='Binned Avg')
    plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=yerr_asym_us,
                 fmt='o', color='red', alpha=0.5)
    plt.xlabel('Orbital Phase')
    plt.ylabel(maglabel)
    plt.gca().invert_yaxis()
    plt.title(f'{magstring} Our Period: {best_period_hours} hrs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/home/kmc249/folded_curves/{key}_our_period.png')
    
    plt.show(block=False)
    
    
    # Plot
    plt.figure(figsize=(8,4))
    plt.scatter(phase, table[magstring], s=15, color='gray', label='Data')
    plt.scatter(phase + 1, table[magstring], s=15, color='gray', alpha=0.5)
    
    plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=yerr_asym_them,
                 fmt='o', color='red', label='Binned Avg')
    plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=yerr_asym_them,
                 fmt='o', color='red', alpha=0.5)
    
    plt.xlabel('Orbital Phase')
    plt.ylabel(maglabel)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'{magstring} Their Period: {P*24} hrs')
    plt.tight_layout()
    plt.savefig(f'/home/kmc249/folded_curves/{key}_their_period.png')
    plt.show()


    #crappy 2-sin cruve fit
    
    '''
    # Our binned data
    xdata = binned['our phase bin'].astype(float).values
    ydata = binned['mean'].values
    yerr = binned['std'].values
    
    # Define a two-sine model
    def ellipsoidal_model(phase, A1, A2, delta, m0):
        return A1*np.sin(2*np.pi*phase) + A2*np.sin(4*np.pi*phase + delta) + m0
    
    # Initial guesses
    A1_guess = 0.05  # small amplitude at orbital frequency
    A2_guess = 0.1   # ellipsoidal amplitude
    delta_guess = 0
    m0_guess = np.mean(ydata)
    
    p0 = [A1_guess, A2_guess, delta_guess, m0_guess]
    
    # Fit
    popt, pcov = curve_fit(ellipsoidal_model, xdata, ydata, sigma=yerr, p0=p0)
    
    # Extract parameters
    A1_fit, A2_fit, delta_fit, m0_fit = popt
    print(f"A1 = {A1_fit:.3f}, A2 = {A2_fit:.3f}, delta = {delta_fit:.3f}, mean mag = {m0_fit:.3f}")
    
    # Plot
    phase_fit = np.linspace(0, 1, 500)
    mag_fit = ellipsoidal_model(phase_fit, *popt)
    
    plt.figure(figsize=(8,4))
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='o', color='red', label='Binned data')
    plt.plot(phase_fit, mag_fit, color='blue', label='Ellipsoidal fit')
    plt.xlabel('Orbital Phase')
    plt.ylabel(maglabel)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title('us')
    plt.tight_layout()
    plt.savefig(f'/home/kmc249/folded_curves/{key}_our_2sin_fit.png')
    plt.show(block=False)
    
    
    # their binned data
    xdata = binned_them['their phase bin'].astype(float).values
    ydata = binned_them['mean'].values
    yerr = binned_them['std'].values
    
    # Fit
    popt, pcov = curve_fit(ellipsoidal_model, xdata, ydata, sigma=yerr, p0=p0)
    
    # Extract parameters
    A1_fit, A2_fit, delta_fit, m0_fit = popt
    print(f"A1 = {A1_fit:.3f}, A2 = {A2_fit:.3f}, delta = {delta_fit:.3f}, mean mag = {m0_fit:.3f}")
    
    # Plot
    phase_fit = np.linspace(0, 1, 500)
    mag_fit = ellipsoidal_model(phase_fit, *popt)
    
    plt.figure(figsize=(8,4))
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='o', color='red', label='Binned data')
    plt.plot(phase_fit, mag_fit, color='blue', label='Ellipsoidal fit')
    plt.xlabel('Orbital Phase')
    plt.ylabel(maglabel)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title('them')
    plt.tight_layout()
    plt.savefig(f'/home/kmc249/folded_curves/{key}_their_2sin_fit.png')
    
    plt.show()

    '''
