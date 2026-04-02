#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:58:20 2025

@author: kmc249
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates
from astropy.io import fits
import datetime as dt

from astropy.timeseries import LombScargle

table=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv')


#table=pd.read_csv('/home/kmc249/Downloads/psf_fluxes_with_extras.csv', low_memory=False)

table['nice time'] = pd.to_datetime(table['time'])

standards=pd.read_csv('/home/kmc249/Downloads/BEST_ens_stds_info.csv')


def f(x, a, c):
    return a*np.log10(x)+c

fig, axes = plt.subplots(figsize=(8, 8))
xdata, ydata=[],[]

for e in table.columns:
    if e not in ['nice time', 'time', 'filename', '1418','1069','1105','1320', 'aql mag']:
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes

        x = np.std(flux_safe)
        y = -2.5 * np.log10(np.nanmean(flux_safe))
        if e not in ['aql']:
            xdata.append(x)
            ydata.append(y)
        axes.scatter(y, x)
        if e == 'aql': 
            a=1
        elif e == 'neighbor':
            a=1
        else:
            a=0.2
        axes.annotate(
            str(e),
            (y, x),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color='red',
            alpha=a

        )


axes.set_ylabel('std of flux')
axes.set_xlabel('mag')

popt, pcov = curve_fit(f, np.array(xdata), np.array(ydata))
x_arr=np.linspace(np.min(xdata), np.max(xdata),150)
axes.plot( f(x_arr, *popt),x_arr, 'g--')
axes.invert_xaxis()
#plt.savefig('/home/kmc249/Downloads/aql_ensemble_variability.png', dpi=250)
plt.show(block=False)



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
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes

        x = -2.5 * np.log10(np.nanmean(flux_safe))
        if x>-10:
            badlist.append(int(e))
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
#plt.savefig('/home/kmc249/Downloads/aql_ensemble_to_stds_psf.png', dpi=250)
plt.show()


fig, axes = plt.subplots(figsize=(8, 8))
for e in table.columns:
    if  e not in ['nice time','time', 'filename', '1320', 'aql mag']:
        try:
            row=standards.loc[standards['num int']==int(e)]
        except:
            continue
        if len(row)<1:
            continue
        x=row['g'].iloc[0]-row['r'].iloc[0]
        flux = table[e].values
        # Only use positive fluxes
        flux_safe = flux[flux > 0]
        if len(flux_safe) == 0:
            continue  # skip columns with no valid fluxes

        #y = -2.5 * np.log10(np.nanmean(flux_safe))-row['r'].iloc[0]
        y=np.abs(row['r'].iloc[0]-(slope*(-2.5 * np.log10(np.nanmean(flux_safe)))+intercept))
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
#plt.savefig('/home/kmc249/Downloads/aql_resids_stds_color.png', dpi=250)
plt.show()

print('badlist: ',badlist)


plt.figure(figsize=(12,3))

exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','1418','1069','1105', '1320', 'aql mag']

to_sum = [table[name] for name in table.columns if name not in exclude_cols]
avg=np.nanmean(to_sum)

#getting ave ens magnitude overall
total_avg=-2.5*np.log10(avg)
offset=total_avg+intercept


for id, row in table.iterrows():
    # average flux of comparison stars only
    to_sum = [row[name] for name in table.columns if name not in exclude_cols]
    avg = np.nanmean(to_sum)
    #h1=plt.scatter(row['nice time'], -2.5*np.log10(avg), s=15, color='gray',label='mean ens mag')

    for name in table.columns:
        if name  in ['aql']:#['nice time','time','filename']:
            flux = row[name]
            # skip non-positive fluxes
            if flux <= 0 or np.isnan(flux):
                continue
            mags = -2.5 * np.log10(flux) +2.5*np.log10(avg)+offset
            if mags<19:
                table.at[id, 'aql mag']=mags
                plt.scatter(row['nice time'], mags, marker='.', color='k',label=f'{name}', s=15)

#plt.legend(handles=handles, labels=labels)
plt.ylabel('Pan-STARRS r')
plt.ylim(20,15)


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
plt.savefig('/home/kmc249/Downloads/aql_lc_psf_try1.png', dpi=250)
plt.show()



# Split date threshold
split_date = dt.datetime(2012, 1, 1)

# Figure setup
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
exclude_cols = ['nice time', 'time', 'filename', 'aql', 'neighbor', '1418', '1069', '1105', '1320']

# --- Compute overall average flux and offset ---
to_sum = [table[name] for name in table.columns if name not in exclude_cols]
avg = np.nanmean(to_sum)
total_avg = -2.5 * np.log10(avg)
offset = total_avg + intercept

# --- Loop over both subplots (top = before 2012, bottom = after) ---
for ax_idx, (ax, date_filter) in enumerate(zip(
    axes,
    [table['nice time'] < split_date, table['nice time'] >= split_date]
)):
    sub_table = table[date_filter]

    for id, row in sub_table.iterrows():
        # average flux of comparison stars only
        to_sum = [row[name] for name in table.columns if name not in exclude_cols]
        avg = np.nanmean(to_sum)

        for name in table.columns:
            if name in ['aql']:
                flux = row[name]
                if flux <= 0 or np.isnan(flux):
                    continue
                mags = -2.5 * np.log10(flux) + 2.5 * np.log10(avg) + offset
                if mags < 19:
                    ax.scatter(row['nice time'], mags, marker='.', color='k', s=15)

    # Formatting
    ax.set_ylabel('Pan-STARRS r')
    ax.set_ylim(20, 15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Secondary x-axis (MJD) ---
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_locs = ax.get_xticks()
    tick_dates = mdates.num2date(tick_locs)
    tick_mjds = Time(tick_dates).mjd

    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])

    # Position of MJD axis below main axis
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax.xaxis.set_ticks_position('top')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.1)
#plt.savefig('/home/kmc249/Downloads/aql_lc_psf_split.png', dpi=250)
plt.show()




#periodogramming things
#table.to_csv('/home/kmc249/Downloads/psf_fluxes_with_extras.csv', index=False)

#mask outbursts and low 
table=table.loc[(table['aql mag']>18.1) & (table['aql mag']<18.6)]

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

fall, pall = LombScargle(times, table['aql mag']-np.nanmean(table['aql mag'])).autopower()
power = LombScargle(times, table['aql mag']-np.nanmean(table['aql mag'])).power(frequency)

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
plt.show(block=False)

fig, ax = plt.subplots()
ax.plot(frequency, power)
plt.show(block=False)

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
nbins = 25
bins = np.linspace(0, 1, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Assign each phase to a bin
table['our phase bin'] = pd.cut(table['our phase'], bins=bins, include_lowest=True, labels=bin_centers)
table['their phase bin'] = pd.cut(table['their phase'], bins=bins, include_lowest=True, labels=bin_centers)

# Compute mean and std per bin
binned = table.groupby('our phase bin')['aql mag'].agg(['mean','std']).reset_index()
binned_them = table.groupby('their phase bin')['aql mag'].agg(['mean','std']).reset_index()

# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase2, table['aql mag'], s=15, color='gray', label='Data')
plt.scatter(phase2 + 1, table['aql mag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=binned['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=binned['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('Pan-STARRS r')
plt.ylim(18.8, 18.1)
plt.title(f'Our Period: {best_period_hours} hrs')
plt.legend()
plt.tight_layout()
plt.show(block=False)


# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase, table['aql mag'], s=15, color='gray', label='Data')
plt.scatter(phase + 1, table['aql mag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('Pan-STARRS r')
plt.ylim(18.8, 18.1)
plt.legend()
plt.title(f'Their Period: {P*24} hrs')
plt.tight_layout()
plt.show()


'''
#crappy 2-sin cruve fit
from scipy.optimize import curve_fit
import numpy as np

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
plt.ylabel('Pan-STARRS r')
plt.gca().invert_yaxis()
plt.legend()
plt.title('us')
plt.tight_layout()
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
plt.ylabel('Pan-STARRS r')
plt.gca().invert_yaxis()
plt.legend()
plt.title('them')
plt.tight_layout()
plt.show()
'''