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

from astropy.timeseries import LombScargle

#read in 1.3 and 1 m stuff
file1 = "/neta/xrb/AqlX-1/1m/ir/proc/J/AqlX1_Jband_lightcurve_mjd_fixed.ecsv"
file2 = "/neta/xrb/AqlX-1/1.3m/ir/proc/J/AqlX1_Jband_lightcurve_mjd_fixed.ecsv"
rtable = pd.read_csv('/home/kmc249/Downloads/rough_aql_r.csv')
t1 = Table.read(file1, format="ascii.ecsv")
t2 = Table.read(file2, format="ascii.ecsv")
df1 = t1.to_pandas()
df2 = t2.to_pandas()
table = pd.concat([df1, df2], ignore_index=True)
mask = np.zeros(len(table), dtype=bool)
mask2 = np.zeros(len(rtable), dtype=bool)

#getting nice time
t = Time(table['MJD'], format='mjd')
table['nice time'] = t.to_datetime()
rtable['nice time'] = pd.to_datetime(rtable['nice time'])
rtable['MJD'] = Time(rtable['nice time']).mjd
mjd = np.asarray(table['MJD'], dtype=float)
mjd2 = np.asarray(rtable['MJD'], dtype=float)


#only get stuff in quiescence
quiescence=pd.read_csv('/home/kmc249/Downloads/quiescence_mjd_ranges_v5.csv')
for start, end in zip(quiescence['q_start_mjd'], quiescence['q_end_mjd']):
    mask |= (mjd >= start) & (mjd <= end)
table = table[mask]

for start, end in zip(quiescence['q_start_mjd'], quiescence['q_end_mjd']):
    mask2 |= (mjd2 >= start) & (mjd2 <= end)
rtable=rtable[mask2]
#%%
###J band stuff
####HEY KATIE????
#mask anything sneaking in from outbursts?? or lower stuff?
table=table.loc[(table['Jmag']>15.6) & (table['Jmag']<17.3)]


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

fall, pall = LombScargle(times, table['Jmag']-np.nanmean(table['Jmag'])).autopower(maximum_frequency=2)
power = LombScargle(times, table['Jmag']-np.nanmean(table['Jmag'])).power(frequency)

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
plt.figure(figsize=(8,4))
plt.plot(period_hours_sortedall, power_sortedall)
plt.xlabel('Period (hours)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram (all freq, capped)')
plt.axvline(x=P*24,alpha=0.5, color='red')
plt.xlim(12, 45)
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
binned = table.groupby('our phase bin')['Jmag'].agg(['mean','std']).reset_index()
binned_them = table.groupby('their phase bin')['Jmag'].agg(['mean','std']).reset_index()

# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase2, table['Jmag'], s=15, color='gray', label='Data')
plt.scatter(phase2 + 1, table['Jmag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=binned['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=binned['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('J mag')
plt.gca().invert_yaxis()
plt.title(f'Our Period: {best_period_hours} hrs')
plt.legend()
plt.tight_layout()
plt.show(block=False)


# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase, table['Jmag'], s=15, color='gray', label='Data')
plt.scatter(phase + 1, table['Jmag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('J mag')
plt.gca().invert_yaxis()
plt.legend()
plt.title(f'Their Period: {P*24} hrs')
plt.tight_layout()
plt.show()

#%%
###R band stuff
####HEY KATIE????
#mask anything sneaking in from outbursts?? or lower stuff?
#rtable=rtable.loc[(rtable['aql mag']>15.6) & (table['aql mag']<17.3)]


#periodogramming things
baseline=rtable['nice time'].max()-rtable['nice time'].min()
base_days=baseline.total_seconds() / 3600 /24
print(base_days)

#folded??
P = 0.789498  # period in days
times = Time(rtable['nice time']).mjd
t0 = times.min()
phase = ((times - t0) / P) % 1
rtable['their phase']=phase

##periodograms
min_frequency = 24/19.5
max_frequency = 24/18.5

deltaf=P/base_days/4
print('DELTA F:', deltaf)
print(np.abs(min_frequency-max_frequency)/10000)

# --- CLEAN R BAND DATA ---

# force numeric
rtable['aql mag'] = pd.to_numeric(rtable['aql mag'], errors='coerce')

times = Time(rtable['nice time']).mjd
mag = rtable['aql mag'].values

# remove NaNs
mask_clean = np.isfinite(times) & np.isfinite(mag)

times = times[mask_clean]
mag = mag[mask_clean]

# optional: remove outliers (recommended)
mask_physical = (mag > 15.5) & (mag < 18)
times = times[mask_physical]
mag = mag[mask_physical]

# mean subtract
mag = mag - np.mean(mag)

frequency = np.arange(min_frequency, max_frequency, deltaf)

ls = LombScargle(times, mag)

fall, pall = ls.autopower(maximum_frequency=2)
power = ls.power(frequency)
print(rtable['aql mag'])

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
plt.figure(figsize=(8,4))
plt.plot(period_hours_sortedall, power_sortedall)
plt.xlabel('Period (hours)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram (all freq, capped)')
plt.axvline(x=P*24,alpha=0.5, color='red')
plt.xlim(12, 45)
plt.show(block=False)

fig, ax = plt.subplots()
ax.plot(fall, pall)
plt.show(block=False)

best_frequency = frequency[np.argmax(power)]
P2 = 1 / best_frequency
best_period_hours = P2 * 24
print(best_period_hours)

phase2 = ((times - t0) / P2) % 1
rtable['our phase']=phase2

# Number of bins
nbins = 25
bins = np.linspace(0, 1, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Assign each phase to a bin
rtable['our phase bin'] = pd.cut(rtable['our phase'], bins=bins, include_lowest=True, labels=bin_centers)
rtable['their phase bin'] = pd.cut(rtable['their phase'], bins=bins, include_lowest=True, labels=bin_centers)

# Compute mean and std per bin
binned = rtable.groupby('our phase bin')['aql mag'].agg(['mean','std']).reset_index()
binned_them = rtable.groupby('their phase bin')['aql mag'].agg(['mean','std']).reset_index()

# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase2, rtable['aql mag'], s=15, color='gray', label='Data')
plt.scatter(phase2 + 1, rtable['aql mag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=binned['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=binned['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('R mag')
plt.gca().invert_yaxis()
plt.title(f'Our Period: {best_period_hours} hrs')
plt.legend()
plt.tight_layout()
plt.show(block=False)


# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase, rtable['aql mag'], s=15, color='gray', label='Data')
plt.scatter(phase + 1, rtable['aql mag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=binned_them['std'],
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('R mag')
plt.gca().invert_yaxis()
plt.legend()
plt.title(f'Their Period: {P*24} hrs')
plt.tight_layout()
plt.show()


#%%
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
plt.ylabel('J mag')
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
plt.ylabel('J mag')
plt.gca().invert_yaxis()
plt.legend()
plt.title('them')
plt.tight_layout()
plt.show()
