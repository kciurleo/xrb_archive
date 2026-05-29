#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:59:23 2026

@author: kmc249
"""

from astropy.table import Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time

#Outburst list
full = pd.read_csv("/home/kmc249/Downloads/full_outbursts.csv")
mini = pd.read_csv("/home/kmc249/Downloads/mini_outbursts.csv")

#K BAND
file2 = "/neta/xrb/AqlX-1/1.3m/ir/proc/K/AqlX1_Kband_lightcurve_mjd_fixed.ecsv"

#Read files

df2 = Table.read(file2, format="ascii.ecsv").to_pandas()
K_df = df2

#From literature
frac_e= 0.41
sigma_frac_e= 0.09

frac_a=1-frac_e

sigma_frac_a = sigma_frac_e

#Mask out quiescence
intervals = list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))

#Call everything quiescent, then remove anything inside the outbursts/mini outbursts
mask = np.ones(len(K_df), dtype=bool)

for start, end in intervals:
    mask &= ~((K_df["MJD"] >= start) & (K_df["MJD"] <= end))

K_quiescent = K_df[mask].copy()

#Convert all magnitudes to flux so I can take an average
K_quiescent['flux']=10**(-0.4 * K_quiescent["Kmag"].values)

#We're getting the average in flux space
F_tot = K_quiescent['flux'].mean()

#Convert the error in magnitude to an error in flux
K_quiescent['flux_err'] = K_quiescent['flux'] * np.log(10) * 0.4 * K_quiescent['e_Kmag']

#Error on our mean flux value
sigma_F_tot = K_quiescent['flux_err'].std()

#Find fluxes of a and e that contribute to the mean quiescent K mag
F_a = frac_a * F_tot
F_e = frac_e * F_tot

#and their associated errors
sigma_F_a = np.sqrt(
    (F_tot * sigma_frac_a)**2 +
    (frac_a * sigma_F_tot)**2
)

sigma_F_e = np.sqrt(
    (F_tot * sigma_frac_e)**2 +
    (frac_e * sigma_F_tot)**2
)

#Now subtract this from all the K band
K_corrected = K_df.copy()
t = Time(K_corrected['MJD'].values, format='mjd')
K_corrected['nice time'] = t.to_datetime()

#Convert to flux space, with error propogation
K_corrected['K_flux']= 10**(-0.4 * K_df["Kmag"].values)

#Subtract the constant in flux space, with errors
K_corrected['F_corr_orig'] = K_corrected['K_flux'] - F_a

#Alternate version where we assume that the flux of e (aql) is some constant fraction. True in quiescence.
K_corrected['F_corr_alt'] = frac_e * K_corrected['K_flux']

#Now, make the real F_corr whichever of the two is higher
#K_corrected['F_corr'] = np.maximum(K_corrected['F_corr_orig'],K_corrected['F_corr_alt'])

#Actually go back to the old way of doing just subtraction
K_corrected['F_corr']=K_corrected['F_corr_orig']
K_corrected.loc[K_corrected['F_corr'] < 0, 'F_corr'] = np.nan

#Convert back to magnitudes just to print the averages
m_a = -2.5 * np.log10(F_a)
m_e = -2.5 * np.log10(F_e)
sigma_m_a = (2.5 / np.log(10)) * (sigma_F_a / F_a)
sigma_m_e = (2.5 / np.log(10)) * (sigma_F_e / F_e)

print(f"K magnitude of a: {m_a:.3f}+/-{sigma_m_a:.3f}")
print(f"K magnitude of e (Aql): {m_e:.3f}+/-{sigma_m_e:.3f}")

#new way of magnitude errors
K_corrected['flux_err'] = K_corrected['K_flux'] * np.log(10) * 0.4 * K_corrected['e_Kmag']
K_corrected['e_Kmag_shifted']=(2.5 / np.log(10)) * (K_corrected['flux_err'] / K_corrected['F_corr'])
sigmafluxescorr=np.sqrt(K_corrected['flux_err']**2+sigma_F_a**2)


#Convert back to magnitude, with associated errors
K_corrected["Kmag_corr"] = -2.5 * np.log10(K_corrected['F_corr'])
K_corrected["Kmag Divided Version"] = -2.5 * np.log10(K_corrected['F_corr_alt'])
#K_corrected['e_Rmag_corr']=np.sqrt(sigma_m_a**2+K_corrected['e_Rmag'])
K_corrected['e_Kmag_corr']=2.5/np.log(10)*sigmafluxescorr/K_corrected['F_corr']
#K_corrected['e_Kmag_corr']=np.sqrt(sigma_m_a**2+K_corrected['e_Kmag_shifted']**2)
print(K_corrected.head(10)[['Kmag', 'e_Kmag', 'Kmag_corr', 'e_Kmag_corr', 'K_flux']])
print('flux of a: ', F_a)
print('number of nan values: ', K_corrected["Kmag_corr"].isna().sum())

#%%
###Plotting
K_corrected = K_corrected.sort_values('nice time')
ymin = min(K_corrected['Kmag_corr'].min(), K_corrected['Kmag'].min())
ymax = max(K_corrected['Kmag_corr'].max(), K_corrected['Kmag'].max())

fig, axes = plt.subplots(
    2, 1, figsize=(16, 16),
    gridspec_kw={'height_ratios': [1,1]}
)

# get start and end times
t_start = K_corrected['nice time'].min()
t_end   = K_corrected['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=3)  # 4 chunks = 5 edges

# now split K_corrected into 4 chunks by time range
time_chunks = [K_corrected[(K_corrected['nice time'] >= edges[i]) & (K_corrected['nice time'] < edges[i+1])]
               for i in range(2)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = K_corrected[(K_corrected['nice time'] >= edges[-2]) & (K_corrected['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (K_corrected['nice time'] >= tmin) & (K_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(K_corrected.loc[mask1, 'nice time'],
                    K_corrected.loc[mask1,  'Kmag_corr'], yerr=np.abs(K_corrected.loc[mask1,  'e_Kmag_corr']), 
                    fmt='.', color='red', markersize=3, label='Corrected')
    
    ax_main.errorbar(K_corrected.loc[mask1, 'nice time'],
                    K_corrected.loc[mask1,  'Kmag'], yerr=K_corrected.loc[mask1,  'e_Kmag'],
                    fmt='.', color='black', markersize=3, label='Uncorrected')
    
    
    ax_main.set_xlim(tmin, tmax)
    #ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    ax_main.set_ylim(ymax, ymin) 
    
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')

plt.tight_layout()
#plt.savefig('/home/kmc249/Downloads/K_corrected.png', dpi=300)
plt.show()

K_corrected[['Date', 'MJD', 'Kmag_corr', 'e_Kmag_corr', 'Kmag', 'e_Kmag', 'Ncomp', 'Nmatch', 'stack_median', "Kmag Divided Version"]].to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_K_corrected_lc.csv', index=False)
#%%

fig, axes = plt.subplots(
    8, 1, figsize=(16, 24),
    gridspec_kw={'height_ratios': [1,1,1,1,1,1,1,1]}
)

# get start and end times
t_start = K_corrected['nice time'].min()
t_end   = K_corrected['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=9)  # 4 chunks = 5 edges

# now split K_corrected into 4 chunks by time range
time_chunks = [K_corrected[(K_corrected['nice time'] >= edges[i]) & (K_corrected['nice time'] < edges[i+1])]
               for i in range(8)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = K_corrected[(K_corrected['nice time'] >= edges[-2]) & (K_corrected['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (K_corrected['nice time'] >= tmin) & (K_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(K_corrected.loc[mask1, 'nice time'],
                    K_corrected.loc[mask1,  'F_corr'], yerr=0.,#yerr=np.abs(K_corrected.loc[mask1,  'e_Kmag_corr']), 
                    fmt='.', color='red', markersize=3, label='Corrected')
    
    ax_main.errorbar(K_corrected.loc[mask1, 'nice time'],
                    K_corrected.loc[mask1,  'K_flux'], yerr=0,
                    fmt='.', color='black', markersize=3, label='Uncorrected')
    
    
    ax_main.set_xlim(tmin, tmax)
    #ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    #ax_main.set_ylim(ymax, ymin) 
    
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')

plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(10,10))
plt.errorbar(K_corrected['Kmag'], K_corrected['Kmag_corr'], xerr=K_corrected['e_Kmag'], yerr=K_corrected['e_Kmag_corr'], fmt='.')
plt.xlabel('K mag')
plt.ylabel('K mag corrected')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()