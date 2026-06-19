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

#H BAND

file2 = "/neta/xrb/AqlX-1/1.3m/ir/proc/H/AqlX1_Hband_lightcurve_mjd_fixed.ecsv"

#Read files

df2 = Table.read(file2, format="ascii.ecsv").to_pandas()
H_df = df2

#Get stacked flux info
H_fit=pd.read_csv('/home/kmc249/current_best_H_grid_fit.csv')
f_a=H_fit.loc[H_fit['name']=='a']['flux_fit'].values[0]
err_a=H_fit.loc[H_fit['name']=='a']['flux_err'].values[0]
f_e=H_fit.loc[H_fit['name']=='e']['flux_fit'].values[0]
err_e=H_fit.loc[H_fit['name']=='e']['flux_err'].values[0]

#fractions and error propogation of fractions
stacked_f = (f_a + f_e)
frac_a = f_a / stacked_f
frac_e = f_e / stacked_f

#Error propogation, they come out to be the same
sigma_frac_a = np.sqrt(
    (f_e / stacked_f**2)**2 * err_a**2 +
    (f_a / stacked_f**2)**2 * err_e**2
)
sigma_frac_e = sigma_frac_a 

#Mask out quiescence
intervals = list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))

#Call everything quiescent, then remove anything inside the outbursts/mini outbursts
mask = np.ones(len(H_df), dtype=bool)

for start, end in intervals:
    mask &= ~((H_df["MJD"] >= start) & (H_df["MJD"] <= end))

H_quiescent = H_df[mask].copy()

#Convert all magnitudes to flux so I can take an average
H_quiescent['flux']=10**(-0.4 * H_quiescent["Hmag"].values)

#We're getting the average in flux space
F_tot = H_quiescent['flux'].mean()

#Convert the error in magnitude to an error in flux
H_quiescent['flux_err'] = H_quiescent['flux'] * np.log(10) * 0.4 * H_quiescent['e_Hmag']

#Error on our mean flux value
sigma_F_tot = H_quiescent['flux_err'].std()

#Find fluxes of a and e that contribute to the mean quiescent H mag
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

#Now subtract this from all the H band
H_corrected = H_df.copy()
t = Time(H_corrected['MJD'].values, format='mjd')
H_corrected['nice time'] = t.to_datetime()

#Convert to flux space, with error propogation
H_corrected['H_flux']= 10**(-0.4 * H_df["Hmag"].values)

#Subtract the constant in flux space, with errors
H_corrected['F_corr_orig'] = H_corrected['H_flux'] - F_a

#Alternate version where we assume that the flux of e (aql) is some constant fraction. True in quiescence.
H_corrected['F_corr_alt'] = frac_e * H_corrected['H_flux']

#Now, make the real F_corr whichever of the two is higher
#H_corrected['F_corr'] = np.maximum(H_corrected['F_corr_orig'],H_corrected['F_corr_alt'])

#Actually go back to the old way of doing just subtraction
H_corrected['F_corr']=H_corrected['F_corr_orig']
H_corrected.loc[H_corrected['F_corr'] < 0, 'F_corr'] = np.nan

#Convert back to magnitudes just to print the averages
m_a = -2.5 * np.log10(F_a)
m_e = -2.5 * np.log10(F_e)
sigma_m_a = (2.5 / np.log(10)) * (sigma_F_a / F_a)
sigma_m_e = (2.5 / np.log(10)) * (sigma_F_e / F_e)

print(f"H magnitude of a: {m_a:.3f}+/-{sigma_m_a:.3f}")
print(f"H magnitude of e (Aql): {m_e:.3f}+/-{sigma_m_e:.3f}")

#new way of magnitude errors
H_corrected['flux_err'] = H_corrected['H_flux'] * np.log(10) * 0.4 * H_corrected['e_Hmag']
H_corrected['e_Hmag_shifted']=(2.5 / np.log(10)) * (H_corrected['flux_err'] / H_corrected['F_corr'])
sigmafluxescorr=np.sqrt(H_corrected['flux_err']**2+sigma_F_a**2)


#Convert back to magnitude, with associated errors
H_corrected["Hmag_corr"] = -2.5 * np.log10(H_corrected['F_corr'])
H_corrected["Hmag Divided Version"] = -2.5 * np.log10(H_corrected['F_corr_alt'])
#H_corrected['e_Rmag_corr']=np.sqrt(sigma_m_a**2+H_corrected['e_Rmag'])
H_corrected['e_Hmag_corr']=2.5/np.log(10)*sigmafluxescorr/H_corrected['F_corr']
#H_corrected['e_Hmag_corr']=np.sqrt(sigma_m_a**2+H_corrected['e_Hmag_shifted']**2)
print(H_corrected.head(10)[['Hmag', 'e_Hmag', 'Hmag_corr', 'e_Hmag_corr', 'H_flux']])
print('flux of a: ', F_a)
print('number of nan values: ', H_corrected["Hmag_corr"].isna().sum())

#%%
###Plotting
H_corrected = H_corrected.sort_values('nice time')
ymin = min(H_corrected['Hmag_corr'].min(), H_corrected['Hmag'].min())
ymax = max(H_corrected['Hmag_corr'].max(), H_corrected['Hmag'].max())

fig, axes = plt.subplots(
    2, 1, figsize=(16, 16),
    gridspec_kw={'height_ratios': [1,1]}
)

# get start and end times
t_start = H_corrected['nice time'].min()
t_end   = H_corrected['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=3)  # 4 chunks = 5 edges

# now split H_corrected into 4 chunks by time range
time_chunks = [H_corrected[(H_corrected['nice time'] >= edges[i]) & (H_corrected['nice time'] < edges[i+1])]
               for i in range(2)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = H_corrected[(H_corrected['nice time'] >= edges[-2]) & (H_corrected['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (H_corrected['nice time'] >= tmin) & (H_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(H_corrected.loc[mask1, 'nice time'],
                    H_corrected.loc[mask1,  'Hmag_corr'], yerr=np.abs(H_corrected.loc[mask1,  'e_Hmag_corr']), 
                    fmt='.', color='red', markersize=3, label='Corrected')
    
    ax_main.errorbar(H_corrected.loc[mask1, 'nice time'],
                    H_corrected.loc[mask1,  'Hmag'], yerr=H_corrected.loc[mask1,  'e_Hmag'],
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
#plt.savefig('/home/kmc249/Downloads/H_corrected.png', dpi=300)
plt.show()

H_corrected[['Date', 'MJD', 'Hmag_corr', 'e_Hmag_corr', 'Hmag', 'e_Hmag', 'Ncomp', 'Nmatch', 'stack_median', "Hmag Divided Version"]].to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_H_corrected_lc.csv', index=False)

#%%
plt.figure(figsize=(10,10))
plt.errorbar(H_corrected['Hmag'], H_corrected['Hmag_corr'], xerr=H_corrected['e_Hmag'], yerr=H_corrected['e_Hmag_corr'], fmt='.')
plt.xlabel('H mag')
plt.ylabel('H mag corrected')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()