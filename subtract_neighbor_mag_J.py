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

#J BAND
file1 = "/neta/xrb/AqlX-1/1m/ir/proc/J/AqlX1_Jband_lightcurve_mjd_fixed.ecsv"
file2 = "/neta/xrb/AqlX-1/1.3m/ir/proc/J/AqlX1_Jband_lightcurve_mjd_fixed.ecsv"

#Read files
df1 = Table.read(file1, format="ascii.ecsv").to_pandas()
df2 = Table.read(file2, format="ascii.ecsv").to_pandas()
J_df = pd.concat([df1, df2], ignore_index=True)

#Get stacked flux info
J_fit=pd.read_csv('/home/kmc249/current_best_J_grid_fit.csv')
f_a=J_fit.loc[J_fit['name']=='a']['flux_fit'].values[0]
err_a=J_fit.loc[J_fit['name']=='a']['flux_err'].values[0]
f_e=J_fit.loc[J_fit['name']=='e']['flux_fit'].values[0]
err_e=J_fit.loc[J_fit['name']=='e']['flux_err'].values[0]

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
mask = np.ones(len(J_df), dtype=bool)

for start, end in intervals:
    mask &= ~((J_df["MJD"] >= start) & (J_df["MJD"] <= end))

J_quiescent = J_df[mask].copy()

#Convert all magnitudes to flux so I can take an average
J_quiescent['flux']=10**(-0.4 * J_quiescent["Jmag"].values)

#We're getting the average in flux space
F_tot = J_quiescent['flux'].mean()

#Convert the error in magnitude to an error in flux
J_quiescent['flux_err'] = J_quiescent['flux'] * np.log(10) * 0.4 * J_quiescent['e_Jmag']

#Error on our mean flux value
sigma_F_tot = J_quiescent['flux_err'].std()

#Find fluxes of a and e that contribute to the mean quiescent J mag
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

#Now subtract this from all the J band
J_corrected = J_df.copy()
t = Time(J_corrected['MJD'].values, format='mjd')
J_corrected['nice time'] = t.to_datetime()

#Convert to flux space, with error propogation
J_corrected['J_flux']= 10**(-0.4 * J_df["Jmag"].values)

#Subtract the constant in flux space, with errors
J_corrected['F_corr_orig'] = J_corrected['J_flux'] - F_a

#Alternate version where we assume that the flux of e (aql) is some constant fraction. True in quiescence.
J_corrected['F_corr_alt'] = frac_e * J_corrected['J_flux']

#Now, make the real F_corr whichever of the two is higher
#J_corrected['F_corr'] = np.maximum(J_corrected['F_corr_orig'],J_corrected['F_corr_alt'])

#Actually go back to the old way of doing just subtraction
J_corrected['F_corr']=J_corrected['F_corr_orig']
J_corrected.loc[J_corrected['F_corr'] < 0, 'F_corr'] = np.nan

#Convert back to magnitudes just to print the averages
m_a = -2.5 * np.log10(F_a)
m_e = -2.5 * np.log10(F_e)
sigma_m_a = (2.5 / np.log(10)) * (sigma_F_a / F_a)
sigma_m_e = (2.5 / np.log(10)) * (sigma_F_e / F_e)

print(f"J magnitude of a: {m_a:.3f}+/-{sigma_m_a:.3f}")
print(f"J magnitude of e (Aql): {m_e:.3f}+/-{sigma_m_e:.3f}")

#new way of magnitude errors
J_corrected['flux_err'] = J_corrected['J_flux'] * np.log(10) * 0.4 * J_corrected['e_Jmag']
J_corrected['e_Jmag_shifted']=(2.5 / np.log(10)) * (J_corrected['flux_err'] / J_corrected['F_corr'])
sigmafluxescorr=np.sqrt(J_corrected['flux_err']**2+sigma_F_a**2)


#Convert back to magnitude, with associated errors
J_corrected["Jmag_corr"] = -2.5 * np.log10(J_corrected['F_corr'])
J_corrected["Jmag Divided Version"] = -2.5 * np.log10(J_corrected['F_corr_alt'])
#J_corrected['e_Rmag_corr']=np.sqrt(sigma_m_a**2+J_corrected['e_Rmag'])
J_corrected['e_Jmag_corr']=2.5/np.log(10)*sigmafluxescorr/J_corrected['F_corr']
#J_corrected['e_Jmag_corr']=np.sqrt(sigma_m_a**2+J_corrected['e_Jmag_shifted']**2)
print(J_corrected.head(10)[['Jmag', 'e_Jmag', 'Jmag_corr', 'e_Jmag_corr', 'J_flux']])
print('flux of a: ', F_a)
print('number of nan values: ', J_corrected["Jmag_corr"].isna().sum())

#%%
###Plotting
J_corrected = J_corrected.sort_values('nice time')
ymin = min(J_corrected['Jmag_corr'].min(), J_corrected['Jmag'].min())
ymax = max(J_corrected['Jmag_corr'].max(), J_corrected['Jmag'].max())

fig, axes = plt.subplots(
    8, 1, figsize=(16, 24),
    gridspec_kw={'height_ratios': [1,1,1,1,1,1,1,1]}
)

# get start and end times
t_start = J_corrected['nice time'].min()
t_end   = J_corrected['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=9)  # 4 chunks = 5 edges

# now split J_corrected into 4 chunks by time range
time_chunks = [J_corrected[(J_corrected['nice time'] >= edges[i]) & (J_corrected['nice time'] < edges[i+1])]
               for i in range(8)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = J_corrected[(J_corrected['nice time'] >= edges[-2]) & (J_corrected['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (J_corrected['nice time'] >= tmin) & (J_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'Jmag_corr'], yerr=np.abs(J_corrected.loc[mask1,  'e_Jmag_corr']), 
                    fmt='.', color='red', markersize=3, label='Corrected')
    
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'Jmag'], yerr=J_corrected.loc[mask1,  'e_Jmag'],
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
#plt.savefig('/home/kmc249/Downloads/J_corrected.png', dpi=300)
plt.show()

#J_corrected[['Date', 'MJD', 'Jmag_corr', 'e_Jmag_corr', 'Jmag', 'e_Jmag', 'Ncomp', 'Nmatch', 'stack_median', "Jmag Divided Version"]].to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_J_corrected_lc.csv', index=False)
#%%

fig, axes = plt.subplots(
    8, 1, figsize=(16, 24),
    gridspec_kw={'height_ratios': [1,1,1,1,1,1,1,1]}
)

# get start and end times
t_start = J_corrected['nice time'].min()
t_end   = J_corrected['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=9)  # 4 chunks = 5 edges

# now split J_corrected into 4 chunks by time range
time_chunks = [J_corrected[(J_corrected['nice time'] >= edges[i]) & (J_corrected['nice time'] < edges[i+1])]
               for i in range(8)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = J_corrected[(J_corrected['nice time'] >= edges[-2]) & (J_corrected['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (J_corrected['nice time'] >= tmin) & (J_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'F_corr'], yerr=0.,#yerr=np.abs(J_corrected.loc[mask1,  'e_Jmag_corr']), 
                    fmt='.', color='red', markersize=3, label='Corrected')
    
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'J_flux'], yerr=0,
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
plt.errorbar(J_corrected['Jmag'], J_corrected['Jmag_corr'], xerr=J_corrected['e_Jmag'], yerr=J_corrected['e_Jmag_corr'], fmt='.')
plt.xlabel('J mag')
plt.ylabel('J mag corrected')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()