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

band='I'
#R BAND
file1 = f'/home/kmc249/Downloads/full_aphot_{band}_lc_04_20.csv'

#Read files
J_df =pd.read_csv(file1, low_memory=False)
J_df['nice time'] = pd.to_datetime(J_df['nice time'], errors='coerce')  # NaT for invalid

# Convert to MJD using astropy
mjds = Time(J_df['nice time'].values).mjd

# Add as a new column
J_df['MJD'] = mjds

#Get stacked flux info
'''
J_fit=pd.read_csv('/home/kmc249/current_best_R_grid_fit.csv')
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
'''

#I
frac_e= 0.225#0.118
sigma_frac_e= 0.015#0.011

#V
'''
frac_e= 0.118
sigma_frac_e=0.011
'''
frac_a=1-frac_e
sigma_frac_a= sigma_frac_e
#Mask out quiescence
intervals = list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))

#Call everything quiescent, then remove anything inside the outbursts/mini outbursts
mask = np.ones(len(J_df), dtype=bool)

for start, end in intervals:
    mask &= ~((J_df["MJD"] >= start) & (J_df["MJD"] <= end))

J_quiescent = J_df[mask].copy()

#Convert all magnitudes to flux so I can take an average
J_quiescent['flux']=10**(-0.4 * J_quiescent["Rmag"].values)

#Convert that to flux space
F_tot = J_quiescent['flux'].mean()
J_quiescent['flux_err'] = J_quiescent['flux'] * np.log(10) * 0.4 * J_quiescent['e_Rmag']
sigma_F_tot = J_quiescent['flux_err'].std() / np.sqrt(len(J_quiescent))

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
J_corrected['J_flux']= 10**(-0.4 * J_df["Rmag"].values)

#Subtract the constant in flux space, with errors
J_corrected['F_corr_orig'] = J_corrected['J_flux'] - F_a

#Alternate version where we assume that the flux of e (aql) is some constant fraction. True in quiescence.
J_corrected['F_corr_alt'] = frac_e * J_corrected['J_flux']

#Now, make the real F_corr whichever of the two is higher
J_corrected['F_corr'] = np.maximum(J_corrected['F_corr_orig'],J_corrected['F_corr_alt'])

#Convert back to magnitudes just to print the averages
m_a = -2.5 * np.log10(F_a)
m_e = -2.5 * np.log10(F_e)
sigma_m_a = (2.5 / np.log(10)) * (sigma_F_a / F_a)
sigma_m_e = (2.5 / np.log(10)) * (sigma_F_e / F_e)

print(f"J magnitude of a: {m_a:.3f}+/-{sigma_m_a:.3f}")
print(f"J magnitude of e (Aql): {m_e:.3f}+/-{sigma_m_e:.3f}")


#Convert back to magnitude, with associated errors
J_corrected["Rmag_corr"] = -2.5 * np.log10(J_corrected['F_corr'])
J_corrected['e_Rmag_corr']=np.sqrt(sigma_m_a**2+J_corrected['e_Rmag']**2)
print(J_corrected.head(10)[['Rmag', 'e_Rmag', 'Rmag_corr', 'e_Rmag_corr', 'J_flux']])
print('flux of a: ', F_a)
print('number of nan values: ', J_corrected["Rmag_corr"].isna().sum())

#%%
###Plotting
J_corrected = J_corrected.sort_values('nice time')
ymin = min(J_corrected['Rmag_corr'].min(), J_corrected['Rmag'].min())
ymax = max(J_corrected['Rmag_corr'].max(), J_corrected['Rmag'].max())

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
    if chunk.empty:
        continue
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (J_corrected['nice time'] >= tmin) & (J_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'Rmag_corr'], yerr=np.abs(J_corrected.loc[mask1,  'e_Rmag_corr']), 
                    fmt='.', color='sienna', markersize=3, label='Corrected')
    
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'Rmag'], yerr=J_corrected.loc[mask1,  'e_Rmag'],
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

J_corrected[['nice time', 'MJD', 'Rmag_corr', 'e_Rmag_corr', 'Rmag', 'e_Rmag', 'filename']].to_csv(f'/neta/xrb/AqlX-1/product/AqlX-1_{band}_corrected_lc_4_27.csv', index=False)
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
    if chunk.empty:
        continue
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (J_corrected['nice time'] >= tmin) & (J_corrected['nice time'] <= tmax)
    
    # --- MAIN LIGHT CURVE ---
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'F_corr'], yerr=0.,#yerr=np.abs(J_corrected.loc[mask1,  'e_Rmag_corr']), 
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
plt.errorbar(J_corrected['Rmag'], J_corrected['Rmag_corr'], xerr=J_corrected['e_Rmag'], yerr=J_corrected['e_Rmag_corr'], fmt='.')
plt.xlabel('R mag')
plt.ylabel('R mag corrected')
#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()
plt.ylim(29,14.2)
plt.xlim(29,14.2)
plt.show()