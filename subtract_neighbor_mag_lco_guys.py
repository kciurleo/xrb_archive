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

band='V'
ftype='orac'
#R BAND
file1 = f'/home/kmc249/Downloads/{band}_usable_{ftype}.txt'
J_df=pd.read_csv(
    file1,
    delim_whitespace=True,   # or sep='\s+'
    comment='#',            # ignore the header comment line
    names=["MJD", "mag", "mag_err", "flag"]
)
print(J_df)
t = Time(J_df["MJD"].values, format='mjd')
J_df["nice time"] = t.to_datetime()

#Get stacked flux info
if band=='R' or band=='rp':
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


#I
if band=='I' or band=='ip':
    frac_e= 0.225#0.118
    sigma_frac_e= 0.015#0.011
    
    frac_a=1-frac_e
    sigma_frac_a= sigma_frac_e

#V
if band=='V' or band =='gp':
    frac_e= 0.118
    sigma_frac_e=0.011

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
J_quiescent['flux']=10**(-0.4 * J_quiescent["mag"].values)

#Convert that to flux space
F_tot = J_quiescent['flux'].median() #J_quiescent['flux'].mean()
print('FTOT',F_tot)
print('FMEAN', J_quiescent['flux'].mean())
print('FMEDIAN',J_quiescent['flux'].median())
J_quiescent['flux_err'] = J_quiescent['flux'] * np.log(10) * 0.4 * J_quiescent['mag_err']
sigma_F_tot = J_quiescent['flux'].std() / np.sqrt(len(J_quiescent))

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
J_corrected['J_flux']= 10**(-0.4 * J_df["mag"].values)

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
m_combined=-2.5 * np.log10(F_tot)


print(f"J magnitude of a: {m_a:.3f}+/-{sigma_m_a:.3f}")
print(f"J magnitude of e (Aql): {m_e:.3f}+/-{sigma_m_e:.3f}")


#new way of magnitude errors
J_corrected['flux_err'] = J_corrected['J_flux'] * np.log(10) * 0.4 * J_corrected['mag_err']
J_corrected['mag_err_shifted']=(2.5 / np.log(10)) * (J_corrected['flux_err'] / J_corrected['F_corr'])
sigmafluxescorr=np.sqrt(J_corrected['flux_err']**2+sigma_F_a**2)
J_corrected['charles suggested flux err']=sigmafluxescorr


#Convert back to magnitude, with associated errors
J_corrected["mag_corr"] = -2.5 * np.log10(J_corrected['F_corr'])
#J_corrected['mag_err_corr']=np.sqrt(sigma_m_a**2+J_corrected['mag_err'])#J_corrected['mag_err_shifted']**2)
J_corrected['mag_err_corr']=2.5/np.log(10)*sigmafluxescorr/J_corrected['F_corr']
#J_corrected['mag_err_corr']=np.sqrt(sigma_m_a**2+J_corrected['mag_err_shifted']**2)

'''
#Convert back to magnitude, with associated errors
J_corrected["mag_corr"] = -2.5 * np.log10(J_corrected['F_corr'])
J_corrected['mag_err_corr']=np.sqrt(sigma_m_a**2+J_corrected['mag_err_shifted']**2)
'''
J_corrected["mag Divided Version"] = -2.5 * np.log10(J_corrected['F_corr_alt'])

print(J_corrected.head(10)[['mag', 'mag_err', 'mag_corr', 'mag_err_corr', 'J_flux']])
print('flux of a: ', F_a)
print('number of nan values: ', J_corrected["mag_corr"].isna().sum())

#%%
###Plotting
J_corrected = J_corrected.sort_values('nice time')
ymin = min(J_corrected['mag_corr'].min(), J_corrected['mag'].min())
ymax = max(J_corrected['mag_corr'].max(), J_corrected['mag'].max())

fig, axes = plt.subplots(
    5, 1, figsize=(16, 20),
    gridspec_kw={'height_ratios': [1,1,1,1,1]}
)

# get start and end times
t_start = J_corrected['nice time'].min()
t_end   = J_corrected['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=6)  # 4 chunks = 5 edges

# now split J_corrected into 4 chunks by time range
time_chunks = [J_corrected[(J_corrected['nice time'] >= edges[i]) & (J_corrected['nice time'] < edges[i+1])]
               for i in range(5)]

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
                    J_corrected.loc[mask1,  'mag_corr'], yerr=np.abs(J_corrected.loc[mask1,  'mag_err_corr']), 
                    fmt='.', color='sienna', markersize=3, label='Corrected')
    
    ax_main.errorbar(J_corrected.loc[mask1, 'nice time'],
                    J_corrected.loc[mask1,  'mag'], yerr=J_corrected.loc[mask1,  'mag_err'],
                    fmt='.', color='black', markersize=3, label='Uncorrected')
    
    
    ax_main.set_xlim(tmin, tmax)
    #ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    ax_main.set_ylim(ymax, ymin) 
    
    ax_main.axhline(m_combined, alpha=0.5, color='blue', linestyle='--')
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')

plt.tight_layout()
#plt.savefig('/home/kmc249/Downloads/J_corrected.png', dpi=300)
plt.show()

J_corrected_cols=J_corrected[['MJD', 'mag_corr', 'mag_err_corr', 'flag', 'mag', 'mag_err','mag Divided Version']]
print(J_corrected_cols)
header = f"# MJD corrected {band} MAG corrected uncertainty upperlimitflag {band} MAG uncertainty alt corrected MAG"

with open(f'/neta/xrb/AqlX-1/product/just_subtracted_shifted/{band}_usable_{ftype}_corrected.txt', "w") as f:
    f.write(header + "\n")
    
    J_corrected_cols.to_csv(
        f,
        sep=" ",
        index=False,
        header=False,
    )
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
plt.axvline(m_a, label='mean quiescent neighbor magnitude', color='r', linestyle='--')
plt.errorbar(J_corrected['mag'], J_corrected['mag_corr'], xerr=J_corrected['mag_err'], yerr=J_corrected['mag_err_corr'], fmt='.')
plt.xlabel('R mag')
plt.ylabel('R mag corrected')
#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()

plt.legend()
plt.ylim(25,15)
plt.xlim(25,15)
plt.show()
