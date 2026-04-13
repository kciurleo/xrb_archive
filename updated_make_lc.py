#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:47:39 2026

@author: kmc249
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates
import glob
from collections import OrderedDict

#read in dfs
filelist=glob.glob('/home/kmc249/Downloads/phot_fluxes_13m_yr*')
dflist=[pd.read_csv(i) for i in filelist]
table1 = pd.concat(dflist, ignore_index=True)

wideR = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_wideR_apr_07.csv', low_memory=False)
onem=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_apr_07.csv', low_memory=False)

#psf fluxes are table2
table2=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv', low_memory=False)
table3=pd.read_csv('/home/kmc249/Downloads/psf_fluxes_2011.csv')
table2 = pd.concat([table2, table3], ignore_index=True)

onempsf=pd.read_csv('/home/kmc249/Downloads/1m_psf_fluxes_all.csv', low_memory=False)
widepsf=pd.read_csv('/home/kmc249/Downloads/1m_wideR_psf_fluxes.csv', low_memory=False)

#this has the Panstarrs-r band mags of the standard stars
standards=pd.read_csv('/home/kmc249/Downloads/BEST_ens_stds_info.csv')

#info about quiescence
quiescence=pd.read_csv('/home/kmc249/Downloads/quiescence_mjd_ranges_v5.csv')
quiescence['start_dt'] = pd.to_datetime(
    Time(quiescence['q_start_mjd'].values, format='mjd').to_datetime()
)
quiescence['end_dt'] = pd.to_datetime(
    Time(quiescence['q_end_mjd'].values, format='mjd').to_datetime()
)

#lco mags
lco1=pd.read_csv('/home/kmc249/Downloads/R_usable_banzai_lt25.txt', sep=r'\s+', header=None, comment='#', names=['MJD', 'R_mag', 'uncertainty', 'upperlimitflag'])
lco2=pd.read_csv('/home/kmc249/Downloads/R_usable_orac_lt25.txt', sep=r'\s+', header=None, comment='#', names=['MJD', 'R_mag', 'uncertainty', 'upperlimitflag'])
lco = pd.concat([lco1, lco2], ignore_index=True)

t = Time(lco['MJD'].values, format='mjd')
lco['nice time'] = t.to_datetime()

#dict of tables
tables = {
    'PSF':   {'df': table2, 'color':'blue'},
    'PSF 1m': {'df': onempsf, 'color':'royalblue'},
    'PSF Wide': {'df': widepsf, 'color':'lightseagreen'},
    'AP':    {'df': table1, 'color':'crimson'},
    'AP 1m': {'df': onem, 'color':'violet'},
    'AP Wide': {'df': wideR, 'color':'mediumpurple'},
    'LCO':   {'df': lco, 'color':'green'}
}

#####DELETE LATER

import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np

filelist = glob.glob('/home/kmc249/Downloads/phot_fluxes_13m_yr_08_apsize_*.csv')

tables = {}

# generate evenly spaced colors
colors = plt.cm.winter(np.linspace(0, 1, len(filelist)))

for file, color in zip(filelist, colors):
    df = pd.read_csv(file, low_memory=False)
    
    fname = os.path.basename(file)
    match = re.search(r'apsize_(\d+\.?\d*)', fname)
    
    if match:
        apsize = match.group(1)
        key = f'Ap={apsize}'
        
        tables[key] = {
            'df': df,
            'color': color
        }
tables = dict(sorted(
    tables.items(),
    key=lambda x: float(x[0].split('=')[1])
))

######



#make time usable
for name, info in tables.items():
    if name == 'LCO':
        continue
    table = info['df']
    table['nice time'] = pd.to_datetime(table['time'])


#psf information for the ensemble. should not use. weird.
hiresphot=pd.read_csv("/home/kmc249/best_r_ensemble.csv")

def f(x, a, c):
    return a*np.log10(x)+c

def fline(x, a, c):
    return a*np.log10(x)+c
for tbname, info in tables.items():
    table = info['df']

    if tbname != 'LCO':
        #old way using mean
        xdata3=[]
        ydata3=[]
        badlist=[]
        fig, axes = plt.subplots(figsize=(8, 8))
        for e in table.columns:
            
            if  e not in ['nice time','time', 'filename', '413','1320','a','b','c','d','410']:
                try:
                    row=standards.loc[standards['num int']==int(e)]
                except:
                    continue
                if len(row)<1:
                    continue
                y=row['r'].iloc[0]
                flux = table[e].values.astype(float)
    
                #if there's a bad flux
                valid = (flux > 0) & (~np.isnan(flux))
                n_invalid = (~valid).sum()
                n_valid = valid.sum()
                #print(f"{e}: invalid={n_invalid}, valid={n_valid}")
                #printing the invald, there's only a few
                
                #get the mean and replace the bad values with mean
                mean_flux = np.mean(flux[valid])
                flux_filled = flux.copy()
                flux_filled[~valid] = mean_flux
                
                #compute magnitudes
                x = np.mean(-2.5 * np.log10(flux_filled))
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
        plt.title(tbname)
        #plt.savefig('/Users/katieciurleo/Downloads/yalestuff/aql_ensemble_to_stds_psf.png', dpi=250)
        plt.show()
        
        info['slope'] = slope
        info['intercept'] = intercept
        
        #make lc
        exclude_cols = ['nice time','time', 'filename', 'aql','neighbor','a','b','c','d','1418','1069','1105', '1320', 'aql mag','ave mag', '413', '410']
        ensemble_cols = [
            c for c in table.columns
            if c not in exclude_cols and c.isdigit()
        ]
        print(ensemble_cols)
        
        ensemble_ids = [int(c) for c in ensemble_cols]

        ensemble_r = standards.loc[
            standards['num int'].isin(ensemble_ids), 'r'
        ]
        #panstarrs r mag of ensemble
        ensemble_r_mean=ensemble_r.mean()
        
        table['aql mag'] = np.nan       # pre-create column
        table['ave mag'] = np.nan      # if needed for table2
        plt.figure(figsize=(12,3))

        for id, row in table.iterrows():
            # average flux of comparison stars only
            to_sum = np.array([row[name] for name in table.columns if name not in exclude_cols])
            
            avgmag=np.nanmean(-2.5*np.log10(to_sum))
            table.at[id, 'ave mag']=slope*avgmag+intercept
            #h1=plt.scatter(row['nice time'], -2.5*np.log10(avg), s=15, color='gray',label='mean ens mag')
            #delta between panstarrs r mag and ensemble average magnitude
            delta=ensemble_r_mean-avgmag
        
            for name in table.columns:
                if name  in ['aql']:#['nice time','time','filename']:
                    flux = row[name]
                    # skip non-positive fluxes
                    if flux <= 0 or np.isnan(flux):
                        continue
                    mags = slope*(-2.5 * np.log10(flux) + delta)
                    table.at[id, 'aql mag']=mags

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
        plt.title(tbname)
        plt.show()
        
#%%
#getting errors
for tbname, info in tables.items():
    if tbname=='LCO':
        continue
    table=info['df']
    # build one combined mask
    mask = np.zeros(len(table), dtype=bool)
    info['quiescence_mask']=mask
    
    for start, end in zip(quiescence['start_dt'], quiescence['end_dt']):
        mask |= (table['nice time'] >= start) & (table['nice time'] <= end)

    # compute overall mean
    mean_quiescent = table.loc[mask, 'aql mag'].mean()
    info['aql_mean_quiescent_mag']=mean_quiescent
    print(mean_quiescent)
    
    x_vals = []
    y_vals = []
    cols_used = []

    for col in table.columns:
        if col not in ['filename', 'time',  'neighbor', '413', '1320', 'nice time', 'aql mag', 'ave mag', 'error', '410']:
            if col == 'aql':
                # use quiescence mask to select only rows in quiescence
                flux_safe = table.loc[mask, col].values.copy()
                flux_safe[flux_safe <= 0] = np.nan
                ave_mag = table.loc[mask, 'ave mag'].values
            else:
                # for other columns, use positive fluxes or replace with the average
                flux = table[col].values.astype(float)
                valid = (flux > 0) & (~np.isnan(flux))
                mean_flux = np.nanmean(flux[valid])
                flux_safe = flux.copy()
                flux_safe[~valid] = mean_flux
                ave_mag = table['ave mag'].values
                
            mag_safe=info['slope']*(-2.5 * np.log10(flux_safe))+info['intercept']
            print(np.mean(mag_safe))
            residuals = ave_mag - mag_safe
            x = np.nanstd(residuals)
            #x = np.std(mag_safe)#/np.sqrt(len(mag_safe))
            y = np.mean(-2.5 * np.log10(flux_safe))
            x_vals.append(x)
            y_vals.append(y)
            cols_used.append(col)

    
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
    closest_idx = [i for i in closest_idx if cols_used[i] != 'aql'][:5]

    #print(constanterror)
    print(tbname)
    print("Stars closest to aql:", cols_used[closest_idx])
    constanterror = np.mean(x_vals[closest_idx])
    #print(constanterror)
    print("Stars closest to aql:", cols_used[closest_idx])
    print("Average x (std) of 4 closest stars:", constanterror)
    info['error mag']=constanterror
    table['error']=constanterror


    plt.figure(figsize=(8,6))

    for x, y, label in zip(x_vals, y_vals, cols_used):
        plt.scatter(y, x, color='black', s=20)
        
        if label == 'aql':
            plt.scatter(y, x, color='red', s=40, zorder=3)
            plt.annotate('Aql', (y, x), xytext=(5,5),
                         textcoords='offset points', color='red', fontsize=10)
        else:
            plt.annotate(label, (y, x), xytext=(3,3),
                         textcoords='offset points', fontsize=7, alpha=0.6)
    
    plt.xlabel('Mean instr. mag')
    plt.ylabel('Std of residuals (ave ens mag - mag)')
    plt.gca().invert_xaxis()  # brighter stars on left (astronomy convention)
    plt.title(f'{tbname}: Scatter vs Magnitude')
    plt.tight_layout()
    plt.show()
#%%
plt.figure(figsize=(12,3))
for tbname, info in tables.items():
    table = info['df']
    if tbname=='LCO':
        plt.errorbar(table['nice time'], table['R_mag'], yerr=table['uncertainty'],fmt='.', color=info['color'], label=f'{tbname}')
    else:
        plt.scatter(table['nice time'], table['aql mag'], marker='.', c=info['color'], label=f'{tbname}', s=15)
            
#handles = [h2, h3]
#labels = ['aql', 'ens (offset)']
plt.legend()
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
plt.title(tbname)
plt.show()


fig, axes = plt.subplots(
    8, 1, figsize=(28, 24)
)


# get start and end times
all_times = pd.concat([info['df']['nice time'].dropna() for info in tables.values()])
tmin = all_times.min()
tmax = all_times.max()

edges = pd.date_range(start=tmin, end=tmax, periods=9)  # 8 chunks = 9 edges

for i, ax_main in enumerate(axes):
    chunk_start = edges[i]
    chunk_end = edges[i+1]
    
    # Plot all tables in this time range
    for tbname, info in tables.items():
        table = info['df'].dropna(subset=['nice time'])
        mask = (table['nice time'] >= chunk_start) & (table['nice time'] <= chunk_end)
        
        if not mask.any():
            continue
        
        if tbname == 'LCO':
            ax_main.errorbar(
                table.loc[mask, 'nice time'],
                table.loc[mask, 'R_mag'],
                yerr=table.loc[mask,'uncertainty'],
                fmt='.',
                color=info['color'],
                label=tbname
            )
        else:
            ax_main.errorbar(
                table.loc[mask, 'nice time'],
                table.loc[mask, 'aql mag'],
                fmt='.',
                yerr=info['error mag'],
                color=info['color'],
                label=tbname,
            )
    
    ax_main.set_xlim(chunk_start, chunk_end)
    ax_main.invert_yaxis()
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

#legend shenanigans
handles, labels = [], []

for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
unique = OrderedDict(zip(labels, handles))
axes[-1].legend(unique.values(), unique.keys())

plt.tight_layout()
plt.savefig('/home/kmc249/Downloads/full_lccomp.png', dpi=300)
plt.show()


#%%
#do any of the math to get r wide on the same scale?

final_R = pd.concat([table1, onem, wideR], ignore_index=True)
final_R = final_R.sort_values('nice time')
final_R[['nice time', 'Rmag','e_Rmag', 'filename']] = final_R[['nice time','aql mag', 'error', 'filename']]
final_R[['nice time', 'Rmag','e_Rmag', 'filename']].to_csv('/home/kmc249/Downloads/full_aphot_lc_04_08.csv', index=False)


#%%

#problem solving
idns = [258, 104, 395]

plt.figure(figsize=(7,5))

for idn in idns:

    apsizes = []   # 🔥 RESET HERE (critical fix)
    aql_fluxes = []
    star_fluxes = []

    for name, info in tables.items():
        df = info['df']

        apsize = float(name.split('=')[1])

        aql = df['aql'].astype(float).values
        star = df[str(idn)].astype(float).values

        aql[aql <= 0] = np.nan
        star[star <= 0] = np.nan

        aql_med = np.nanmedian(aql)
        star_med = np.nanmedian(star)

        apsizes.append(apsize)
        aql_fluxes.append(aql_med)
        star_fluxes.append(star_med)

    apsizes_arr = np.array(apsizes)
    aql_fluxes = np.array(aql_fluxes)
    star_fluxes = np.array(star_fluxes)

    order = np.argsort(apsizes_arr)

    aps = apsizes_arr[order]
    aql_fluxes = aql_fluxes[order]
    star_fluxes = star_fluxes[order]

    aql_fluxes /= np.nanmax(aql_fluxes)
    star_fluxes /= np.nanmax(star_fluxes)

    plt.plot(aps, star_fluxes, 'o-', label=f'Star {idn}')

# Aql once (IMPORTANT: recompute safely or reuse last loop)
plt.plot(aps, aql_fluxes, 'k-o', label='Aql')

plt.xlabel('Aperture radius')
plt.ylabel('Normalized flux')
plt.title('Flux vs Aperture Size')
plt.legend()
plt.grid(alpha=0.3)
plt.show()