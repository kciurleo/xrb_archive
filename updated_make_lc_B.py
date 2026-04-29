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
#filelist=glob.glob('/home/kmc249/Downloads/phot_fluxes_13m_yr*')
filelist=glob.glob('/home/kmc249/Downloads/phot_fluxes_13m_yr*_apsize_8.0.csv')

dflist=[pd.read_csv(i) for i in filelist]
table1 = pd.concat(dflist, ignore_index=True)
bband=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_B_apsize_8.0.csv')
vband=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_V_apsize_8.0.csv')
iband=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_I_apsize_8.0.csv')
onev=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_V_apsize_8.0.csv')
onei=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_I_apsize_8.0.csv')

#wideR = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_wideR_apr_07.csv', low_memory=False)
wideR = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_wideR_apr_17_rap_8.csv', low_memory=False)

#onem=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_apr_07.csv', low_memory=False)
onem=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_apr_17_rap_8.csv', low_memory=False)

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
    #'PSF':   {'df': table2, 'color':'blue'},
    #'PSF 1m': {'df': onempsf, 'color':'royalblue'},
    #'PSF Wide': {'df': widepsf, 'color':'lightseagreen'},
    'R band':    {'df': table1, 'color':'crimson'},
    'R 1m': {'df': onem, 'color':'violet'},
    'R Wide': {'df': wideR, 'color':'mediumpurple'},
    'LCO':   {'df': lco, 'color':'black'},
    'B band': {'df': bband, 'color':'blue'},
    'V band': {'df': vband, 'color':'green'},
    'I band': {'df': iband, 'color':'saddlebrown'},
    'V 1m': {'df': onev, 'color':'yellowgreen'},
    'I 1m': {'df': onei, 'color':'chocolate'}
}

#read in list of bad files and get rid of them
bad_set = set(pd.read_csv('/neta/xrb/AqlX-1/temp/AqlX-1_R_badimgs.csv')['filename'])

for key, entry in tables.items():
    if key == 'LCO':
        continue
    df = entry['df']
    
    # remove rows where filename is in bad_files
    entry['df'] = df[~df['filename'].isin(bad_set)].reset_index(drop=True)

#####DELETE LATER
'''
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
'''
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

def fline(x, c):
    return np.log10(x)+c
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
                if tbname=='B band':
                    y=row['BP'].iloc[0]
                elif tbname=='V band':
                    y=row['Gaia'].iloc[0]
                elif tbname=='I band':
                    y=row['i'].iloc[0]
                else:
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
        #slope, intercept, r, p, se =linregress(xdata3, ydata3)
        intercept = np.mean(np.array(ydata3) - np.array(xdata3))
        slope = 1.0
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
        
        if tbname=='B band':
            ensemble_r = standards.loc[
                standards['num int'].isin(ensemble_ids), 'BP'
            ]
            sidelabel='Gaia BP'
        elif tbname=='V band':
            ensemble_r = standards.loc[
                standards['num int'].isin(ensemble_ids), 'Gaia'
            ]
            sidelabel='Gaia'
        elif tbname=='I band':
            ensemble_r = standards.loc[
                standards['num int'].isin(ensemble_ids), 'i'
            ]
            sidelabel='PanSTARRS i'
        else:
            ensemble_r = standards.loc[
                standards['num int'].isin(ensemble_ids), 'r'
            ]
            sidelabel='PanSTARRS r'
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
        plt.ylabel(sidelabel)
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
    #if tbname!='AP 1m':
        continue
    table=info['df']
    #get rid of any inf values
    table.replace([np.inf, -np.inf], np.nan, inplace=True)
    # build one combined mask
    mask = np.zeros(len(table), dtype=bool)
    
    for start, end in zip(quiescence['start_dt'], quiescence['end_dt']):
        mask |= (table['nice time'] >= start) & (table['nice time'] <= end)
    info['quiescence_mask']=mask
    # compute overall mean
    mean_quiescent = np.nanmean(table.loc[mask, 'aql mag'])
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
            print(np.nanmean(mag_safe))
            residuals = ave_mag - mag_safe
            x = np.nanstd(residuals)
            #x = np.std(mag_safe)#/np.sqrt(len(mag_safe))
            y = np.nanmean(-2.5 * np.log10(flux_safe))+info['intercept']
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
    print("Average x (std) of 5 closest stars:", constanterror)
    info['error aql ave mag']=constanterror


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
    
    plt.xlabel('Mean PanSTARRS mag')
    plt.ylabel('Std of residuals (ave ens mag - mag)')
    plt.gca().invert_xaxis()  # brighter stars on left (astronomy convention)
    plt.title(f'{tbname}: Scatter vs Magnitude')
    plt.tight_layout()
    plt.show()
    
    #errors per point
    star_stats = {
        col: {"mean_mag": y, "std": x}
        for col, x, y in zip(cols_used, x_vals, y_vals)
        if col != 'aql'
    }
    
    # prepare array for per-row errors
    row_errors = np.full(len(table), np.nan)
    
    for i, row in table.iterrows():
        aql_mag = row['aql mag']
        
        if np.isnan(aql_mag):
            continue
    
        # compute distance to each star's mean magnitude
        dists = []
        for col, stats in star_stats.items():
            d = abs(stats["mean_mag"] - aql_mag)
            dists.append((col, d))
    
        # sort and take 5 closest
        closest = sorted(dists, key=lambda x: x[1])[:3]
        
        # get their stds
        closest_stds = [star_stats[col]["std"] for col, _ in closest]
        
        # average std → error for this row
        row_errors[i] = np.nanmean(closest_stds)
    
    # assign back to table
    table['error'] = row_errors

#%%

#do any of the math to get r wide on the same scale?
#quiescent wideR mean

wideRmask = tables['R Wide']['quiescence_mask']
q_wide=np.nanmedian(tables['R Wide']['df'].loc[wideRmask, 'aql mag'])

#quiescent 1m R mean
oneRmask = tables['R 1m']['quiescence_mask']
q_one=np.nanmedian(tables['R 1m']['df'].loc[oneRmask, 'aql mag'])

#get difference and shift wideR
print(q_wide)
print(q_one)

diff=q_one-q_wide
print(diff)

tables['R Wide']['df']['aql mag']=tables['R Wide']['df']['aql mag']+diff

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
                yerr=table.loc[mask, 'error'],
                color=info['color'],
                label=tbname,
            )
    
    ax_main.set_xlim(chunk_start, chunk_end)
    ax_main.set_ylim(21.5,14.5)
    #ax_main.invert_yaxis()
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
plt.savefig('/home/kmc249/Downloads/full_lccomp_with_B.png', dpi=300)
plt.show()


#%%
#save

final_R = pd.concat([tables['R band']['df'], tables['R 1m']['df'], tables['R Wide']['df']], ignore_index=True)
final_R = final_R.sort_values('nice time')
final_R[['nice time', 'Rmag','e_Rmag', 'filename']] = final_R[['nice time','aql mag', 'error', 'filename']]
final_R[['nice time', 'Rmag','e_Rmag', 'filename']].to_csv('/home/kmc249/Downloads/full_aphot_lc_04_20.csv', index=False)

final_B = tables['B band']['df'].sort_values('nice time')
final_B[['nice time', 'Rmag','e_Rmag', 'filename']] = final_B[['nice time','aql mag', 'error', 'filename']]
final_B[['nice time', 'Rmag','e_Rmag', 'filename']].to_csv('/home/kmc249/Downloads/full_aphot_B_lc_04_20.csv', index=False)


final_V = pd.concat([tables['V band']['df'], tables['V 1m']['df']], ignore_index=True)
final_V = final_V.sort_values('nice time')
final_V[['nice time', 'Rmag','e_Rmag', 'filename']] = final_V[['nice time','aql mag', 'error', 'filename']]
final_V[['nice time', 'Rmag','e_Rmag', 'filename']].to_csv('/home/kmc249/Downloads/full_aphot_V_lc_04_20.csv', index=False)


final_I = pd.concat([tables['I band']['df'], tables['I 1m']['df']], ignore_index=True)
final_I = final_I.sort_values('nice time')
final_I[['nice time', 'Rmag','e_Rmag', 'filename']] = final_I[['nice time','aql mag', 'error', 'filename']]
final_I[['nice time', 'Rmag','e_Rmag', 'filename']].to_csv('/home/kmc249/Downloads/full_aphot_I_lc_04_20.csv', index=False)



#%%
'''
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
'''
#%%
#bigplot
import matplotlib.pyplot as plt

for table in [final_R, final_V, final_I, final_B]:
    table['nice time'] = pd.to_datetime(table['time'], errors='coerce')
    
for table in [final_R, final_V, final_I, final_B]:
    table.dropna(subset=['nice time'], inplace=True)

fig, ax = plt.subplots(figsize=(14, 4))

ax.errorbar(final_R['nice time'], final_R['Rmag'],
            yerr=final_R['e_Rmag'],
            fmt='.', color='crimson', label='R', alpha=0.8)

ax.errorbar(final_V['nice time'], final_V['Rmag'],
            yerr=final_V['e_Rmag'],
            fmt='.', color='green', label='V', alpha=0.8)

ax.errorbar(final_I['nice time'], final_I['Rmag'],
            yerr=final_I['e_Rmag'],
            fmt='.', color='chocolate', label='I', alpha=0.8)

ax.errorbar(final_B['nice time'], final_B['Rmag'],
            yerr=final_B['e_Rmag'],
            fmt='.', color='blue', label='B', alpha=0.8)

ax.invert_yaxis()
ax.set_ylabel("Magnitude")
ax.set_xlabel("Time")
ax.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

bands = [
    ("B", final_B, "blue"),
    ("V", final_V, "green"),
    ("R", final_R, "crimson"),
    ("I", final_I, "chocolate"),
]

for ax, (label, df, color) in zip(axes, bands):
    ax.errorbar(
        df['nice time'],
        df['Rmag'],
        yerr=df['e_Rmag'],
        fmt='.',
        color=color,
        alpha=0.8,
        elinewidth=0.8,
        capsize=0
    )

    ax.set_ylabel(f"{label} mag")
    ax.invert_yaxis()
    ax.set_title(f"{label} band")

axes[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()


#%%
#reading in lco files
def read_uncorrected_txt(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["MJD", "mag", "mag_err", "flag"]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

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
            "alt_mag_corr"
        ]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

#%%
corrected = {}

for band in ['V', 'R', 'I']:
    df = pd.read_csv(f'/neta/xrb/AqlX-1/product/AqlX-1_{band}_corrected_lc_4_27.csv')
    df['nice time'] = pd.to_datetime(df['nice time'], errors='coerce')
    df = df.dropna(subset=['nice time'])
    corrected[band] = df
    
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True)#, sharey='row')

bands = [
    ("B", final_B, "blue"),
    ("V", final_V, "green"),
    ("R", final_R, "crimson"),
    ("I", final_I, "chocolate"),
]

for i, (label, df_old, color) in enumerate(bands):
    

    df_new = corrected.get(label)
        


    # ---- LEFT: original ----
    axes[i, 0].errorbar(
        df_old['nice time'],
        df_old['Rmag'],
        yerr=df_old['e_Rmag'],
        fmt='.',
        color=color,
        alpha=0.8
    )

    axes[i, 0].set_ylabel(f"{label} mag")

    if df_new is None:
        print(f"Skipping corrected {label} (not found)")
        continue
    # ---- RIGHT: corrected ----
    axes[i, 1].errorbar(
        df_new['nice time'],
        df_new['Rmag_corr'],
        yerr=df_new['e_Rmag'],
        fmt='.',
        color=color,
        alpha=0.8
    )
# ---- invert ONCE per row ----
for i in range(4):
    ax_left = axes[i, 0]
    ax_right = axes[i, 1]

    # get current limits after plotting
    ymin, ymax = ax_left.get_ylim()
    if i>0:
        ymin2, ymax = ax_right.get_ylim()


    # flip them manually (this is the key)
    ax_left.set_ylim(ymax, ymin)
    ax_right.set_ylim(ymax, ymin)
axes[0, 1].set_title("Corrected")
axes[0, 0].set_title("Original")
axes[-1, 0].set_xlabel("Time")
axes[-1, 1].set_xlabel("Time")

plt.tight_layout()
plt.show()


#%%

#R band
R_LCO_banzai = read_uncorrected_txt('/home/kmc249/Downloads/R_usable_banzai.txt')  
R_LCO_orac = read_uncorrected_txt('/home/kmc249/Downloads/R_usable_orac.txt')
R_LCO=pd.concat([R_LCO_banzai, R_LCO_orac], ignore_index=True)
Rp_LCO = read_uncorrected_txt('/home/kmc249/Downloads/rp_usable_banzai.txt')     

#R band corr
R_LCO_banzai_corr = read_corrected_txt('/home/kmc249/Downloads/R_usable_banzai_corrected.txt')  
R_LCO_orac_corr = read_corrected_txt('/home/kmc249/Downloads/R_usable_orac_corrected.txt')
R_LCO_corr=pd.concat([R_LCO_banzai_corr, R_LCO_orac_corr], ignore_index=True)
Rp_LCO_corr = read_corrected_txt('/home/kmc249/Downloads/rp_usable_banzai_corrected.txt') 

#ip band
ip_LCO_banzai = read_uncorrected_txt('/home/kmc249/Downloads/ip_usable_banzai.txt')  
ip_LCO_orac = read_uncorrected_txt('/home/kmc249/Downloads/ip_usable_orac.txt')
ip_LCO=pd.concat([ip_LCO_banzai, ip_LCO_orac], ignore_index=True)  

#ip band corr
ip_LCO_banzai_corr = read_corrected_txt('/home/kmc249/Downloads/ip_usable_banzai_corrected.txt')  
ip_LCO_orac_corr = read_corrected_txt('/home/kmc249/Downloads/ip_usable_orac_corrected.txt')
ip_LCO_corr=pd.concat([ip_LCO_banzai_corr, ip_LCO_orac_corr], ignore_index=True)


#v band
v_LCO_banzai = read_uncorrected_txt('/home/kmc249/Downloads/V_usable_banzai.txt')  
v_LCO_orac = read_uncorrected_txt('/home/kmc249/Downloads/V_usable_orac.txt')
v_LCO=pd.concat([v_LCO_banzai, v_LCO_orac], ignore_index=True)  

#v band corr
v_LCO_banzai_corr = read_corrected_txt('/home/kmc249/Downloads/V_usable_banzai_corrected.txt')  
v_LCO_orac_corr = read_corrected_txt('/home/kmc249/Downloads/V_usable_orac_corrected.txt')
v_LCO_corr=pd.concat([v_LCO_banzai_corr, v_LCO_orac_corr], ignore_index=True)


#%%

#R band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

R_new = corrected.get("R")
    # ---- LEFT: original ----
axes[0].errorbar(
    final_R['nice time'],
    final_R['Rmag'],
    yerr=final_R['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    R_LCO['nice time'],
    R_LCO['mag'],
    yerr=R_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[0].errorbar(
    Rp_LCO['nice time'],
    Rp_LCO['mag'],
    yerr=Rp_LCO['mag_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)


axes[0].set_ylabel("R mag")
axes[1].set_ylabel("R mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    R_new['nice time'],
    R_new['Rmag_corr'],
    yerr=R_new['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    R_LCO_corr['nice time'],
    R_LCO_corr['mag_corr'],
    yerr=R_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[1].errorbar(
    Rp_LCO_corr['nice time'],
    Rp_LCO_corr['mag_corr'],
    yerr=Rp_LCO_corr['mag_corr_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)

# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()


#%%


#V band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

V_new = corrected.get("V")
    # ---- LEFT: original ----
axes[0].errorbar(
    final_V['nice time'],
    final_V['Rmag'],
    yerr=final_V['e_Rmag'],
    fmt='.',
    color='green',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    v_LCO['nice time'],
    v_LCO['mag'],
    yerr=v_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO V'
)


axes[0].set_ylabel("V mag")
axes[1].set_ylabel("V mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    V_new['nice time'],
    V_new['Rmag_corr'],
    yerr=V_new['e_Rmag'],
    fmt='.',
    color='green',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    v_LCO_corr['nice time'],
    v_LCO_corr['mag_corr'],
    yerr=v_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO V'
)


# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()


#%%


#I band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

I_new = corrected.get("I")
    # ---- LEFT: original ----
axes[0].errorbar(
    final_I['nice time'],
    final_I['Rmag'],
    yerr=final_I['e_Rmag'],
    fmt='.',
    color='chocolate',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    ip_LCO['nice time'],
    ip_LCO['mag'],
    yerr=ip_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO ip'
)


axes[0].set_ylabel("I mag")
axes[1].set_ylabel("I mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    I_new['nice time'],
    I_new['Rmag_corr'],
    yerr=I_new['e_Rmag'],
    fmt='.',
    color='chocolate',
    alpha=0.8,
    label='SMARTS'#v band

)

#LCO stuff
axes[1].errorbar(
    ip_LCO_corr['nice time'],
    ip_LCO_corr['mag_corr'],
    yerr=ip_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO ip'
)


# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()

#%%
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

#find quiescence values for all tables
quiescent_tables = {}

for name, table in {
    "final_R": final_R,
    "final_V": final_V,
    "final_I": final_I
}.items():

    table = table.copy()
    table["MJD"] = Time(table["nice time"]).mjd

    quiescent_tables[name] = get_quiescent(table, intervals)

#find quiescence values for all tables which have MJD already
extra_tables = {
    "R_new": R_new,
    "V_new": V_new,
    "I_new": I_new,
    "Rp_LCO": Rp_LCO,
    "Rp_LCO_corr": Rp_LCO_corr,
    "R_LCO": R_LCO,
    "R_LCO_corr": R_LCO_corr,
    "ip_LCO": ip_LCO,
    "ip_LCO_corr": ip_LCO_corr,
    "v_LCO": v_LCO,
    "v_LCO_corr": v_LCO_corr,
}

for name, table in extra_tables.items():
    quiescent_tables[name] = get_quiescent(table, intervals)
    
#%%

#get mean quiescent values and comparisons
#uncorrected
uncoorr_R_SMARTS=quiescent_tables['final_R']['Rmag'].mean()
uncoorr_R_LCO=quiescent_tables['R_LCO']['mag'].mean()
uncoorr_Rp_LCO=quiescent_tables['Rp_LCO']['mag'].mean()

uncoorr_V_SMARTS=quiescent_tables['final_V']['Rmag'].mean()
uncoorr_V_LCO=quiescent_tables['v_LCO']['mag'].mean()

uncoorr_I_SMARTS=quiescent_tables['final_I']['Rmag'].mean()
uncoorr_ip_LCO=quiescent_tables['ip_LCO']['mag'].mean()

#corrected
coorr_R_SMARTS=quiescent_tables['R_new']['Rmag'].mean()
coorr_R_LCO=quiescent_tables['R_LCO_corr']['mag'].mean()
coorr_Rp_LCO=quiescent_tables['Rp_LCO_corr']['mag'].mean()

coorr_V_SMARTS=quiescent_tables['V_new']['Rmag'].mean()
coorr_V_LCO=quiescent_tables['v_LCO_corr']['mag'].mean()

coorr_I_SMARTS=quiescent_tables['I_new']['Rmag'].mean()
coorr_ip_LCO=quiescent_tables['ip_LCO_corr']['mag'].mean()


#print differences
print('--- R BAND ---')
print(uncoorr_R_SMARTS)
print(uncoorr_R_LCO)
print(uncoorr_Rp_LCO)
print('us-lco')
print(uncoorr_R_SMARTS-uncoorr_R_LCO)
print(uncoorr_R_SMARTS-uncoorr_Rp_LCO)
print('rp-lco')
print(uncoorr_Rp_LCO-uncoorr_R_LCO)
print('')

print('--- I BAND ---')
print(uncoorr_I_SMARTS)
print(uncoorr_ip_LCO)
print('us-lco')
print(uncoorr_I_SMARTS-uncoorr_ip_LCO)
print('')

print('--- V BAND ---')
print(uncoorr_V_SMARTS)
print(uncoorr_V_LCO)
print('us-lco')
print(uncoorr_V_SMARTS-uncoorr_V_LCO)
print('')

#%%

#correct the LCO band
offsets = {
    "R_LCO": uncoorr_R_SMARTS - uncoorr_R_LCO,
    "Rp_LCO": uncoorr_R_SMARTS - uncoorr_Rp_LCO,
    "v_LCO": uncoorr_V_SMARTS - uncoorr_V_LCO,
    "ip_LCO": uncoorr_I_SMARTS - uncoorr_ip_LCO,
}

R_LCO_corr["mag_shifted"] = R_LCO_corr["mag"] - offsets['R_LCO']
Rp_LCO_corr["mag_shifted"] = Rp_LCO_corr["mag"] - offsets['Rp_LCO']
v_LCO_corr["mag_shifted"] = v_LCO_corr["mag"] - offsets['v_LCO']
ip_LCO_corr["mag_shifted"] = ip_LCO_corr["mag"] - offsets['ip_LCO']


#R band
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)#, sharey='row')

R_new = corrected.get("R")
    # ---- LEFT: original ----
axes[0].errorbar(
    final_R['nice time'],
    final_R['Rmag'],
    yerr=final_R['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[0].errorbar(
    R_LCO['nice time'],
    R_LCO['mag'],
    yerr=R_LCO['mag_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[0].errorbar(
    Rp_LCO['nice time'],
    Rp_LCO['mag'],
    yerr=Rp_LCO['mag_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)


axes[0].set_ylabel("R mag")
axes[1].set_ylabel("R mag")


# ---- RIGHT: corrected ----
axes[1].errorbar(
    R_new['nice time'],
    R_new['Rmag_corr'],
    yerr=R_new['e_Rmag'],
    fmt='.',
    color='crimson',
    alpha=0.8,
    label='SMARTS'
)

#LCO stuff
axes[1].errorbar(
    R_LCO_corr['nice time'],
    R_LCO_corr['mag_shifted'],
    yerr=R_LCO_corr['mag_corr_err'],
    fmt='.',
    color='black',
    alpha=0.8,
    label='LCO R'
)

axes[1].errorbar(
    Rp_LCO_corr['nice time'],
    Rp_LCO_corr['mag_shifted'],
    yerr=Rp_LCO_corr['mag_corr_err'],
    fmt='.',
    color='gray',
    alpha=0.8,
    label='LCO rp'
)

# ---- invert ONCE per row ----

ax_left = axes[0]
ax_right = axes[1]

# get current limits after plotting
ymin, ymax = ax_left.get_ylim()
ymin2, ymax = ax_right.get_ylim()


# flip them manually (this is the key)
ax_left.set_ylim(ymax, ymin)
ax_right.set_ylim(ymax, ymin)
axes[1].set_title("Corrected")
axes[0].set_title("Original")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.legend()
plt.show()
