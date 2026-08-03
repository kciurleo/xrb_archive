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

#table1:
bad_guys=['/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010705.0027.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0717_1898.067.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1009_1098.010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1120_2198.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0515_1699.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.057.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.061.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.066.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.069.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990805.0017.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100423.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141031.0025.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd180527.0142.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd180628.0094.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990727.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020511.0037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050620.0131.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd080327.0142.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd080426.0131.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100330.0175.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110318.0139.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110608.0137.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd150730.0062.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0807_0898.046.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0815_1698.039.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0818_1998.033.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0824_2598.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0905_0698.029.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0926_2798.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0928_2998.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1012_1398.006.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1022_2398.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1029_3098.003.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1117_1898.002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1118_1998.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0317_1899.041.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0320_2199.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0511_1299.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.053.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990728.0013.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000406.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0035.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000522.0032.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000629.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000819.0024.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001002.0016.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001015.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001122.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001123.0002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001124.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001125.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010428.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010624.0087.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010708.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010902.0010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010912.0011.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011112.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011120.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011121.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020329.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020426.0048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020505.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd030307.0213.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050721.0079.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd090524.0062.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141026.0030.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd170818.0067.fits']

table1=table1[~table1['filename'].isin(bad_guys)]



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
    if tbname!='R band':
        continue

    if tbname != 'LCO':
        #old way using mean
        xdata3=[]
        ydata3=[]
        badlist=[]
        fig, axes = plt.subplots(figsize=(8, 8))
        for e in table.columns:
            
            if  e not in ['nice time','time', 'filename', '413','1320','a','b','c','d','410', '820','271','641', '525']:
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
        exclude_cols = ['ave instr mag','zeropoint','nice time','time', 'filename', 'aql','neighbor','a','b','c','d','1418','1069','1105', '1320', 'aql mag','ave mag', '413', '410', '820','271','641', '525']
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
        table['ave instr mag'] = np.nan
        table['zeropoint'] = np.nan
        plt.figure(figsize=(12,3))

        for id, row in table.iterrows():
            # average flux of comparison stars only
            to_sum = np.array([row[name] for name in table.columns if name not in exclude_cols])
            
            avgmag=np.nanmean(-2.5*np.log10(to_sum))
            table.at[id, 'ave instr mag']=avgmag
            #h1=plt.scatter(row['nice time'], -2.5*np.log10(avg), s=15, color='gray',label='mean ens mag')
            #delta between panstarrs r mag and ensemble average magnitude
            delta=ensemble_r_mean-avgmag
            table.at[id, 'zeropoint']=delta
            table.at[id, 'ave mag']=avgmag+delta
        
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
errors_for_now=[]
for tbname, info in tables.items():
    if tbname!='R band':
    #if tbname=='LCO':
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
    print('Mean quiescent value:',mean_quiescent)
    
    x_vals = []
    y_vals = []
    cols_used = []
    '''
    for col in table.columns:
        if col not in ['ave instr mag','zeropoint','filename', 'time',  'neighbor', '413', '1320', 'nice time', 'aql mag', 'ave mag', 'error', '410', '820','271','641', '525']:
            if col == 'aql':
                # use quiescence mask to select only rows in quiescence
                flux_safe = table.loc[mask, col].values.copy()
                flux_safe[flux_safe <= 0] = np.nan
                ave_mag = table.loc[mask, 'ave mag'].values
                ave_instr_mag = table.loc[mask, 'ave instr mag'].values
                zeropoints=table.loc[mask, 'zeropoint'].values
            else:
                # for other columns, use positive fluxes or replace with the average
                flux = table[col].values.astype(float)
                valid = (flux > 0) & (~np.isnan(flux))
                mean_flux = np.nanmean(flux[valid])
                flux_safe = flux.copy()
                flux_safe[~valid] = mean_flux
                ave_mag = table['ave mag'].values
                ave_instr_mag = table['ave instr mag'].values
                zeropoints=table['zeropoint'].values
                
            #mag_safe=info['slope']*(-2.5 * np.log10(flux_safe))+info['intercept']
            mag_safe=(-2.5 * np.log10(flux_safe))+zeropoints
            #print(np.nanmean(mag_safe))
            #residuals = ave_mag - mag_safe
            residuals = ave_instr_mag - (-2.5 * np.log10(flux_safe))
            x = np.nanstd(residuals)
            #x = np.std(mag_safe)#/np.sqrt(len(mag_safe))
            y = np.nanmean(-2.5 * np.log10(flux_safe))+info['intercept']
            #x_vals.append(x)
            y_vals.append(y)
            cols_used.append(col)


            # mask of replaced points
            replaced_mask = ~valid
        
            # remove NaN residuals for plotting
            finite = np.isfinite(residuals)
        
            times = table.loc[finite, 'nice time']
            residuals_plot = residuals[finite]
            replaced_plot = replaced_mask[finite]
        
            # make figure with side histogram
            fig, (ax1, ax2) = plt.subplots(
                1, 2,
                figsize=(14,3),
                gridspec_kw={'width_ratios':[4,1]}
            )
        
            # --------------------------
            # LEFT PANEL: residuals vs time
            # --------------------------
        
            # normal points
            ax1.scatter(
                times[~replaced_plot],
                residuals_plot[~replaced_plot],
                color='k',
                s=15,
                label='Measured flux'
            )
        
            # replaced points
            ax1.scatter(
                times[replaced_plot],
                residuals_plot[replaced_plot],
                color='red',
                s=25,
                label='Repputting table['nice time'] = pd.to_datetime(table['nice time']) didn't worklaced with mean flux'
            )
        
            ax1.set_ylabel('Differential magnitude (ensemble - star)')
            ax1.set_title(f'Star: {col}')
            ax1.invert_yaxis()
        
            ax1.legend(loc='best', fontsize=8)
        
            # format dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
            # secondary MJD axis
            ax_top = ax1.twiny()
            ax_top.set_xlim(ax1.get_xlim())
        
            tick_locs = ax1.get_xticks()
            tick_dates = mdates.num2date(tick_locs)
            tick_mjds = Time(tick_dates).mjd
        
            ax_top.set_xticks(tick_locs)
            ax_top.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])
        
            ax_top.xaxis.set_ticks_position('bottom')
            ax1.xaxis.set_ticks_position('top')
            ax_top.xaxis.set_label_position('bottom')
        
            # --------------------------
            # RIGHT PANEL: histogram
            # --------------------------
        
            ax2.hist(
                residuals[np.isfinite(residuals)],
                bins=30,
                orientation='horizontal',
                color='gray',
                alpha=0.7
            )
        
            ax2.set_xlabel('N')
            ax2.set_title(r'$\sigma =$' + f' {np.nanstd(residuals):.3f}')
        
            # match y-axis range
            ax2.set_ylim(ax1.get_ylim())
        
            plt.tight_layout()
            #plt.savefig(f'/home/kmc249/Downloads/ens_star_diff_mags/{col}.png', dpi=200)
            plt.show()
            
            # finite residuals only
            r = residuals[np.isfinite(residuals)]
            
            # mean and sigma
            mu = np.nanmean(r)
            sigma = np.nanstd(r)
            
            # fraction within 1 sigma of the mean
            frac_within_1sigma = np.mean(np.abs(r - mu) < sigma)
            
            # S such that 68% are within mean +/- S
            S68 = np.nanpercentile(np.abs(r - mu), 68)
            
            print(f'{col}')
            print(f'  sigma = {sigma:.4f}')
            print(f'  fraction within 1 sigma = {frac_within_1sigma:.3f}')
            print(f'  S68 = {S68:.4f}')
            x_vals.append(S68)
    '''
    all_outliers = []
    for col in table.columns:

        if col not in ['ave instr mag','zeropoint','filename',
                       'time','neighbor','413','1320',
                       'nice time','aql mag','ave mag',
                       'error','410', '820','271','641', '525']:
    
            # ----------------------------
            # SELECT WHICH TABLE TO USE
            # ----------------------------
            if col == 'aql':
                table_use = table.loc[mask].copy()
            else:
                table_use = table.copy()
    
            # ----------------------------
            # GET FLUXES
            # ----------------------------
            flux = table_use[col].values.astype(float)
    
            valid = (flux > 0) & np.isfinite(flux)
    
            flux_safe = flux.copy()
    
            if col == 'aql':
                # keep bad values as NaN
                flux_safe[~valid] = np.nan
            else:
                # replace bad values with mean
                mean_flux = np.nanmean(flux[valid])
                flux_safe[~valid] = mean_flux
    
            # ----------------------------
            # ANCILLARY ARRAYS
            # ----------------------------
            ave_mag = table_use['ave mag'].values
            ave_instr_mag = table_use['ave instr mag'].values
            zeropoints = table_use['zeropoint'].values
            times_all = table_use['nice time'].values
    
            # ----------------------------
            # COMPUTE RESIDUALS
            # ----------------------------
            mag_safe = (-2.5 * np.log10(flux_safe)) + zeropoints
    
            residuals = ave_instr_mag - (-2.5 * np.log10(flux_safe))
    
            x = np.nanstd(residuals)
            y = np.nanmean(-2.5 * np.log10(flux_safe)) + info['intercept']
    
            y_vals.append(y)
            cols_used.append(col)
    
            # ----------------------------
            # PLOTTING MASKS
            # ----------------------------
            replaced_mask = ~valid
    
            finite = np.isfinite(residuals)
    
            times = times_all[finite]
            residuals_plot = residuals[finite]
            replaced_plot = replaced_mask[finite]
            
            # finite residuals only
            r = residuals[np.isfinite(residuals)]
            
            # mean and sigma
            mu = np.nanmean(r)
            sigma = np.nanstd(r)
            
            # fraction within 1 sigma of the mean
            frac_within_1sigma = np.mean(np.abs(r - mu) < sigma)
            
            # S such that 68% are within mean +/- S
            S68 = np.nanpercentile(np.abs(r - mu), 68)
            
            outlier_mask = np.abs(residuals - mu) > 5*S68
            
            outlier_df = pd.DataFrame({
                'filename': table_use['filename'].values,
                'time': table_use['nice time'].values,
                'star': col,
                'residual': residuals
            })
            
            outlier_df = outlier_df[outlier_mask]
            
            all_outliers.append(outlier_df)
            
            #print(f'{col}')
            #print(f'  sigma = {sigma:.4f}')
            #print(f'  fraction within 1 sigma = {frac_within_1sigma:.3f}')
            #print(f'  S68 = {S68:.4f}')
            x_vals.append(S68)
            
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
    #print("Stars closest to aql:", cols_used[closest_idx])
    constanterror = np.mean(x_vals[closest_idx])
    #print(constanterror)
    #print("Stars closest to aql:", cols_used[closest_idx])
    #print("Average x (std) of 5 closest stars:", constanterror)
    info['error aql ave mag']=constanterror
    errors_for_now.append(constanterror)

    def poly4(x,  c, d, e):
        return  c*x**2 + d*x + e #a*x**4 + b*x**3 + c*x**2 + d*x + e
    
    finite_fit = np.isfinite(y_vals) & np.isfinite(x_vals)

    xfit = y_vals[finite_fit]
    yfit = x_vals[finite_fit]
    
    # fit
    popt, pcov = curve_fit(poly4, xfit, yfit)
    
    # fitted curve
    xgrid = np.linspace(np.min(xfit), np.max(xfit), 5000)
    ygrid = poly4(xgrid, *popt)
    idx_min = np.argmin(ygrid)

    mag_min = xgrid[idx_min]
    err_min = ygrid[idx_min]
    
    #print("Minimum at mag =", mag_min)
    #print("Minimum error =", err_min)
    
    yfloor = ygrid.copy()

    bright = xgrid < mag_min
    
    yfloor[bright] = err_min

    plt.figure(figsize=(8,6))

    for x, y, label in zip(x_vals, y_vals, cols_used):
        
        if label == 'aql':
            plt.scatter(y, x, color='red', s=40, zorder=3)
            plt.annotate('Aql', (y, x), xytext=(5,5),
                         textcoords='offset points', color='red', fontsize=10)
            #print('cool')
        else:
            plt.scatter(y, x, color='black', s=20)
            plt.annotate(label, (y, x), xytext=(3,3),
                         textcoords='offset points', fontsize=7, alpha=0.6)
            
       # raw quartic
    plt.plot(xgrid, ygrid,
             color='dodgerblue',
             lw=2,
             label='Quartic fit')
    
    # adopted floor model
    plt.plot(xgrid, yfloor,
             color='red',
             lw=3,
             label='Adopted error model')
    
    # minimum point
    plt.scatter(mag_min, err_min,
                color='red',
                s=80,
                zorder=5)
    
    plt.xlabel('Mean PanSTARRS mag')
    #plt.ylabel('Std of residuals (ave ens mag - mag)')
    plt.ylabel('S68 of residuals (ave ens mag - mag)')
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
    '''
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
        closest = sorted(dists, key=lambda x: x[1])[:5]
        
        # get their stds
        closest_stds = [star_stats[col]["std"] for col, _ in closest]
        
        # average std → error for this row
        row_errors[i] = np.nanmean(closest_stds)
    
    # assign back to table
    table['error'] = row_errors
    '''
    def error_model(mag):

        mag = np.asarray(mag)
    
        vals = poly4(mag, *popt)
    
        # brighter than minimum magnitude
        bright = mag < mag_min
    
        # impose constant floor only there
        vals[bright] = err_min
    
        return vals
    
    # evaluate model at each Aql magnitude
    row_errors = error_model(table['aql mag'].values)
    
    # preserve NaNs where magnitude is NaN
    row_errors[~np.isfinite(table['aql mag'].values)] = np.nan
    
    table['error'] = row_errors
    
    #temporarily, where are those outliers?
    all_outliers = pd.concat(all_outliers, ignore_index=True)
    
    counts = (
        all_outliers
        .groupby('filename')
        .size()
        .sort_values(ascending=False)
    )
    
    print(counts.tail(50))
    print(len(counts))
    print('out of ',len(table))
    
    
#%%
# --------------------------------------------------------
# Recompute Aql magnitudes using a nightly ensemble subset
# --------------------------------------------------------
#%%
# --------------------------------------------------------
# Recompute Aql magnitudes using a nightly ensemble subset
# --------------------------------------------------------

table = tables['R band']['df']

table['aql mag old'] = table['aql mag']
table['aql mag subset'] = np.nan
table['reduced_ensemble'] = False
table['n_ensemble_used'] = np.nan

# dictionary of PanSTARRS magnitudes
ensemble_catalog = {}

for star in ensemble_cols:
    row = standards.loc[standards['num int'] == int(star)]
    if len(row):
        ensemble_catalog[star] = row['r'].iloc[0]

for i, row in table.iterrows():

    instr_mags = []
    catalog_mags = []

    for star in ensemble_cols:

        flux = row[star]

        if np.isfinite(flux) and (flux > 0):

            instr_mags.append(-2.5*np.log10(flux))
            catalog_mags.append(ensemble_catalog[star])

    if len(instr_mags) == 0:
        continue

    table.at[i, 'n_ensemble_used'] = len(instr_mags)

    if len(instr_mags) < len(ensemble_cols):
        table.at[i, 'reduced_ensemble'] = True

    nightly_instr_mean = np.mean(instr_mags)
    nightly_catalog_mean = np.mean(catalog_mags)

    delta = nightly_catalog_mean - nightly_instr_mean

    aql_flux = row['aql']

    if np.isfinite(aql_flux) and (aql_flux > 0):

        table.at[i,'aql mag subset'] = (
            -2.5*np.log10(aql_flux) + delta
        )

# --------------------------------------------------------
# Scatter plot: old vs new
# --------------------------------------------------------

good = (
    np.isfinite(table['aql mag old']) &
    np.isfinite(table['aql mag subset'])
)


plt.figure(figsize=(7,7))

plt.errorbar(
    table.loc[good,'aql mag old'],
    table.loc[good,'aql mag subset'],
    xerr=table.loc[good,'error'],
    yerr=table.loc[good,'error'],
    fmt='.',
    ms=4,
    alpha=0.6,
    capsize=0,
    elinewidth=0.5
)
lims = [
    min(table.loc[good,'aql mag old'].min(),
        table.loc[good,'aql mag subset'].min()),
    max(table.loc[good,'aql mag old'].max(),
        table.loc[good,'aql mag subset'].max())
]
plt.plot(lims, lims, 'r--')

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.xlabel("Original")
plt.ylabel("Nightly subset")

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Difference vs time
# --------------------------------------------------------

diff = table['aql mag subset'] - table['aql mag old']

plt.figure(figsize=(12,4))

plt.scatter(
    table.loc[~table['reduced_ensemble'],'nice time'],
    diff[~table['reduced_ensemble']],
    s=8,
    color='royalblue',
    label='Full ensemble'
)

plt.scatter(
    table.loc[table['reduced_ensemble'],'nice time'],
    diff[table['reduced_ensemble']],
    s=30,
    facecolors='none',
    edgecolors='red',
    linewidth=1.2,
    label='Reduced ensemble'
)

plt.axhline(0,color='k',ls='--')

plt.ylabel("Subset − Original (mag)")
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Light curves + residuals
# --------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2,1,
    figsize=(18,7),
    sharex=True,
    gridspec_kw={'height_ratios':[3,1]}
)

good_old = np.isfinite(table['aql mag old'])
good_new = np.isfinite(table['aql mag subset'])

normal = ~table['reduced_ensemble']
reduced = table['reduced_ensemble']

# ----- Original light curve

ax1.errorbar(
    table.loc[good_old,'nice time'],
    table.loc[good_old,'aql mag old'],
    yerr=table.loc[good_old,'error'],
    fmt='.',
    color='k',
    ms=4,
    alpha=0.7,
    capsize=0,
    label='Original'
)

ax1.scatter(
    table.loc[reduced & good_old,'nice time'],
    table.loc[reduced & good_old,'aql mag old'],
    s=25,
    facecolors='none',
    edgecolors='red',
    linewidth=1,
    label='Original (reduced ensemble)'
)

# ----- New light curve
'''
ax1.scatter(
    table.loc[normal & good_new,'nice time'],
    table.loc[normal & good_new,'aql mag subset'],
    s=5,
    color='red',
    alpha=0.7,
    label='Nightly subset'
)
'''

ax1.scatter(
    table.loc[reduced & good_new,'nice time'],
    table.loc[reduced & good_new,'aql mag subset'],
    s=25,
    facecolors='none',
    edgecolors='blue',
    linewidth=1,
    label='Nightly subset (reduced ensemble)'
)

ax1.invert_yaxis()
ax1.set_ylabel("Magnitude")
ax1.legend(fontsize=8)

# ----- Residuals
# ----- Residuals

both = good_old & good_new

resid = (
    table.loc[both,'aql mag subset']
    - table.loc[both,'aql mag old']
)

reduced_both = table.loc[both,'reduced_ensemble']

# normal nights
ax2.errorbar(
    table.loc[both & ~table['reduced_ensemble'],'nice time'],
    resid[~reduced_both],
    yerr=0,#table.loc[both & ~table['reduced_ensemble'],'error'],
    fmt='.',
    ms=4,
    color='royalblue',
    capsize=0,
    alpha=0.7
)

# reduced ensemble nights (red circles)
ax2.errorbar(
    table.loc[both & table['reduced_ensemble'],'nice time'],
    resid[reduced_both],
    yerr=0,#table.loc[both & table['reduced_ensemble'],'error'],
    fmt='o',
    ms=5,
    color='red',
    markerfacecolor='none',
    capsize=0,
    linewidth=1,
    label='Reduced ensemble'
)

ax2.axhline(0,color='k',ls='--')

ax2.set_ylabel("Subset − Original (mag)")
ax2.set_xlabel("Date")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Summary
# --------------------------------------------------------

print(f"Reduced ensemble used on {table['reduced_ensemble'].sum()} "
      f"of {len(table)} exposures.")

print("\nNumber of comparison stars used:")

print(table['n_ensemble_used'].value_counts().sort_index())

print("\nMedian difference:", np.nanmedian(diff))
print("RMS difference:", np.nanstd(diff))
print("Largest difference:", np.nanmax(np.abs(diff)))



#%%

###EVEN STRICTER
# --------------------------------------------------------
# Nightly ensemble with bad-fit star rejection
# --------------------------------------------------------

table = tables['R band']['df']

table['aql mag old'] = table['aql mag']
table['aql mag clipped'] = np.nan
table['n_ensemble_used'] = np.nan
table['rejected_stars'] = None


# -----------------------------
# Catalog magnitudes
# -----------------------------

ensemble_catalog = {}

for star in ensemble_cols:
    row = standards.loc[standards['num int'] == int(star)]
    if len(row):
        ensemble_catalog[star] = row['r'].iloc[0]


# -----------------------------
# Build historical star residuals
# -----------------------------
# Each star's typical offset from catalog

star_offsets = {}

for star in ensemble_cols:

    flux = table[star].values.astype(float)

    good = np.isfinite(flux) & (flux > 0)

    if good.sum() < 10:
        continue

    instr_mag = -2.5*np.log10(flux[good])

    residual = instr_mag - ensemble_catalog[star]

    center = np.median(residual)

    S68 = np.percentile(
        np.abs(residual-center),
        68
    )

    star_offsets[star] = {
        'center': center,
        'S68': S68
    }


# threshold
sigma_cut = 4


# -----------------------------
# Loop over exposures
# -----------------------------

for i, row in table.iterrows():

    star_residuals = {}
    
    # calculate residual of every usable star
    for star in ensemble_cols:

        flux = row[star]

        if not np.isfinite(flux) or flux <= 0:
            continue

        if star not in star_offsets:
            continue

        instr_mag = -2.5*np.log10(flux)

        residual = (
            instr_mag
            - ensemble_catalog[star]
        )

        # subtract typical star offset
        residual -= star_offsets[star]['center']

        star_residuals[star] = residual


    if len(star_residuals) < 3:
        continue


    # ------------------------------------------------
    # Find image-wide zeropoint shift
    # ------------------------------------------------
    # If all stars move together, this captures it

    image_offset = np.median(
        list(star_residuals.values())
    )


    # residual relative to image
    relative_residuals = {
        s: r-image_offset
        for s,r in star_residuals.items()
    }


    rejected = []


    # ------------------------------------------------
    # Reject only stars that disagree with ensemble
    # ------------------------------------------------

    for star, resid in relative_residuals.items():

        limit = sigma_cut * star_offsets[star]['S68']

        if np.abs(resid) > limit:
            rejected.append(star)


    # save rejected stars
    table.at[i,'rejected_stars'] = rejected


    # remaining stars

    good_stars = [
        s for s in star_residuals.keys()
        if s not in rejected
    ]


    if len(good_stars) == 0:
        continue


    table.at[i,'n_ensemble_used'] = len(good_stars)


    # ------------------------------------------------
    # Compute ensemble zeropoint
    # ------------------------------------------------

    instr_mags = []
    catalog_mags = []


    for star in good_stars:

        flux=row[star]

        instr_mags.append(
            -2.5*np.log10(flux)
        )

        catalog_mags.append(
            ensemble_catalog[star]
        )


    nightly_instr_mean = np.mean(instr_mags)
    nightly_catalog_mean = np.mean(catalog_mags)

    delta = (
        nightly_catalog_mean
        - nightly_instr_mean
    )


    # Aql magnitude

    aql_flux=row['aql']

    if np.isfinite(aql_flux) and aql_flux > 0:

        table.at[i,'aql mag clipped'] = (
            -2.5*np.log10(aql_flux)
            + delta
        )

# --------------------------------------------------------
# Compare old ensemble vs clipped ensemble
# --------------------------------------------------------

good = (
    np.isfinite(table['aql mag old']) &
    np.isfinite(table['aql mag clipped'])
)


# --------------------------------------------------------
# Old vs new scatter
# --------------------------------------------------------

plt.figure(figsize=(7,7))

plt.errorbar(
    table.loc[good,'aql mag old'],
    table.loc[good,'aql mag clipped'],
    xerr=table.loc[good,'error'],
    yerr=table.loc[good,'error'],
    fmt='.',
    ms=4,
    alpha=0.6,
    capsize=0,
    elinewidth=0.5
)

lims = [
    min(
        table.loc[good,'aql mag old'].min(),
        table.loc[good,'aql mag clipped'].min()
    ),
    max(
        table.loc[good,'aql mag old'].max(),
        table.loc[good,'aql mag clipped'].max()
    )
]

plt.plot(
    lims,
    lims,
    'r--',
    label='1:1'
)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.xlabel("Original ensemble")
plt.ylabel("Sigma-clipped ensemble")

plt.title("Old vs clipped ensemble")

plt.legend()
plt.tight_layout()
plt.show()



# --------------------------------------------------------
# Difference vs time
# --------------------------------------------------------

diff = (
    table['aql mag clipped']
    - table['aql mag old']
)


plt.figure(figsize=(12,4))


# normal images

normal = (
    table['rejected_stars']
    .apply(lambda x: len(x)==0 if isinstance(x,list) else True)
)


plt.scatter(
    table.loc[normal,'nice time'],
    diff[normal],
    s=8,
    color='royalblue',
    label='No rejected stars'
)


# images where stars were removed

rejected = ~normal


plt.scatter(
    table.loc[rejected,'nice time'],
    diff[rejected],
    s=35,
    facecolors='none',
    edgecolors='red',
    linewidth=1.2,
    label='Rejected ensemble stars'
)


plt.axhline(
    0,
    color='k',
    ls='--'
)

plt.ylabel(
    "Clipped − Original (mag)"
)

plt.legend()

plt.tight_layout()
plt.show()



# --------------------------------------------------------
# Light curves + residuals
# --------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(18,7),
    sharex=True,
    gridspec_kw={'height_ratios':[3,1]}
)


good_old = np.isfinite(table['aql mag old'])
good_new = np.isfinite(table['aql mag clipped'])


# which exposures rejected stars?

rejected = table['rejected_stars'].apply(
    lambda x: len(x)>0 if isinstance(x,list) else False
)


# -----------------------------
# Original light curve
# -----------------------------

ax1.errorbar(
    table.loc[good_old,'nice time'],
    table.loc[good_old,'aql mag old'],
    yerr=table.loc[good_old,'error'],
    fmt='.',
    color='k',
    ms=4,
    alpha=0.6,
    capsize=0,
    label='Original ensemble'
)


# mark affected images

ax1.scatter(
    table.loc[rejected & good_old,'nice time'],
    table.loc[rejected & good_old,'aql mag old'],
    s=30,
    facecolors='none',
    edgecolors='red',
    linewidth=1,
    label='Rejected stars'
)



# -----------------------------
# Clipped light curve
# -----------------------------

ax1.scatter(
    table.loc[good_new,'nice time'],
    table.loc[good_new,'aql mag clipped'],
    s=6,
    color='royalblue',
    alpha=0.6,
    label='Sigma clipped ensemble'
)


ax1.scatter(
    table.loc[rejected & good_new,'nice time'],
    table.loc[rejected & good_new,'aql mag clipped'],
    s=35,
    facecolors='none',
    edgecolors='red',
    linewidth=1
)


ax1.invert_yaxis()

ax1.set_ylabel(
    "Magnitude"
)

ax1.legend(
    fontsize=8
)



# -----------------------------
# Residual panel
# -----------------------------

both = (
    good_old &
    good_new
)


resid = (
    table.loc[both,'aql mag clipped']
    -
    table.loc[both,'aql mag old']
)


rejected_both = rejected[both]


# normal exposures

ax2.scatter(
    table.loc[both & ~rejected,'nice time'],
    resid[~rejected_both],
    s=8,
    color='royalblue'
)


# rejected exposures

ax2.scatter(
    table.loc[both & rejected,'nice time'],
    resid[rejected_both],
    s=35,
    facecolors='none',
    edgecolors='red',
    linewidth=1
)


ax2.axhline(
    0,
    color='k',
    ls='--'
)


ax2.set_ylabel(
    "Clipped - Original"
)

ax2.set_xlabel(
    "Date"
)


plt.tight_layout()
plt.show()



# --------------------------------------------------------
# Summary statistics
# --------------------------------------------------------

n_rejected = table['rejected_stars'].apply(
    lambda x: len(x) if isinstance(x,list) else 0
)


print(
    f"Images with rejected ensemble stars: "
    f"{(n_rejected>0).sum()} / {len(table)}"
)

print(
    f"Median difference: {np.nanmedian(diff):.5f} mag"
)

print(
    f"RMS difference: {np.nanstd(diff):.5f} mag"
)

print(
    f"Largest difference: "
    f"{np.nanmax(np.abs(diff)):.5f} mag"
)


print("\nRejected star counts:")
print(
    n_rejected.value_counts()
    .sort_index()
)


#%%

#An attempt at per night
star_means = {}

for col in ensemble_cols:

    flux = table[col].astype(float).values

    valid = (flux > 0) & np.isfinite(flux)

    mag = np.full(len(flux), np.nan)
    mag[valid] = (
        -2.5*np.log10(flux[valid])
        + table.loc[valid, 'zeropoint'].values
    )

    star_means[col] = np.nanmean(mag)
    
row_errors = np.full(len(table), np.nan)

N_CLOSEST = 5

for i, row in table.iterrows():

    aql_mag = row['aql mag']

    if not np.isfinite(aql_mag):
        continue

    # ----------------------------------
    # find stars closest to Aql's current magnitude
    # ----------------------------------

    dists = []

    for star, mean_mag in star_means.items():

        d = abs(mean_mag - aql_mag)

        dists.append((star, d))

    dists.sort(key=lambda x: x[1])

    closest_stars = [s for s, _ in dists[:N_CLOSEST]]

    # ----------------------------------
    # compute residuals for those stars
    # ----------------------------------

    residuals = []

    for star in closest_stars:

        flux = row[star]

        if not np.isfinite(flux):
            continue

        if flux <= 0:
            continue

        star_mag = (
            -2.5*np.log10(flux)
            + row['zeropoint']
        )

        residuals.append(
            star_mag - star_means[star]
        )

    residuals = np.array(residuals)

    if len(residuals) < 5:
        continue

    center = np.nanmedian(residuals)

    S68 = np.nanpercentile(
        np.abs(residuals - center),
        68
    )

    row_errors[i] = S68
    
    
table['error_pernight'] = row_errors
   

#%%

def error_model(mag):

    mag = np.asarray(mag)

    vals = poly4(mag, *popt)

    vals = np.where(mag < mag_min, err_min, vals)

    return vals
#%%

# Per-night edited:
# For each image, bin ensemble stars by magnitude, compute S68 within
# each magnitude bin, then shift the global error model horizontally.

from scipy.optimize import minimize_scalar

BIN_WIDTH = 0.3
MIN_STARS_PER_BIN = 4

# --------------------------------------------------------
# Fit horizontal shift
# --------------------------------------------------------

def fit_shift(bin_mag, bin_s68):

    def objective(shift):

        pred = error_model(bin_mag + shift)

        return np.nansum((bin_s68 - pred)**2)

    res = minimize_scalar(
        objective,
        bounds=(-2,2),
        method='bounded'
    )

    return res.x

star_mean_mag = {}

for star in ensemble_cols:

    flux = table[star].astype(float).values

    valid = (flux > 0) & np.isfinite(flux)

    mags = (
        -2.5*np.log10(flux[valid])
        + table.loc[valid,'zeropoint'].values
    )

    star_mean_mag[star] = np.nanmean(mags)

# --------------------------------------------------------
# Compute one error estimate per image
# --------------------------------------------------------

row_errors = np.full(len(table), np.nan)

for i, row in table.iterrows():

    star_mag = []
    star_residual = []

    # ----------------------------------
    # residual for every ensemble star
    # ----------------------------------

    for star in ensemble_cols:

        flux = row[star]

        if (not np.isfinite(flux)) or (flux <= 0):
            continue

        mag = (
            -2.5*np.log10(flux)
            + row['zeropoint']
        )

        residual = mag - star_mean_mag[star]

        star_mag.append(mag)
        star_residual.append(residual)

    star_mag = np.array(star_mag)
    star_residual = np.array(star_residual)

    if len(star_mag) < MIN_STARS_PER_BIN:
        continue

    # ----------------------------------
    # magnitude bins
    # ----------------------------------

    bins = np.arange(
        np.floor(star_mag.min()),
        np.ceil(star_mag.max()) + BIN_WIDTH,
        BIN_WIDTH
    )

    inds = np.digitize(star_mag, bins)

    bin_centers = []
    bin_s68 = []

    for b in np.unique(inds):

        use = inds == b

        if np.sum(use) < MIN_STARS_PER_BIN:
            continue

        center = np.nanmean(star_mag[use])

        r = star_residual[use]

        med = np.nanmedian(r)

        s68 = np.nanpercentile(
            np.abs(r - med),
            68
        )

        bin_centers.append(center)
        bin_s68.append(s68)

    bin_centers = np.array(bin_centers)
    bin_s68 = np.array(bin_s68)

    if len(bin_centers) < 2:
        continue

    # ----------------------------------
    # fit horizontal shift
    # ----------------------------------

    shift = fit_shift(bin_centers, bin_s68)

    # ----------------------------------
    # estimate Aql uncertainty
    # ----------------------------------

    if np.isfinite(row['aql mag']):

        row_errors[i] = error_model(
            row['aql mag'] + shift
        )

table['error_pernight_edited'] = row_errors

#%%

#temporarily...plot aql errors over time in R band

plt.figure(figsize=(12,3))
for tbname, info in tables.items():
    if tbname!='R band':
    #if tbname=='LCO':
        continue
    table = info['df']
    #plt.errorbar(table['nice time'].values, table['error_pernight'], yerr=0,fmt='.', color='blue', label='Per night errors')
    #plt.errorbar(table['nice time'].values, table['error'], yerr=0,fmt='.', color=info['color'], label='Global errors')
    #plt.errorbar(table['nice time'].values, table['error_pernight_edited'], yerr=0,fmt='.', color='green', label='Per night errors edited')
    
    
    plt.errorbar(table['nice time'].values, table['aql mag'], yerr=table['error_pernight'],fmt='none', color='blue', label='Per night errors')
    plt.errorbar(table['nice time'].values, table['aql mag'], yerr=table['error'],fmt='none', color=info['color'], label='Global errors')
    plt.errorbar(table['nice time'].values, table['aql mag'], yerr=table['error_pernight_edited'],fmt='none', color='green', label='Per night errors edited')
    
      

       
#handles = [h2, h3]
#labels = ['aql', 'ens (offset)']
plt.legend()
plt.ylabel('Pan-STARRS r')
#plt.ylim(20,16.5)
#plt.yscale('log')
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
#plt.savefig('/home/kmc249/Downloads/full_lccomp_with_B.png', dpi=300)
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
