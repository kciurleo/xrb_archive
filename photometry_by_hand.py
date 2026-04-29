#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:34:22 2026

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
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from astropy.io import fits


onem=pd.read_csv('/home/kmc249/Downloads/phot_fluxes_1m_apr_07.csv', low_memory=False)
wideR = pd.read_csv('/home/kmc249/Downloads/phot_fluxes_wideR_apr_07.csv', low_memory=False)

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

tables = {
    #'PSF':   {'df': table2, 'color':'blue'},
    #'PSF 1m': {'df': onempsf, 'color':'royalblue'},
    'R Wide': {'df': wideR, 'color':'pink'},
    #'R interest': {'df': df, 'color':'red'},
    'R 1m': {'df': onem, 'color':'violet'},

}

#read in list of bad files and get rid of them
bad_set = set(pd.read_csv('/neta/xrb/AqlX-1/temp/AqlX-1_R_badimgs.csv')['filename'])

for key, entry in tables.items():
    if key == 'LCO':
        continue
    df = entry['df']
    
    # remove rows where filename is in bad_files
    entry['df'] = df[~df['filename'].isin(bad_set)].reset_index(drop=True)
    



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
interest=tables['R 1m']['df']
df=interest.loc[interest['nice time'].dt.year==2001]
df = df.sort_values('nice time')
stacked_ensemble=pd.read_csv("/home/kmc249/best_r_ensemble.csv")
stacked_ensemble["x_fit"]=stacked_ensemble["x_fit"]/2
stacked_ensemble["y_fit"]=stacked_ensemble["y_fit"]/2

neighborhood=pd.read_csv('/home/kmc249/current_best_R_grid_fit.csv')
neighborhood["x_fit"]=neighborhood["x_fit"]/2+231
neighborhood["y_fit"]=neighborhood["y_fit"]/2+231
stacked_aql=neighborhood.loc[neighborhood['name']=='e']

# difference with previous and next points
diff_prev = (df['aql mag'] - df['aql mag'].shift(1)).abs()
diff_next = (df['aql mag'] - df['aql mag'].shift(-1)).abs()

# flag points that differ by >2 from BOTH neighbors
blue_mask = (diff_prev > 1) & (diff_next > 1)
outliers = df[blue_mask]
'''
for col in outliers.columns:
    try:
        coluint=int(col)
        #print(outliers[col].mean())
        #print(df[col].mean())
        print((outliers[col].mean()-df[col].mean())/df[col].mean())
    except:
        continue
'''
#%%
#plotting

for id, row in outliers.iterrows():
    print(row['filename'])
    print(id)
    print(row['aql']/row['800'])
    final_data=fits.getdata(row['filename'])
    #aperture photometry instead
    #r_ap = 5.0
    r_ap = 8.0
    
    #using fitted positions from PSF photometry
    positions = np.transpose((stacked_ensemble['x_fit'], stacked_ensemble['y_fit']))
    
    #Manually add the aql position
    aql_position = np.array([[
        stacked_aql['x_fit'].iloc[0],
        stacked_aql['y_fit'].iloc[0]
    ]])
    positions = np.vstack([positions, aql_position])
    
    #do the ap phot
    apertures = CircularAperture(positions, r=r_ap)
    aper_phot = aperture_photometry(final_data, apertures)
    
    # map results by ID
    ids = list(stacked_ensemble['id'])
    ids.append('aql')
    fluxes = aper_phot['aperture_sum']
    
    flux_dict = dict(zip(ids, fluxes))
    
    
    #plot to check apertures
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(final_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(8, 8))
    plt.imshow(final_data, origin='lower', cmap='gray', norm=norm)
    
    # plot apertures
    apertures.plot(color='red', lw=1.5, alpha=0.8)
    
    # optionally label IDs
    for i, (x, y) in enumerate(positions):
        plt.annotate(
            str(ids[i]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            color='yellow',
            fontsize=8
        )
    plt.show()
    print()
#%%
#get guys before and after these weirdo points
true_outliers=outliers.loc[outliers['ave mag']<19]
print(true_outliers)

df = df.sort_values("nice time")

positions = df.index.get_indexer(true_outliers.index)

expanded_positions = set()
for pos in positions:
    expanded_positions.update([pos - 1, pos, pos + 1])

expanded_positions = [p for p in expanded_positions if 0 <= p < len(df)]

result = df.iloc[sorted(expanded_positions)]
print(result['nice time'])

for id, row in result.iterrows():
    print(row['filename'])
    print(id)
    print(row['aql']/row['800'])
    final_data=fits.getdata(row['filename'])
    #aperture photometry instead
    #r_ap = 5.0
    r_ap = 8.0
    
    #using fitted positions from PSF photometry
    positions = np.transpose((stacked_ensemble['x_fit'], stacked_ensemble['y_fit']))
    
    #Manually add the aql position
    aql_position = np.array([[
        stacked_aql['x_fit'].iloc[0],
        stacked_aql['y_fit'].iloc[0]
    ]])
    positions = np.vstack([positions, aql_position])
    
    #do the ap phot
    apertures = CircularAperture(positions, r=r_ap)
    aper_phot = aperture_photometry(final_data, apertures)
    
    # map results by ID
    ids = list(stacked_ensemble['id'])
    ids.append('aql')
    fluxes = aper_phot['aperture_sum']
    
    flux_dict = dict(zip(ids, fluxes))
    
    
    #plot to check apertures
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(final_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(8, 8))
    plt.imshow(final_data, origin='lower', cmap='gray', norm=norm)
    
    # plot apertures
    apertures.plot(color='red', lw=1.5, alpha=0.8)
    
    # optionally label IDs
    for i, (x, y) in enumerate(positions):
        plt.annotate(
            str(ids[i]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            color='yellow',
            fontsize=8
        )
    plt.savefig(f'/home/kmc249/Downloads/img_{row["nice time"]}.png', dpi=300)
    plt.show()
    print()

#%%

plt.figure(figsize=(8,8))
cols=['66', '80', '104', '116', '120',
       '187', '215', '244', '258', '271', '290', '295', '318', '371', '395',
       '410', '467', '492', '506', '525', '641', '681', '729', '744', '800',
       '820']
for id, row in result.iterrows():
    yvals=[]
    for col in cols:
        yvals.append(row[col])
    ploty=np.array(yvals)/np.mean(np.array(yvals))
    xvals=np.full(len(yvals), row['aql']/np.mean(np.array(yvals)))
    plt.scatter(xvals,ploty, label=row['nice time'])
    for i in range(len(cols)):
        plt.annotate(
            str(cols[i]),
            (xvals[i], ploty[i]),
            textcoords="offset points",
            xytext=(5, 5),
            color='black',
            fontsize=8
        )
plt.legend()
plt.xlabel('aql flux rel to ens mean')
plt.ylabel('ens star flux rel to ens mean')
plt.show()

#%%

corr=pd.read_csv('/neta/xrb/AqlX-1/product/AqlX-1_R_corrected_lc.csv')
corr['nice time']=pd.to_datetime(corr['nice time'])
corr=corr.loc[corr['nice time'].dt.year==1998]
plt.figure(figsize=(12,4))
plt.scatter(corr['nice time'], corr['Rmag_corr'])
plt.gca().invert_yaxis()
plt.show()

#%%
for id, row in corr.loc[corr['Rmag_corr']>24].iterrows():
    print(row['filename'])

    final_data=fits.getdata(row['filename'])
    #aperture photometry instead
    #r_ap = 5.0
    r_ap = 8.0
    
    #using fitted positions from PSF photometry
    positions = np.transpose((stacked_ensemble['x_fit'], stacked_ensemble['y_fit']))
    
    #Manually add the aql position
    aql_position = np.array([[
        stacked_aql['x_fit'].iloc[0],
        stacked_aql['y_fit'].iloc[0]
    ]])
    positions = np.vstack([positions, aql_position])
    
    #do the ap phot
    apertures = CircularAperture(positions, r=r_ap)
    aper_phot = aperture_photometry(final_data, apertures)
    
    # map results by ID
    ids = list(stacked_ensemble['id'])
    ids.append('aql')
    fluxes = aper_phot['aperture_sum']
    
    flux_dict = dict(zip(ids, fluxes))
    
    
    #plot to check apertures
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(final_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(8, 8))
    plt.imshow(final_data, origin='lower', cmap='gray', norm=norm)
    
    # plot apertures
    apertures.plot(color='red', lw=1.5, alpha=0.8)
    
    # optionally label IDs
    for i, (x, y) in enumerate(positions):
        plt.annotate(
            str(ids[i]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            color='yellow',
            fontsize=8
        )
    plt.show()
    print()