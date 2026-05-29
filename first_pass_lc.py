#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:42:20 2026

@author: kmc249
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from astropy.table import QTable, Table
from scipy.ndimage import map_coordinates
from photutils.psf import extract_stars
from astropy.nddata import NDData
from smith_utils import *
from photutils.datasets import make_model_image
import glob
from scipy.optimize import curve_fit
import os
import pandas as pd
import matplotlib.dates as mdates
from astropy.time import Time


target='J1752-223'

outdir = f'/neta/xrb/{target}/product/first_pass_lightcurves'

#reference
reffile=glob.glob(f'/neta/xrb/{target}/product/*_ref_stars_kciurleo.csv')[0]
refstars=pd.read_csv(reffile)
eids = refstars.loc[refstars['type'] != 'target'].index.tolist()

#to iterate over
optical_bands=['B','V','R','I']
telescopes=['1.3m', '1m']

#function definition
def poly4(x, a, b, c, d, e):
    return  a*x**4 + b*x**3 + c*x**2 + d*x + e #a*x**4 + b*x**3 + c*x**2 + d*x + e

def error_model(mag, popt, mag_min, err_min):
    mag = np.asarray(mag)

    vals = poly4(mag, *popt)

    # brighter than minimum magnitude
    bright = mag < mag_min

    # impose constant floor only there
    vals[bright] = err_min

    return vals


for tele in telescopes:
    #Only do the things we have, and do them separately.
    print(f'Trying {tele}')
    if not os.path.exists(f'/neta/xrb/{target}/{tele}/'):
        print(f'Skipping {tele}')
        continue
    
    
    for band in optical_bands:
        trimdir=f'/neta/xrb/{target}/{tele}/opt/rccd/{band}_trimmed/'
        if not os.path.exists(trimdir):
            print(f'Skipping {tele} {band}')
            continue
        print(f'Working on {tele} {band}')
        infile = f'{outdir}/{target}_{tele}_{band}_first_pass_phot.csv'
        table=pd.read_csv(infile, low_memory=False)
        table['nice time']=pd.to_datetime(table['time'])
        
        #Whole Ensemble Comparison Across Time
        fig, axes = plt.subplots(figsize=(8, 8))
        instrmags=[]
        refmags=[]
        
        #get the right comparison
        if band=='B':
            bandstr= 'BP'
        elif band=='V':
            bandstr= 'Gaia'
        elif band=='I':
            bandstr= 'i'
        elif band=='R':
            bandstr= 'r'
                
        for e in eids:
            row=refstars.loc[int(e)]
    
            refmag=row[bandstr]
            
            #Find valid flux
            flux = table[str(e)].values.astype(float)
            valid = (flux > 0) & (~np.isnan(flux))
            flux_filled=flux[valid]
            
            #Compute instrumental magnitudes and
            instrmag = np.mean(-2.5 * np.log10(flux_filled))
            instrmags.append(instrmag)
            refmags.append(refmag)
            axes.scatter(instrmag, refmag)
            axes.annotate(
                str(e),
                (instrmag,refmag),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color='red',
                alpha=0.5
            )
    
        axes.set_xlabel('instr mag of standard stars')
        axes.set_ylabel(f'{bandstr} mag')
        #slope, intercept, r, p, se =linregress(xdata3, ydata3)
        intercept = np.mean(np.array(refmags) - np.array(instrmags))
        slope = 1.0
        instrmag_arr=np.linspace(np.min(instrmags), np.max(instrmags))
        axes.plot(instrmag_arr, slope*instrmag_arr+intercept, 'g--', label=f'y={np.round(slope,2)}x+{np.round(intercept, 2)}')
        axes.invert_yaxis()
        axes.invert_xaxis()
        plt.legend()
        plt.title(f'{tele} {band}')
        plt.savefig(f'{outdir}/{target}_{tele}_{band}_magcal.png', dpi=200)
        plt.show()
        
        
        #Main light curve
        #Getting the mean ensemble calibrated magnitude
        ensemble_r_mean = refstars.loc[eids, bandstr].mean()     


        #Now doing the conversions night by night
        table['target mag'] = np.nan
        table['ave mag'] = np.nan
        table['ave instr mag'] = np.nan
        table['zeropoint'] = np.nan

        for id, row in table.iterrows():
            # average flux of comparison stars only
            eid_cols = [str(e) for e in eids]
            to_sum = row[eid_cols].to_numpy(dtype=float)
            
            avgmag=np.nanmean(-2.5*np.log10(to_sum))
            table.at[id, 'ave instr mag']=avgmag

            #delta between calibrated mag and ensemble average magnitude
            delta=ensemble_r_mean-avgmag
            table.at[id, 'zeropoint']=delta
            table.at[id, 'ave mag']=avgmag+delta
        
            flux = row['target']
            # skip non-positive fluxes
            if flux <= 0 or np.isnan(flux):
                continue
            mags = slope*(-2.5 * np.log10(flux) + delta)
            table.at[id, 'target mag']=mags
        
        #Getting Errors
        s68s=[]
        mean_cal_mags=[]
        for col in eid_cols:
                #Get the fluxes
                flux = table[col].values.astype(float)
                valid = (flux > 0) & np.isfinite(flux)
                flux_safe = flux.copy()
        
                if col == 'target':
                    # keep bad values as NaN
                    flux_safe[~valid] = np.nan
                else:
                    # replace bad values with mean
                    mean_flux = np.nanmean(flux[valid])
                    flux_safe[~valid] = mean_flux
        

                ave_instr_mag = table['ave instr mag'].values
                zeropoints = table['zeropoint'].values
        
                #Residuals
                mag_safe = (-2.5 * np.log10(flux_safe)) + zeropoints
                residuals = ave_instr_mag - (-2.5 * np.log10(flux_safe))
        
                mean_cal_mag = np.nanmean(-2.5 * np.log10(flux_safe))+intercept
        
                mean_cal_mags.append(mean_cal_mag)
        
                # ----------------------------
                # PLOTTING MASKS
                # ----------------------------
                replaced_mask = ~valid
        
                finite = np.isfinite(residuals)
        
                residuals_plot = residuals[finite]
                replaced_plot = replaced_mask[finite]
                
                # finite residuals only
                r = residuals[np.isfinite(residuals)]
                
                #mean residual
                mu = np.nanmean(r)
                
                # S such that 68% are within mean +/- S
                S68 = np.nanpercentile(np.abs(r - mu), 68)
                
                s68s.append(S68)
        
        #For just the target point:
        flux = table['target'].values.astype(float)

        valid = (flux > 0) & np.isfinite(flux)
        
        flux_safe = flux.copy()
        flux_safe[~valid] = np.nan
        
        ave_instr_mag = table['ave instr mag'].values
        zeropoints = table['zeropoint'].values
        
        mag_safe = (-2.5 * np.log10(flux_safe)) + zeropoints
        residuals = ave_instr_mag - (-2.5 * np.log10(flux_safe))
        
        r = residuals[np.isfinite(residuals)]
        
        mu = np.nanmean(r)
        target_s68 = np.nanpercentile(np.abs(r - mu), 68)
        
        target_mean_mag = np.nanmean(-2.5 * np.log10(flux_safe)) + intercept
        

        #Now the whol error array        
        s68s = np.array(s68s)
        mean_cal_mags = np.array(mean_cal_mags)
        
        #Fit the polynomial to the errors
        finite_fit = np.isfinite(mean_cal_mags) & np.isfinite(s68s)

        xfit = mean_cal_mags[finite_fit]
        yfit = s68s[finite_fit]
        
        popt, pcov = curve_fit(poly4, xfit, yfit)

        xgrid = np.linspace(np.min(xfit), np.max(xfit), 5000)
        ygrid = poly4(xgrid, *popt)
        idx_min = np.argmin(ygrid)

        mag_min = xgrid[idx_min]
        err_min = ygrid[idx_min]
        
        #Extended floor
        yfloor = ygrid.copy()
        bright = xgrid < mag_min
        yfloor[bright] = err_min

        #Make plot
        plt.figure(figsize=(8,6))

        for x, y, label in zip(mean_cal_mags, s68s, eid_cols):
            

            plt.scatter(x, y, color='black', s=20)
            plt.annotate(label, (x, y), xytext=(3,3),
                         textcoords='offset points', fontsize=7, alpha=0.6)
            
        plt.scatter(target_mean_mag, target_s68,
            color='red', s=40, zorder=3)

        plt.annotate(f'{target}',
                     (target_mean_mag, target_s68),
                     xytext=(5,5),
                     textcoords='offset points',
                     color='red',
                     fontsize=10)
                
           # raw quartic
        plt.plot(xgrid, ygrid,
                 color='dodgerblue',
                 lw=2,
                 label='Quadratic fit')
        
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
        
        plt.xlabel('Mean calibrated mag')
        #plt.ylabel('Std of residuals (ave ens mag - mag)')
        plt.ylabel('S68 of residuals (ave ens mag - mag)')
        plt.gca().invert_xaxis()  # brighter stars on left (astronomy convention)
        plt.title(f'{tele} {band}: Scatter vs Magnitude')
        plt.tight_layout()
        plt.savefig(f'{outdir}/{target}_{tele}_{band}_errors_scatter.png', dpi=200)
        plt.show()
        
        
        # prepare array for per-row errors
        row_errors = np.full(len(table), np.nan)
        

        # evaluate model at each target magnitude
        row_errors = error_model(table['target mag'].values, popt, mag_min, err_min)
        
        # preserve NaNs where magnitude is NaN
        row_errors[~np.isfinite(table['target mag'].values)] = np.nan
        
        table['error'] = row_errors
        
        
        
        #Plotting the full light curve with errors
        
        fig, ax1 = plt.subplots(figsize=(12, 3))
        
        valid = table['target mag'].notna()
        
        ax1.errorbar(
            table.loc[valid, 'nice time'],
            table.loc[valid, 'target mag'],
            yerr=table.loc[valid, 'error'],
            fmt='.',
            color='k',
            markersize=15
        )
        
        ax1.set_ylabel(bandstr)
        ax1.invert_yaxis()
        
        # --- Primary x-axis: date ---
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # --- Secondary x-axis: MJD ---
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        
        tick_locs = ax1.get_xticks()
        tick_dates = mdates.num2date(tick_locs)
        tick_mjds = Time(tick_dates).mjd
        
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels([f'{mjd:.1f}' for mjd in tick_mjds])
        
        # Axis placement
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax1.xaxis.set_ticks_position('top')
        
        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()
        
        plt.title(f'{tele} {band}')
        plt.savefig(f'{outdir}/{target}_{tele}_{band}_lc.png', dpi=200)
        plt.show()
        
#Also add one more plot to look at - the reference stars on the image  
