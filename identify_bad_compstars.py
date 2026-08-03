#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:26:16 2026

@author: kmc249
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
import os
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ImageNormalize, ZScaleInterval, SinhStretch, AsinhStretch



target='NovaMusca'
compstars=[154,192,167,98 ] #If a SMARTS object with published already, ids of compstars
ignore_error_issue=False #Sometimes, the 4D fit goes crazy, so we need to do a 2D poly fit instead
limit_magdiff=0.25 #limit used for distance from line drawn in calib vs instr
limit_s68_fraction = 0.5   # allow 50% larger scatter than expected
manual_bad_stars=[]


outdir = f'/neta/xrb/{target}/product/first_pass_lightcurves'
savedir='/neta/xrb/PRODUCTS/'

#reference
reffile=glob.glob(f'/neta/xrb/{target}/product/*_ref_stars_kciurleo.csv')[0]
refstars=pd.read_csv(reffile)
all_eids = refstars.loc[refstars['type'] != 'target'].index.tolist()

#to iterate over
optical_bands=['B','V','R','I']
telescopes=['1.3m', '1m']

#function definition
if ignore_error_issue:
    def poly4(x, c, d, e):
        return  c*x**2 + d*x + e 
else:
    def poly4(x, a, b, c, d, e):
        return  a*x**4 + b*x**3 + c*x**2 + d*x + e 

def error_model(mag, popt, mag_min, err_min):
    mag = np.asarray(mag)

    vals = poly4(mag, *popt)

    # brighter than minimum magnitude
    bright = mag < mag_min

    # impose constant floor only there
    vals[bright] = err_min

    return vals

#transformation functions from gaia and panstarrs to johnson's cousins
#https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.T7
#https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99#apj425122fd6

def get_mag_from_gaia(gaia, bp, rp, dgaia, dbp, drp, band):
    '''
    given Gaia, BP, RP mags, as well as error in Gaia, BP, RP, return band and associated error
    '''
    
    coeffs = {
            'V': (-0.01760, -0.006860, -0.1732, 0.045858),
            'R': (-0.003226, 0.3833, -0.1345, 0.04840),
            'I': (0.02085,0.7419,-0.09631, 0.04956),
    }

    a, b, d, scatter = coeffs[band]

    c = bp - rp

    mag = gaia - (a + b*c + d*c**2)

    color_term = b + 2*d*c
    c_err = np.sqrt(dbp**2 + drp**2)

    mag_err = np.sqrt(
        dgaia**2 +
        color_term**2 * c_err**2 +
        scatter**2
    )

    return mag, mag_err


def get_mag_from_panstarrs(i, g, r, di, dg, dr, band):
    '''
    given Pan-STARRS i, g, r mags and errors, return synthesized band and associated error
    '''

    coeffs = {
        'I': ('i', -0.366, -0.136, -0.018, 0.017),
        'R': ('r', -0.137,  0.108, -0.029, 0.015),
        'V': ('r',  0.005,  0.462,  0.013, 0.012),
        'B': ('g',  0.212,  0.556,  0.034, 0.032),
    }

    ref_band, a, b, c, scatter = coeffs[band]

    color = g - r

    # Select the magnitude and error of the reference band
    ref_mag = {
        'i': i,
        'g': g,
        'r': r
    }[ref_band]

    ref_err = {
        'i': di,
        'g': dg,
        'r': dr
    }[ref_band]

    mag = ref_mag + a + b*color + c*color**2

    color_term = b + 2*c*color
    color_err = np.sqrt(dg**2 + dr**2)

    mag_err = np.sqrt(
        ref_err**2 +
        color_term**2 * color_err**2 +
        scatter**2
    )

    return mag, mag_err


#calculate johnsons values for all the refstars
for band in ['V', 'R', 'I']:
    refstars[f'{band}_gaia'], refstars[f'd{band}_gaia'] = get_mag_from_gaia(
        refstars['Gaia'],
        refstars['BP'],
        refstars['RP'],
        refstars['dGaia'],
        refstars['dBP'],
        refstars['dRP'],
        band
    )

for band in ['B', 'V', 'R', 'I']:
    refstars[f'{band}_panstarrs'], refstars[f'd{band}_panstarrs'] = get_mag_from_panstarrs(
        refstars['i'],
        refstars['g'],
        refstars['r'],
        refstars['di'],
        refstars['dg'],
        refstars['dr'],
        band
    )


error_file = f'/neta/xrb/{target}/product/{target}_error_models.txt'

if os.path.exists(error_file):
    os.remove(error_file)
#%%new
for tele in telescopes:
    # Only do the things we have, and do them separately.
    print(f'Trying {tele}')
    if not os.path.exists(f'/neta/xrb/{target}/{tele}/'):
        print(f'Skipping {tele}')
        continue

    for refsystem in ['gaia', 'panstarrs']:

        if refsystem == 'gaia':
            ref_bands = ['V', 'R', 'I']
        else:
            ref_bands = ['B', 'V', 'R', 'I']

        for band in ref_bands:

            bad_star_list = []
            if manual_bad_stars:
                bad_star_list.extend(manual_bad_stars)

            trimdir = f'/neta/xrb/{target}/{tele}/opt/rccd/{band}_trimmed/'
            if not os.path.exists(trimdir):
                print(f'Skipping {tele} {band}')
                continue

            print(f'Working on {tele} {band} using {refsystem}')

            try:
                infile = f'{outdir}/{target}_{tele}_{band}_first_pass_phot.csv'
                table = pd.read_csv(infile, low_memory=False)
            except:
                print('phot file doesnt exist!!! Something went wrong earlier')
                continue

            if len(table) == 0:
                print('phot file is empty!!! Something went wrong earlier')
                continue

            table['nice time'] = pd.to_datetime(
                table['time'],
                format='mixed',
                utc=True,
                errors='coerce'
            ).dt.tz_localize(None)

            table['nice time'] = pd.Series(
                table['nice time'].values,
                dtype='datetime64[ns]'
            )

            # Reference magnitude column
            ref_col = f'{band}_{refsystem}'

            if ref_col not in refstars.columns:
                print(f'Missing {ref_col}, skipping')
                continue

            # Whole Ensemble Comparison Across Time
            instrmags = []
            refmags = []

            # Make sure no weird stars make it into our ensemble
            eids = all_eids.copy()

            good_mask = refstars.loc[eids, ref_col] != 0
            eids = list(np.array(eids)[good_mask.values])

            for e in eids:
                row = refstars.loc[int(e)]

                refmag = row[ref_col]

                # Find valid flux
                flux = table[str(e)].values.astype(float)
                valid = (flux > 0) & (~np.isnan(flux))
                flux_filled = flux[valid]

                if len(flux_filled) == 0:
                    continue

                instrmag = np.mean(-2.5 * np.log10(flux_filled))

                instrmags.append(instrmag)
                refmags.append(refmag)

            if len(instrmags) == 0:
                print('No valid stars')
                continue

            intercept = np.nanmean(np.array(refmags) - np.array(instrmags))
            slope = 1.0

            instrmag_arr = np.linspace(
                np.nanmin(instrmags),
                np.nanmax(instrmags)
            )

            for e, refmag, instrmag in zip(eids, refmags, instrmags):

                magdiff = refmag - (instrmag + intercept)

                if abs(magdiff) > limit_magdiff:
                    bad_star_list.append(e)

            good_mask = [e not in bad_star_list for e in eids]
            bad_mask = [e in bad_star_list for e in eids]

            fig, axes = plt.subplots(figsize=(8, 8))

            # Good stars
            axes.scatter(
                np.array(instrmags)[good_mask],
                np.array(refmags)[good_mask],
                color='k'
            )

            # Bad stars
            axes.scatter(
                np.array(instrmags)[bad_mask],
                np.array(refmags)[bad_mask],
                color='gray',
                alpha=0.5
            )

            axes.set_xlabel(
                f'Uncalibrated {tele} {band} Magnitude'
            )

            axes.set_ylabel(
                f'Calibrated {band} Magnitude ({refsystem})'
            )

            axes.plot(
                instrmag_arr,
                slope*instrmag_arr + intercept,
                'g--',
                lw=2,
                label=f'y={np.round(slope,2)}x+{np.round(intercept,2)}'
            )

            axes.invert_yaxis()
            axes.invert_xaxis()

            plt.legend(loc='upper left')
            #SAVE FOR NOW DELETE LATER KT
            plt.savefig(f'/home/kmc249/Downloads/calibration_plots/{target}_{band}_{refsystem}.png', dpi=300)
            plt.show()
for band in ['V', 'R', 'I']:
    
    gaia_col = f'{band}_gaia'
    ps_col = f'{band}_panstarrs'

    # only compare stars with valid measurements
    mask = (
        (refstars[gaia_col] != 0) &
        (refstars[ps_col] != 0) &
        np.isfinite(refstars[gaia_col]) &
        np.isfinite(refstars[ps_col])
    )

    gaia_mag = refstars.loc[mask, gaia_col]
    ps_mag = refstars.loc[mask, ps_col]

    # Difference
    diff = ps_mag - gaia_mag

    # Statistics
    mean_offset = np.mean(diff)
    scatter = np.std(diff)
    median_offset = np.median(diff)

    # Outlier rejection
    bad = np.abs(diff - median_offset) > 3 * scatter

    print(f'--- {band} ---')
    print(f'Number of stars: {len(diff)}')
    print(f'Bad stars (>3 sigma): {np.sum(bad)}')
    print(f'Bad fraction: {np.sum(bad)/len(diff):.2%}')
    print(f'Mean PS - Gaia offset: {mean_offset:.4f} mag')
    print(f'Median PS - Gaia offset: {median_offset:.4f} mag')
    print(f'Scatter: {scatter:.4f} mag')

    # Optional plot
    plt.figure(figsize=(6,5))
    plt.scatter(gaia_mag, diff, s=10, color='k')
    plt.axhline(0, color='k', linestyle='--')
    plt.axhline(mean_offset, color='b', linestyle='-',
                label=f'mean={mean_offset:.3f}')
    plt.axhline(median_offset, color='r', linestyle='--',
                label=f'median={median_offset:.3f}')
    plt.xlabel(f'{band} Gaia magnitude')
    plt.ylabel(f'{band} PanSTARRS - Gaia')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title(f'{band}: PanSTARRS vs Gaia')
    plt.savefig(f'/home/kmc249/Downloads/calibration_plots/{target}_{band}_comparison.png', dpi=300)
    
    plt.show()

#%%old
for tele in telescopes:
    #Only do the things we have, and do them separately.
    print(f'Trying {tele}')
    if not os.path.exists(f'/neta/xrb/{target}/{tele}/'):
        print(f'Skipping {tele}')
        continue
    
    
    for band in optical_bands:
        bad_star_list=[]
        if manual_bad_stars:
            bad_star_list.extend(manual_bad_stars)
        trimdir=f'/neta/xrb/{target}/{tele}/opt/rccd/{band}_trimmed/'
        if not os.path.exists(trimdir):
            print(f'Skipping {tele} {band}')
            continue
        print(f'Working on {tele} {band}')
        try:
            infile = f'{outdir}/{target}_{tele}_{band}_first_pass_phot.csv'
            table=pd.read_csv(infile, low_memory=False)
        except:
            print('phot file doesnt exist!!! Something went wrong earlier')
            continue
        if len(table)==0:
            print('phot file is empty!!! Something went wrong earlier')
            continue
        table['nice time'] = pd.to_datetime(
            table['time'],
            format='mixed',
            utc=True,
            errors='coerce'
        ).dt.tz_localize(None)
        
        # force real datetime64 dtype
        table['nice time'] = pd.Series(
            table['nice time'].values,
            dtype='datetime64[ns]'
        )
        
        #Whole Ensemble Comparison Across Time
        instrmags=[]
        refmags=[]
        
   
        #Make sure no weird stars make it into our ensemble
        eids = all_eids.copy()

        good_mask = refstars.loc[eids, band] != 0
        eids = list(np.array(eids)[good_mask.values])

                
        for e in eids:
            row=refstars.loc[int(e)]
    
            refmag=row[band]
            
            #Find valid flux
            flux = table[str(e)].values.astype(float)
            valid = (flux > 0) & (~np.isnan(flux))
            flux_filled=flux[valid]
            
            #Compute instrumental magnitudes and
            instrmag = np.mean(-2.5 * np.log10(flux_filled))
            instrmags.append(instrmag)
            refmags.append(refmag)
            
        #slope, intercept, r, p, se =linregress(xdata3, ydata3)
        intercept = np.nanmean(np.array(refmags) - np.array(instrmags))
        slope = 1.0
        instrmag_arr=np.linspace(np.nanmin(instrmags), np.nanmax(instrmags))
        for e, refmag, instrmag in zip(eids, refmags, instrmags):

            magdiff = refmag - (instrmag + intercept)
        
            if abs(magdiff) > limit_magdiff:
                bad_star_list.append(e)

        


        good_mask = [e not in bad_star_list for e in eids]
        bad_mask = [e in bad_star_list for e in eids]
        
        fig, axes = plt.subplots(figsize=(8, 8))
        
        # Good stars
        axes.scatter(
            np.array(instrmags)[good_mask],
            np.array(refmags)[good_mask],
            color='k'
        )
        
        # Bad stars
        axes.scatter(
            np.array(instrmags)[bad_mask],
            np.array(refmags)[bad_mask],
            color='gray',
            alpha=0.5
        )
        
        '''
        for i, e in enumerate(eids):
            axes.annotate(
                str(e),
                (instrmags[i], refmags[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color='red' if e in bad_star_list else 'black',
                alpha=0.5
            )
        '''
        
        axes.set_xlabel(f'Uncalibrated {tele} {band} Magnitude')
        axes.set_ylabel(f'Calibrated {band} Magnitude')
        axes.plot(instrmag_arr, slope*instrmag_arr+intercept, 'g--',lw=2, label=f'y={np.round(slope,2)}x+{np.round(intercept, 2)}')
        axes.invert_yaxis()
        axes.invert_xaxis()
        plt.legend(loc='upper left')
        #plt.savefig(f'{outdir}/{target}_{tele}_{band}_magcal.png', dpi=200)
        plt.show()

        #Main light curve
        #Getting the mean ensemble calibrated magnitude
        ensemble_r_mean = refstars.loc[eids, band].mean()     


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
        
        
        
        # ------------------------------------
        # Second-pass bad star identification
        # ------------------------------------
        
        # Keep stars already rejected from the magnitude calibration
        bad_star_list = list(bad_star_list)
        
        
        for col, mag, s68 in zip(eid_cols, mean_cal_mags, s68s):
        
            # Skip stars already identified as bad
            eid = int(col)
        
            if eid in bad_star_list:
                continue
        
            # skip invalid values
            if not np.isfinite(mag) or not np.isfinite(s68):
                continue
        
        
        #Fit the polynomial to the errors
        # IDs corresponding to each point
        eid_array = np.array([int(e) for e in eid_cols])
        
        # Exclude previously identified bad stars
        not_bad = ~np.isin(eid_array, bad_star_list)
        
        finite_fit = (
            np.isfinite(mean_cal_mags) &
            np.isfinite(s68s) &
            not_bad
        )
        
        xfit = mean_cal_mags[finite_fit]
        yfit = s68s[finite_fit]
        
        popt= least_squares(
            lambda params: poly4(xfit, *params) - yfit,
            x0=np.zeros(5),
            loss='soft_l1'
        ).x
        
        xgrid = np.linspace(np.min(xfit), np.max(xfit), 5000)
        ygrid = poly4(xgrid, *popt)
        idx_min = np.argmin(ygrid)

        mag_min = xgrid[idx_min]
        err_min = ygrid[idx_min]
        
        #Extended floor
        yfloor = ygrid.copy()
        bright = xgrid < mag_min
        yfloor[bright] = err_min
        
        # ------------------------------------
        # Second-pass bad star identification
        # ------------------------------------
        

        # Red error model interpolation
        error_model_interp = interp1d(
            xgrid,
            yfloor,
            bounds_error=False,
            fill_value=np.nan
        )
        
        
        for col, mag, s68 in zip(eid_cols, mean_cal_mags, s68s):
        
            eid = int(col)
        
            # keep first-pass bad stars
            if eid in bad_star_list:
                continue
        
            if not np.isfinite(mag) or not np.isfinite(s68):
                continue
        
            # Expected scatter from the red line
            expected_s68 = error_model_interp(mag)
            fractional_excess = (s68 - expected_s68) / max(expected_s68, 1e-3)
        
            if fractional_excess > limit_s68_fraction:
                bad_star_list.append(eid)
        
      


        #Make plot
        plt.figure(figsize=(8,6))

        good_mask = [int(e) not in bad_star_list for e in eid_cols]
        bad_mask = [int(e) in bad_star_list for e in eid_cols]
        
        plt.figure(figsize=(8,6))
        
        # good stars
        plt.scatter(
            mean_cal_mags[good_mask],
            s68s[good_mask],
            color='black',
            s=20
        )
        
        # bad stars
        plt.scatter(
            mean_cal_mags[bad_mask],
            s68s[bad_mask],
            color='gray',
            alpha=0.5,
            s=20
        )
        
        for x, y, label in zip(mean_cal_mags, s68s, eid_cols):
            plt.annotate(
                label,
                (x, y),
                xytext=(3,3),
                textcoords='offset points',
                fontsize=7,
                color='red' if int(label) in bad_star_list else 'black',
                alpha=0.6
            )
            
        plt.scatter(target_mean_mag, target_s68,
            color='red', s=40, zorder=3)

        plt.annotate(f'{target}',
                     (target_mean_mag, target_s68),
                     xytext=(5,5),
                     textcoords='offset points',
                     color='red',
                     fontsize=10)
                
        # adopted floor model
        plt.plot(xgrid, yfloor,
                 color='red',
                 lw=3,
                 label='Adopted error model')
        
        # minimum point
        plt.scatter(mag_min, err_min,
                    color='red',
                    s=30,
                    zorder=5)
        
        plt.xlabel('Mean calibrated mag')
        #plt.ylabel('Std of residuals (ave ens mag - mag)')
        plt.ylabel('S68 of residuals (ave ens mag - mag)')
        plt.gca().invert_xaxis()  # brighter stars on left (astronomy convention)
        plt.title(f'{tele} {band}: Scatter vs Magnitude')
        plt.tight_layout()

        plt.show()


        # ------------------------------------
        # Final error model fit using GOOD stars only
        # ------------------------------------
        
        good_mask = np.array([int(e) not in bad_star_list for e in eid_cols])
        
        finite_final_fit = (
            np.isfinite(mean_cal_mags) &
            np.isfinite(s68s) &
            good_mask
        )
        
        xfit_final = mean_cal_mags[finite_final_fit]
        yfit_final = s68s[finite_final_fit]
        
        # redo polynomial fit
        popt, pcov = curve_fit(poly4, xfit_final, yfit_final)
        
        
        # Generate final smooth error curve
        xgrid = np.linspace(np.min(xfit_final),
                            np.max(xfit_final),
                            5000)
        
        ygrid = poly4(xgrid, *popt)
        
        
        # Find minimum error floor
        idx_min = np.argmin(ygrid)
        
        mag_min = xgrid[idx_min]
        err_min = ygrid[idx_min]
        
        
        # Apply bright-end floor
        yfloor = ygrid.copy()
        
        bright = xgrid < mag_min
        yfloor[bright] = err_min
        
        
        # prepare array for per-row errors
        row_errors = np.full(len(table), np.nan)
        

        # evaluate model at each target magnitude
        row_errors = error_model(table['target mag'].values, popt, mag_min, err_min)
        
        # preserve NaNs where magnitude is NaN
        row_errors[~np.isfinite(table['target mag'].values)] = np.nan
        
        table['error'] = row_errors
        
        # ------------------------------------
        # Plot FINAL error model
        # ------------------------------------
        
        plt.figure(figsize=(8,6))
        
        # final masks
        good_mask = np.array([int(e) not in bad_star_list for e in eid_cols])
        bad_mask = np.array([int(e) in bad_star_list for e in eid_cols])

        # stars used in final fit
        plt.scatter(
            mean_cal_mags[good_mask],
            s68s[good_mask],
            color='black',
            s=25,
            label='Good stars'
        )
        
        # rejected stars
        plt.scatter(
            mean_cal_mags[bad_mask],
            s68s[bad_mask],
            color='gray',
            alpha=0.5,
            s=25,
            label='Rejected stars'
        )

        
        # target
        plt.scatter(
            target_mean_mag,
            target_s68,
            color='red',
            s=60,
            zorder=5,
            label=target
        )
        
        plt.annotate(
            target,
            (target_mean_mag, target_s68),
            xytext=(5,5),
            textcoords='offset points',
            color='red'
        )
        
        
        plt.axhline(err_min, color='gray', alpha=0.5, linestyle='--', lw=2)
        
        # final error model
        plt.plot(
            xgrid,
            yfloor,
            color='red',
            lw=3,
            label='Final error model'
        )
        
        
        # minimum error floor
        plt.scatter(
            mag_min,
            err_min,
            color='red',
            s=50,
            zorder=6,
            label=f'floor={err_min:.3f}'
        )
        
        
        plt.xlabel(f'Calibrated {tele} {band} Magnitude')
        plt.ylabel('S68 of Residuals')
        plt.gca().invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(f'{outdir}/{target}_{tele}_{band}_errors_scatter.png', dpi=200)
        plt.show()
        
        # ------------------------------------
        # Save star quality flags for this band
        # ------------------------------------
        
        quality_col = f'{band}_quality'
        
        # initialize column if it does not exist
        if quality_col not in refstars.columns:
            refstars[quality_col] = 'unknown'
        
        for eid in eids:
            if eid in bad_star_list:
                refstars.loc[eid, quality_col] = 'bad'
            else:
                refstars.loc[eid, quality_col] = 'good'
        
        #force to be good
        for eid in compstars:
            refstars.loc[eid, quality_col] = 'good'
            refstars.loc[eid, 'type'] = 'comp'
        
        #write error file

        with open(error_file, "a") as f:
        
            f.write(f"{tele}_{band}\n")
            f.write("popt = " + ",".join(map(str, popt)) + "\n")
            f.write(f"mag_min = {mag_min}\n")
            f.write(f"err_min = {err_min}\n")
            f.write("\n")

          
outfile = f'/neta/xrb/{target}/product/{target}_ref_stars_quality.csv'

refstars.to_csv(outfile, index=False)

print(f"Saved updated reference star table to {outfile}")


#%%
#Look at just the good stars

optical_bands = ['B','V','R','I']

for band in optical_bands:

    file = f'{savedir}stacked_images_optical/{target}_{band}_stacked.fits'

    if not os.path.exists(file):
        print(f'Missing {file}, skipping')
        continue

    print(f'Plotting {band}')

    # -----------------------------
    # Load stacked FITS
    # -----------------------------
    data = fits.getdata(file)

    #get wcs from the wcs fits, which should be identical
    wcs = WCS(fits.getheader(glob.glob(f'/neta/xrb/{target}/product/*wcs.fits')[0]))

    pixscale_deg = proj_plane_pixel_scales(wcs)
    pixscale_arcsec = pixscale_deg * 3600

    ny, nx = data.shape

    width_arcsec = nx * pixscale_arcsec[0]
    height_arcsec = ny * pixscale_arcsec[1]

    width_arcmin = width_arcsec / 60
    height_arcmin = height_arcsec / 60


    # -----------------------------
    # Quality masks
    # -----------------------------
    quality_col = f'{band}_quality'

    if quality_col not in refstars.columns:
        print(f'Missing {quality_col}, skipping')
        continue


    good_mask = (
        (refstars[quality_col] == 'good') &
        (refstars['type'] != 'target')
    )

    target_mask = refstars['type'] == 'target'


    # -----------------------------
    # Display
    # -----------------------------
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    norm = ImageNormalize(
        vmin=vmin,
        vmax=vmax,
        stretch=AsinhStretch()
    )


    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(
        data,
        cmap='gray_r',
        origin='lower',
        norm=norm
    )


    # -----------------------------
    # Good stars with EIDs
    # -----------------------------
    good_rows = refstars.loc[good_mask]

    ax.scatter(
        good_rows['xpix'],
        good_rows['ypix'],
        s=25,
        marker='x',
        color='gold',
        linewidths=1,
        label='Good stars'
    )


    for eid, row in good_rows.iterrows():

        ax.annotate(
            str(eid),
            xy=(row['xpix'], row['ypix']),
            xytext=(5,5),
            textcoords='offset points',
            color='dodgerblue',
            fontsize=8,
            alpha=0.9
        )


    # -----------------------------
    # Target
    # -----------------------------
    target_rows = refstars.loc[target_mask]

    if len(target_rows) > 0:

        target_row = target_rows.iloc[0]

        ax.annotate(
            target,
            xy=(target_row['xpix'], target_row['ypix']),
            xytext=(-25,20),
            textcoords='offset points',
            color='red',
            fontsize=12,
            fontweight='bold',
            arrowprops=dict(
                arrowstyle='-',
                color='red',
                lw=2
            )
        )


    ax.set_title(
        f'{target} {band} Finding Chart, FOV: {width_arcmin:.2f}$^\\prime$ x {height_arcmin:.2f}$^\\prime$'
    )

    ax.axis('off')
    plt.tight_layout()

    # optional save
    # plt.savefig(
    #     f'{savedir}/finding_charts_optical/{target}_1.3m_{band}_finding_chart.png',
    #     dpi=200,
    #     bbox_inches='tight'
    # )

    plt.show()

