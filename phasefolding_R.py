#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:21:41 2026

@author: kmc249
"""
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates
from astropy.io import fits
import datetime as dt

from astropy.timeseries import LombScargle

#read in 1.3 and 1 m stuff

table=pd.read_csv('/neta/xrb/AqlX-1/product/AqlX-1_R_corrected_lc.csv', low_memory=False)

#getting nice time
t = Time(table['MJD'], format='mjd')
table['nice time'] = t.to_datetime()



#cutting off 1m data for now
# define cutoff date
cutoff = pd.Timestamp('2003-01-01')

# keep only rows on or after 2003
table = table[table['nice time'] >= cutoff]




mask = np.zeros(len(table), dtype=bool)
mjd = np.asarray(table['MJD'], dtype=float)

#only get stuff in quiescence
quiescence=pd.read_csv('/home/kmc249/Downloads/quiescence_mjd_ranges_v5.csv')
for start, end in zip(quiescence['q_start_mjd'], quiescence['q_end_mjd']):
    mask |= (mjd >= start) & (mjd <= end)
table = table[mask]

#%%
###J band stuff
####HEY KATIE????
#mask anything sneaking in from outbursts?? or lower stuff?
table=table.loc[(table['Rmag']>17.5) & (table['Rmag']<18.5)]


#periodogramming things
baseline=table['nice time'].max()-table['nice time'].min()
base_days=baseline.total_seconds() / 3600 /24
print(base_days)

#folded??
P = 0.789498  # period in days
times = Time(table['nice time']).mjd
t0 = times.min()
phase = ((times - t0) / P) % 1
table['their phase']=phase

##periodograms
min_frequency = 24/19.5
max_frequency = 24/18.5

deltaf=P/base_days/4
print('DELTA F:', deltaf)
print(np.abs(min_frequency-max_frequency)/10000)

frequency = np.arange(min_frequency, max_frequency, deltaf)

fall, pall = LombScargle(times, table['Rmag']-np.nanmean(table['Rmag'])).autopower(maximum_frequency=2)
power = LombScargle(times, table['Rmag']-np.nanmean(table['Rmag'])).power(frequency)

# Convert frequency to period in hours
period_hours = 24 / frequency
sorted_idx = np.argsort(period_hours)
period_hours_sorted = period_hours[sorted_idx]
power_sorted = power[sorted_idx]

# Plot periodogram in period units
plt.figure(figsize=(8,4))
plt.plot(period_hours_sorted, power_sorted)
plt.xlabel('Period (hours)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.axvline(x=P*24,alpha=0.5, color='red')
plt.show(block=False)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(frequency, power)
plt.show(block=False)

# samme thing for fall pall
pall_hours = 24 / fall
sorted_idxall = np.argsort(pall_hours)
period_hours_sortedall = pall_hours[sorted_idxall]
power_sortedall = pall[sorted_idxall]

# Plot periodogram in period units
plt.figure(figsize=(8,4))
plt.plot(period_hours_sortedall, power_sortedall)
plt.xlabel('Period (hours)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram (all freq, capped)')
plt.axvline(x=P*24,alpha=0.5, color='red')
plt.xlim(12, 45)
plt.show(block=False)

fig, ax = plt.subplots()
ax.plot(fall, pall)
plt.show(block=False)

best_frequency = frequency[np.argmax(power)]
P2 = 1 / best_frequency
best_period_hours = P2 * 24
print(best_period_hours)

phase2 = ((times - t0) / P2) % 1
table['our phase']=phase2

# Number of bins
nbins = 16
bins = np.linspace(0, 1, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Assign each phase to a bin
table['our phase bin'] = pd.cut(table['our phase'], bins=bins, include_lowest=True, labels=bin_centers)
table['their phase bin'] = pd.cut(table['their phase'], bins=bins, include_lowest=True, labels=bin_centers)

# Compute mean and std per bin
binned = table.groupby('our phase bin')['Rmag'].agg(['mean','std']).reset_index()
binned_them = table.groupby('their phase bin')['Rmag'].agg(['mean','std']).reset_index()

yerr_up_us = []
yerr_down_us = []

for center in binned['our phase bin']:
    # Get all points in this phase bin
    points = table.loc[table['our phase bin'] == center, 'Rmag'].values
    if len(points) == 0:
        yerr_up_us.append(0)
        yerr_down_us.append(0)
        continue
    
    mean_bin = np.mean(points)
    
    # Points above/below mean
    above = points[points > mean_bin]
    below = points[points < mean_bin]
    
    # 68% confidence ~ 1 sigma
    sigma_up = np.percentile(above, 68.3) - mean_bin if len(above) > 0 else 0
    sigma_down = mean_bin - np.percentile(below, 31.7) if len(below) > 0 else 0
    
    yerr_up_us.append(sigma_up)
    yerr_down_us.append(sigma_down)

# Make 2×N array for asymmetric error bars
yerr_asym_us = np.array([yerr_down_us, yerr_up_us])

yerr_up_them = []
yerr_down_them = []

for center in binned['our phase bin']:
    # Get all points in this phase bin
    points = table.loc[table['our phase bin'] == center, 'Rmag'].values
    if len(points) == 0:
        yerr_up_them.append(0)
        yerr_down_them.append(0)
        continue
    
    mean_bin = np.mean(points)
    
    # Points above/below mean
    above = points[points > mean_bin]
    below = points[points < mean_bin]
    
    # 68% confidence ~ 1 sigma
    sigma_up = np.percentile(above, 68.3) - mean_bin if len(above) > 0 else 0
    sigma_down = mean_bin - np.percentile(below, 31.7) if len(below) > 0 else 0
    
    yerr_up_them.append(sigma_up)
    yerr_down_them.append(sigma_down)

# Make 2×N array for asymmetric error bars
yerr_asym_them = np.array([yerr_down_them, yerr_up_them])

# Plot with asymmetric error bars
plt.figure(figsize=(8,4))
plt.scatter(phase2, table['Rmag'], s=15, color='gray', label='Data')
plt.scatter(phase2 + 1, table['Rmag'], s=15, color='gray', alpha=0.5)
plt.errorbar(binned['our phase bin'].astype(float), binned['mean'], yerr=yerr_asym_us,
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned['our phase bin'].astype(float)+1, binned['mean'], yerr=yerr_asym_us,
             fmt='o', color='red', alpha=0.5)
plt.xlabel('Orbital Phase')
plt.ylabel('R mag')
plt.gca().invert_yaxis()
plt.title(f'Our Period: {best_period_hours} hrs')
plt.legend()
plt.tight_layout()

plt.show(block=False)


# Plot
plt.figure(figsize=(8,4))
plt.scatter(phase, table['Rmag'], s=15, color='gray', label='Data')
plt.scatter(phase + 1, table['Rmag'], s=15, color='gray', alpha=0.5)

plt.errorbar(binned_them['their phase bin'].astype(float), binned_them['mean'], yerr=yerr_asym_them,
             fmt='o', color='red', label='Binned Avg')
plt.errorbar(binned_them['their phase bin'].astype(float)+1, binned_them['mean'], yerr=yerr_asym_them,
             fmt='o', color='red', alpha=0.5)

plt.xlabel('Orbital Phase')
plt.ylabel('R mag')
plt.gca().invert_yaxis()
plt.legend()
plt.title(f'Their Period: {P*24} hrs')
plt.tight_layout()
plt.show()

#%%
#crappy 2-sin cruve fit
from scipy.optimize import curve_fit
import numpy as np

# Our binned data
xdata = binned['our phase bin'].astype(float).values
ydata = binned['mean'].values
yerr = binned['std'].values

# Define a two-sine model
def ellipsoidal_model(phase, A1, A2, delta, m0):
    return A1*np.sin(2*np.pi*phase) + A2*np.sin(4*np.pi*phase + delta) + m0

# Initial guesses
A1_guess = 0.05  # small amplitude at orbital frequency
A2_guess = 0.1   # ellipsoidal amplitude
delta_guess = 0
m0_guess = np.mean(ydata)

p0 = [A1_guess, A2_guess, delta_guess, m0_guess]

# Fit
popt, pcov = curve_fit(ellipsoidal_model, xdata, ydata, sigma=yerr, p0=p0)

# Extract parameters
A1_fit, A2_fit, delta_fit, m0_fit = popt
print(f"A1 = {A1_fit:.3f}, A2 = {A2_fit:.3f}, delta = {delta_fit:.3f}, mean mag = {m0_fit:.3f}")

# Plot
phase_fit = np.linspace(0, 1, 500)
mag_fit = ellipsoidal_model(phase_fit, *popt)

plt.figure(figsize=(8,4))
plt.errorbar(xdata, ydata, yerr=yerr, fmt='o', color='red', label='Binned data')
plt.plot(phase_fit, mag_fit, color='blue', label='Ellipsoidal fit')
plt.xlabel('Orbital Phase')
plt.ylabel('R mag')
plt.gca().invert_yaxis()
plt.legend()
plt.title('us')
plt.tight_layout()
plt.show(block=False)


# their binned data
xdata = binned_them['their phase bin'].astype(float).values
ydata = binned_them['mean'].values
yerr = binned_them['std'].values

# Fit
popt, pcov = curve_fit(ellipsoidal_model, xdata, ydata, sigma=yerr, p0=p0)

# Extract parameters
A1_fit, A2_fit, delta_fit, m0_fit = popt
print(f"A1 = {A1_fit:.3f}, A2 = {A2_fit:.3f}, delta = {delta_fit:.3f}, mean mag = {m0_fit:.3f}")

# Plot
phase_fit = np.linspace(0, 1, 500)
mag_fit = ellipsoidal_model(phase_fit, *popt)

plt.figure(figsize=(8,4))
plt.errorbar(xdata, ydata, yerr=yerr, fmt='o', color='red', label='Binned data')
plt.plot(phase_fit, mag_fit, color='blue', label='Ellipsoidal fit')
plt.xlabel('Orbital Phase')
plt.ylabel('R mag')
plt.gca().invert_yaxis()
plt.legend()
plt.title('them')
plt.tight_layout()
plt.show()


#%%
from scipy.ndimage import zoom

keep=set(pd.read_csv('/home/kmc249/Downloads/NEWEST_aql_R_600.0_list.csv')['filename'])
import os
'''
filtered_table = table[
    table['filename'].apply(os.path.basename).isin(keep)
]
'''
filtered_table=table
filenames_by_bin = filtered_table.groupby('our phase bin')['filename'].apply(list)

for phase_bin, files in filenames_by_bin.items():
    print(f"\nPhase bin: {phase_bin}")
    
    kept_images = []
    HDR = fits.getheader(files[0])
    HDR['PHASE'] = phase_bin

    for file in files:
        im = fits.getdata(file)
        hdr = fits.getheader(file)
        im_norm = im / hdr['EXPTIME']
        
        kept_images.append(im_norm)


    if len(kept_images) == 0:
        print("No images kept for this bin, skipping...")
        continue

    # stack only kept images
    image_stack = np.stack(kept_images, axis=2)
    median_image = np.median(image_stack, axis=2)

    median_image = zoom(median_image, 2, order=3)

    fits.writeto(
        f'/home/kmc249/Downloads/Aql_us_phase_{phase_bin}.fits',
        median_image,
        header=HDR,
        overwrite=True
    )    
        
        
        
filenames_by_bin_them = filtered_table.groupby('their phase bin')['filename'].apply(list)

for phase_bin, files in filenames_by_bin_them.items():
    print(f"\nPhase bin: {phase_bin}")
    
    kept_images = []
    HDR = fits.getheader(files[0])
    HDR['PHASE'] = phase_bin

    for file in files:
        im = fits.getdata(file)
        hdr = fits.getheader(file)
        im_norm = im / hdr['EXPTIME']
        
        kept_images.append(im_norm)


    if len(kept_images) == 0:
        print("No images kept for this bin, skipping...")
        continue

    # stack only kept images
    image_stack = np.stack(kept_images, axis=2)
    median_image = np.median(image_stack, axis=2)

    median_image = zoom(median_image, 2, order=3)

    fits.writeto(
        f'/home/kmc249/Downloads/Aql_us_phase_{phase_bin}.fits',
        median_image,
        header=HDR,
        overwrite=True
    )
    fits.writeto(f'/home/kmc249/Downloads/Aql_them_phase_{phase_bin}.fits',median_image,header=HDR, overwrite=True)
        
 #%%
 
#doing psf phot on each image

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from astropy.table import QTable, Table, vstack
from scipy.ndimage import map_coordinates
from photutils.psf import extract_stars
from astropy.nddata import NDData
from astropy.wcs import WCS
from smith_utils import *
from photutils.datasets import make_model_image
import glob
from scipy.optimize import curve_fit
import json
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord



#%%

#load locations of good sources for ePSF
sources=Table.read('/home/kmc249/Downloads/good_sources.vot')
res=2
#filelist=glob.glob('/home/kmc249/Downloads/Aql_us_phase*')
filelist=glob.glob('/home/kmc249/Downloads/Aql_them_phase*')
testimage=fits.getdata(filelist[0])

#put in correct x-y and get rid of epsf guys we won't use
sources["xcentroid"]=sources["xcentroid"]-301
sources["ycentroid"]=sources["ycentroid"]-301
sources["x"]=sources["x"]-301
sources["y"]=sources["y"]-301
mask=((sources['xcentroid']>= 0) &
    (sources['xcentroid']< 512) &
    (sources['ycentroid']>= 0) &
    (sources['ycentroid']< 512))
sources=sources[mask]

sources["xcentroid"]=sources["xcentroid"]*res
sources["ycentroid"]=sources["ycentroid"]*res
sources["x"]=sources["x"]*res
sources["y"]=sources["y"]*res


#info about the neighbor
neighborhood=pd.read_csv('/home/kmc249/current_best_R_grid_fit.csv')
stacked_ensemble=pd.read_csv("/home/kmc249/best_r_ensemble.csv")

#put all the vots in the smaller x-y system, from tthe resolution of x2
#and the neighborhood from the really zoomed in to the normal 512 x 512
neighborhood["x_fit"]=neighborhood["x_fit"]+2*231
neighborhood["y_fit"]=neighborhood["y_fit"]+2*231

#init params to fit the neighborhood, NO FLUX INIT
init_params=neighborhood[['id','x_fit','y_fit','y_err', 'x_err']]
init_params = Table.from_pandas(init_params)



#ensemble fluxes to use for std
nb=[]
aql=[]

#make df
cols=['filename','phase bin', 'a','e']
big_df = pd.DataFrame(0, index=np.arange(len(filelist)), columns=cols)
big_df['filename'] = filelist
big_df['phase bin'] = np.nan
big_df['a'] = np.nan
big_df['e'] = np.nan
big_df['a_err'] = np.nan
big_df['e_err'] = np.nan

showplot=True
nonexistent=[]
problems=[]
for ind, row in big_df.iterrows():
    print(f'working on {ind} of {len(filelist)}')
    file=row['filename']
    print('trying', file)
    try:
        imdata,hdr = fits.getdata(file,header=True)
    except:
        nonexistent.append(file)
        continue
    big_df.at[ind, 'phase bin']=hdr['PHASE']
    print(hdr['PHASE'])

    
    #background subtract data
    sigma_clip=SigmaClip(sigma=3.0)
    bkg_estimator=MedianBackground()
    fullbkg=Background2D(imdata, (20*res,20*res), filter_size=(3*res+1,3*res+1),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkg_sub_full_data=imdata-fullbkg.background
    bkg_sub_full_data = np.nan_to_num(bkg_sub_full_data, nan=0.0)

    #plot sources on data just to double check

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(bkg_sub_full_data)
    vmin2, vmax2 = interval.get_limits(testimage)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
    #plt.imshow(testimage, cmap='viridis', origin='lower', norm=norm2, alpha=0.5)
    plt.scatter(init_params['x_fit'], init_params['y_fit'], marker='x')
    plt.scatter(sources['xcentroid'], sources['ycentroid'], marker='+', c='gold')

    for row in init_params:
        plt.annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )
    plt.show()

    #we need to get rid of the guys that are outside of this image if it's majorly shifted!! Some buffer around,
    #the edges are wonky....or we just trim everything to max shift. temp fix noted below

    #use the given positions of the good PSF stars to generate a new EPSF
    size=21*res+1
    nddata=NDData(data=bkg_sub_full_data)
    good_stars=extract_stars(nddata, sources, size=size)

    #temp fix:
    # Filter out stars with invalid or zero flux

    valid_stars = [star for star in good_stars 
                if np.isfinite(np.sum(star.data)) and np.sum(star.data) > 0]
    
    # filter out stars with the wrong shape
    valid_stars = [star for star in valid_stars if star.data.shape == (size, size)]


    try:
        epsf_input = EPSFStars(valid_stars)
        epsf_builder=EPSFBuilder(oversampling=2, maxiters=10)
        epsf, fitted_stars = epsf_builder(epsf_input)
    except:
        problems.append(file)
        continue
    

    #plot epsf just to see if it worked

    norm=simple_norm(epsf.data, 'log', percent=99.0)

    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='gray')
    plt.show()

    #psf fitting, using init params
    psf_model = epsf
    fitnum=12
    fit_shape=(fitnum*res+1,fitnum*res+1)
    #grouper=SourceGrouper(min_separation=8)
    psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=10, xy_bounds=0.4)
    
    #make the xy pixel fit dist. 0.1, avg error in original fit so it doesn't move
    try:
        phot = psfphot(bkg_sub_full_data, init_params=init_params)
    except ValueError as e:
        raise e

    resid=psfphot.make_residual_image(bkg_sub_full_data)
    model=psfphot.make_model_image(np.shape(bkg_sub_full_data))

    label=True
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(bkg_sub_full_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    fig, axes=plt.subplots(1,3, figsize=(20,10))
    axes[0].imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
    axes[0].scatter(phot['x_fit'], phot['y_fit'], marker='x')
    axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
    axes[2].imshow(resid, cmap='gray', origin='lower', norm=norm)
    if label:
        for row in phot:
            axes[0].annotate(
                str(row['id']), 
                (row['x_fit'], row['y_fit']),
                textcoords="offset points",
                xytext=(5,5),
                fontsize=8,
                color='red'
            )

    plt.show()
    
    row1 = phot[phot['id'] == 2][0]
    row2 = phot[phot['id'] == 4][0]
    
    big_df.at[ind, 'a'] = row1['flux_fit']
    big_df.at[ind, 'a_err']  = row1['flux_err']
    
    big_df.at[ind, 'e'] = row2['flux_fit']
    big_df.at[ind, 'e_err'] = row2['flux_err']
    
    
#%%

for id, row in big_df.iterrows():
    print(row)

plt.figure(figsize=(10,6))
#plt.scatter(big_df['phase bin'], big_df['a'])
#plt.scatter(big_df['phase bin'], big_df['e'])
plt.scatter(big_df['phase bin'], big_df['e']/(big_df['a']+big_df['e']))
plt.show()