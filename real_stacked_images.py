#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:59:53 2026

@author: kmc249
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from photutils.psf import fit_fwhm
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch


###per band, I will make stacked images here.


target='NovaMusca'
fractions = {
    'B': 0.3,
    'V': 0.3,
    'R': 0.3,
    'I': 0.3
}
  #fraction of best frames to take in the stack, default of 0.3

savedir='/neta/xrb/PRODUCTS/'
indir=f'/neta/xrb/{target}/'

reffile=glob.glob(f'/neta/xrb/{target}/product/*_ref_stars_kciurleo.csv')[0]
refstars=pd.read_csv(reffile)
all_eids = refstars.loc[refstars['type'] != 'target'].index.tolist()

telescopes=['1m', '1.3m']

bands=['B','V','R','I']



#get seeing from fwhm helper from photutils
def measure_seeing(data, refstars, fit_shape=15):
    refs=refstars.loc[refstars['type'] != 'target']
    
    # subtract background
    data = data - np.nanmedian(data)

    xypos = list(
        zip(
            refs['xpix'].values,
            refs['ypix'].values
        )
    )

    try:

        fwhms = fit_fwhm(
            data,
            xypos=xypos,
            fit_shape=fit_shape,
            fwhm=3
        )

    except Exception as e:
        print("FWHM fitting failed:", e)
        return np.nan


    # remove failures
    fwhms = fwhms[np.isfinite(fwhms)]
    fwhms = fwhms[fwhms > 0]


    if len(fwhms)==0:
        return np.nan

    return np.median(fwhms)


#measure countrate
def measure_counts(data, refstars, exptime,stamp_size=15):

    fluxes=[]

    half=stamp_size//2


    for _, row in refstars.iterrows():

        x=int(row.xpix)
        y=int(row.ypix)


        stamp=data[
            y-half:y+half+1,
            x-half:x+half+1
        ]


        if stamp.shape != (stamp_size,stamp_size):
            continue


        sky=np.median(stamp)

        flux=np.sum(stamp-sky)


        if np.isfinite(flux) and flux>0:
            fluxes.append(flux/exptime)


    if len(fluxes)==0:
        return np.nan


    return np.median(fluxes)


#take ALL the trimmed files:
quality_tables = {}

for band in bands:

    filelist1 = glob.glob(f'{indir}/1.3m/opt/rccd/{band}_trimmed/*')
    filelist2 = glob.glob(f'{indir}/1m/opt/rccd/{band}_trimmed/*')

    filelist = filelist1 + filelist2

    if len(filelist) == 0:
        continue

    quality = []

    for i, filename in enumerate(filelist):

        print(f'{band}: {i+1}/{len(filelist)}')

        data = fits.getdata(filename)
        exptime = fits.getheader(filename)['EXPTIME']

        seeing = measure_seeing(data, refstars)
        counts = measure_counts(data, refstars, exptime)

        quality.append({
            'file': filename,
            'seeing': seeing,
            'counts': counts
        })

    quality = pd.DataFrame(quality).dropna()

    quality['seeing_score'] = quality['seeing'].min() / quality['seeing']
    quality['count_score'] = quality['counts'] / quality['counts'].max()
    quality['score'] = quality['seeing_score'] * quality['count_score']

    quality = quality.sort_values('score', ascending=False)

    quality_tables[band] = quality

    print(f'\nBest {band} frames')
    print(quality.head())
#%%    
for band in bands:

    if band not in quality_tables:
        continue

    quality = quality_tables[band]

    fraction = fractions[band]

    nkeep = max(1, int(len(quality) * fraction))

    best_images = quality.head(nkeep)['file'].values

    print(f'{band}: stacking {nkeep} images')
    
    #sigma stack
    images=[]
    
    for f in best_images:
    
        images.append(
            fits.getdata(f)
        )
    
    
    images=np.array(images)
    
    
    clipped = sigma_clip(
        images,
        sigma=3,
        axis=0
    )
    
    stacked = np.ma.median(
        clipped,
        axis=0
    )
    
    # convert masked array -> normal ndarray
    stacked = np.asarray(stacked)
    
    #save and plot
    header=fits.getheader(best_images[0])
    header['STACKED']='True'
    header['NUM_IMS']=len(best_images)
    header['STACK_FRAC']=fraction
    

    outfile=f'{savedir}/stacked_images_optical/{target}_{band}_stacked.fits'
    
    
    fits.writeto(outfile, stacked, header=header, overwrite=True)
    
    
    print('Saved:',outfile)
    

    
    # ============================================================
    # PLOT STACK
    # ============================================================
    
    interval=ZScaleInterval()
    
    vmin,vmax=interval.get_limits(
        stacked
    )
    
    
    norm=ImageNormalize(
        vmin=vmin,
        vmax=vmax,
        stretch=AsinhStretch()
    )
    
    
    plt.figure(figsize=(10,10))
    
    plt.imshow(
        stacked,
        cmap='gray_r',
        origin='lower',
        norm=norm
    )
    
    plt.title(
        f'{target} {band} best seeing stack\n'
        f'{len(best_images)} images'
    )
    
    plt.axis('off')
    
    plt.show()
    
    
    
    # ============================================================
    # PLOT WITH REFERENCE STARS
    # ============================================================
    
    plt.figure(figsize=(10,10))
    
    
    plt.imshow(
        stacked,
        cmap='gray_r',
        origin='lower',
        norm=norm
    )
    
    
    plt.scatter(
        refstars.xpix,
        refstars.ypix,
        facecolors='none',
        edgecolors='red',
        s=30
    )
    
    
    for eid,row in refstars.iterrows():
    
        plt.annotate(
            str(eid),
            (row.xpix,row.ypix),
            xytext=(3,3),
            textcoords='offset points',
            color='yellow',
            fontsize=7
        )
    
    
    plt.title(
        f'{target} {band} best seeing stack with EIDs'
    )
    
    plt.axis('off')
    
    plt.show()
