#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:09:03 2026

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

#aperture radius
r_ap = 6.0
fwhm=3.0
target='J1752-223'

outdir = f'/neta/xrb/{target}/product/first_pass_lightcurves'
os.makedirs(outdir, exist_ok=True)

#reference stars
reffile=glob.glob(f'/neta/xrb/{target}/product/*_ref_stars_kciurleo.csv')[0]
refstars=pd.read_csv(reffile)
eids = refstars.loc[refstars['type'] != 'target'].index.tolist()
cols=['filename','time', 'target'] + [str(e) for e in eids]

#label column
source_ids = []

for _, row in refstars.iterrows():
    if row['type'] == 'target':
        source_ids.append('target')
    else:
        #source_ids.append(str(row['objid']))
        source_ids.append(str(_))

#to iterate over
optical_bands=['B','V','R','I']
telescopes=['1.3m', '1m']

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

        #Get a file list
        filelist=sorted(glob.glob(f'{trimdir}*'))
        
        #initialize this df
        band_df = pd.DataFrame(0, index=np.arange(len(filelist)), columns=cols)
        band_df['filename'] = filelist
        band_df['time'] = pd.NaT 
        band_df['target'] = np.nan
        for e in eids:
            band_df[str(e)] = np.nan 
            
        for ind, row in band_df.iterrows():
            print(f'Working on {ind} out of {len(filelist)}')
            file=row['filename']
            #get data
            data=fits.getdata(file)
            hdr=fits.getheader(file)
            #hold onto time
            band_df.at[ind, 'time']=pd.to_datetime(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}")
            
            #background subtraction?
            #background subtract data
            sigma_clip=SigmaClip(sigma=3.0)
            bkg_estimator=MedianBackground()
            fullbkg=Background2D(data, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            bkg_sub_full_data=data-fullbkg.background
            bkg_sub_full_data = np.nan_to_num(bkg_sub_full_data, nan=0.0)
            
            
            #Let DAOstarfinder get actual x y coords with initial conditions?
            '''
            mean, median, std = sigma_clipped_stats(data, sigma=3, maxiters=5)
            
            xycoords = refstars[['xpix', 'ypix']].to_numpy()
            
            daofinder = DAOStarFinder(
                threshold=2 * std,
                fwhm=fwhm,
                xycoords=xycoords
            )
            
            sources = daofinder(bkg_sub_full_data)
            
            sources['source_id'] = source_ids
            
            # aperture photometry
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            '''
            #or do we just use the things we have right now?
            sources=refstars
            sources['source_id'] = source_ids
            positions = np.transpose((refstars['xpix'], refstars['ypix']))
            
            apertures = CircularAperture(positions, r=r_ap)
            
            aper_phot = aperture_photometry(bkg_sub_full_data, apertures)
            
            # attach fluxes to sources table
            sources['flux'] = aper_phot['aperture_sum']
            
            # plot to check apertures
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(data)
            
            norm = ImageNormalize(
                vmin=vmin,
                vmax=vmax,
                stretch=SinhStretch()
            )
            if ind==0:
                #let's only plot the first one for now
                plt.figure(figsize=(8, 8))
                plt.imshow(data, origin='lower', cmap='gray', norm=norm)
                
                # plot apertures
                apertures.plot(color='red', lw=1.5, alpha=0.8)
                
                # label IDs
                for i, (x, y) in enumerate(positions):
                
                    sid = sources['source_id'][i]
                
                    plt.annotate(
                        str(sid),
                        (x, y),
                        textcoords='offset points',
                        xytext=(5, 5),
                        color='yellow',
                        fontsize=8
                    )
                
                plt.show()
            
            #for src in sources: #put this back if you put back the DAOStarFinder thing
            for _, src in sources.iterrows():
                sid = src['source_id']
                flux = src['flux']
            
                band_df.at[ind, sid] = flux
        
        outfile = f'{outdir}/{target}_{tele}_{band}_first_pass_phot.csv'
        band_df.to_csv(outfile, index=False)
        
        print(f'Saved {outfile}')