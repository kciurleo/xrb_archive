#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:50:14 2026

@author: kmc249
"""

from astrometry_helper import *
from lookup_name import *
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
import os
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
import pandas as pd
from smith_utils import *
from photutils.psf import fit_fwhm
from astropy.time import Time

### THIS ONE NEEDS EDITING TO DEAL WITH WEIRD PIXEL CONVERSION STUFF DEPENDING ON SIZE
##and also should be edited to skip over objects with these files already

#quantile window to choose for "good seeing"
q1, q2 = 0.0, 0.45
errors=[]
for target in xrb_list:
    print(target)
    try:
        txtfile = glob.glob(f'/neta/xrb/{target}/temp/{target}*.txt')[0]
        data = {}
        with open(txtfile) as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                key, value = line.split(':', 1)

                data[key.strip()] = value.strip()

        most_files_band = data.get('band')
        ref = data.get('reference_image')
        x_pixel = float(data.get('x_pixel'))
        y_pixel = float(data.get('y_pixel'))
        print(x_pixel, y_pixel)

    except:
        print('no ref info found. skipping')
        continue
    try:
        filelist=glob.glob(f'/neta/xrb/{target}/1.3m/opt/rccd/{most_files_band}/*')
        print(f'filelist is {len(filelist)} files long')
    
            
        data=fits.getdata(ref)
        
        #do some stacking
        #get base info
        IM=fits.getdata(ref)
        HDR=fits.getheader(ref)
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(IM)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    
        plt.imshow(IM, cmap='gray', origin='lower', norm=norm)
    
        plt.scatter(x_pixel,y_pixel, color='red', marker='x', s=3)
        plt.show()
        #align and trim, find fwhm of a handful of stars
        fwhms=pd.DataFrame(columns=['filename', 'maxshift', 'xshift', 'yshift', 'fwhm'])
    
        print('initial align and trim')
        for id, file in enumerate(filelist):
            fwhms.at[id, 'filename']=file
            try:
                im,hdr = fits.getdata(file,header=True)
                mjd=Time(hdr['DATE-OBS']).mjd
                fwhms.at[id, 'exptime']=hdr['EXPTIME']
                fwhms.at[id, 'mjd']=mjd
            except:
                print('bad file', file)
                continue
            #do the fft crossimaging, centered on the star
            xshift, yshift = cross_image(IM, im, int(x_pixel), int(y_pixel), boxsize=400)
            newimg=shift_image(im,xshift,yshift)
            f = fit_fwhm(newimg, xypos=[(x_pixel, y_pixel)], fit_shape=7)
    
            fwhms.at[id, 'fwhm']=f
            fwhms.at[id, 'xshift']=xshift
            fwhms.at[id, 'yshift']=yshift
    
        #pick some percentiles; was gonna do avg and std away, but we have a very skewed distribution here
        p20 = fwhms['fwhm'].quantile(q1)
        p40 = fwhms['fwhm'].quantile(q2)
    
        #mask out so we have just the ones w/fwhms in this range
        mask = ((fwhms['fwhm'] >= p20) & (fwhms['fwhm'] <= p40))
        winners = fwhms[mask]
        winners.reset_index(inplace=True)
        print(f'{len(winners)} files found for deep stack.')
    
        #realign just those of interest, stack, trim, save
        IM, HDR = fits.getdata(winners['filename'][0],header=True)
        image_stack = np.full((1024, 1024, len(winners)), np.nan)
        xshifts = {}
        yshifts = {}
    
        #make stack/realign
        print('final align and stack')
        for id, row in winners.iterrows():
            im,hdr = fits.getdata(row['filename'],header=True)
            im=im[:1024,:1024]
            #do the fft crossimaging, centered on the star
            xshift, yshift = cross_image(IM, im, int(x_pixel), int(y_pixel), boxsize=400)
            newimg=shift_image(im,xshift,yshift)
            image_stack[:,:,id] = newimg
    
            
    
        #median image                    
        median_image = np.nanmedian(image_stack, axis=2)
    
        #add key word to nw HDR
        HDR['STACK']=True
        #save image and also save log of which fits files we threw into this image
        fits.writeto(f'/neta/xrb/{target}/temp/{target}_{most_files_band}_sigmastack.fits',median_image,header=HDR, overwrite=True)
        make_wcs_fits(f'/neta/xrb/{target}/temp/{target}_{most_files_band}_sigmastack.fits', f'/neta/xrb/{target}/product/{target}_{most_files_band}_wcs.fits')
    
        #Plotting just to tell
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(median_image)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    
        plt.imshow(median_image, cmap='gray', origin='lower', norm=norm)
    
        plt.scatter(x_pixel,y_pixel, color='red', marker='x', s=3)
        plt.show()
    except: 
        errors.append(target)

print(errors)

#make_wcs_fits(ref, output_path)