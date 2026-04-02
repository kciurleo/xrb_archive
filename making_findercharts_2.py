#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:23:50 2026

@author: kmc249
"""

###run after the first making_findercharts.py

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


for target in ['IGR17098-3628', 'IGRJ17454', 'IGRJ17544-2619',
'IGR17597-2201', 'J004308.6+411247', 'J1626.6', 'J165408-395636', 'J174716.1-281048', 'MMVel',
'V1408Aql', 'X0614+091']:#xrb_list:
    print(target)
    skipthis_target = False
    try:
        txtfile = glob.glob(f'/neta/xrb/{target}/temp/{target}*.txt')[0]
        data = {}
        with open(txtfile) as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue

                key, value = line.split(':', 1)

                data[key] = value

        most_files_band = data.get('band')
        ref = data.get('reference_image')
        x_pixel = float(data.get('x_pixel'))
        y_pixel = float(data.get('y_pixel'))
        print('skipped')

    except:
        print('no/bad txt file. stacking to image.')
        path=f'/neta/xrb/{target}/1.3m/opt/rccd/'
        #find which optical band has the most data in 1.3m and rccd
        counts = {}
        for band in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(band):
                n_files = len(glob.glob(os.path.join(band, '*')))
                counts[os.path.basename(band)] = n_files
        try:
            most_files_band = max(counts, key=counts.get)
        except:
            print('error with max band: ', counts)
            continue

        print(f'Most files in: {most_files_band}')
        try:
            data=fits.getdata(f'/neta/xrb/{target}/temp/{target}_{most_files_band}_dumbstack.fits')
            print('dumbstack exists. skipping')
            continue
        except:
            print('no dumbstack exists. running.')
        happy=''
        while not 'y' in happy:
            #pull a random image from that filter to use as a reference image. if bad, move to the next image
            filelist=glob.glob(path+f'/{most_files_band}/*')
            print(f'filelist is {len(filelist)} files long')
            found_ref=False
            idx=0
            while not found_ref:
                ref=filelist[idx]
                
                data=fits.getdata(ref)
            
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(data)
                norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
                plt.figure(figsize=(10,10))
                plt.imshow(data, cmap='gray', origin='lower', norm=norm)
                plt.show()
                found_inp=input('is this an okay reference? (y/n)')
                if 'y' in found_inp or 'Y' in found_inp:
                    found_ref=True
                else:
                    idx+=1
            print(f'found a ref image! {ref}')
            
            #do some stacking
            #get base info
            IM=fits.getdata(ref)
            HDR=fits.getheader(ref)
            
            cutto=input('cut to a specific number? type 0 if no cut')
            if cutto!='0':
                filelist=filelist[:int(cutto)]
                
    
            #align and trim
    
            image_stack = np.empty([np.shape(IM)[0],np.shape(IM)[1],len(filelist)])
            center_h, center_w = np.array(IM.shape)//2
            for id, file in enumerate(filelist):
                im,hdr = fits.getdata(file,header=True)
                #do the fft crossimaging, centered on the middle
    
                xshift, yshift = cross_image(IM, im, center_h, center_w, boxsize=400)
                if abs(xshift) > 300 or abs(yshift) > 300:
                    print(f"Skipping {file}: shift too large (x={xshift}, y={yshift})")
                    continue
                newimg=shift_image(im,xshift,yshift)
                #trimimg = newimg[center_h-256:center_h+256, center_w-256:center_w+256]
                try:
                    image_stack[:,:,id]=newimg
                except:
                    print('weird shape: ',newimg.shape)
                    continue
    
            #median image                    
            median_image = np.nanmedian(image_stack, axis=2)
    
    
            #Plotting just to tell
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(median_image)
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
            plt.figure(figsize=(10,10))
            plt.imshow(median_image, cmap='gray', origin='lower', norm=norm)
            plt.show()
            
            happy=input('happy? y/n')
            if 'n' in happy:
                skipthis_input=input('skip this guy? y/n')
                if 'y' in skipthis_input:
                    skipthis_target = True
                    break  # break out of the happy loop
        if skipthis_target:
            print(f'Skipping target {target}')
            continue
        #add key word to nw HDR
        HDR['STACK']=True
        #save image and also save log of which fits files we threw into this image
        fits.writeto(f'/neta/xrb/{target}/temp/{target}_{most_files_band}_dumbstack.fits',median_image,header=HDR, overwrite=True)
        print('saved fits ')