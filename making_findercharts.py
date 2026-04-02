#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:15:56 2026

@author: kmc249
"""
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


#for each target, 
dumb_run=True #if you specifically want to run this on the dumb stacked objects
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

                data[key] = value

        most_files_band = data.get('band')
        ref = data.get('reference_image')
        x_pixel = float(data.get('x_pixel'))
        y_pixel = float(data.get('y_pixel'))
        print('exists - skipped')
        continue
    except:
        print('no basic finding info exists. running.')
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
    
    #pull a random image from that filter to use as a reference image. if bad, move to the next image
    filelist=glob.glob(path+most_files_band+'/*')
    found_ref=False
    if dumb_run:
        try:
            ref=f'/neta/xrb/{target}/temp/{target}_{most_files_band}_dumbstack.fits'
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
        except:
            print('no dumb stack found. skipping')
            continue
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
    print(f'running daophot')
    happy_flag=False
    fwhm = 6.
    thresh=5
    while not happy_flag:
        #run dao starfinder on it and plot the results
        mean, median, std = sigma_clipped_stats(data, sigma=3, maxiters=5)
        mask = np.zeros_like(data, dtype=bool)
        mask[:50, :] = mask[-50:, :] = mask[:, :50] = mask[:, -50:] = 1.0
        daofinder = DAOStarFinder(threshold=thresh * std, fwhm=fwhm)
        sources = daofinder(data, mask=mask)
        
        
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=3 * fwhm)
    
        plt.figure(figsize=(10,10))
        plt.imshow(data, cmap='gray', origin='lower', norm=norm)
        apertures.plot(color="red")
        for i, (x, y) in enumerate(positions):
            plt.text(x+8, y+8, str(i),
                     color='yellow',
                     fontsize=10,
                     ha='center', va='center')
        plt.show()
        star_id=int(input('Which index is the star?'))

    
        #ask user input: which id is the star?
        x, y = positions[star_id]
        star_aperture = CircularAperture((x, y), r=3 * fwhm)
        
        plt.figure(figsize=(10,10))
        plt.imshow(data, cmap='gray', origin='lower', norm=norm)
        plt.title(target)
        star_aperture.plot(color='red', lw=2)
        
        plt.show()
    
        happy=input('happy with this? (y/n)')
        
        if 'y' in happy or 'Y' in happy:
            happy_flag=True
            plt.figure(figsize=(10,10))
            plt.imshow(data, cmap='gray', origin='lower', norm=norm)
            
            star_aperture.plot(color='red', lw=2)
            plt.title(target)
            plt.savefig(f'/neta/xrb/{target}/temp/{target}_{most_files_band}_unstacked_finding_chart.png', dpi=300)
            plt.close()
            #in f'/neta/xrb/{target}/temp' save a txt file with the reference image full file path and the (x,y)
            #tuple pixel location of the object
            # save reference info
            
            txtfile = f'/neta/xrb/{target}/temp/{target}_{most_files_band}_ref.txt'
            
            with open(txtfile, 'w') as f:
                f.write(f'band: {most_files_band}\n')
                f.write(f'reference_image: {ref}\n')
                f.write(f'x_pixel: {x:.2f}\n')
                f.write(f'y_pixel: {y:.2f}\n')
            
            print(f'Saved files for {target}.')

        else:
            happy_flag=False
            skip_obj=input('skip this object? (y/n)')
            if 'y' in skip_obj or 'Y' in skip_obj:
                break
            fwhm = float(input('new fwhm value for daostarfinder? (6. was original)'))
            thresh = float(input('new thresh*std value for daostarfinder? (5 was original)'))

    

