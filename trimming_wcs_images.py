#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:40:30 2026

@author: kmc249
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
import astroalign as aa
from pathlib import Path
from lookup_name import *
from astropy.coordinates import SkyCoord
import astropy.units as u
from astrometry_helper import *
from pathlib import Path
import re

#This JUST makes a trimmed wcs, providing a ref image exists
def make_trimmed_path(untrimmed_ref):
    p = Path(untrimmed_ref)

    band = p.parent.name
    fname = p.name

    new_dir = p.parent.parent / f"{band}_trimmed"
    new_name = f"trim_{fname}"

    return str(new_dir / new_name)


overwrite=True

#loop
for target in ['LMCX-3']:
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
        untrimmed_ref = data.get('reference_image')
        x_pixel = float(data.get('x_pixel'))
        y_pixel = float(data.get('y_pixel'))
        print(x_pixel, y_pixel)
    except:
        print(f'Skipped {target}')
        continue
    print(untrimmed_ref)
    try:
        #data and 512x512 cutout
        ref=make_trimmed_path(untrimmed_ref)
        IM=fits.getdata(ref)
        HDR=fits.getheader(ref)
    except:
        #If the above doesn't work, it's because there's a dumbstack thing instead. So let's trim it and call it okay
        bigIM=fits.getdata(untrimmed_ref)
        HDR=fits.getheader(untrimmed_ref)
        
        size = 512
        hs = size // 2 
        IM = bigIM[int(y_pixel)-hs:int(y_pixel)+hs, int(x_pixel)-hs:int(x_pixel)+hs]
        fits.writeto(f'/neta/xrb/{target}/temp/{target}_{most_files_band}_dumbstack_trimmed.fits', IM, HDR, overwrite=True)
        ref=f'/neta/xrb/{target}/temp/{target}_{most_files_band}_dumbstack_trimmed.fits'
    '''
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(IM)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,10))
    plt.imshow(IM, cmap='gray', origin='lower', norm=norm)
    plt.scatter(x_pixel, y_pixel, marker='x')
    plt.show()
    
    
    #cutout should be Roughly centered on the star
    vmin, vmax = interval.get_limits(cutout)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,10))
    plt.imshow(cutout, cmap='gray', origin='lower', norm=norm)
    plt.scatter(256, 256, marker='x')
    plt.title(target)
    plt.show()
    '''
    outfile=f'/neta/xrb/{target}/product/{target}_{most_files_band}_wcs.fits'
    
    #check if a trimmed version already exists, and skip if we want to
    if not overwrite and outfile.exists():
        print('trim wcs file exists!')
        continue
    
    #get ra and dec in degrees
    coord = SkyCoord(HDR['RA'], HDR['DEC'], unit=(u.hourangle, u.deg))

    ra_deg = coord.ra.deg
    dec_deg = coord.dec.deg
    
    make_centered_wcs_fits(ref, outfile, ra_deg, dec_deg)