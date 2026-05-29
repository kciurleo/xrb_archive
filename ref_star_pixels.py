#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:34:07 2026

@author: kmc249
"""
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from lookup_name import *
from astropy.coordinates import SkyCoord
import astropy.units as u
import re
import shutil
import pandas as pd
import glob

#Lining up the refernce stars with detected stars on the image
target='J1808.4-3658'
shortname=re.split(r'[-−.]', target)[0]

#if the folder is in downloads, move it to the archive
try:
    file=f'/home/kmc249/Downloads/{shortname}_ref_stars_kciurleo.csv'
    refstars=pd.read_csv(file)
    shutil.move(file, f'/neta/xrb/{target}/product/{shortname}_ref_stars_kciurleo.csv')
    print('moved file.')
except:
    print('file in archive already.')
    file=f'/neta/xrb/{target}/product/{shortname}_ref_stars_kciurleo.csv'
    refstars=pd.read_csv(file)

#trimmed wcs file    
wcsfile=glob.glob(f'/neta/xrb/{target}/product/*wcs.fits')[0]
data=fits.getdata(wcsfile)
wcs = WCS(fits.getheader(wcsfile))

#%%
#Converting sky coordinates to pixel values
sky = SkyCoord(ra=refstars['RA'].values * u.deg,
                  dec=refstars['Dec'].values * u.deg,
                  frame='icrs')

xpix, ypix = wcs.world_to_pixel(sky)

refstars['xpix'] = xpix
refstars['ypix'] = ypix
refstars['skycoords']=sky

#label all the stars as reference stars and mask out those which are within 10 pix of the border
ny, nx = data.shape
pad = 10
mask = (
    (refstars['xpix'] > pad) &
    (refstars['xpix'] < nx - pad) &
    (refstars['ypix'] > pad) &
    (refstars['ypix'] < ny - pad)
)
refstars = refstars[mask].reset_index(drop=True)
refstars['type'] = 'ref'

#%%

#Plot just to check
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())

plt.figure(figsize=(8,8))

plt.imshow(data, cmap='gray', origin='lower', norm=norm)
plt.scatter(refstars['xpix'], refstars['ypix'],
           s=40, facecolors='none', edgecolors='red',
           linewidths=1.)

for i, row in refstars.iterrows():
    plt.text(row['xpix'] + 8,
             row['ypix'] + 8,
             str(i),
             color='yellow',
             fontsize=5,
             ha='center',
             va='center')
    
#This should be the target
plt.scatter(256,256, marker='x')

plt.show()

#%%
#HUMAN INTERVENTION NEEDED.
try:
    #If our target is one of the ref stars, list it
    target_id=int(input('Target id?'))
    refstars.loc[target_id, 'type'] = 'target'
except:
    print('No id. adding new row at the center.')
    #Or add a row for our target if it isn't already in the table. 
    target_row = {
        'RA': np.nan,
        'Dec': np.nan,
        'xpix': 256,
        'ypix': 256,
        'type': 'target'
    }
    refstars = pd.concat([refstars, pd.DataFrame([target_row])], ignore_index=True)

#Save the new csv
refstars.to_csv(f'/neta/xrb/{target}/product/{shortname}_ref_stars_kciurleo.csv', index=False)