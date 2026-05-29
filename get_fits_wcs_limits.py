#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:42:59 2026

@author: kmc249
"""

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import glob

target='J1753'
file=glob.glob(f'/neta/xrb/{target}/product/*wcs.fits')[0]

# Load FITS
data=fits.getdata(file)
wcs = WCS(fits.getheader(file))

# Image shape
ny, nx = data.shape

# Corner pixels (note: FITS pixels are 1-indexed in WCS, but astropy handles this)
corners_pix = np.array([
    [0, 0],
    [0, nx-1],
    [ny-1, 0],
    [ny-1, nx-1]
])

# Convert pixel -> world (RA/Dec)
ra_dec = wcs.all_pix2world(corners_pix, 0)  # origin=0 for numpy indexing

ra = ra_dec[:, 0]
dec = ra_dec[:, 1]
print(target)
print("RA range:", ra.min(), ra.max())
print("Dec range:", dec.min(), dec.max())