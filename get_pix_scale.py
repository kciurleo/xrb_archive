#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:21:36 2025

@author: kmc249
"""

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

h = fits.open("/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits")
w = WCS(h[0].header)

# returns scale per pixel in degrees/pixel
scales_deg = proj_plane_pixel_scales(w)

# Convert to arcsec/pixel
scales_arcsec = scales_deg * 3600 * u.arcsec

print(scales_arcsec)
