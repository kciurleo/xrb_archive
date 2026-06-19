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

with fits.open("/neta/xrb/LMCX-3/product/LMCX-3_V_wcs.fits") as h:
    data = h[0].data
    w = WCS(h[0].header)

# image dimensions
ny, nx = data.shape

# pixel scales
pixscale = (proj_plane_pixel_scales(w) * u.deg).to(u.arcsec)

# FOV along each axis
fov_x = nx * pixscale[0]
fov_y = ny * pixscale[1]

print(f"Image size: {nx} x {ny} pixels")
print(f"Pixel scale: {pixscale}")
print(f"FOV: {fov_x.to(u.arcmin):.2f} × {fov_y.to(u.arcmin):.2f}")