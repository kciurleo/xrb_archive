#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:59:27 2026

@author: kmc249
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch,AsinhStretch, simple_norm
from astropy.table import QTable, Table, vstack
from scipy.ndimage import map_coordinates
from photutils.psf import extract_stars
from astropy.nddata import NDData
from astropy.wcs import WCS
from smith_utils import *
from photutils.datasets import make_model_image
import glob
from scipy.optimize import curve_fit
import json
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord

zoomphot=pd.read_csv('/home/kmc249/current_best_J_grid_fit.csv')
opticalinits=pd.read_csv('/home/kmc249/current_best_R_grid_fit.csv')

#IR image
bkg_sub_full_data,hdr = fits.getdata('/neta/xrb/AqlX-1/temp/Aql_X-1_J_stack_aligned_to_opt.fits',header=True)
res=2
zoomdata=bkg_sub_full_data[2*231:2*281,2*231:2*281]

#opt image
imdataopt,hdropt = fits.getdata('/home/kmc249/Downloads/hires_master.fits',header=True)
imdatazoom=imdataopt[231*res:281*res,231*res:281*res]


interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
vminhr, vmaxhr = interval.get_limits(imdatazoom)
normhires = ImageNormalize(vmin=vminhr, vmax=vmaxhr, stretch=SinhStretch())
fig, axes=plt.subplots(1,1, figsize=(20,20))
axes.imshow(imdatazoom,  origin='lower', norm=normhires, cmap='gray')
axes.imshow(zoomdata, origin='lower', norm=norm,  alpha=0.8, cmap='Reds')

axes.scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes.scatter(opticalinits['x_fit'], opticalinits['y_fit'], marker='.', c='cyan')
axes.set_xlim(20,80)
axes.set_ylim(20,80)

for _, row in zoomphot.iterrows():
    axes.annotate(
        str(row['name']), 
        (row['x_fit'], row['y_fit']),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=8,
        color='red'
    )
    
for _, row in opticalinits.iterrows():
    axes.annotate(
        str(row['name']), 
        (row['x_fit'], row['y_fit']),
        textcoords="offset points",
        xytext=(5,-5),
        fontsize=8,
        color='cyan'
    )

plt.show()


#%%

#wcs things


w_low = WCS(fits.getheader(
    "/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER1.fits"
))
from copy import deepcopy

w_hi = deepcopy(w_low)

# CRPIX scales with the image dimensions
w_hi.wcs.crpix *= res

# CD matrix (or CDELT) scales inversely
if w_hi.wcs.has_cd():
    w_hi.wcs.cd /= res
else:
    w_hi.wcs.cdelt /= res


x0 = 231 * res
y0 = 231 * res

w_zoom = deepcopy(w_hi)
w_zoom.wcs.crpix -= [x0, y0]


#%%
import matplotlib.patheffects as pe

# -------------------------------------------------------
# Figure with WCS axes (RA/Dec)
# -------------------------------------------------------
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=w_zoom)

# Optical image
ax.imshow(imdatazoom, origin='lower', norm=normhires, cmap='gray')

# IR contours
positive = zoomdata[zoomdata > 0]

levels = np.geomspace(
    np.percentile(positive, 25),
    np.percentile(positive, 99.9),
    15
)

ax.contour(
    zoomdata,
    levels=levels,
    colors='indianred',
    linewidths=1.2
)

# -------------------------------------------------------
# Scatter points
# -------------------------------------------------------
ax.scatter(
    zoomphot['x_fit'], zoomphot['y_fit'],
    marker='x',
    c='red',
    s=80,
    linewidths=2
)

ax.scatter(
    opticalinits['x_fit'], opticalinits['y_fit'],
    marker='+',
    c='dodgerblue',
    s=80
)

ax.set_xlim(20, 80)
ax.set_ylim(20, 80)

# -------------------------------------------------------
# RA / Dec formatting
# -------------------------------------------------------
ra = ax.coords['ra']
dec = ax.coords['dec']

ra.set_major_formatter('hh:mm:ss')
dec.set_major_formatter('dd:mm:ss')

ra.set_ticks_position('b')
ra.set_ticklabel_position('b')

dec.set_ticks_position('l')
dec.set_ticklabel_position('l')

# Optional grid
# ax.coords.grid(color='white', alpha=0.3, ls=':')

# -------------------------------------------------------
# Labels
# -------------------------------------------------------
fontsize = 18

for _, row in zoomphot.iterrows():
    txt = ax.annotate(
        str(row['name']),
        (row['x_fit'], row['y_fit']),
        xytext=(3,4),
        textcoords='offset points',
        fontsize=fontsize,
        color='red',
        ha='left',
        va='bottom'
    )

    txt.set_path_effects([
        pe.Stroke(linewidth=0.5, foreground='white'),
        pe.Normal()
    ])

for _, row in opticalinits.iterrows():
    txt = ax.annotate(
        str(row['name']),
        (row['x_fit'], row['y_fit']),
        xytext=(-3,-4),
        textcoords='offset points',
        fontsize=fontsize,
        color='dodgerblue',
        ha='right',
        va='top'
    )

    txt.set_path_effects([
        pe.Stroke(linewidth=0.5, foreground='white'),
        pe.Normal()
    ])

# -------------------------------------------------------
# Overlay pixel axes on top/right
# -------------------------------------------------------
ax_pix = fig.add_axes(ax.get_position(), frameon=False)

ax_pix.set_xlim(ax.get_xlim())
ax_pix.set_ylim(ax.get_ylim())

ax_pix.patch.set_alpha(0)

ax_pix.xaxis.set_ticks_position('top')
ax_pix.yaxis.set_ticks_position('right')

ax_pix.tick_params(
    top=True,
    labeltop=True,
    bottom=False,
    labelbottom=False,
    right=True,
    labelright=True,
    left=False,
    labelleft=False
)

ticks = np.arange(20, 81, 10)

ax_pix.set_xticks(ticks)
ax_pix.set_yticks(ticks)

# Pixel coordinates starting at 0 and divided by 2
labels = ((ticks - 20) / 2).astype(int)

ax_pix.set_xticklabels(labels)
ax_pix.set_yticklabels(labels)


ax_pix.xaxis.set_label_position("top")


ax_pix.yaxis.set_label_position("right")

ra.set_axislabel("")
dec.set_axislabel("")


ra = ax.coords['ra']
dec = ax.coords['dec']

ra.set_major_formatter('hh:mm:ss')
dec.set_major_formatter('dd:mm:ss')

# More RA ticks
ra.set_ticks(number=5)


# Keep Dec as desired
dec.set_ticks(spacing=5*u.arcsec)

# Aql X-1 coordinates (J2000)
aqlx1 = SkyCoord(
    "19h11m16.4s",
    "-00d38m41s",
    frame="icrs"
)

# Convert sky coordinates -> pixel coordinates in cropped image
x_aql, y_aql = w_zoom.world_to_pixel(aqlx1)

print("Aql X-1 pixel position:", x_aql, y_aql)



plt.show()
