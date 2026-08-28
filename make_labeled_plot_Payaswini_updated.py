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
import matplotlib.patheffects as pe

zoomphot=pd.read_csv('/home/kmc249/current_best_J_grid_fit.csv')
opticalinits=pd.read_csv('/home/kmc249/current_best_R_grid_fit.csv')

#IR image
bkg_sub_full_data,hdr = fits.getdata('/neta/xrb/AqlX-1/temp/Aql_X-1_J_stack_aligned_to_opt.fits',header=True)
res=2
zoomdata=bkg_sub_full_data[2*231:2*281,2*231:2*281]

#opt image
imdataopt,hdropt = fits.getdata('/home/kmc249/Downloads/hires_master_withwcs.fits',header=True)
imdatazoom=imdataopt[231*res:281*res,231*res:281*res]


interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
vminhr, vmaxhr = interval.get_limits(imdatazoom)
normhires = ImageNormalize(vmin=vminhr, vmax=vmaxhr-200, stretch=SinhStretch())
fig, axes=plt.subplots(1,1, figsize=(20,20))
axes.imshow(imdatazoom,  origin='lower', norm=normhires, cmap='gray')

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
# WCS from plate-solved hires image

from copy import deepcopy

# This FITS now contains the correct WCS
w_hi = WCS(hdropt)

# Crop origin in hires pixels
x0 = 231 * res
y0 = 231 * res

# WCS for cropped image
w_zoom = deepcopy(w_hi)
w_zoom.wcs.crpix -= [x0, y0]



#%%


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
fontsize = 20

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
    labelleft=False,
    labelsize=14,     # <-- bigger pixel labels
    width=1.5,        # thicker tick marks
    length=8          # longer tick marks
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


ra.set_ticks(number=2)
dec.set_ticks(number=8)

# Increase font size
ra.tick_params(
    labelsize=14,
    width=1.5,
    length=8,
    direction='in'
)

dec.tick_params(
    labelsize=14,
    width=1.5,
    length=8,
    direction='in'
)

#Bottom / left (WCS)
ra.set_axislabel('Right Ascension (J2000)', fontsize=18)
dec.set_axislabel('Declination (J2000)', fontsize=18, minpad=-0.5)

# Top / right (pixel axes)
ax_pix.set_xlabel('X (Pixels)', fontsize=18, labelpad=12)
ax_pix.set_ylabel('Y (Pixels)', fontsize=18, labelpad=22, rotation=270)

ax_pix.xaxis.set_label_position('top')
ax_pix.yaxis.set_label_position('right')


from astropy.wcs.utils import proj_plane_pixel_scales
# ------------------------------------------------------------
# Pixel scale from the WCS
# ------------------------------------------------------------

hires_scale_deg = np.mean(proj_plane_pixel_scales(w_hi))
hires_scale_arcsec = hires_scale_deg * 3600.0

# Desired angular length of scale bar
scale_arcsec = 1.0

# Convert arcseconds -> HI-RES image pixels
scale_bar_pixels = scale_arcsec / hires_scale_arcsec


# ============================================================
# ARCSECOND SCALE BAR
# ============================================================

bar_x0 = 68
bar_y0 = 28
bar_x1 = bar_x0 + scale_bar_pixels

# Main bar
ax.plot(
    [bar_x0, bar_x1],
    [bar_y0, bar_y0],
    color='white',
    linewidth=2,
    solid_capstyle='butt',
    zorder=20
)

# End caps
cap_height = 0.8

ax.plot(
    [bar_x0, bar_x0],
    [bar_y0 - cap_height, bar_y0 + cap_height],
    color='white',
    linewidth=2,
    zorder=20
)

ax.plot(
    [bar_x1, bar_x1],
    [bar_y0 - cap_height, bar_y0 + cap_height],
    color='white',
    linewidth=2,
    zorder=20
)

# Scale label
txt = ax.text(
    (bar_x0 + bar_x1) / 2,
    bar_y0 - 3,
    f'{scale_arcsec:g}"',
    color='white',
    fontsize=16,
    fontweight='bold',
    ha='center',
    va='bottom',
    zorder=21
)

txt.set_path_effects([
    pe.Stroke(linewidth=2, foreground='black'),
    pe.Normal()
])

# ============================================================
# SIMPLE NORTH / EAST INDICATOR
# North = up, East = left
# ============================================================

# Position of the compass origin
compass_x = 68+scale_bar_pixels
compass_y = 33

# Length of arrows in image pixels
compass_length = 7

# North = UP
ax.annotate(
    '',
    xy=(compass_x, compass_y + compass_length),
    xytext=(compass_x, compass_y),
    arrowprops=dict(
        arrowstyle='->',
        color='white',
        linewidth=2
    ),
    zorder=20
)


# East = LEFT
ax.annotate(
    '',
    xy=(compass_x - compass_length, compass_y),
    xytext=(compass_x, compass_y),
    arrowprops=dict(
        arrowstyle='->',
        color='white',
        linewidth=2
    ),
    zorder=20
)

# N label
txt = ax.text(
    compass_x,
    compass_y + compass_length + 1,
    'N',
    color='white',
    fontsize=16,
    fontweight='bold',
    ha='center',
    va='bottom',
    zorder=21
)

txt.set_path_effects([
    pe.Stroke(linewidth=2, foreground='black'),
    pe.Normal()
])

# E label
txt = ax.text(
    compass_x - compass_length - 1,
    compass_y,
    'E',
    color='white',
    fontsize=16,
    fontweight='bold',
    ha='right',
    va='center',
    zorder=21
)

txt.set_path_effects([
    pe.Stroke(linewidth=2, foreground='black'),
    pe.Normal()
])
plt.savefig('/home/kmc249/Downloads/ir_opt_neighborhood.png', dpi=300)
plt.show()
