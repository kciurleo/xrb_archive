#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:49:29 2026

@author: kmc249
"""
from astropy.io import fits
import numpy as np
import astroalign as aa
import pandas as pd
import glob
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
from reproject import reproject_interp
from astrometry_helper import *
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture

def register_with_flips(src, ref, **kwargs):
    """
    Try registering src → ref with astroalign.
    If it fails, try horizontal, vertical, and both flips.
    Returns aligned image and footprint, or (None, None) if all fail.
    """

    flip_funcs = [
        lambda x: x,                     # no flip
        np.fliplr,                       # horizontal
        np.flipud,                       # vertical
        lambda x: np.flipud(np.fliplr(x))# both
    ]

    for func in flip_funcs:
        try:
            test_img = func(src)
            aligned, footprint = aa.register(test_img, ref, **kwargs)
            return aligned, footprint
        except Exception as exc:
            print(exc)
            pass

    raise RuntimeError("astroalign failed for all flip orientations.")


#first, find the conversion between 1m and main data
band='K'
#file='/neta/xrb/AqlX-1/temp/Aql_J_stack.fits'
file=f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack_NEW_MEDIAN_SCALED.fits'
base='/home/kmc249/Downloads/hires_master.fits' #the hires master
'''
img = np.asarray(fits.getdata(base), dtype='<f8')
inp_img = np.asarray(fits.getdata(file), dtype='<f8')


#get wcs for each of these
make_wcs_fits(file, f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack_WCS.fits')
#make_wcs_fits(base, '/neta/xrb/AqlX-1/temp/Aql_R_hires_WCS.fits')


ir = fits.open(f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack_WCS.fits')[0]
optical = fits.open('/neta/xrb/AqlX-1/temp/Aql_R_hires_WCS.fits')[0]

img_aligned, footprint = reproject_interp(ir, optical.header)

#fits.writeto("image1_reprojected.fits", array, hdu2.header, overwrite=True)
'''
jfits = fits.getdata(file).astype('<f8')
jfits=np.rot90(jfits, k=3)
rfits = fits.getdata(base).astype('<f8')
## new way of shifting: using ensemble stars
#run dao starfinder on it and plot the results
mean, median, std = sigma_clipped_stats(jfits, sigma=3, maxiters=5)
rmean, rmedian, rstd = sigma_clipped_stats(rfits, sigma=3, maxiters=5)

jdaofinder = DAOStarFinder(threshold=5*std, fwhm=6.)
jsources = jdaofinder(jfits)

rdaofinder = DAOStarFinder(threshold=5*rstd, fwhm=6.)
rsources = rdaofinder(rfits)


label=True
fig, axes = plt.subplots(1, 2, figsize=(14,7))
jfits_rot = np.rot90(jfits, k=3)
# ---- J image ----
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(jfits)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())

jpositions = np.transpose((jsources['xcentroid'], jsources['ycentroid']))
japertures = CircularAperture(jpositions, r=6.)

ax = axes[0]
ax.imshow(jfits, cmap='gray', origin='lower', norm=norm)
if label:
    japertures.plot(ax=ax, color="red")
    
    for i, (x, y) in enumerate(jpositions):
        ax.text(x+8, y+8, str(i),
                color='yellow',
                fontsize=8,
                ha='center', va='center')

ax.set_title(f"{band} band")


# ---- R image ----
interval = ZScaleInterval()
vmin2, vmax2 = interval.get_limits(rfits)
norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())

rpositions = np.transpose((rsources['xcentroid'], rsources['ycentroid']))
rapertures = CircularAperture(rpositions, r=6.)

ax = axes[1]
ax.imshow(rfits, cmap='gray', origin='lower', norm=norm2)
if label:
    rapertures.plot(ax=ax, color="red")
    
    for i, (x, y) in enumerate(rpositions):
        ax.text(x+8, y+8, str(i),
                color='yellow',
                fontsize=8,
                ha='center', va='center')

ax.set_title("R band")

plt.tight_layout()
plt.show()

#r, j
#this is for the old image
'''
matches = [
    (490, 280),
    (521, 279),
    (497, 136),
    (531, 132),
    (805,156),
    (591, 57),
    (320, 220),
    (194,198),
    (223,184),
    (195,178),
    (166,183)
]
'''
#r, j matches
'''
matches = [
    (490, 314),
    (521, 312),
    (497, 148),
    (531, 144),
    (805, 171),
    (591, 61),
    (320, 249),
    (194, 223),
    (223, 208),
    (195, 196),
    (166, 203)
]
'''
'''
#r, h matches
matches = [
    (580,177),
    (555,174),
    (531,170),
    (790,227),
    (753,217),
    (747,228),
    (359,75),
    (303,64)
]
'''
#r, k matches
matches = [
    (580,115),
    (555,112),
    (531,107),
    (753,142),
    (771,154),
    (485,99),
    (359,60),
    (320,53),
    (306,55)
]

src = np.array([rpositions[i] for i, j in matches])
dst = np.array([jpositions[j] for i, j in matches])

src = np.array([rpositions[i] for i, j in matches])
dst = np.array([jpositions[j] for i, j in matches])

#plotting to check, gonna rotate
h, w = jfits.shape

dst_rot = np.zeros_like(dst)
dst_rot[:,0] = h - 1 - dst[:,1]   # new x
dst_rot[:,1] = dst[:,0]           # new y
jfits_rot = np.rot90(jfits, k=3)



fig, axes = plt.subplots(1, 2, figsize=(12,6))

# ---- R image (source) ----
ax = axes[0]
ax.imshow(rfits, cmap='gray', origin='lower', norm=norm2)
ax.scatter(src[:,0], src[:,1], s=80, facecolors='none', edgecolors='red')

for k, (x, y) in enumerate(src):
    ax.text(x+8, y+8, str(k), color='yellow', fontsize=10)

ax.set_title("R band (matched stars)")

# ---- J image (target) ----
ax = axes[1]
ax.imshow(jfits_rot, cmap='gray', origin='lower', norm=norm)
ax.scatter(dst_rot[:,0], dst_rot[:,1], s=80, facecolors='none', edgecolors='red')

for k, (x, y) in enumerate(dst_rot):
    ax.text(x+8, y+8, str(k), color='yellow', fontsize=10)

ax.set_title(f"{band} band (matched stars)")

plt.tight_layout()
plt.show()

tform = aa.estimate_transform('affine', dst, src)

aligned, footprint = aa.apply_transform(tform, jfits, rfits)


aql=pd.read_csv("/home/kmc249/new_final_phot_aql_hires.csv")
neighborhood=pd.read_csv("/home/kmc249/new_final_phot_hires.csv")

plt.figure(figsize=(10,10))
plt.imshow(aligned, cmap='gray', origin='lower', norm=norm)
plt.scatter(aql['x_fit']+206, aql['y_fit']+206, marker='.', s=2, c='red', label='aql')
for name, group in neighborhood.groupby('name'):
    plt.scatter(group['x_0'], group['y_0'], marker='.', s=2, label=name)

plt.legend()
plt.show()

#final comparison
#%%

interval = ZScaleInterval()
fig, ax = plt.subplots(figsize=(12,12))  # create an Axes object

ax.imshow(footprint, origin='lower', cmap='gray_r')
ax.imshow(aligned, cmap='Oranges', origin='lower', norm=norm, alpha=0.5)

ax.imshow(rfits, cmap='Blues', origin='lower', norm=norm2, alpha=0.5)

ax.scatter(src[:,0], src[:,1],
           facecolors='none', edgecolors='red',
           s=80)

for k, (x, y) in enumerate(src):
    ax.text(x+8, y+8, str(k), color='red', fontsize=10)

plt.show()

fits.writeto(f"/neta/xrb/AqlX-1/temp/Aql_X-1_{band}_stack_aligned_to_opt.fits", aligned, fits.getheader(file), overwrite=True)