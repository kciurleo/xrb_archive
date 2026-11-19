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
from astropy.wcs import WCS
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

#file='/neta/xrb/AqlX-1/temp/Aql_J_stack.fits'

base = (
    "/neta/xrb/GX339-4/product/"
    "GX339-4_I_wcs.fits"
)


file = (
    "/neta/xrb/GX339-4/temp/"
    "GX339-4_J_stacked.fits"
)
output_file = (
    "/neta/xrb/GX339-4/temp/"
    "real_aligned_IR.fits"
)

# Preserve original, unrotated IR image
original_ir = fits.getdata(file).astype('<f8')

# Rotate for registration
jfits = np.rot90(original_ir, k=3)

ny_ir, nx_ir = original_ir.shape
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

ax.set_title(f"IR")


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

ax.set_title("Opt")

plt.tight_layout()
plt.show()


#nova musca matches opt, ir
matches = [
    (390, 112),
    (344, 94),
    (354,102),
    (239,58),
    (240, 62),
    (138, 25),
    (139, 26),
    (123, 18)

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

ax.set_title("Optical (matched stars)")

# ---- J image (target) ----
ax = axes[1]
ax.imshow(jfits_rot, cmap='gray', origin='lower', norm=norm)
ax.scatter(dst_rot[:,0], dst_rot[:,1], s=80, facecolors='none', edgecolors='red')

for k, (x, y) in enumerate(dst_rot):
    ax.text(x+8, y+8, str(k), color='yellow', fontsize=10)

ax.set_title(f"IR (matched stars)")

plt.tight_layout()
plt.show()

tform = aa.estimate_transform('affine', dst, src)



# ============================================================
# CREATE ORIGINAL IR IMAGE WITH OPTICAL WCS
# ============================================================

print("\n============================================================")
print("CREATING IR IMAGE WITH OPTICAL WCS")
print("============================================================")

# ------------------------------------------------------------
# 1. Load optical WCS
# ------------------------------------------------------------

optical_header = fits.getheader(base)
optical_wcs = WCS(optical_header).celestial

if not optical_wcs.has_celestial:
    raise RuntimeError("Optical image does not have a celestial WCS.")

optical_cd = optical_wcs.pixel_scale_matrix

# ------------------------------------------------------------
# 2. Extract affine transformation
# ------------------------------------------------------------

# tform maps:
#
#   ROTATED IR pixels -> optical pixels
#
# The affine transformation is:
#
#   optical = M @ rotated_IR + t

M = tform.params[:2, :2]
t = tform.params[:2, 2]

print("\nRotated IR -> optical transformation:")
print(M)
print("Translation:", t)

# ------------------------------------------------------------
# 3. Account for the original 90-degree clockwise rotation
# ------------------------------------------------------------

# Your IR image was rotated using:
#
#   jfits = np.rot90(original_ir, k=3)
#
# Therefore:
#
#   x_rot = ny_ir - 1 - y_original
#   y_rot = x_original
#
# In matrix form:
#
#   rotated = R @ original + r

R = np.array([
    [0.0, -1.0],
    [1.0,  0.0]
])

r = np.array([
    ny_ir - 1.0,
    0.0
])

# ------------------------------------------------------------
# 4. Compose transformations
# ------------------------------------------------------------

# We want:
#
# ORIGINAL IR -> ROTATED IR -> OPTICAL
#
# Therefore:
#
#   optical = A_ir @ original_IR + t_ir

A_ir = M @ R

t_ir = M @ r + t

print("\nOriginal IR -> optical transformation:")
print(A_ir)
print("Translation:", t_ir)

# ------------------------------------------------------------
# 5. Choose reference pixel in ORIGINAL IR image
# ------------------------------------------------------------

# FITS CRPIX is 1-based.

ir_crpix = np.array([
    (nx_ir - 1) / 2.0 + 1.0,
    (ny_ir - 1) / 2.0 + 1.0
])

ir_ref_zero = ir_crpix - 1.0

# Corresponding optical pixel (zero-based).

optical_ref_zero = A_ir @ ir_ref_zero + t_ir

print("\nReference optical pixel:")
print(optical_ref_zero)

# ------------------------------------------------------------
# 6. Calculate sky coordinate at reference pixel
# ------------------------------------------------------------

ra_ref, dec_ref = optical_wcs.pixel_to_world_values(
    optical_ref_zero[0],
    optical_ref_zero[1]
)

print("\nReference sky coordinates:")
print(f"RA:  {ra_ref:.10f}")
print(f"DEC: {dec_ref:.10f}")

# ------------------------------------------------------------
# 7. Construct new IR WCS
# ------------------------------------------------------------

# Transform the optical CD matrix into original IR coordinates.

ir_cd = optical_cd @ A_ir

ir_wcs = WCS(naxis=2)

ir_wcs.wcs.crpix = ir_crpix
ir_wcs.wcs.crval = [ra_ref, dec_ref]
ir_wcs.wcs.cd = ir_cd

ir_wcs.wcs.ctype = optical_wcs.wcs.ctype
ir_wcs.wcs.cunit = optical_wcs.wcs.cunit

if optical_wcs.wcs.radesys:
    ir_wcs.wcs.radesys = optical_wcs.wcs.radesys

if optical_wcs.wcs.equinox:
    ir_wcs.wcs.equinox = optical_wcs.wcs.equinox

print("\nNew IR WCS:")
print(ir_wcs)

# ------------------------------------------------------------
# 8. Add WCS to ORIGINAL IR header
# ------------------------------------------------------------



ir_header = fits.getheader(file).copy()

# Remove old WCS keywords, if present.
wcs_prefixes = (
    "CTYPE", "CRPIX", "CRVAL", "CDELT",
    "CUNIT", "CD", "PC", "CROTA",
    "PV", "PS"
)

for key in list(ir_header.keys()):
    if key.startswith(wcs_prefixes):
        del ir_header[key]

for key in [
    "WCSAXES", "WCSNAME", "RADESYS",
    "EQUINOX", "LONPOLE", "LATPOLE"
]:
    if key in ir_header:
        del ir_header[key]

# Add newly constructed WCS.
ir_header.update(ir_wcs.to_header())

ir_header["HISTORY"] = (
    "WCS transferred from optical image using manually matched stars."
)

ir_header["HISTORY"] = (
    "Original IR image pixels preserved; no resampling performed."
)

ir_header["HISTORY"] = (
    "Registration: 90 degree clockwise rotation + affine transformation."
)

# ------------------------------------------------------------
# 9. Write original IR image with new WCS
# ------------------------------------------------------------

fits.writeto(
    output_file,
    original_ir,
    header=ir_header,
    overwrite=True
)

print("\nSuccessfully wrote:")
print(output_file)


aligned, footprint = aa.apply_transform(tform, jfits, rfits)



plt.figure(figsize=(10,10))
plt.imshow(aligned, cmap='gray', origin='lower', norm=norm)


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


# ------------------------------------------------------------
# Normalize each image independently
# ------------------------------------------------------------

opt_norm = ImageNormalize(
    rfits,
    interval=ZScaleInterval()
)

ir_norm = ImageNormalize(
    aligned,
    interval=ZScaleInterval()
)


opt_scaled = opt_norm(rfits)

ir_scaled = ir_norm(aligned)


# ------------------------------------------------------------
# Replace NaN/inf
# ------------------------------------------------------------

opt_scaled = np.nan_to_num(
    opt_scaled,
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
)

ir_scaled = np.nan_to_num(
    ir_scaled,
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
)


# ------------------------------------------------------------
# Clip
# ------------------------------------------------------------

opt_scaled = np.clip(
    opt_scaled,
    0,
    1
)

ir_scaled = np.clip(
    ir_scaled,
    0,
    1
)


# ------------------------------------------------------------
# Construct RGB
#
# RED   = optical
# GREEN = IR
# BLUE  = IR
#
# Optical only -> red
# IR only      -> cyan
# overlap      -> white/gray
# ------------------------------------------------------------

rgb = np.zeros(
    (*rfits.shape, 3)
)

rgb[:, :, 0] = opt_scaled
rgb[:, :, 1] = ir_scaled
rgb[:, :, 2] = ir_scaled


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.imshow(
    rgb,
    origin="lower",
    interpolation="none"
)



ax.set_title(
    "Optical + Registered IR Overlay\n"
    "Red = Optical   Cyan = IR"
)

ax.set_xlabel(
    "X pixel"
)

ax.set_ylabel(
    "Y pixel"
)

plt.tight_layout()

plt.show()


