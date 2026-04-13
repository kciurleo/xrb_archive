#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:18:51 2026

@author: kmc249
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.profiles import RadialProfile

# --- LOAD IMAGE ---
fname = '/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits'
hdu = fits.open(fname)
data = hdu[0].data.astype(float)

ny, nx = data.shape

# --- Find stars ---
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
daofind = DAOStarFinder(fwhm=3.0, threshold=8.0*std)
sources = daofind(data - median)

# --- Parameters ---
max_offset = 0.5       # pixels, peak tolerance
edge_margin = 15       # pixels, minimum distance from edge
cutout_size = 3        # cutout size for peak check

filtered_sources = []

for star in sources:
    x_star = star['xcentroid']
    y_star = star['ycentroid']
    
    # Skip stars too close to edge
    if (x_star < edge_margin or x_star > nx - edge_margin or
        y_star < edge_margin or y_star > ny - edge_margin):
        continue
    
    # Define cutout safely
    x0 = int(x_star) - cutout_size // 2
    x1 = x0 + cutout_size
    y0 = int(y_star) - cutout_size // 2
    y1 = y0 + cutout_size

    cutout = data[y0:y1, x0:x1]
    
    # Safety check
    if cutout.size == 0 or cutout.shape[0] < cutout_size or cutout.shape[1] < cutout_size:
        continue
    
    # Check if peak is at center
    dy, dx = np.unravel_index(np.argmax(cutout), cutout.shape)
    if abs(dx - cutout_size//2) <= max_offset and abs(dy - cutout_size//2) <= max_offset:
        filtered_sources.append(star)

print(f"Kept {len(filtered_sources)} stars out of {len(sources)}")

#normalized
plt.figure()
for i in range(len(filtered_sources)):
    x_star = sources['xcentroid'][i]
    y_star = sources['ycentroid'][i]
    
    yy, xx = np.indices(data.shape)
    r = np.sqrt((xx - x_star)**2 + (yy - y_star)**2)
    
    r_flat = r.ravel()
    flux_flat = data.ravel()
    
    r_int = r_flat.astype(int)
    r_max = 10
    
    radial_profile = np.bincount(r_int, weights=flux_flat) / np.bincount(r_int)
    radial_profile = radial_profile[:r_max]
    
    # --- NORMALIZE ---
    radial_profile /= radial_profile.max()   # peak normalization
    
    plt.plot(np.arange(len(radial_profile)), radial_profile)

plt.xlabel("Radius (pixels)")
plt.ylabel("Normalized Flux")
plt.title("Normalized Radial Profiles (Peak = 1)")
plt.show()

#%%

plt.figure()
i=278
x_star = sources['xcentroid'][i]
y_star = sources['ycentroid'][i]

yy, xx = np.indices(data.shape)
r = np.sqrt((xx - x_star)**2 + (yy - y_star)**2)

r_flat = r.ravel()
flux_flat = data.ravel()

r_int = r_flat.astype(int)
r_max = 10

radial_profile = np.bincount(r_int, weights=flux_flat) / np.bincount(r_int)
radial_profile = radial_profile[:r_max]

# --- NORMALIZE ---
radial_profile /= radial_profile.max()   # peak normalization

plt.plot(np.arange(len(radial_profile)), radial_profile)

plt.xlabel("Radius (pixels)")
plt.ylabel("Normalized Flux")
plt.title("Aql")
plt.show()
#%%
# Use your filtered_sources to avoid plotting bad stars
positions = [(star['xcentroid'], star['ycentroid']) for star in sources]

# Define aperture radius
aperture_radius = 5  # pixels

# Create circular apertures
apertures = CircularAperture(positions, r=aperture_radius)

# Plot image
plt.figure(figsize=(10,10))
plt.imshow(data, cmap='gray', origin='lower', 
           vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))

# Overlay apertures
apertures.plot(color='red', lw=1.5, alpha=0.6)

# Add IDs next to stars
for i, (x, y) in enumerate(positions):
    plt.text(x + 1, y + 1, str(i), color='yellow', fontsize=8)

plt.title("Filtered Stars with Apertures and IDs")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.show()


'''
# Example: remove stars with indices 2, 5, and 7
indices_to_remove = [2, 5, 7]

# Keep only stars whose index is NOT in indices_to_remove
filtered_sources = [star for i, star in enumerate(filtered_sources) if i not in indices_to_remove]

print(f"Remaining stars: {len(filtered_sources)}")
'''

#%%
# --- Compute radial profiles using photutils RadialProfile ---
plt.figure()
r_max = 10

for star in filtered_sources[:50]:  # plot up to 50 stars
    x_star = star['xcentroid']
    y_star = star['ycentroid']

    # Use a circular aperture for radial profile
    aperture = CircularAperture((x_star, y_star), r=r_max)
    rp = RadialProfile(data, (x_star, y_star), np.arange(0,10))
    rp_data = rp.profile  # average flux in annuli
    r_values = rp.radii

    # Normalize peak to 1
    #rp_data /= rp_data.max()

    min_len = min(len(r_values), len(rp_data))
    plt.plot(r_values[:min_len], rp_data[:min_len], alpha=0.1)

star_index = 278  # for example, the 6th star in filtered_sources
star = sources[star_index]

x_star = star['xcentroid']
y_star = star['ycentroid']

# Use a circular aperture for radial profile
aperture = CircularAperture((x_star, y_star), r=r_max)
rp = RadialProfile(data, (x_star, y_star), np.arange(0,r_max))
rp_data = rp.profile  # average flux in annuli
r_values = rp.radii

# Normalize peak to 1
#rp_data /= rp_data.max()

min_len = min(len(r_values), len(rp_data))
plt.plot(r_values[:min_len], rp_data[:min_len], alpha=1, label='aql')


plt.yscale('log')
plt.xlabel("Radius (pixels)")
plt.ylabel("Un-normalized Flux")
#plt.title("Normalized Radial Profiles using photutils")
plt.grid(True)
plt.show()