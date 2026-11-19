#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:19:22 2026

@author: kmc249
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

ir_file = "/neta/xrb/NovaMusca/temp/real_aligned_IR.fits"

catalog_file = (
    "/neta/xrb/NovaMusca/product/"
    "NovaMusca_ref_stars_kciurleo.csv"
)

reg_file = (
    "/neta/xrb/NovaMusca/product/"
    "NovaMusca_J_stars.reg"
)


# ------------------------------------------------------------
# Read catalog and select stars with J-band measurements
# ------------------------------------------------------------

catalog = pd.read_csv(catalog_file)

# J = 0 indicates no measurement in this catalog.
j_catalog = catalog[
    (pd.to_numeric(catalog["J"], errors="coerce") > 0)
].copy()

print(f"Total catalog stars: {len(catalog)}")
print(f"Stars with J-band measurements: {len(j_catalog)}")

# Coordinates are already in decimal degrees.
coords = SkyCoord(
    ra=j_catalog["RA"].values * u.deg,
    dec=j_catalog["Dec"].values * u.deg,
    frame="icrs"
)


# ------------------------------------------------------------
# Read IR image and WCS
# ------------------------------------------------------------

with fits.open(ir_file) as hdul:
    ir_data = hdul[0].data
    ir_header = hdul[0].header

ir_wcs = WCS(ir_header).celestial

# Convert catalog RA/Dec to IR pixel coordinates.
x, y = ir_wcs.world_to_pixel(coords)

ny, nx = ir_data.shape

inside = (
    (x >= 0) & (x < nx) &
    (y >= 0) & (y < ny)
)

print(f"J-band stars inside IR image: {inside.sum()} / {len(x)}")


# ------------------------------------------------------------
# Plot catalog positions on IR image
# ------------------------------------------------------------

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection=ir_wcs)

vmin, vmax = np.nanpercentile(ir_data, [5, 99.5])

ax.imshow(
    ir_data,
    origin="lower",
    cmap="gray",
    vmin=vmin,
    vmax=vmax
)

# Plot only stars that fall within the image.
ax.scatter(
    coords.ra.deg[inside],
    coords.dec.deg[inside],
    transform=ax.get_transform("world"),
    s=100,
    facecolors="none",
    edgecolors="red",
    linewidths=1.5,
    label="Catalog stars with J measurements"
)

ax.set_xlabel("Right Ascension")
ax.set_ylabel("Declination")
ax.legend(loc="best")

plt.title("Nova Musca: J-band Catalog Stars on IR Image")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Save calibration stars to CSV
# ------------------------------------------------------------

from astropy.coordinates import Angle

# Keep stars with nonzero J, H, and K measurements that are
# inside the IR image.
valid = (
    inside &
    (pd.to_numeric(j_catalog["J"], errors="coerce").values > 0) &
    (pd.to_numeric(j_catalog["H"], errors="coerce").values > 0) &
    (pd.to_numeric(j_catalog["K"], errors="coerce").values > 0)
)

calib = j_catalog.loc[valid].copy()

# Corresponding IR pixel positions.
calib_x = x[valid]
calib_y = y[valid]

# Convert RA/Dec to sexagesimal strings.
calib_coords = SkyCoord(
    ra=calib["RA"].values * u.deg,
    dec=calib["Dec"].values * u.deg,
    frame="icrs"
)

ra_hms = calib_coords.ra.to_string(
    unit=u.hourangle,
    sep=":",
    precision=2,
    pad=True
)

dec_dms = calib_coords.dec.to_string(
    unit=u.deg,
    sep=":",
    precision=1,
    alwayssign=True,
    pad=True
)

# Build output catalog.
calib_output = pd.DataFrame({
    "name": np.arange(1, len(calib) + 1),
    "objid": calib["objid"].values,
    "RA_deg": calib["RA"].values,
    "Dec_deg": calib["Dec"].values,
    "RA_hms": ra_hms,
    "Dec_dms": dec_dms,
    "x_stack": calib_x,
    "y_stack": calib_y,
    "Jmag": calib["J"].values,
    "e_Jmag": calib["dJ"].values,
    "Hmag": calib["H"].values,
    "e_Hmag": calib["dH"].values,
    "Kmag": calib["K"].values,
    "e_Kmag": calib["dK"].values
})

output_file = "/neta/xrb/NovaMusca/temp/calib_stars_all.csv"

calib_output.to_csv(output_file, index=False, float_format="%.6f")

print(f"\nSaved {len(calib_output)} calibration stars to:")
print(output_file)

print("\nFirst few rows:")
print(calib_output.head())