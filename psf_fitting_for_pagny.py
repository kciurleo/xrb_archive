#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:56:29 2026

@author: kmc249
"""

#import everything
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from astropy.table import QTable, Table, vstack
from photutils.psf import extract_stars
from astropy.nddata import NDData
from photutils.datasets import make_model_image

###PART 1: ESTIMATE BACKGROUND
#load image
file=
imdata,hdr = fits.getdata(file,header=True)

#background subtract data. we need to do this to find our PSF
boxsize=20  #Change these values to experiment
filtsize=3  #Change these values to experiment
sigma_clip=SigmaClip(sigma=3.0)
bkg_estimator=MedianBackground()
fullbkg=Background2D(imdata, (boxsize,boxsize), filter_size=(filtsize,filtsize),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
bkg_sub_full_data=imdata-fullbkg.background

#plot background and check the model looks good
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(fullbkg.background)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
plt.figure(figsize=(10,12))
plt.imshow(fullbkg.background, cmap='gray', origin='lower', norm=norm)
plt.plot()

###PART 2: CHECK TO FIND GOOD STARS FOR A PSF ESTIMATION
#get sources
mean, median, std = sigma_clipped_stats(bkg_sub_full_data, sigma=3, maxiters=5)

#starfinder on image
fwhm=5. #Change if needed
daofinder = DAOStarFinder(threshold=5. * std, fwhm=fwhm)
sources = daofinder(bkg_sub_full_data)

#plot and then look through the sources and find which ones are isolated, definitely stars, 
#and not extremely faint. 
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
plt.figure(figsize=(10,12))
plt.imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
plt.scatter(sources['xcentroid'], sources['ycentroid'], marker='x')

for row in sources:
    plt.annotate(
        str(row['id']), 
        (row['xcentroid'], row['ycentroid']),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=8,
        color='red'
    )
plt.show()


#ids of those stars go in here. the more the better, unless they're bad.
good_star_ids=[242,232,210,316]
#mask edges and mask out any not good stars
ny, nx = np.shape(imdata)
margin=19  #should probably be at least the same as your star size
mask_edges = (
    (sources['xcentroid'] >= margin) &
    (sources['xcentroid'] < nx - margin) &
    (sources['ycentroid'] >= margin) &
    (sources['ycentroid'] < ny - margin)
)

mask_ids = np.isin(sources['id'], good_star_ids)
mask = mask_edges & mask_ids
sources = sources[mask]

#these specific columns are needed for the psf fit
sources['x'] = sources['xcentroid']
sources['y'] = sources['ycentroid']
    
###PART 3: MAKE AN EFFECTIVE PSF
#use the given positions of the good PSF stars to generate a new EPSF
star_size=19 #needs to be odd, size of the box around a star to use
nddata=NDData(data=bkg_sub_full_data)
good_stars=extract_stars(nddata, sources, size=star_size)

#filter out stars with invalid or zero flux
valid_stars = [star for star in good_stars 
            if np.isfinite(np.sum(star.data)) and np.sum(star.data) > 0]

#make the star objects and build the epsf
epsf_input = EPSFStars(valid_stars)
epsf_builder=EPSFBuilder(oversampling=2, maxiters=10)
epsf, fitted_stars = epsf_builder(epsf_input)


#plot epsf just to see if it worked
norm=simple_norm(epsf.data, 'log', percent=99.0)
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='gray')
plt.show()


###PART 4: PERFORM PSF FITTING
#actual PSF fitting
psf_model = epsf
fit_shape=(11,11) #Change these values to experiment
min_separation=8  #Change these values to experiment
aperture_radius=8 #Change these values to experiment

finder=DAOStarFinder(threshold=5. * std, fwhm=fwhm)
grouper=SourceGrouper(min_separation=min_separation)

psfphot=PSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=aperture_radius)
phot=psfphot(imdata)
resid=psfphot.make_residual_image(imdata)
model=psfphot.make_model_image(np.shape(imdata))

#plot data. if your model is good, your residuals should look uniform.
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(imdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(phot['x_fit'], phot['y_fit'], marker='x')
axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(resid, cmap='gray', origin='lower', norm=norm)

for row in phot:
    axes[0].annotate(
        str(row['id']), 
        (row['x_fit'], row['y_fit']),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=8,
        color='red'
    )

plt.show()


#If you just want to fit a few stars, e.g. not use DAOfinder to fit ALL the stars (which sometimes takes a long time)
'''
#You need a table with the following columns, or at the very least an x and y column
inits = Table.from_pandas(pd.read_csv("filename"))
inits=opticalinits[['x_init', 'y_init', 'id','group_id']]
phot=psfphot(imdata, init_params=inits)

'''