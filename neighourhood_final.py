#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:50:50 2025

@author: kmc249
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
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
#%%

#parameters for some of the drawing line functions
length=15
num_points=500
id1=0
id2=3
id3=1
id4=2

def individual_star_models(psf_model, phottable, shape, star_indices=None):
    """
    Build individual star model images for selected fitted sources.
    Works for both ePSFModel and analytic PRF models.
    
    Parameters
    ----------
    psf_model : astropy.modeling.Fittable2DModel
        The PSF/PRF model (e.g., EPSFModel, CircularGaussianPRF).
    phottable : QTable
        Photometry results table with fitted parameters.
    shape : tuple
        Shape of the output image (ny, nx).
    star_indices : list of ints or None
        If given, only build models for these rows in phottable.
    """
    star_models = []
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]

    if star_indices is None:
        star_indices = range(len(phottable))

    for idx in star_indices:
        row = phottable[idx]
        model = psf_model.copy()
        for pname in model.param_names:
            if pname in row.colnames:
                setattr(model, pname, row[pname])
        star_models.append((idx, model(x, y)))
    return star_models



# build models only for the cross-section star pairs

def line_eq(x,x1,y1,x2,y2):
    return y1+(y2-y1)/(x2-x1)*(x-x1)

def line_profile(id1, id2, phottable):
    x1, y1 = np.abs(phottable['x_fit'][id2]+phottable['x_fit'][id1])/2, np.abs(phottable['y_fit'][id2]+phottable['y_fit'][id1])/2
    x_vals = np.linspace(x1-length/2, x1+length/2, num_points)
    y_vals = line_eq(x_vals, phottable['x_fit'][id1], phottable['y_fit'][id1],phottable['x_fit'][id2], phottable['y_fit'][id2])
    distance=np.sqrt((x_vals-x_vals[0])**2+(y_vals-y_vals[0])**2)
    return(x_vals, y_vals, distance)


def plotstuff(phot, model, resid, data):
    x_vals, y_vals, distance=line_profile(id1,id2,phot)
    x_vals2, y_vals2, distance2=line_profile(id3,id4,phot)
    
    fig, axes=plt.subplots(2,2, figsize=(18,8))
    
    # ===== First pair =====
    axes[0,0].plot(distance, map_coordinates(data, [y_vals, x_vals], order=1), label='data')
    axes[0,0].plot(distance, map_coordinates(model, [y_vals, x_vals], order=1), label='model1 total (epsf)')
    
    axes[0,0].legend()
    
    axes[1,0].scatter(distance, map_coordinates(data, [y_vals, x_vals], order=1)
                      - map_coordinates(model, [y_vals, x_vals], order=1), label='model1 resid')
    axes[1,0].axhline(0, color='black')
    axes[1,0].legend()
    
    # ===== Second pair =====
    axes[0,1].plot(distance2, map_coordinates(data, [y_vals2, x_vals2], order=1), label='data')
    axes[0,1].plot(distance2, map_coordinates(model, [y_vals2, x_vals2], order=1), color='g', label='model1 total (epsf)')
    
    
    axes[0,1].legend()
    
    axes[1,1].scatter(distance2, map_coordinates(data, [y_vals2, x_vals2], order=1)
                      - map_coordinates(model, [y_vals2, x_vals2], order=1), label='model1 resid')
    
    axes[1,1].axhline(0, color='black')
    axes[1,1].legend()
    
    plt.show()
    
    fig, axes=plt.subplots(1,3, figsize=(20,10))
    axes[0].imshow(data, cmap='gray', origin='lower', norm=norm)
    axes[0].scatter(phot['x_fit'], phot['y_fit'], marker='x')
    axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
    for i in [0,1,2]:
        axes[i].plot(x_vals, y_vals, color='orange')
        axes[i].plot(x_vals2, y_vals2, color='g')
    axes[0].set_xlim(0,50)
    axes[0].set_ylim(0,50)
    axes[2].imshow(resid, cmap='gray', origin='lower', norm=norm)
    if label:
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
#%%

#load data
stacked_trim_str='/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits'
#stacked_trim_str='/home/kmc249/Downloads/lores_master.fits'
stacked_full = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits')
stacked_trim = fits.getdata('/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits')
#stacked_trim = fits.getdata('/home/kmc249/Downloads/lores_master.fits')

w = WCS(fits.getheader('/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER1.fits'))

#load locations of good sources for ePSF
sources=Table.read('/home/kmc249/Downloads/good_sources.vot')

#put in correct x-y and get rid of epsf guys we won't use
sources["xcentroid"]=sources["xcentroid"]-301
sources["ycentroid"]=sources["ycentroid"]-301
sources["x"]=sources["x"]-301
sources["y"]=sources["y"]-301
mask=((sources['xcentroid']>= 0) &
    (sources['xcentroid']< 512) &
    (sources['ycentroid']>= 0) &
    (sources['ycentroid']< 512))
sources=sources[mask]

#info about the ensemble
stacked_ensemble=Table.read('/home/kmc249/Downloads/ensemble_info.vot')
eids = list(stacked_ensemble['id'])

#put all the vots in the smaller x-y system
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]-301
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]-301


#init params to fit the neighbor and ensemble
init_params=stacked_ensemble['id','group_id','flux_init','x_fit','y_fit','flux_fit']


imdata,hdr = fits.getdata(stacked_trim_str,header=True)

#background subtract data
sigma_clip=SigmaClip(sigma=3.0)
bkg_estimator=MedianBackground()
fullbkg=Background2D(imdata, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
bkg_sub_full_data=imdata-fullbkg.background

#plot sources on data just to double check

interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
vmin2, vmax2 = interval.get_limits(stacked_trim)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
plt.figure(figsize=(10,12))
plt.imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
plt.imshow(stacked_trim, cmap='viridis', origin='lower', norm=norm2, alpha=0.5)
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


#we need to get rid of the guys that are outside of this image if it's majorly shifted!! Some buffer around,
#the edges are wonky....or we just trim everything to max shift. temp fix noted below

#use the given positions of the good PSF stars to generate a new EPSF
size=19
nddata=NDData(data=bkg_sub_full_data)
good_stars=extract_stars(nddata, sources, size=size)

#temp fix:
# Filter out stars with invalid or zero flux

valid_stars = [star for star in good_stars 
            if np.isfinite(np.sum(star.data)) and np.sum(star.data) > 0]

epsf_input = EPSFStars(valid_stars)
#end

epsf_builder=EPSFBuilder(oversampling=2, maxiters=10)
epsf, fitted_stars = epsf_builder(epsf_input)


#plot epsf just to see if it worked

norm=simple_norm(epsf.data, 'log', percent=99.0)

plt.imshow(epsf.data, norm=norm, origin='lower', cmap='gray')
plt.show()


#psf fitting the exact same way I used to for the "just the bit around aql x-1 512x512 square"
#in the original version of psf_fit.py

#THIS IS JUST FOR THE ENSEMBLE STARS, KT I THINK YOU WANT TO CHANGE
psf_model = epsf
fit_shape=(7,7)
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8, xy_bounds=0.1)
#make the xy pixel fit dist. 0 so it doesn't move
ensphot=psfphot(bkg_sub_full_data, init_params=init_params)
ensresid=psfphot.make_residual_image(bkg_sub_full_data)
ensmodel=psfphot.make_model_image(np.shape(bkg_sub_full_data))

label=True
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(ensphot['x_fit'], ensphot['y_fit'], marker='x')
axes[1].imshow(ensmodel, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(ensresid, cmap='gray', origin='lower', norm=norm)
if label:
    for row in ensphot:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()

#get ave ens flux:
ave_ens_flux=np.nanmean(ensphot['flux_fit'])
#%%

###here's where we only do the bit right around aql x1:
#first step: just do detect magic
zoomdata=bkg_sub_full_data[231:281,231:281]

finder=DAOStarFinder(6.0, 2.0)
grouper=SourceGrouper(min_separation=8)
psfphot=PSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8)

zoomphot=psfphot(zoomdata)
zoomresid=psfphot.make_residual_image(zoomdata)
zoommodel=psfphot.make_model_image(np.shape(zoomdata))
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax-150, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes[1].imshow(zoommodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
axes[2].imshow(zoomresid, cmap='gray', origin='lower', norm=norm)
if label:
    for row in zoomphot:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()
print(zoomphot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit']])

#keep_ids = [21, 23, 24]
keep_ids = [17,19,20]

mask = np.isin(zoomphot['id'], keep_ids)

zoom_init= zoomphot[mask]['x_init', 'x_fit', 'y_init', 'y_fit', 'flux_fit']
zoom_init['name']=['b','a','c']
print(zoom_init)

#%%
#then do by hand 2nd iterative photometry to get d and e

zoomphot2=psfphot(zoomresid)
zoomresid2=psfphot.make_residual_image(zoomresid)
zoommodel2=psfphot.make_model_image(np.shape(zoomdata))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomresid, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot2['x_fit'], zoomphot2['y_fit'], marker='x')
axes[1].imshow(zoommodel2, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
axes[2].imshow(zoomresid2, cmap='gray', origin='lower', norm=norm)
if label:
    for row in zoomphot2:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()

keep_ids2 = [21]#, 19]

mask = np.isin(zoomphot2['id'], keep_ids2)

zoom_init2= zoomphot2[mask]['x_fit', 'y_fit', 'flux_fit']
zoom_init2['name']=['e']#, 'd']

#zoom_init2['x_fit']=[26, 28]
#zoom_init2['y_fit']=[24.8, 25]


zoom_init2 = vstack([zoom_init, zoom_init2])
print(zoom_init2)
#
#%%
#then, using those 5 as the initial conditions, run it again
#five_init=zoom_init2['x_fit', 'y_fit', 'flux_fit']
five_init=QTable()
five_init['x']=[18.510958634533807,24.104306907227482,31.154202003458273, 26, 28.]
five_init['y']=[20.62908720657539,23.834092810985606,25.154174418180144, 24.8, 25]
five_init['name']=['b','a','c', 'aql', 'd']

fivephot=psfphot(zoomdata, init_params=five_init)
fiveresid=psfphot.make_residual_image(zoomdata)
fivemodel=psfphot.make_model_image(np.shape(zoomdata))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(fivephot['x_fit'], fivephot['y_fit'], marker='x')
axes[1].imshow(fivemodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
axes[2].imshow(fiveresid, cmap='gray', origin='lower', norm=norm)
if label:
    for row in fivephot:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()
fivephot['name']=['b','a','c','e','d']
print(fivephot[['name','x_init','x_fit', 'y_init', 'y_fit', 'flux_fit']])
#%%
#checking separation
# reference star
ref_row = fivephot[fivephot['name'] == 'a'][0]
x_ref = ref_row['x_fit'].item()
y_ref = ref_row['y_fit'].item()
sky1 = w.pixel_to_world(x_ref, y_ref)
sky2 = w.pixel_to_world(fivephot['x_fit'], fivephot['y_fit'])
seps = sky1.separation(sky2)  # returns Quantity array in degrees

# print results
for name, sep in zip(fivephot['name'], seps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")
#%%

#then, we'll call a final! subtract a and run this on the 4 resid guys and those are the 
test_params=QTable()
test_params['x_0'] = [x_ref]
test_params['y_0'] = [y_ref]
test_params['flux'] = [ref_row['flux_fit'].item()]

#model just a and subtract, rerun phot
just_a=make_model_image(np.shape(zoomdata), psf_model, test_params)
without_a=zoomdata-just_a
four_init = fivephot[fivephot['name'] != 'a']
print(four_init)
fourphot=psfphot(without_a, init_params=four_init)
fourresid=psfphot.make_residual_image(without_a)
fourmodel=psfphot.make_model_image(np.shape(without_a))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(without_a, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(fourphot['x_fit'], fourphot['y_fit'], marker='x')
axes[1].imshow(fourmodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
axes[2].imshow(fourresid, cmap='gray', origin='lower', norm=norm)
if label:
    for row in fourphot:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()
print(fourphot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'flags']])
fourphot['name']=['b','c','e','d']
sky22 = w.pixel_to_world(fourphot['x_fit'], fourphot['y_fit'])
seps2 = sky1.separation(sky22)  # returns Quantity array in degrees
for name, sep in zip(fourphot['name'], seps2):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")
     
#%%
#combine into one table:
final_init=vstack([fourphot, QTable([ref_row])])
cols_to_remove = ['x_init', 'y_init', 'flux_init', 'group_size']
for col in cols_to_remove:
    if col in final_init.colnames:
        final_init.remove_column(col)
final_init.rename_column('x_fit',   'x_init')
final_init.rename_column('y_fit',   'y_init')
final_init.rename_column('flux_fit','flux_init')
print(final_init[['x_init', 'y_init', 'flux_init', 'name']])

finalphot=psfphot(zoomdata, init_params=final_init)
finalresid=psfphot.make_residual_image(zoomdata)
finalmodel=psfphot.make_model_image(np.shape(zoomdata))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(finalphot['x_fit'], finalphot['y_fit'], marker='x')
axes[1].imshow(finalmodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
axes[2].imshow(finalresid, cmap='gray', origin='lower', norm=norm)
if label:
    for row in finalphot:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()

#comparing these two
print(finalphot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'flags', 'id']])

finalphot['name']=['b','a','c','e','d']
ref_row2 = finalphot[finalphot['name'] == 'a'][0]

x_ref2 = ref_row2['x_fit'].item()
y_ref2 = ref_row2['y_fit'].item()
sky12 = w.pixel_to_world(x_ref2, y_ref2)
sky23 = w.pixel_to_world(finalphot['x_fit'], finalphot['y_fit'])
seps3 = sky12.separation(sky23)  # returns Quantity array in degrees
for name, sep in zip(finalphot['name'], seps3):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f}")

flux_a = finalphot[finalphot['name'] == 'a']['flux_fit'].item()
flux_e = finalphot[finalphot['name'] == 'e']['flux_fit'].item()
total_flux = flux_a + flux_e

rel_flux_a = flux_a / total_flux
rel_flux_e = flux_e / total_flux

print(f"Relative flux of 'a' (a / (a+e)) in finalphot: {rel_flux_a:.3f}")
print(f"Relative flux of 'e' (e / (a+e)) in finalphot: {rel_flux_e:.3f}")

print(fivephot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'flags', 'id']])
for name, sep in zip(fivephot['name'], seps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f}")
    
flux_a_5 = fivephot[fivephot['name'] == 'a']['flux_fit'].item()
flux_e_5 = fivephot[fivephot['name'] == 'e']['flux_fit'].item()
total_flux_5 = flux_a_5 + flux_e_5

rel_flux_a_5 = flux_a_5 / total_flux_5
rel_flux_e_5 = flux_e_5 / total_flux_5

print(f"Relative flux of 'a' (a / (a+e)) in fivephot: {rel_flux_a_5:.3f}")
print(f"Relative flux of 'e' (e / (a+e)) in fivephot: {rel_flux_e_5:.3f}")


#%%
#plotting and comparing stuff, we need a separate version of just a for final
test_params2=QTable()
test_params2['x_0'] = [x_ref2]
test_params2['y_0'] = [y_ref2]
test_params2['flux'] = [ref_row2['flux_fit'].item()]

#model just a and subtract, rerun phot
just_a_final=make_model_image(np.shape(zoomdata), psf_model, test_params2)

plotstuff(finalphot, finalmodel, finalresid, zoomdata)
plotstuff(fivephot, fivemodel, fiveresid, zoomdata)
plotstuff(finalphot, finalmodel-just_a_final, finalresid, zoomdata-just_a_final)
plotstuff(fivephot, fivemodel-just_a, fiveresid, zoomdata-just_a)

#%%
#getting the scaled fluxes for everything, save x, y, and relative flux to a csv file
#using finalphot as the correct one instead of fivephot for now
final_phot=finalphot

#we need to put x and y back in the trimmed coordinates, so:

final_phot['scaled flux']=final_phot['flux_fit']/ave_ens_flux
final_phot['x_0']=final_phot['x_fit']+231
final_phot['y_0']=final_phot['y_fit']+231
final_phot[['name','x_0', 'y_0','scaled flux', 'x_err','y_err','flux_err']]

aql = final_phot[final_phot['name'] == 'e']
final_phot = final_phot[final_phot['name'] != 'e']


final_phot=final_phot[['name','x_0', 'y_0','scaled flux', 'x_err','y_err','flux_err']]

final_phot.write("/home/kmc249/final_phot.csv", format="csv", overwrite=True)
aql.write("/home/kmc249/final_phot_aql.csv", format="csv", overwrite=True)
print('flux error aql: ', aql['flux_err'].value[0])
print('max pos. error aql, in 512x512 land: ',np.nanmax(list(aql['y_err'])+list(aql['x_err'])))

