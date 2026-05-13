#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:24:19 2025

@author: kmc249
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:50:50 2025

@author: kmc249
"""


#### This file does the psf fitting to the hires image in the R band for aql.
####

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
length=30
num_points=1000
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
    #for i in [0,1,2]:
        #axes[i].plot(x_vals, y_vals, color='orange')
        #axes[i].plot(x_vals2, y_vals2, color='g')
    axes[0].set_xlim(0,100)
    axes[0].set_ylim(0,100)
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

#resolution is 2, hires
res=2
#load data, hires comes from interpolating x2


#this one is untrimmed original (1024x1024), which the "good star" selection for psf fitting was done on.
stacked_full = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits')

#this one is the 512x512 cutout of the original image
stacked_trim_zoom = fits.getdata('/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits')

#this one is the 512x512 cutout of the original image, zoomed in to be hi res 1024x1024
stacked_trim_str='/home/kmc249/Downloads/hires_master.fits'
stacked_trim = fits.getdata('/home/kmc249/Downloads/hires_master.fits')

#this is the same as the hires but has the wcs. just a slightly different stack, from an earlier time
w = WCS(fits.getheader('/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER1.fits'))

#%%
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

sources["xcentroid"]=sources["xcentroid"]*res
sources["ycentroid"]=sources["ycentroid"]*res
sources["x"]=sources["x"]*res
sources["y"]=sources["y"]*res

#info about the ensemble
stacked_ensemble=Table.read('/home/kmc249/Downloads/ensemble_info.vot')
eids = list(stacked_ensemble['id'])
### Getting rid of the two stars that are not going to be useful (one is 'negative', one is variable):
ids_to_remove = [1320, 413]
ensmask = ~np.isin(stacked_ensemble['id'], ids_to_remove)
stacked_ensemble = stacked_ensemble[ensmask]    


#put all the vots in the smaller x-y system
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]-301
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]-301
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]*res
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]*res
stacked_ensemble["x_fit"]=stacked_ensemble["x_fit"]*res
stacked_ensemble["y_fit"]=stacked_ensemble["y_fit"]*res


#init params to fit the ensemble
init_params=stacked_ensemble['id','group_id','flux_init','x_fit','y_fit','flux_fit']

imdata,hdr = fits.getdata(stacked_trim_str,header=True)

#background subtract data
sigma_clip=SigmaClip(sigma=3.0)
bkg_estimator=MedianBackground()
fullbkg=Background2D(imdata, (20*res,20*res), filter_size=(3*res+1,3*res+1),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
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
#%%
#we need to get rid of the guys that are outside of this image if it's majorly shifted!! Some buffer around,
#the edges are wonky....or we just trim everything to max shift. 

ids_to_remove_from_epsf = [566,618,634,626,687,53,571,625,621,515,584,467,388,466,450,385,347,280,245,203,185,240,175,168,285,277,382,313,365,404]
epsfmask = ~np.isin(sources['id'], ids_to_remove_from_epsf)
sources = sources[epsfmask]   


#use the given positions of the good PSF stars to generate a new EPSF
size=19*res+1
#size=21*res+1
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
#%%

#fit to the ensemble stars; fit_shape has been tested to be the best
psf_model = epsf
fitnum=13
fit_shape=(fitnum*res+1,fitnum*res+1)
#fit_shape=(int(size/2-0.5),int(size/2-0.5))
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8*res)
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
ensphot.write("/home/kmc249/ens_phot_apr_02.csv", format="csv", overwrite=True)
#get ave ens flux:
ave_ens_flux=np.nanmean(ensphot['flux_fit'])
#%%
print(ave_ens_flux)

#%%

###here's where we only do the bit right around aql x1:
#first step: just detect as many stars as we can
zoomdata=bkg_sub_full_data[231*res:281*res,231*res:281*res]

finder=DAOStarFinder(5.0, 8.0)
grouper=SourceGrouper(min_separation=16)
psfphot=PSFPhotometry(psf_model, fit_shape, grouper=grouper, finder=finder, aperture_radius=8*res)

zoomphot=psfphot(zoomdata)
zoomresid=psfphot.make_residual_image(zoomdata)
zoommodel=psfphot.make_model_image(np.shape(zoomdata))
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax-150, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes[1].imshow(zoommodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
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
keep_ids = [19,21,22]

mask = np.isin(zoomphot['id'], keep_ids)

zoom_init= zoomphot[mask]['x_init', 'x_fit', 'y_init', 'y_fit', 'flux_fit', 'group_id']
zoom_init['name']=['b','a','c']
print(zoom_init)
#%%
#then to the residual image, fit again in that area to see if we pull any other stars
#I did that in the NON zoomed in version to get the five init below.
#actually, I think what I did was from the position of A, where should E and D be based on triangle math 
#from the chevalier paper?


#%%
#then, using those 5 as the initial conditions, run it again
#five_init=zoom_init2['x_fit', 'y_fit', 'flux_fit']
five_init=QTable()
five_init['x']=[18.510958634533807,24.104306907227482,31.154202003458273, 26, 28.]
five_init['y']=[20.62908720657539,23.834092810985606,25.154174418180144, 24.8, 25]
five_init['name']=['b','a','c', 'e', 'd']
five_init['x']=five_init['x']*2
five_init['y']=five_init['y']*2
five_init['group_id']=int(1)
print(five_init['group_id'])

fivephot=psfphot(zoomdata, init_params=five_init)
fiveresid=psfphot.make_residual_image(zoomdata)
fivemodel=psfphot.make_model_image(np.shape(zoomdata))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(fivephot['x_fit'], fivephot['y_fit'], marker='x')
axes[1].imshow(fivemodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
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
print(fivephot[['name','x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'id','group_id']])
#%%
#checking separation
# reference star
x_scaled = fivephot['x_fit'] / res
y_scaled = fivephot['y_fit'] / res



ref_row = fivephot[fivephot['name'] == 'a'][0]
x_ref = ref_row['x_fit'].item()/res
y_ref = ref_row['y_fit'].item()/res
sky1 = w.pixel_to_world(x_ref, y_ref)
sky2 = w.pixel_to_world(x_scaled, y_scaled)
seps = sky1.separation(sky2)  # returns Quantity array in degrees

# print results
for name, sep in zip(fivephot['name'], seps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")
    
sigma_x_ref = ref_row['x_err'] / res
sigma_y_ref = ref_row['y_err'] / res

# nominal sky coordinate of reference
sky_ref = w.pixel_to_world(x_ref, y_ref)

# numerical derivatives for reference
dx = 1e-3
dy = 1e-3
dra_dx_ref = (w.pixel_to_world(x_ref + dx, y_ref).ra.deg - sky_ref.ra.deg) / dx
dra_dy_ref = (w.pixel_to_world(x_ref, y_ref + dy).ra.deg - sky_ref.ra.deg) / dy
ddec_dx_ref = (w.pixel_to_world(x_ref + dx, y_ref).dec.deg - sky_ref.dec.deg) / dx
ddec_dy_ref = (w.pixel_to_world(x_ref, y_ref + dy).dec.deg - sky_ref.dec.deg) / dy

sigma_ra_ref = np.sqrt((dra_dx_ref * sigma_x_ref)**2 + (dra_dy_ref * sigma_y_ref)**2)
sigma_dec_ref = np.sqrt((ddec_dx_ref * sigma_x_ref)**2 + (ddec_dy_ref * sigma_y_ref)**2)

# loop over all stars
x_scaled = fivephot['x_fit'] / res
y_scaled = fivephot['y_fit'] / res
sigma_x = fivephot['x_err'] / res
sigma_y = fivephot['y_err'] / res

for name, x, y, sx, sy in zip(fivephot['name'], x_scaled, y_scaled, sigma_x, sigma_y):
    sky = w.pixel_to_world(x, y)
    
    # numerical derivatives for this star
    dra_dx = (w.pixel_to_world(x + dx, y).ra.deg - sky.ra.deg) / dx
    dra_dy = (w.pixel_to_world(x, y + dy).ra.deg - sky.ra.deg) / dy
    ddec_dx = (w.pixel_to_world(x + dx, y).dec.deg - sky.dec.deg) / dx
    ddec_dy = (w.pixel_to_world(x, y + dy).dec.deg - sky.dec.deg) / dy

    sigma_ra = np.sqrt((dra_dx * sx)**2 + (dra_dy * sy)**2)
    sigma_dec = np.sqrt((ddec_dx * sx)**2 + (ddec_dy * sy)**2)
    
    # separation
    sep = sky_ref.separation(sky)
    
    # propagated error in arcsec
    sigma_sep = np.sqrt(sigma_ra_ref**2 + sigma_dec_ref**2 + sigma_ra**2 + sigma_dec**2) * 3600
    
    print(f"{name} separation: {sep.to('arcsec'):.3f} ± {sigma_sep:.3f} arcsec")
#%%
#So fivephot is the version whose initial condition is a combination of finding stars in our image
#and information from literature.

#final phot is going to be a slightly different versions, where we go further and subtract a from the image 
#(based on what we got from fivephot) and refit the other four, then use those two combined as an init param
#then, we'll call a final! subtract a and run this on the 4 resid guys and those are the 
test_params=QTable()
test_params['x_0'] = [x_ref*res]
test_params['y_0'] = [y_ref*res]
test_params['flux'] = [ref_row['flux_fit'].item()]

#model just a and subtract, rerun phot
just_a=make_model_image(np.shape(zoomdata), psf_model, test_params)
without_a=zoomdata-just_a
four_init = fivephot[fivephot['name'] != 'a']
four_init = four_init[four_init['name'] != 'd']

print(four_init[['x_fit', 'y_fit', 'flux_fit', 'flags']])

fourphot=psfphot(without_a, init_params=four_init)
fourresid=psfphot.make_residual_image(without_a)
fourmodel=psfphot.make_model_image(np.shape(without_a))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(without_a, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(fourphot['x_fit'], fourphot['y_fit'], marker='x')
axes[1].imshow(fourmodel, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
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
fourphot['name']=['b','c','e']
sky22 = w.pixel_to_world(fourphot['x_fit']/res, fourphot['y_fit']/res)
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
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
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
print(finalphot[['x_fit', 'y_fit', 'flux_fit','x_init','y_init', 'flags', 'id']])
print(finalphot.columns)
finalphot['name']=['b','a','c','e']
ref_row2 = finalphot[finalphot['name'] == 'a'][0]

x_ref2 = ref_row2['x_fit'].item()/res
y_ref2 = ref_row2['y_fit'].item()/res
sky12 = w.pixel_to_world(x_ref2, y_ref2)
sky23 = w.pixel_to_world(finalphot['x_fit']/res, finalphot['y_fit']/res)
seps3 = sky12.separation(sky23)  # returns Quantity array in degrees

# loop over all stars

sigma_x_final = finalphot['x_err'] / res
sigma_y_final = finalphot['y_err'] / res

print('finalphot')
    
for name, x, y, sx, sy in zip(finalphot['name'], finalphot['x_fit']/res, finalphot['y_fit']/res, sigma_x_final, sigma_y_final):
    sky = w.pixel_to_world(x, y)
    
    # numerical derivatives for this star
    dra_dx = (w.pixel_to_world(x + dx, y).ra.deg - sky.ra.deg) / dx
    dra_dy = (w.pixel_to_world(x, y + dy).ra.deg - sky.ra.deg) / dy
    ddec_dx = (w.pixel_to_world(x + dx, y).dec.deg - sky.dec.deg) / dx
    ddec_dy = (w.pixel_to_world(x, y + dy).dec.deg - sky.dec.deg) / dy

    sigma_ra = np.sqrt((dra_dx * sx)**2 + (dra_dy * sy)**2)
    sigma_dec = np.sqrt((ddec_dx * sx)**2 + (ddec_dy * sy)**2)
    
    # separation
    sep = sky_ref.separation(sky)
    
    # propagated error in arcsec
    sigma_sep = np.sqrt(sigma_ra_ref**2 + sigma_dec_ref**2 + sigma_ra**2 + sigma_dec**2) * 3600
    
    print(f"{name} separation: {sep.to('arcsec'):.3f} ± {sigma_sep:.3f} arcsec")

flux_a = finalphot[finalphot['name'] == 'a']['flux_fit'].item()
flux_e = finalphot[finalphot['name'] == 'e']['flux_fit'].item()
flux_a_err = finalphot[finalphot['name'] == 'a']['flux_err'].item()
flux_e_err = finalphot[finalphot['name'] == 'e']['flux_err'].item()

total_flux = flux_a + flux_e

rel_flux_a = flux_a / total_flux
rel_flux_e = flux_e / total_flux
rel_flux_e_err = rel_flux_e*np.sqrt((flux_a_err/flux_a)**2+(flux_e_err/flux_e)**2)

print(f"Relative flux of 'a' (a / (a+e)) in finalphot: {rel_flux_a:.3f}")
print(f"Relative flux of 'e' (e / (a+e)) in finalphot: {rel_flux_e:.3f}+/-{rel_flux_e_err:.3f}")


flux_a_5 = fivephot[fivephot['name'] == 'a']['flux_fit'].item()
flux_e_5 = fivephot[fivephot['name'] == 'e']['flux_fit'].item()
flux_a_5_err = fivephot[fivephot['name'] == 'a']['flux_err'].item()
flux_e_5_err = fivephot[fivephot['name'] == 'e']['flux_err'].item()

total_flux_5 = flux_a_5 + flux_e_5

rel_flux_a_5 = flux_a_5 / total_flux_5
rel_flux_e_5 = flux_e_5 / total_flux_5
rel_flux_e_5_err = rel_flux_e_5*np.sqrt((flux_a_5_err/flux_a_5)**2+(flux_e_5_err/flux_e_5)**2)
print('fivephot')
print(f"Relative flux of 'a' (a / (a+e)) in fivephot: {rel_flux_a_5:.3f}")
print(f"Relative flux of 'e' (e / (a+e)) in fivephot: {rel_flux_e_5:.3f}+/-{rel_flux_e_5_err:.3f}")

for name, x, y, sx, sy in zip(fivephot['name'], x_scaled, y_scaled, sigma_x, sigma_y):
    sky = w.pixel_to_world(x, y)
    
    # numerical derivatives for this star
    dra_dx = (w.pixel_to_world(x + dx, y).ra.deg - sky.ra.deg) / dx
    dra_dy = (w.pixel_to_world(x, y + dy).ra.deg - sky.ra.deg) / dy
    ddec_dx = (w.pixel_to_world(x + dx, y).dec.deg - sky.dec.deg) / dx
    ddec_dy = (w.pixel_to_world(x, y + dy).dec.deg - sky.dec.deg) / dy

    sigma_ra = np.sqrt((dra_dx * sx)**2 + (dra_dy * sy)**2)
    sigma_dec = np.sqrt((ddec_dx * sx)**2 + (ddec_dy * sy)**2)
    
    # separation
    sep = sky_ref.separation(sky)
    
    # propagated error in arcsec
    sigma_sep = np.sqrt(sigma_ra_ref**2 + sigma_dec_ref**2 + sigma_ra**2 + sigma_dec**2) * 3600
    
    print(f"{name} separation: {sep.to('arcsec'):.3f} ± {sigma_sep:.3f} arcsec")

#%%
#plotting and comparing stuff, we need a separate version of just a for final
test_params2=QTable()
test_params2['x_0'] = [x_ref2*res]
test_params2['y_0'] = [y_ref2*res]
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
final_phot=fivephot
#final_phot.write("/home/kmc249/radec_finalphot_hires.csv", format="csv", overwrite=True)

#we need to put x and y back in the trimmed coordinates, so:

final_phot['scaled flux']=final_phot['flux_fit']/ave_ens_flux
final_phot['x_0']=final_phot['x_fit']/res+231
final_phot['y_0']=final_phot['y_fit']/res+231
final_phot[['name','x_0', 'y_0','scaled flux', 'x_err','y_err','flux_err']]

aql = final_phot[final_phot['name'] == 'e']
final_phot = final_phot[final_phot['name'] != 'e']


final_phot=final_phot[['name','x_0', 'y_0','scaled flux', 'x_err','y_err','flux_err']]

#final_phot.write("/home/kmc249/new_final_phot_hires.csv", format="csv", overwrite=True)
#aql.write("/home/kmc249/new_final_phot_aql_hires.csv", format="csv", overwrite=True)
print('flux error aql: ', aql['flux_err'].value[0])
print('max pos. error aql, in 512x512 land: ',np.nanmax(list(aql['y_err'])+list(aql['x_err']))/res)

interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
vmin2, vmax2 = interval.get_limits(stacked_trim)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
plt.figure(figsize=(10,12))
plt.imshow(stacked_trim_zoom, cmap='grey', origin='lower', norm=norm2, alpha=1)
plt.scatter(final_phot['x_0'], final_phot['y_0'], marker='x')
plt.scatter(aql['x_0'], aql['y_0'], marker='x')

plt.show()

print(final_phot['scaled flux'])

print(final_phot)


#%%
#outside so I don't have to rerun every time
epsf_cache={}

#%% dumb cha thin
whatiputin=five_init[[row['name'] != 'd' for row in five_init]]
#whatiputin=five_init

def get_epsf(size):
    """
    Return cached EPSF if available, otherwise build and cache it.
    """
    if size in epsf_cache:
        return epsf_cache[size]
    
    print(f"Building EPSF for size={size}")
    
    nddata = NDData(data=bkg_sub_full_data)
    good_stars = extract_stars(nddata, sources, size=size)
    
    valid_stars = [star for star in good_stars 
                   if np.isfinite(np.sum(star.data)) and np.sum(star.data) > 0]
    
    epsf_input = EPSFStars(valid_stars)
    epsf_builder = EPSFBuilder(oversampling=2, maxiters=10)
    epsf, _ = epsf_builder(epsf_input)
    
    epsf_cache[size] = epsf
    
    return epsf

def make_star_mask(shape, phot_table, radius, exclude_names=None):
    """
    Create a boolean mask selecting pixels near fitted stars.
    
    Parameters
    ----------
    exclude_names : list or None
        Star names to exclude from the mask (e.g., ['d'])
    """
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    mask = np.zeros(shape, dtype=bool)
    
    for row in phot_table:
        if exclude_names is not None and row['name'] in exclude_names:
            continue
        
        x0 = row['x_fit']
        y0 = row['y_fit']
        
        r2 = (x - x0)**2 + (y - y0)**2
        mask |= (r2 <= radius**2)
    
    return mask


def compute_chi2(data, model, resid, phot_table, radius=10, read_noise=5.0):
    """
    Approximate chi^2 using Poisson + read noise. Masked for around the pixels we care about
    """
    mask = make_star_mask(
        data.shape,
        phot_table,
        radius,
        exclude_names=['d']   # <-- KEY CHANGE
    )
    
    noise2 = np.abs(model) + read_noise**2
    
    chi2 = np.nansum((resid[mask]**2) / noise2[mask])
    n_pix = np.sum(mask)
    
    return chi2, n_pix

# ===== SET THESE =====
sep_lit = 0.46      # arcsec (example)
sep_err_lit = 0.01

flux_lit = 0.17     # e / (a+e)
flux_err_lit = 0.02

def compute_score(chi2_img, n_pix, n_params,
                  sep, sep_err,
                  flux, flux_err,
                  w_sep=2.0, w_flux=1.0):
    
    # reduced chi2
    chi2_red = chi2_img / (n_pix - n_params)
    
    # external constraints
    chi2_sep = ((sep - sep_lit) / sep_err_lit)**2
    chi2_flux = ((flux - flux_lit) / flux_err_lit)**2
    
    total_score = chi2_red + w_sep*chi2_sep + w_flux*chi2_flux
    
    return {
        "chi2_red": chi2_red,
        "chi2_sep": chi2_sep,
        "chi2_flux": chi2_flux,
        "score": total_score
    }
def run_fit(size, fitnum, ap_radius, return_full=False):
    
    # ===== rebuild EPSF =====
    psf_model = get_epsf(size)
    fit_shape = (fitnum*res+1, fitnum*res+1)
    
    # ===== photometry (ONLY fivephot now) =====
    psfphot = PSFPhotometry(
        psf_model,
        fit_shape,
        aperture_radius=ap_radius*res,
        grouper=SourceGrouper(min_separation=16)
    )
    
    result = psfphot(zoomdata, init_params=whatiputin)

    # re-attach names (Photutils drops them)
    result['name'] = whatiputin['name']
    
    model = psfphot.make_model_image(np.shape(zoomdata))
    resid = psfphot.make_residual_image(zoomdata)
    
    # ===== χ² (mask radius tied to fit scale) =====
    chi2, n_pix = compute_chi2(
        zoomdata,
        model,
        resid,
        result,
        radius=fitnum*res
    )
    
    # ===== separation =====
    ref = result[result['name'] == 'a'][0]
    sky_ref = w.pixel_to_world(ref['x_fit']/res, ref['y_fit']/res)
    
    e_star = result[result['name'] == 'e'][0]
    sky_e = w.pixel_to_world(e_star['x_fit']/res, e_star['y_fit']/res)
    
    sep = sky_ref.separation(sky_e).arcsec
    sep_err = 0.01  # keep your placeholder or replace later
    
    # ===== flux ratio =====
    flux_a = ref['flux_fit']
    flux_e = e_star['flux_fit']
    
    flux = flux_e / (flux_a + flux_e)
    
    flux_err = flux * np.sqrt(
        (ref['flux_err']/flux_a)**2 +
        (e_star['flux_err']/flux_e)**2
    )
    
    result_dict = {
        "size": size,
        "fitnum": fitnum,
        "ap_radius": ap_radius,
        "chi2": chi2,
        "n_pix": n_pix,
        "n_params": len(result)*3,
        "sep": sep,
        "sep_err": sep_err,
        "flux": flux,
        "flux_err": flux_err,
        "x_e": e_star['x_fit'],
        "y_e": e_star['y_fit'],
        "flux_a": flux_a,
        "flux_e": flux_e,
        "phot_table": result,
    }
    
    if return_full:
        result_dict["phot_table"] = result
        result_dict["model"] = model
        result_dict["resid"] = resid
    
    return result_dict

#%%
#grid search
sizes = [13, 15, 17, 19, 21]
fitnums = [10, 11, 12, 13, 14, 15]
ap_radii = [3, 4, 5, 6, 7, 8, 9, 10]

results = []

for s in sizes:
    for f in fitnums:
        for ap in ap_radii:
            try:
                result_dict = run_fit(s, f, ap)
                #return_full if I want the pics with it
                
                score_dict = compute_score(
                    result_dict["chi2"],
                    result_dict["n_pix"],
                    result_dict["n_params"],
                    result_dict["sep"],
                    result_dict["sep_err"],
                    result_dict["flux"],
                    result_dict["flux_err"]
                )
                
                result_dict.update(score_dict)
                results.append(result_dict)
                
                print(f"Done: size={s}, fitnum={f}, ap={ap}, score={result_dict['score']:.3f}")
            

            except Exception as e:
                print(f"FAILED: size={s}, fitnum={f}, ap={ap}", e)
                             
                
#%%
df = pd.DataFrame(results)

# best overall
best = df.sort_values("score").iloc[0]
print("\nBEST MODEL:")
print(best)

# check dependence on aperture radius
print("\nAverage score by aperture radius:")
print(df.groupby("ap_radius")["score"].mean())

# stability check
print("\nTop 10 solutions:")
print(df.sort_values("score").head(10))

#%%
top10 = df.sort_values("score").head(10)
'''
for i, row in top10.iterrows():
    print(f"\nPlotting: size={row['size']}, fitnum={row['fitnum']}, ap={row['ap_radius']}, score={row['score']:.3f}")
    
    res_full = run_fit(
        int(row['size']),
        int(row['fitnum']),
        int(row['ap_radius']),
        return_full=True
    )
    
    phot = res_full["phot_table"]
    model = res_full["model"]
    resid = res_full["resid"]
    
    # reuse your plotting function
    plotstuff(phot, model, resid, zoomdata)
    plt.suptitle(f"size={row['size']}, fitnum={row['fitnum']}, ap={row['ap_radius']}, score={row['score']:.3f}")
'''
#%%
top10 = df.sort_values("score").head(10)
data = top10  # or df

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.ravel()  # flatten for easy indexing

# ===== LEFT COLUMN: positions/separation =====
axes[0].hist(data['x_e'], bins=20)
axes[0].set_xlabel("x position of e (pixels)")
axes[0].set_ylabel("Count")
axes[0].set_title("x position of e")

axes[2].hist(data['y_e'], bins=20)
axes[2].set_xlabel("y position of e (pixels)")
axes[2].set_ylabel("Count")
axes[2].set_title("y position of e")

axes[4].hist(data['sep'], bins=20)
axes[4].set_xlabel("Separation (arcsec)")
axes[4].set_ylabel("Count")
axes[4].set_title("Separation")

# ===== RIGHT COLUMN: fluxes =====
axes[5].hist(data['flux'], bins=20)
axes[5].set_xlabel("Flux ratio e / (a+e)")
axes[5].set_ylabel("Count")
axes[5].set_title("Flux ratio")

# Individual fluxes (if you have them in your run_fit output)
axes[3].hist([res['flux_a'] for res in data.to_dict('records')], bins=20)
axes[3].set_xlabel("Flux of a")
axes[3].set_ylabel("Count")
axes[3].set_title("Flux of a")

axes[1].hist([res['flux_e'] for res in data.to_dict('records')], bins=20)
axes[1].set_xlabel("Flux of e")
axes[1].set_ylabel("Count")
axes[1].set_title("Flux of e")

plt.tight_layout()
plt.show()

#%%

#the current best fit
# Extract the photometry table from the best model
best_phot_table = best['phot_table']

#%%

import seaborn as sns

# Pick relevant columns
cols = ['size', 'fitnum', 'ap_radius', 'score', 'sep', 'flux']

sns.pairplot(df[cols], corner=True, diag_kind='hist', plot_kws={'alpha':0.7})
plt.suptitle("Parameter correlations and clustering", y=1.02)
plt.show()

#%%

best_result= run_fit(21, 12, 10, return_full=True)
phot = best_result["phot_table"]
model = best_result["model"]
resid = best_result["resid"]

neighbors=phot[phot['name']=='e']

test_params=neighbors[['id','group_id', 'group_size', 'name']]
test_params['x_0'], test_params['y_0'], test_params['flux']=neighbors['x_fit'], neighbors['y_fit'],neighbors['flux_fit']
print(test_params)
test=make_model_image(np.shape(zoomdata), psf_model, test_params)


# reuse your plotting function
#plotstuff(phot, model, resid, zoomdata)
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(zoomdata)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(phot['x_fit'], phot['y_fit'], marker='x')
#axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
axes[1].imshow(test, cmap='gray', origin='lower', norm=norm)
#for i in [0,1,2]:
    #axes[i].plot(x_vals, y_vals, color='orange')
    #axes[i].plot(x_vals2, y_vals2, color='g')
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
axes[2].imshow(zoomdata-test, cmap='gray', origin='lower', norm=norm)
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

#do the actual ensemble fit with the best result too
print(ave_ens_flux)
#fit to the ensemble stars; fit_shape has been tested to be the best
psf_model = epsf_cache[21]
fitnum=12
fit_shape=(fitnum*res+1,fitnum*res+1)
#fit_shape=(int(size/2-0.5),int(size/2-0.5))
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=10*res)
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
#ensphot.write("/home/kmc249/best_r_ensemble.csv", format="csv", overwrite=True)
#get ave ens flux:
ave_ens_flux=np.nanmean(ensphot['flux_fit'])
#%%
print(ave_ens_flux)

#getting reletavie fluxes to the ave ensemble and then saving
best_phot_table['stacked_flux_factor']=best_phot_table['flux_fit']/ave_ens_flux
# Keep only the columns you need for initialization
init_params_to_save = best_phot_table[['x_fit','x_err', 'y_fit','y_err', 'flux_fit','flux_err', 'name', 'group_id','id', 'group_size', 'stacked_flux_factor']].to_pandas()
init_params_to_save.to_csv('/home/kmc249/current_best_R_grid_fit.csv', index=False)



#%%