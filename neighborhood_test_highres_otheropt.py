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
    for i in [0,1,2]:
        axes[i].plot(x_vals, y_vals, color='orange')
        axes[i].plot(x_vals2, y_vals2, color='g')
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
res=2
band='V'
#load data
#stacked_trim_str='/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits'
#stacked_trim_str=f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits'
stacked_full = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits')
stacked_trim_zoom = fits.getdata('/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits')
#stacked_trim = fits.getdata(f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits')
stacked_trim = fits.getdata(f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits')
from scipy.ndimage import zoom
#stacked_trim = zoom(stacked_trim, 2, order=3)
hiresopt=fits.getdata('/home/kmc249/Downloads/hires_master.fits')

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

sources["xcentroid"]=sources["xcentroid"]*res
sources["ycentroid"]=sources["ycentroid"]*res
sources["x"]=sources["x"]*res
sources["y"]=sources["y"]*res

#making a new ensemble
#too variable + too close to edge/clearly multiple stars + way less bright than aql x-1
exclude_ids = [634, 566, 618, 444] +[704, 536, 385, 218, 168, 175, 484, 185, 203,388, 687]+[162, 163, 167, 171, 172, 174, 177, 179, 180, 188, 190, 191, 193, 196, 197, 198, 199, 200, 201, 204, 207, 210, 212, 220, 222, 223, 228, 229, 235, 236, 239, 240, 244, 248, 249, 250, 251, 253, 258, 265, 266, 268, 269, 270, 271, 274, 276, 278, 283, 286, 289, 290, 291, 293, 294, 298, 299, 301, 302, 303, 306, 310, 311, 313, 315, 316, 322, 323, 325, 326, 332, 336, 337, 342, 343, 345, 347, 349, 353, 356, 357, 358, 363, 370, 371, 374, 375, 377, 378, 380, 383, 384, 391, 392, 393, 400, 402, 404, 408, 409, 414, 415, 416, 419, 420, 422, 424, 425, 430, 431, 432, 433, 436, 437, 439, 447, 450, 452, 454, 455, 457, 462, 464, 465, 466, 468, 472, 475, 480, 481, 485, 488, 490, 491, 492, 494, 495, 499, 500, 504, 506, 507, 509, 513, 514, 515, 518, 519, 521, 522, 524, 527, 529, 531, 534, 535, 538, 539, 543, 549, 550, 554, 555, 556, 557, 558, 560, 561, 562, 565, 568, 571, 572, 575, 576, 577, 580, 584, 587, 594, 595, 596, 598, 599, 601, 602, 605, 606, 607, 608, 610, 613, 617, 621, 627, 628, 630, 636, 638, 639, 645, 647, 649, 652, 657, 659, 660, 661, 663, 664, 668, 669, 673, 678, 680, 682, 683, 685, 689, 691, 695, 697, 699, 700, 701, 707, 708, 709]

mask2 = ~np.isin(sources['id'], exclude_ids)

new_ensemble = sources[mask2].copy()
new_eids = list(new_ensemble['id'])
new_init_params = new_ensemble['id', 'flux', 'x', 'y']
#end making a new ensemble

#info about the ensemble
stacked_ensemble=Table.read('/home/kmc249/Downloads/ensemble_info.vot')
eids = list(stacked_ensemble['id'])


#put all the vots in the smaller x-y system
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]-301
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]-301
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]*res
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]*res
stacked_ensemble["x_fit"]=stacked_ensemble["x_fit"]*res
stacked_ensemble["y_fit"]=stacked_ensemble["y_fit"]*res


#init params to fit the neighbor and ensemble
init_params=stacked_ensemble['id','group_id','flux_init','x_fit','y_fit','flux_fit']

imdata,hdr = fits.getdata(f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits',header=True)
#imdata = zoom(imdata, 2, order=3)
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

#we need to get rid of the guys that are outside of this image if it's majorly shifted!! Some buffer around,
#the edges are wonky....or we just trim everything to max shift. temp fix noted below

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


#psf fitting the exact same way I used to for the "just the bit around aql x-1 512x512 square"
#in the original version of psf_fit.py

#THIS IS JUST FOR THE ENSEMBLE STARS, KT I THINK YOU WANT TO CHANGE
psf_model = epsf
fit_shape=(11*res+1,11*res+1)
#fit_shape=(int(size/2-0.5),int(size/2-0.5))
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8*res)#, xy_bounds=0.1*res)
#make the xy pixel fit dist. 0 so it doesn't move
ensphot=psfphot(bkg_sub_full_data, init_params=new_init_params)
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
#ensphot.write("/home/kmc249/new_ens_phot_hires.csv", format="csv", overwrite=True)
#get ave ens flux:
ave_ens_flux=np.nanmean(ensphot['flux_fit'])
#%%
print(ave_ens_flux)
print("EPSF sum:", np.sum(epsf.data))
#%%

###here's where we only do the bit right around aql x1:
#first step: just do detect magic
zoomdata=bkg_sub_full_data[231*res:281*res,231*res:281*res]
opticalinits = Table.from_pandas(pd.read_csv("/home/kmc249/radec_finalphot_hires.csv"))
opticalinits=opticalinits[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'id','group_id']]
opticalinits['name']=['b','a','c','e']

finder=DAOStarFinder(0.1, 8.0)
grouper=SourceGrouper(min_separation=12)
psfphot=PSFPhotometry(psf_model, fit_shape, grouper=grouper, finder=finder, aperture_radius=8*res)

zoomphot=psfphot(zoomdata, init_params=opticalinits)
zoomresid=psfphot.make_residual_image(zoomdata)
zoommodel=psfphot.make_model_image(np.shape(zoomdata))
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
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
print(zoomphot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'flux_err']])
#%%
#keep_ids = [21, 23, 24]
#keep_ids = [17,19,20]

mask = np.isin(zoomphot['id'], keep_ids)

zoom_init= zoomphot[mask]['x_init', 'x_fit', 'y_init', 'y_fit', 'flux_fit']
#zoom_init['name']=['b','a','c']
print(zoom_init)
#

#%%
'''
finder=DAOStarFinder(threshold=5. * std, fwhm=6.)
grouper=SourceGrouper(min_separation=8)
#psfphot=IterativePSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8, maxiters=3)

psfphot=PSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8)


#optical overplot
imdataopt,hdropt = fits.getdata('/home/kmc249/Downloads/hires_master.fits',header=True)
imdatazoom=imdataopt[231*res:281*res,231*res:281*res]

opticalinits = Table.from_pandas(pd.read_csv("/home/kmc249/radec_finalphot_hires.csv"))
opticalinits=opticalinits[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'id','group_id']]
opticalinits['name']=['b','a','c','e']


zoomphot=psfphot(zoomdata, init_params=opticalinits)
zoomresid=psfphot.make_residual_image(zoomdata)
zoommodel=psfphot.make_model_image(np.shape(zoomdata))
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
vminhr, vmaxhr = interval.get_limits(imdatazoom)
normhires = ImageNormalize(vmin=vminhr, vmax=vmaxhr, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].imshow(imdatazoom,  origin='lower', norm=normhires, alpha=0.5, cmap='viridis')
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
#axes[0].scatter(aql['x_fit'], aql['y_fit'], marker='.', c='green', label='aql')
#for name, group in neighborhood.groupby('name'):
#    axes[0].scatter(group['x_0']-206, group['y_0']-206, marker='.',c='green', label=name)
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
axes[1].imshow(zoommodel, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(zoomresid, cmap='gray', origin='lower', norm=norm)
#axes[2].imshow(hiresopt, cmap='gray', origin='lower', norm=normhires)

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
#keep_ids = [8, 6, 29, 57]
keep_ids = [1,2,3,4]
mask = np.isin(zoomphot['id'], keep_ids)

zoom_init= zoomphot[mask]#['x_init', 'x_fit', 'y_init', 'y_fit', 'flux_fit']
#zoom_init=zoomphot['x_init', 'x_fit', 'y_init', 'y_fit', 'flux_fit']
#zoom_init['name']=['a','b','e','c']
zoom_init['name']=['b','a','e','c']
print(zoom_init[['flux_fit','flux_err','name']])
'''
#%%
#then, using those 5 as the initial conditions, run it again
#five_init=zoom_init2['x_fit', 'y_fit', 'flux_fit']
five_init=QTable()
five_init['x']=[18.510958634533807,24.104306907227482,31.154202003458273, 26, 28.]
five_init['y']=[20.62908720657539,23.834092810985606,25.154174418180144, 24.8, 25]
five_init['name']=['b','a','c', 'aql', 'd']
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
#%%

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
for name, sep in zip(finalphot['name'], seps3):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")

flux_a = finalphot[finalphot['name'] == 'a']['flux_fit'].item()
flux_e = finalphot[finalphot['name'] == 'e']['flux_fit'].item()
total_flux = flux_a + flux_e

rel_flux_a = flux_a / total_flux
rel_flux_e = flux_e / total_flux

print(f"Relative flux of 'a' (a / (a+e)) in finalphot: {rel_flux_a:.3f}")
print(f"Relative flux of 'e' (e / (a+e)) in finalphot: {rel_flux_e:.3f}")

print(fivephot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit', 'flags', 'id']])
for name, sep in zip(fivephot['name'], seps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")

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
final_phot=finalphot
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