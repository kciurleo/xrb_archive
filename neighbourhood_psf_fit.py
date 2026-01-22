#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from astropy.table import QTable, Table
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


#load data
stacked_trim_str='/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits'
stacked_full = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits')
stacked_trim = fits.getdata('/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits')

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

for file in [stacked_trim_str]:
    imdata,hdr = fits.getdata(file,header=True)

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
    phot=psfphot(bkg_sub_full_data, init_params=init_params)
    resid=psfphot.make_residual_image(bkg_sub_full_data)
    model=psfphot.make_model_image(np.shape(bkg_sub_full_data))

    label=True
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(bkg_sub_full_data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    fig, axes=plt.subplots(1,3, figsize=(20,10))
    axes[0].imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
    axes[0].scatter(phot['x_fit'], phot['y_fit'], marker='x')
    axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
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
    
    ###here's where we only do the bit right around aql x1:

    zoomdata=bkg_sub_full_data[231:281,231:281]
    zoom_init=QTable()
    zoom_init['x']=[18.510958634533807,24.104306907227482,31.154202003458273, 26, 28.]
    zoom_init['y']=[20.62908720657539,23.834092810985606,25.154174418180144, 24.8, 25]
    zoom_init['name']=['b','a','c', 'aql', 'd']

    finder=DAOStarFinder(6.0, 2.0)
    grouper=SourceGrouper(min_separation=8)
    psfphot=PSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8)
    
    zoomphot=psfphot(zoomdata)
    zoomresid=psfphot.make_residual_image(zoomdata)
    zoommodel=psfphot.make_model_image(np.shape(zoomdata))
    
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
    
    
    #by hand 2nd iterative photometry

    zoomphot=psfphot(zoomresid)
    zoomresid=psfphot.make_residual_image(zoomdata)
    zoommodel=psfphot.make_model_image(np.shape(zoomdata))
    
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
    
#put back to normal coords
zoomphot['name'] = zoom_init['name']
zoomphot['x_fit'] = zoomphot['x_fit']+231
zoomphot['y_fit'] = zoomphot['y_fit']+231



label=False
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(resid, cmap='gray', origin='lower', norm=norm)
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

print(zoomphot[['x_init','x_fit', 'y_init', 'y_fit', 'id']])

#referencing a, how far away?


# reference star
ref_row = zoomphot[zoomphot['name'] == 'a'][0]
x_ref = ref_row['x_fit'].item()
y_ref = ref_row['y_fit'].item()
sky1 = w.pixel_to_world(x_ref, y_ref)
sky2 = w.pixel_to_world(zoomphot['x_fit'], zoomphot['y_fit'])
seps = sky1.separation(sky2)  # returns Quantity array in degrees

# print results
for name, sep in zip(zoomphot['name'], seps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")
    
#get separations by hand    
hand_x=np.array([18.510958634533807,24.104306907227482,31.154202003458273, 27.49707687254557, 29.79843875499086])+231
hand_y=np.array([20.62908720657539, 23.834092810985606,25.154174418180144, 24.800687567445088, 25.946833285054062])+231
skyother=w.pixel_to_world(24.104306907227482+231, 23.834092810985606+231)
handsky = w.pixel_to_world(hand_x, hand_y)
otherseps = skyother.separation(handsky) 
zoomphot['x_fit'] = zoomphot['x_fit']-231
zoomphot['y_fit'] = zoomphot['y_fit']-231



#%%
for name, sep in zip (zoom_init['name'], otherseps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")

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

#more stuff?
length=15
num_points=100
def line_eq(x,x1,y1,x2,y2):
    return y1+(y2-y1)/(x2-x1)*(x-x1)

def line_profile(id1, id2, phottable):
    x1, y1 = np.abs(phottable['x_fit'][id2]+phottable['x_fit'][id1])/2, np.abs(phottable['y_fit'][id2]+phottable['y_fit'][id1])/2
    x_vals = np.linspace(x1-length/2, x1+length/2, num_points)
    y_vals = line_eq(x_vals, phottable['x_fit'][id1], phottable['y_fit'][id1],phottable['x_fit'][id2], phottable['y_fit'][id2])
    distance=np.sqrt((x_vals-x_vals[0])**2+(y_vals-y_vals[0])**2)
    return(x_vals, y_vals, distance)
    

x_vals, y_vals, distance=line_profile(0,3,zoomphot)
x_vals2, y_vals2, distance2=line_profile(1,2,zoomphot)

fig, axes=plt.subplots(2,2, figsize=(18,8))

# ===== First pair =====
axes[0,0].plot(distance, map_coordinates(zoomdata, [y_vals, x_vals], order=1), label='data')
axes[0,0].plot(distance, map_coordinates(zoommodel, [y_vals, x_vals], order=1), label='model1 total (epsf)')

axes[0,0].legend()

axes[1,0].scatter(distance, map_coordinates(zoomdata, [y_vals, x_vals], order=1)
                  - map_coordinates(zoommodel, [y_vals, x_vals], order=1), label='model1 resid')
axes[1,0].axhline(0, color='black')
axes[1,0].legend()

# ===== Second pair =====
axes[0,1].plot(distance2, map_coordinates(zoomdata, [y_vals2, x_vals2], order=1), label='data')
axes[0,1].plot(distance2, map_coordinates(zoommodel, [y_vals2, x_vals2], order=1), label='model1 total (epsf)')


axes[0,1].legend()

axes[1,1].scatter(distance2, map_coordinates(zoomdata, [y_vals2, x_vals2], order=1)
                  - map_coordinates(zoommodel, [y_vals2, x_vals2], order=1), label='model1 resid')

axes[1,1].axhline(0, color='black')
axes[1,1].legend()

plt.show()

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes[1].imshow(zoommodel, cmap='gray', origin='lower', norm=norm)
for i in [0,1,2]:
    axes[i].plot(x_vals, y_vals, color='orange')
    axes[i].plot(x_vals2, y_vals2, color='g')
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

print(zoomphot['name'])
print(zoomphot['flux_fit'])
#%%
neighbor=zoomphot[zoomphot['name']=='a']
#make a model of star a
test_params=neighbor['id','name','group_id', 'group_size','local_bkg','npixfit','qfit','cfit','flags']
test_params['x_0'], test_params['y_0'], test_params['flux']=neighbor['x_fit'].value[0], neighbor['y_fit'].value[0],neighbor['flux_fit'].value[0],
test=make_model_image(np.shape(zoomdata), psf_model, test_params)

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes[1].imshow(zoommodel-test, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
for i in [0,1,2]:
    axes[i].plot(x_vals, y_vals, color='orange')
    axes[i].plot(x_vals2, y_vals2, color='g')
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
fig, axes=plt.subplots(2,2, figsize=(18,8))
# ===== First pair =====
axes[0,0].plot(distance, map_coordinates(zoomdata-test, [y_vals, x_vals], order=1), label='data')
axes[0,0].plot(distance, map_coordinates(zoommodel-test, [y_vals, x_vals], order=1), label='model1 total (epsf)')

axes[0,0].legend()

axes[1,0].scatter(distance, map_coordinates(zoomdata-test, [y_vals, x_vals], order=1)
                  - map_coordinates(zoommodel-test, [y_vals, x_vals], order=1), label='model1 resid')
axes[1,0].axhline(0, color='black')
axes[1,0].legend()

# ===== Second pair =====
axes[0,1].plot(distance2, map_coordinates(zoomdata-test, [y_vals2, x_vals2], order=1), label='data')
axes[0,1].plot(distance2, map_coordinates(zoommodel-test, [y_vals2, x_vals2], order=1), label='model1 total (epsf)')


axes[0,1].legend()

axes[1,1].scatter(distance2, map_coordinates(zoomdata-test, [y_vals2, x_vals2], order=1)
                  - map_coordinates(zoommodel-test, [y_vals2, x_vals2], order=1), label='model1 resid')

axes[1,1].axhline(0, color='black')
axes[1,1].legend()

plt.show()

#%%

###even more iterative:
withouta=zoomdata-test

withouta_init=QTable()
withouta_init['x']=[18.510958634533807,31.154202003458273, 26, 28.]
withouta_init['y']=[20.62908720657539,25.154174418180144, 24.8, 25]
withouta_init['name']=['b','c', 'aql', 'd']


#make the xy pixel fit dist. 0 so it doesn't move
finder=DAOStarFinder(6.0, 2.0)
grouper=SourceGrouper(min_separation=8)
psfphot=PSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8)

zoomphot2=psfphot(withouta, init_params=withouta_init)
zoomresid2=psfphot.make_residual_image(withouta)
zoommodel2=psfphot.make_model_image(np.shape(withouta))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(withouta, cmap='gray', origin='lower', norm=norm)
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
print(zoomphot2[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit']])


sky22 = w.pixel_to_world(zoomphot2['x_fit']+231, zoomphot2['y_fit']+231)
seps2 = sky1.separation(sky22)  # returns Quantity array in degrees

# print results
for name, sep in zip(withouta_init['name'], seps2):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")
    
#%%


x_vals, y_vals, distance=line_profile(0,2,zoomphot2)
x_vals2, y_vals2, distance2=line_profile(2,1,zoomphot2)

fig, axes=plt.subplots(2,2, figsize=(18,8))

# ===== First pair =====
axes[0,0].plot(distance, map_coordinates(withouta, [y_vals, x_vals], order=1), label='data')
axes[0,0].plot(distance, map_coordinates(zoommodel2, [y_vals, x_vals], order=1), label='model1 total (epsf)')

axes[0,0].legend()

axes[1,0].scatter(distance, map_coordinates(withouta, [y_vals, x_vals], order=1)
                  - map_coordinates(zoommodel2, [y_vals, x_vals], order=1), label='model1 resid')
axes[1,0].axhline(0, color='black')
axes[1,0].legend()

# ===== Second pair =====
axes[0,1].plot(distance2, map_coordinates(withouta, [y_vals2, x_vals2], order=1), label='data')
axes[0,1].plot(distance2, map_coordinates(zoommodel2, [y_vals2, x_vals2], order=1), label='model1 total (epsf)')


axes[0,1].legend()

axes[1,1].scatter(distance2, map_coordinates(withouta, [y_vals2, x_vals2], order=1)
                  - map_coordinates(zoommodel2, [y_vals2, x_vals2], order=1), label='model1 resid')

axes[1,1].axhline(0, color='black')
axes[1,1].legend()

plt.show()

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(withouta, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(zoomphot2['x_fit'], zoomphot2['y_fit'], marker='x')
axes[1].imshow(zoommodel2, cmap='gray', origin='lower', norm=norm)
axes[0].set_xlim(0,50)
axes[0].set_ylim(0,50)
for i in [0,1,2]:
    axes[i].plot(x_vals, y_vals, color='orange')
    axes[i].plot(x_vals2, y_vals2, color='g')
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