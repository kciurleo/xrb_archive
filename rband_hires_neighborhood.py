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
band='I'
#load data
#stacked_trim_str='/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits'
stacked_trim_str=f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits'
stacked_full = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits')
stacked_trim_zoom = fits.getdata('/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits')
#stacked_trim = fits.getdata(f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits')
stacked_trim = fits.getdata(f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits')
hiresopt=fits.getdata('/home/kmc249/Downloads/hires_master.fits')

w = WCS(fits.getheader('/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER1.fits'))


#opticalinits = Table.from_pandas(pd.read_csv("/home/kmc249/radec_finalphot_hires.csv"))
opticalinits = Table.from_pandas(pd.read_csv("/home/kmc249/current_best_grid_fit.csv"))
opticalinits=opticalinits[['x_fit','y_fit','name','group_id','id','group_size']]
opticalinits['name']=['b','a','c','e']

'''
#putting a pin on the sources from optical
#load locations of good sources for ePSF
sources=Table.read('/home/kmc249/Downloads/good_sources.vot')

#put in correct x-y and get rid of epsf guys we won't use
sources["xcentroid"]=sources["xcentroid"]-301
sources["ycentroid"]=sources["ycentroid"]-301
sources["x"]=sources["x"]-301
sources["y"]=sources["y"]-301
mask=((sources['xcentroid']>= 125) &
    (sources['xcentroid']< 475) &
    (sources['ycentroid']>= 50) &
    (sources['ycentroid']< 390))
sources=sources[mask]

sources["xcentroid"]=sources["xcentroid"]*res
sources["ycentroid"]=sources["ycentroid"]*res
sources["x"]=sources["x"]*res
sources["y"]=sources["y"]*res

#making a new ensemble
#too variable + too close to edge/clearly multiple stars + way less bright than aql x-1
exclude_ids = [595]+[634, 566, 618, 444] +[704, 536, 385, 218, 168, 175, 484, 185, 203,388, 687]+[162, 163, 167, 171, 172, 174, 177, 179, 180, 188, 190, 191, 193, 196, 197, 198, 199, 200, 201, 204, 207, 210, 212, 220, 222, 223, 228, 229, 235, 236, 239, 240, 244, 248, 249, 250, 251, 253, 258, 265, 266, 268, 269, 270, 271, 274, 276, 278, 283, 286, 289, 290, 291, 293, 294, 298, 299, 301, 302, 303, 306, 310, 311, 313, 315, 316, 322, 323, 325, 326, 332, 336, 337, 342, 343, 345, 347, 349, 353, 356, 357, 358, 363, 370, 371, 374, 375, 377, 378, 380, 383, 384, 391, 392, 393, 400, 402, 404, 408, 409, 414, 415, 416, 419, 420, 422, 424, 425, 430, 431, 432, 433, 436, 437, 439, 447, 450, 452, 454, 455, 457, 462, 464, 465, 466, 468, 472, 475, 480, 481, 485, 488, 490, 491, 492, 494, 495, 499, 500, 504, 506, 507, 509, 513, 514, 515, 518, 519, 521, 522, 524, 527, 529, 531, 534, 535, 538, 539, 543, 549, 550, 554, 555, 556, 557, 558, 560, 561, 562, 565, 568, 571, 572, 575, 576, 577, 580, 584, 587, 594, 595, 596, 598, 599, 601, 602, 605, 606, 607, 608, 610, 613, 617, 621, 627, 628, 630, 636, 638, 639, 645, 647, 649, 652, 657, 659, 660, 661, 663, 664, 668, 669, 673, 678, 680, 682, 683, 685, 689, 691, 695, 697, 699, 700, 701, 707, 708, 709]

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
'''
imdata,hdr = fits.getdata(stacked_trim_str,header=True)

#background subtract data
sigma_clip=SigmaClip(sigma=3.0)
bkg_estimator=MedianBackground()
fullbkg=Background2D(imdata, (20*res,20*res), filter_size=(3*res+1,3*res+1),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
bkg_sub_full_data=imdata-fullbkg.background

#get sources

mean, median, std = sigma_clipped_stats(bkg_sub_full_data, sigma=3, maxiters=5)

daofinder = DAOStarFinder(threshold=6. * std, fwhm=5.)
sources = daofinder(bkg_sub_full_data)


#right here we need to get rid of any bad psf stars:
#bad_ids = [231,228,213,227,202,221,219,194,183,180,178,176,371,177,171,165,174, 225, 173, 153]
#bad_ids=[592,545,386,500,502]
#bad_ids=[524,514,507,491,420,494,460,467,462,470,455,477,436,552,551,456,517,525,521,549,547,479,485,422,418,325,310,397,393,400,329,450,457,472,464,468,460,478,487,479,495,385,372,543,3379,360,352,340,346,337,358,349,295,291,329,335,326,336,110,325,544,528,498,490,504,518,519,523,520,501,505,489,410,402,473,376,378,398,395,343,347,332,341,295,292,298,367,369,357,361,355,285,254,252,265,217,211,204,215,203,138,139,190,206,224,219,200,196,173,112,68,58,59,1,9,7,2,23,90,80,65,78,66,75,101,107,105,23,25,28,29,37,38,66,78,70,75,63,64,90,80,73,71,69,65,60,55,181,43,111,87,118,56,48,26,17,41,32,54,52,19,16,36,22,35,24,11,10,20,177,169,210,213,172,175,155,160,146,264,251,289,279,267,284,261,91,97,100]
#bad_ids=[719,714,709,685,639,663,653,703,712,713,673,674,650,658,645,661,611,610,622,629,599,543,533,549,577,553,558,502,497,446,431,373,391,387,388,453,454,429,535,441,458,632,634,688,619,620,629,623,636,648,669,629,641,631,638,659,667,666,668,684,693,715,721,643,636,635,637,630,707,706,711,614,575,571,573,602,582,585,564,559,555,556,551,545,514,523,540,532,518,516,542,554,474,467,490,482,420,421,448,435,380,353,368,370,404,407,346,435,340,337,330,308,297,316,288,286,283,284,327,329,336,304,302,258,263,298,310,270,266,255,243,233,221,217,229,231,188,174,165,164,153,38,51,54,44,33,43,20,63,80,85,139,138,114,123,120,115,19,34,62,70,61,104,151,229,258,373,333,334,328,325,282,267,280,275,295,335,338,300,299,306,321,244,211,216,229,207,215,208,213,200,187,189,192,179,173,160,172,161,158,150,118,132,135,124,122,106,119,101,94,65,68,26,24,35,72,82,76,37,45,96,59,60,29,55,53,62,90,93,97,128,91,83]
bad_ids=[]
# --- Edge mask ---

mask_edges = (
    (sources['xcentroid'] >= 30) &
    (sources['xcentroid'] < 1024-30) &
    (sources['ycentroid'] >= 30) &
    (sources['ycentroid'] < 1024-30)
)


sources = sources[mask_edges]

# --- ID mask ---
mask_ids = ~np.isin(sources['id'], bad_ids)

# --- Filtered sources ---
sources = sources[mask_ids]
'''

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
mask_edges = (
    (sources['xcentroid'] >= 30) &
    (sources['xcentroid'] < 1024-30) &
    (sources['ycentroid'] >= 30) &
    (sources['ycentroid'] < 1024-30)
)


ids_to_remove_from_epsf = [566,618,634,626,687,53,571,625,621,515,584,467,388,466,450,385,347,280,245,203,185,240,175,168,285,277,382,313,365,404]
epsfmask = ~np.isin(sources['id'], ids_to_remove_from_epsf)
sources = sources[epsfmask]  

'''    
sources['x'] = sources['xcentroid']
sources['y'] = sources['ycentroid']

#plot just to check

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


'''
#looping over stars
for i, star in enumerate(good_stars):
    star_id = sources['id'][i]   # match star to its ID
    
    # Skip invalid stars if you want
    if not (np.isfinite(np.sum(star.data)) and np.sum(star.data) > 0):
        continue

    plt.figure(figsize=(4,4))
    
    norm = simple_norm(star.data, 'log', percent=99.0)
    plt.imshow(star.data, origin='lower', cmap='gray', norm=norm)
    
    plt.title(f"Star ID: {star_id}")
    plt.colorbar()
    plt.show()
'''

sources.write('/home/kmc249/Downloads/good_sources_I.vot', overwrite=True, format='votable')

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
##testing testing testing
psf_model = epsf
fit_shape=(11,11)
zoomdata=bkg_sub_full_data[2*231:2*281,2*231:2*281]

aql=pd.read_csv("/home/kmc249/new_final_phot_aql_hires.csv")
neighborhood=pd.read_csv("/home/kmc249/new_final_phot_hires.csv")

finder=DAOStarFinder(threshold=5. * std, fwhm=6.)
grouper=SourceGrouper(min_separation=8)
#psfphot=IterativePSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8, maxiters=3)

psfphot=PSFPhotometry(psf_model, fit_shape, finder=finder, grouper=grouper, aperture_radius=8)


#optical overplot
imdataopt,hdropt = fits.getdata('/home/kmc249/Downloads/hires_master.fits',header=True)
imdatazoom=imdataopt[231*res:281*res,231*res:281*res]


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
zoom_init['name']=opticalinits['name']
print(zoom_init[['flux_fit','flux_err','name']])

#%%
#hiresw= WCS(fits.getheader('/home/kmc249/Downloads/hires_master.fits'))
x_scaled = zoom_init['x_fit'] / res
y_scaled = zoom_init['y_fit'] / res

ref_row = zoom_init[zoom_init['name'] == 'a'][0]
x_ref = ref_row['x_fit'].item()/res
y_ref = ref_row['y_fit'].item()/res
sky1 = w.pixel_to_world(x_ref, y_ref)
sky2 = w.pixel_to_world(x_scaled, y_scaled)
seps = sky1.separation(sky2)  # returns Quantity array in degrees

e_row = zoom_init[zoom_init['name'] == 'e'][0]
flux_ratio=e_row['flux_fit']/(ref_row['flux_fit']+e_row['flux_fit'])

print(f'flux ratio: {flux_ratio}')
# print results
for name, sep in zip(zoom_init['name'], seps):
    print(f"{name} separation from reference: {sep.to('arcsec'):.3f} arcsec")

#%%

vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
vminhr, vmaxhr = interval.get_limits(imdatazoom)
normhires = ImageNormalize(vmin=vminhr, vmax=vmaxhr, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].imshow(imdatazoom,  origin='lower', norm=normhires, alpha=0.5, cmap='viridis')
axes[0].scatter(zoomphot['x_fit'], zoomphot['y_fit'], marker='x')
axes[0].scatter(opticalinits['x_fit'], opticalinits['y_fit'], marker='.', c='cyan')
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
    
for row in opticalinits:
    axes[0].annotate(
        str(row['name']), 
        (row['x_fit'], row['y_fit']),
        textcoords="offset points",
        xytext=(5,-5),
        fontsize=8,
        color='cyan'
    )

plt.show()
#%%

#copied and pasted from the optical stuff
epsf_cache={}

#%% dumb cha thin
whatiputin=opticalinits

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

##we don't know flux_lit for B so ignore
#flux_lit = 0.17     # e / (a+e)
#flux_err_lit = 0.02

def compute_score(chi2_img, n_pix, n_params,
                  sep, sep_err,
                  flux, flux_err,
                  w_sep=1.0, w_flux=1.0):
    
    # reduced chi2
    chi2_red = chi2_img / (n_pix - n_params)
    
    # external constraints
    chi2_sep = ((sep - sep_lit) / sep_err_lit)**2
    #chi2_flux = ((flux - flux_lit) / flux_err_lit)**2
    
    total_score = chi2_red + w_sep*chi2_sep #+ w_flux*chi2_flux
    
    return {
        "chi2_red": chi2_red,
        "chi2_sep": chi2_sep,
        #"chi2_flux": chi2_flux,
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
data = df#top10  # or df

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

# Keep only the columns you need for initialization
init_params_to_save = best_phot_table[['x_fit', 'y_fit', 'flux_fit', 'name', 'group_id','id', 'group_size']].to_pandas()
init_params_to_save.to_csv(f'/home/kmc249/current_best_{band}_grid_fit.csv', index=False)

#%%

import seaborn as sns

# Pick relevant columns
cols = ['size', 'fitnum', 'ap_radius', 'score', 'sep', 'flux']

sns.pairplot(df[cols], corner=True, diag_kind='hist', plot_kws={'alpha':0.7})
plt.suptitle("Parameter correlations and clustering", y=1.02)
plt.show()

#%%

best_result= run_fit(17, 11, 9, return_full=True)

phot = best_result["phot_table"]
model = best_result["model"]
resid = best_result["resid"]

# reuse your plotting function
vmin, vmax = interval.get_limits(bkg_sub_full_data)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
vminhr, vmaxhr = interval.get_limits(imdatazoom)
normhires = ImageNormalize(vmin=vminhr, vmax=vmaxhr, stretch=SinhStretch())
fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(zoomdata, cmap='gray', origin='lower', norm=norm)
axes[0].imshow(imdatazoom,  origin='lower', norm=normhires, alpha=0.5, cmap='viridis')
axes[0].scatter(phot['x_fit'], phot['y_fit'], marker='x')
#axes[0].scatter(aql['x_fit'], aql['y_fit'], marker='.', c='green', label='aql')
#for name, group in neighborhood.groupby('name'):
#    axes[0].scatter(group['x_0']-206, group['y_0']-206, marker='.',c='green', label=name)
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)
axes[1].imshow(model, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(resid, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(opticalinits['x_fit'], opticalinits['y_fit'], marker='.', c='cyan')
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
    
for row in opticalinits:
    axes[0].annotate(
        str(row['name']), 
        (row['x_fit'], row['y_fit']),
        textcoords="offset points",
        xytext=(5,-5),
        fontsize=8,
        color='cyan'
    )

plt.show()

#%%

#info about the ensemble
stacked_ensemble=Table.read('/home/kmc249/Downloads/ensemble_info.vot')
eids = list(stacked_ensemble['id'])
### Getting rid of the two stars that are not going to be useful (one is 'negative', one is variable):
ids_to_remove = [1320, 413, 410]
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
init_params=stacked_ensemble['id','group_id','x_fit','y_fit',]

#do the actual ensemble fit with the best result too
#fit to the ensemble stars; fit_shape has been tested to be the best
psf_model = epsf_cache[17]
fitnum=11
fit_shape=(fitnum*res+1,fitnum*res+1)
#fit_shape=(int(size/2-0.5),int(size/2-0.5))
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=9*res)
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
ensphot.write("/home/kmc249/best_I_ensemble.csv", format="csv", overwrite=True)
#get ave ens flux:
ave_ens_flux=np.nanmean(ensphot['flux_fit'])
#%%
print(ave_ens_flux)

#getting reletavie fluxes to the ave ensemble and then saving
best_phot_table['stacked_flux_factor']=best_phot_table['flux_fit']/ave_ens_flux
# Keep only the columns you need for initialization
init_params_to_save = best_phot_table[['x_fit','x_err', 'y_fit','y_err', 'flux_fit','flux_err', 'name', 'group_id','id', 'group_size', 'stacked_flux_factor']].to_pandas()
init_params_to_save.to_csv('/home/kmc249/current_best_I_grid_fit.csv', index=False)