#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:12:29 2026

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
from astropy.table import QTable, Table
from scipy.ndimage import map_coordinates
from photutils.psf import extract_stars
from astropy.nddata import NDData
from smith_utils import *
from photutils.datasets import make_model_image
import glob
from scipy.optimize import curve_fit
import json
import pandas as pd
from astropy.time import Time
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

#%%
#read in main df
fwhms=pd.read_csv('/home/kmc249/test_data/temp_aql_shifts.csv', low_memory=False)

bad_list=np.array(['130811.0067', '140529.0030','130504.0083','070524.0061','070809.0032','060710.0057', 
                   '070411.0052','070413.0070','030329.0212', '130504.0083','170730.0019', '171008.0030',
                   '130820.0064', '130820.0064', '150408.0159','051010.0021','050517.0149','050719.0101',
                   '040531.0042','120906.0021','090928.0041','070729.0050','030321.0139','030307.0212',
                   '160825.0042', '160904.0021', '170604.0070'])
bad_list='rccd'+bad_list+'.fits'

input_df=fwhms.loc[~fwhms['filename'].isin(bad_list)]

#for just 2008 purposes
input_df = fwhms.loc[
    (~fwhms['filename'].isin(bad_list)) &
    (fwhms['filename'].str.startswith('rccd08'))
]

filelist='/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_'+np.array(input_df['filename'])

stacked_trim = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits')

#load locations of good sources for ePSF
sources=Table.read('/home/kmc249/Downloads/good_sources.vot')
aql=pd.read_csv("/home/kmc249/new_final_phot_aql_hires.csv")
neighborhood=pd.read_csv("/home/kmc249/new_final_phot_hires.csv")
neighbors=neighborhood['name']

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

#making a new ensemble
#too variable + too close to edge/clearly multiple stars + way less bright than aql x-1
exclude_ids = [634, 566, 618, 444] +[704, 536, 385, 218, 168, 175, 484, 185, 203,388, 687]+[162, 163, 167, 171, 172, 174, 177, 179, 180, 188, 190, 191, 193, 196, 197, 198, 199, 200, 201, 204, 207, 210, 212, 220, 222, 223, 228, 229, 235, 236, 239, 240, 244, 248, 249, 250, 251, 253, 258, 265, 266, 268, 269, 270, 271, 274, 276, 278, 283, 286, 289, 290, 291, 293, 294, 298, 299, 301, 302, 303, 306, 310, 311, 313, 315, 316, 322, 323, 325, 326, 332, 336, 337, 342, 343, 345, 347, 349, 353, 356, 357, 358, 363, 370, 371, 374, 375, 377, 378, 380, 383, 384, 391, 392, 393, 400, 402, 404, 408, 409, 414, 415, 416, 419, 420, 422, 424, 425, 430, 431, 432, 433, 436, 437, 439, 447, 450, 452, 454, 455, 457, 462, 464, 465, 466, 468, 472, 475, 480, 481, 485, 488, 490, 491, 492, 494, 495, 499, 500, 504, 506, 507, 509, 513, 514, 515, 518, 519, 521, 522, 524, 527, 529, 531, 534, 535, 538, 539, 543, 549, 550, 554, 555, 556, 557, 558, 560, 561, 562, 565, 568, 571, 572, 575, 576, 577, 580, 584, 587, 594, 595, 596, 598, 599, 601, 602, 605, 606, 607, 608, 610, 613, 617, 621, 627, 628, 630, 636, 638, 639, 645, 647, 649, 652, 657, 659, 660, 661, 663, 664, 668, 669, 673, 678, 680, 682, 683, 685, 689, 691, 695, 697, 699, 700, 701, 707, 708, 709]

mask2 = ~np.isin(sources['id'], exclude_ids)

new_ensemble = sources[mask2].copy()
new_eids = list(new_ensemble['id'])
new_init_params = new_ensemble['id', 'flux', 'x', 'y']
#end making a new ensemble

#info about the neighbor
#stacked_ensemble=Table.read('/home/kmc249/Downloads/ensemble_info.vot')
#eids = list(stacked_ensemble['id'])


#flux factor (and error eventually)
stacked_flux_factors=neighborhood['scaled flux']

#init params to fit the neighbor and ensemble
#init_params=stacked_ensemble['id','flux_init','x_fit','y_fit','flux_fit']

#fallback means for if a star fails to be fit
#fallback=pd.read_csv('/home/kmc249/Downloads/psf_fluxes.csv', low_memory=False)
fallback=pd.read_csv('/home/kmc249/Downloads/new_psf_fluxes_neighborhood_2008_apphot.csv', low_memory=False)

fallback = fallback.drop(columns=['filename', 'time', 'aql'])#, 'neighbor'])
fallback_dict = fallback.mean(skipna=True).to_dict()

#ensemble fluxes to use for std
#NEW CHANGED
ens_d=dict((el,[]) for el in new_eids)

#make df
cols=['filename','time', 'aql'] + [str(n) for n in neighbors] + [str(e) for e in new_eids]
big_df = pd.DataFrame(0, index=np.arange(len(filelist)), columns=cols)
big_df['filename'] = filelist
big_df['time'] = pd.NaT 
big_df['aql'] = np.nan
for n in neighbors:
    big_df[str(n)] = np.nan 
for e in new_eids:
    big_df[str(e)] = np.nan 

showplot=True

#%%
print('-------------')
file=big_df.iloc[0]['filename']
print('trying', file)
#%%
try:
    imdata,hdr = fits.getdata(file,header=True)
except:
    nonexistent.append(file)
big_df.at[ind, 'time']=Time(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}")

#background subtract data
sigma_clip=SigmaClip(sigma=3.0)
bkg_estimator=MedianBackground()
fullbkg=Background2D(imdata, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
bkg_sub_full_data=imdata-fullbkg.background
bkg_sub_full_data = np.nan_to_num(bkg_sub_full_data, nan=0.0)

#plot sources on data just to double check

interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
vmin2, vmax2 = interval.get_limits(stacked_trim)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
plt.figure(figsize=(10,12))
plt.imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
plt.imshow(stacked_trim, cmap='viridis', origin='lower', norm=norm2, alpha=0.5)
plt.scatter(new_init_params['x'], new_init_params['y'], marker='x')
plt.scatter(neighborhood['x_0'], neighborhood['y_0'], marker='.')

for row in new_init_params:
    plt.annotate(
        str(row['id']), 
        (row['x_fit'], row['y_fit']),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=8,
        color='red'
    )
plt.show()
#%%

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

# filter out stars with the wrong shape
valid_stars = [star for star in valid_stars if star.data.shape == (size, size)]



epsf_input = EPSFStars(valid_stars)
epsf_builder=EPSFBuilder(oversampling=2, maxiters=10)
epsf, fitted_stars = epsf_builder(epsf_input)



#plot epsf just to see if it worked

norm=simple_norm(epsf.data, 'log', percent=99.0)

plt.imshow(epsf.data, norm=norm, origin='lower', cmap='gray')
plt.show()

#scale the epsf so that flux totals 1!!!
#epsf.data /= np.sum(epsf.data)
#%%

#####THIS PART SHOULDN'T EVEN BE NEEDED
#print(init_params)
#psf fitting, using init params
psf_model = epsf
fit_shape=(11,11)
#grouper=SourceGrouper(min_separation=8)
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8)#, xy_bounds=0.1)


#make the xy pixel fit dist. 0.1, avg error in original fit so it doesn't move
try:
    phot = psfphot(bkg_sub_full_data, init_params=new_init_params)
except ValueError as e:
    raise e

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
#axes[2].imshow(resid, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
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

#by-hand psf fit to all guys see if it picks up neighbors
print(Table.from_pandas(neighborhood))
phot2 = psfphot(bkg_sub_full_data, init_params=Table.from_pandas(neighborhood))


resid2=psfphot.make_residual_image(bkg_sub_full_data)
model2=psfphot.make_model_image(np.shape(bkg_sub_full_data))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(phot2['x_fit'], phot2['y_fit'], marker='.', alpha=0.2)
axes[1].imshow(model2, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(resid2, cmap='gray', origin='lower', norm=norm)
axes[2].set_xlim(156,356)
axes[2].set_ylim(156,356)
axes[1].set_xlim(156,356)
axes[1].set_ylim(156,356)
axes[0].set_xlim(156,356)
axes[0].set_ylim(156,356)
if label:
    for row in phot2:
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

###END UNNEEDED PART

#ensemble
#ids=[69,79,71,18,7]
ens = phot[np.isin(phot['id'], new_eids)]

#Build an array of fallback (mean) values from the first 
#fallback_vals = np.array([fallback_dict[str(i)] for i in ens['id']])
fallback_vals = np.array([fallback_dict.get(str(i), np.nan) for i in ens['id']])


#Replace NaNs in flux_fit with fallback_vals
filled_flux = np.where(np.isnan(ens['flux_fit']), fallback_vals, ens['flux_fit'])

ave_ens_flux = np.mean(filled_flux)

#get scaled flux of neighbor for this particular exposure
neighborhood['flux']=stacked_flux_factors*ave_ens_flux

#using stacked x and y positions, subtract a psf of this neighbor star
test=make_model_image(np.shape(bkg_sub_full_data), psf_model, Table.from_pandas(neighborhood))


interval = ZScaleInterval()
vmin, vmax = interval.get_limits(bkg_sub_full_data)
vmin2, vmax2 = interval.get_limits(imdata)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())

fig, axes=plt.subplots(1,3, figsize=(20,8))
axes[0].imshow(imdata, cmap='gray', origin='lower', norm=norm2)
axes[0].set_xlim(156,356)
axes[0].set_ylim(156,356)
axes[1].imshow(test, cmap='gray', origin='lower', norm=norm)
axes[1].set_xlim(156,356)
axes[1].set_ylim(156,356)
axes[2].imshow(imdata-test, cmap='gray', origin='lower', norm=norm2)
axes[2].set_xlim(156,356)
axes[2].set_ylim(156,356)
plt.suptitle(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}")
plt.tight_layout()
#plt.savefig(f"/home/kmc249/Downloads/aqlpics/sub_{hdr['DATE-OBS']}T{hdr['TIME-OBS']}.png")
plt.show()
#%%
neighborhood['group_id']=1
neighborhood['group_size']=3

#what are we getting from the test model
finder=DAOStarFinder(6.0, 8.0)
grouper=SourceGrouper(min_separation=12)
psfphot2=PSFPhotometry(psf_model, fit_shape, grouper=grouper, finder=finder, aperture_radius=8)

phot3 = psfphot2(test, init_params=Table.from_pandas(neighborhood))

resid3=psfphot2.make_residual_image(test)
model3=psfphot2.make_model_image(np.shape(test))

fig, axes=plt.subplots(1,3, figsize=(20,10))
axes[0].imshow(test, cmap='gray', origin='lower', norm=norm)
axes[0].scatter(phot3['x_fit'], phot3['y_fit'], marker='.', alpha=0.2)
axes[1].imshow(model3, cmap='gray', origin='lower', norm=norm)
axes[2].imshow(resid3, cmap='gray', origin='lower', norm=norm)
axes[2].set_xlim(156,356)
axes[2].set_ylim(156,356)
axes[1].set_xlim(156,356)
axes[1].set_ylim(156,356)
axes[0].set_xlim(156,356)
axes[0].set_ylim(156,356)
if label:
    for row in phot3:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

plt.show()
print(ave_ens_flux)
print(neighborhood['flux'])
print(phot3['flux_fit'])

#%%
final_data=imdata-test

#hold tight to fluxes of ensemble stars
for e in new_eids:
    ens_d[e].append(ens[ens['id']==e]['flux_fit'].value[0])
    big_df.at[ind, str(e)] = ens[ens['id'] == e]['flux_fit'].value[0]
#and the neighborhood
for n in neighbors:
    big_df.at[ind, str(n)] = neighborhood[neighborhood['name'] == n]['flux'].iloc[0]
#print(big_df[['a','b','c','d']])

'''
#unlike the other version, we now need to model aql separately; change the x-y bounds to reflect the error in original fit
bound=np.nanmax(np.concatenate([aql['y_err'].values, aql['x_err'].values]))
aql_init=Table.from_pandas(aql[['x_0', 'y_0', 'x_err', 'y_err']])
psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8, xy_bounds=bound)
aqlphot=psfphot(final_data, init_params=aql_init)
aqlresid=psfphot.make_residual_image(final_data)
aqlmodel=psfphot.make_model_image(np.shape(final_data))
if showplot:
    
    fig, axes=plt.subplots(1,3, figsize=(20,10))
    axes[0].imshow(final_data, cmap='gray', origin='lower', norm=norm2)
    axes[0].scatter(aqlphot['x_fit'], aqlphot['y_fit'], marker='x')
    axes[1].imshow(aqlmodel, cmap='gray', origin='lower', norm=norm2)
    axes[2].imshow(aqlresid, cmap='gray', origin='lower', norm=norm2)
    axes[0].set_xlim(156,356)
    axes[0].set_ylim(156,356)
    axes[1].set_xlim(156,356)
    axes[1].set_ylim(156,356)
    axes[2].set_xlim(156,356)
    axes[2].set_ylim(156,356)
    for row in aqlphot:
        axes[0].annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )

    plt.show()
print(aqlphot[['x_init','x_fit', 'y_init', 'y_fit', 'flux_fit']])
'''

#####dumb ap phot method
#position stuff
ap_radius = 6.0
annulus_r_in = 8.0
annulus_r_out = 12.0

positions = [(aql['x_0'].iloc[0], aql['y_0'].iloc[0])]

aperture = CircularAperture(positions, r=ap_radius)
annulus = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

#aperture photometry
ap_table = aperture_photometry(final_data, [aperture, annulus])

#do i actually need to do bkg subtraction? ignoring for now
bkg_mean = ap_table['aperture_sum_1'] / annulus.area
bkg_total = bkg_mean * aperture.area

aql_flux = ap_table['aperture_sum_0'] #- bkg_total

#plot
if showplot:
    plt.figure(figsize=(6,6))
    plt.imshow(final_data, cmap='gray', origin='lower', norm=norm2)
    aperture.plot(color='red')
    #annulus.plot(color='blue')
    plt.xlim(156,356)
    plt.ylim(156,356)
    plt.title(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}")
    plt.savefig(f"/home/kmc249/Downloads/aqlpics/ap_{hdr['DATE-OBS']}T{hdr['TIME-OBS']}.png")
    plt.show()


#photometry on top, psf on bottom
big_df.at[ind, 'aql'] = aql_flux[0]
#big_df.at[ind, 'aql']=aqlphot['flux_fit'].value[0]

savefits=True
if savefits:
    hdr['SUBTR']=True
    fits.writeto(f'/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/sub_{file.split("/")[-1]}',final_data, hdr, overwrite=True)
if showplot:
    showyn=input('continue to show plots?')
    if 'y' in showyn:
        showplot=True
    else:
        showplot=False
