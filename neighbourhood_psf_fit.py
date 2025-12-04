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

#load data
neighborhoodmaster='/Users/katieciurleo/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits'
stacked_full = fits.getdata('/Users/katieciurleo/Downloads/AqlX-1_R_600.0_stack.fits')
stacked_trim = fits.getdata('/Users/katieciurleo/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits')

#load locations of good sources for ePSF
sources=Table.read('/Users/katieciurleo/Downloads/yalestuff/good_sources.vot')

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
stacked_ensemble=Table.read('/Users/katieciurleo/Downloads/yalestuff/ensemble_info.vot')
eids = list(stacked_ensemble['id'])

#put all the vots in the smaller x-y system
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]-301
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]-301


#init params to fit the neighbor and ensemble
init_params=stacked_ensemble['id','group_id','flux_init','x_fit','y_fit','flux_fit']

for file in [neighborhoodmaster]:
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

    #####THIS PART SHOULDN'T EVEN BE NEEDED

    #psf fitting, using init params
    psf_model = epsf
    fit_shape=(7,7)
    psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8, xy_bounds=0.1)
    #make the xy pixel fit dist. 0 so it doesn't move
    phot=psfphot(bkg_sub_full_data, init_params=init_params)
    resid=psfphot.make_residual_image(bkg_sub_full_data)
    model=psfphot.make_model_image(np.shape(bkg_sub_full_data))
    '''
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
    '''


    ###END UNNEEDED PART

    #neighbor
    nid=stacked_neighbor['id'][0]
    neighbor=phot[phot['id']==nid]

    #aqlx1
    aid=stacked_aql['id'][0]
    aql_table=phot[phot['id']==aid]

    #ensemble
    #ids=[69,79,71,18,7]
    ens = phot[np.isin(phot['id'], eids)]
    print(ens['flux_fit', 'flux_init'])

    ave_ens_flux=np.nanmean(ens['flux_fit'])
    #get scaled flux of neighbor for this particular exposure
    scaled_n_flux=stacked_flux_factor*ave_ens_flux

    #using stacked x and y positions, subtract a psf of this neighbor star
    test_params=neighbor['id','group_id', 'group_size','local_bkg','npixfit','qfit','cfit','flags']
    test_params['x_0'], test_params['y_0'], test_params['flux']=neighbor['x_fit'].value[0], neighbor['y_fit'].value[0],scaled_n_flux
    test=make_model_image(np.shape(bkg_sub_full_data), psf_model, test_params)

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
    plt.tight_layout()
    plt.savefig(f'/Users/katieciurleo/Downloads/yalestuff/sub_aqlx1/img_{file.split("/")[-1][:-5]}.png')
    #plt.show()

    final_data=imdata-test

    #hold tight to fluxes of ensemble stars
    for e in eids:
        ens_d[e].append(ens[ens['id']==e]['flux_fit'].value[0])
        big_df.at[ind, str(e)] = ens[ens['id'] == e]['flux_fit'].value[0]
        

    #hold onto neighbor flux
    nb.append(scaled_n_flux)
    big_df.at[ind, 'neighbor']=scaled_n_flux

    #modeled nb flux
    nb_modeled.append(neighbor['flux_fit'].value[0])

    #modeled aql flux
    aql.append(aql_table['flux_fit'].value[0])
    big_df.at[ind, 'aql']=aql_table['flux_fit'].value[0]

    savefits=True
    if savefits:
        hdr['SUBTR']=True
        fits.writeto(f'/Users/katieciurleo/Downloads/yalestuff/sub_aqlx1/sub_{file.split("/")[-1]}',final_data, hdr, overwrite=True)








'''
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
# epsf (lots of stars)
ind_models1 = individual_star_models(psf_model, phot, np.shape(data), star_indices=[73, 77])

# Gaussian (just the init_params stars)
ind_models2 = individual_star_models(psf_model2, phot2, np.shape(data), star_indices=[0, 1])


fig, axes=plt.subplots(2,2, figsize=(18,8))

# ===== First pair =====
axes[0,0].plot(distance, map_coordinates(bkg_sub_data, [y_vals, x_vals], order=1), label='data')
axes[0,0].plot(distance, map_coordinates(model, [y_vals, x_vals], order=1), label='model1 total (epsf)')

for idx, im in ind_models1:
    prof = map_coordinates(im, [y_vals, x_vals], order=1)
    axes[0,0].plot(distance, prof, '--', alpha=0.7, label=f'model1 star {idx}')

axes[0,0].legend()

axes[1,0].scatter(distance, map_coordinates(bkg_sub_data, [y_vals, x_vals], order=1)
                  - map_coordinates(model, [y_vals, x_vals], order=1), label='model1 resid')
axes[1,0].scatter(distance, map_coordinates(bkg_sub_data, [y_vals, x_vals], order=1)
                  - map_coordinates(model2, [y_vals, x_vals], order=1), label='model2 resid')
axes[1,0].axhline(0, color='black')
axes[1,0].legend()

# ===== Second pair =====
axes[0,1].plot(distance2, map_coordinates(bkg_sub_data, [y_vals2, x_vals2], order=1), label='data')
axes[0,1].plot(distance2, map_coordinates(model, [y_vals2, x_vals2], order=1), label='model1 total (epsf)')
axes[0,1].plot(distance2, map_coordinates(model2, [y_vals2, x_vals2], order=1), label='model2 total')

for idx, im in ind_models2:
    prof = map_coordinates(im, [y_vals2, x_vals2], order=1)
    axes[0,1].plot(distance2, prof, '--', alpha=0.7, label=f'model2 star {idx}')

axes[0,1].legend()

axes[1,1].scatter(distance2, map_coordinates(bkg_sub_data, [y_vals2, x_vals2], order=1)
                  - map_coordinates(model, [y_vals2, x_vals2], order=1), label='model1 resid')
axes[1,1].scatter(distance2, map_coordinates(bkg_sub_data, [y_vals2, x_vals2], order=1)
                  - map_coordinates(model2, [y_vals2, x_vals2], order=1), label='model2 resid')
axes[1,1].axhline(0, color='black')
axes[1,1].legend()

plt.show()
'''