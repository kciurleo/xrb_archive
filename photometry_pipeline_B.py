import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
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

#read in file list
band='V'
telescope='1m'
target='AqlX-1'

filelist=glob.glob(f'/neta/xrb/{target}/{telescope}/opt/rccd/{band}_trimmed/trim_*')

#ap phot size
r_ap = 8.0

#r band stack just for plotting purposes, this is a 512 x 512
stacked_trim_str = f'/neta/xrb/AqlX-1/temp/Aql_{band}_stack.fits'

#load locations of good sources for ePSF, specific to band
sources=Table.read(f'/home/kmc249/Downloads/good_sources_{band}.vot')


#for now, change to 2 when I get the 512 x 512 images
res=2

#put in correct x-y and get rid of epsf guys we won't use
sources["xcentroid"]=sources["xcentroid"]/res
sources["ycentroid"]=sources["ycentroid"]/res
sources["x"]=sources["x"]/res
sources["y"]=sources["y"]/res

#info about the neighbor
neighborhood=pd.read_csv(f'/home/kmc249/current_best_{band}_grid_fit.csv')
stacked_ensemble=pd.read_csv(f"/home/kmc249/best_{band}_ensemble.csv")
eids = list(stacked_ensemble['id'])

#put all the vots in the smaller x-y system, from tthe resolution of x2
#and the neighborhood from the really zoomed in to the normal 512 x 512
neighborhood["x_fit"]=neighborhood["x_fit"]/res+231
neighborhood["y_fit"]=neighborhood["y_fit"]/res+231
stacked_ensemble["x_fit"]=stacked_ensemble["x_fit"]/res
stacked_ensemble["y_fit"]=stacked_ensemble["y_fit"]/res

#flux factor (and error eventually)
stacked_b=neighborhood.loc[neighborhood['name']=='b']
stacked_c=neighborhood.loc[neighborhood['name']=='c']
neighbors=neighborhood.loc[neighborhood['name'].isin(['b','c'])]
stacked_aql=neighborhood.loc[neighborhood['name']=='e']
stacked_flux_factor_b=stacked_b['stacked_flux_factor'].values[0]
stacked_flux_factor_c=stacked_c['stacked_flux_factor'].values[0]

#init params to fit the ensemble
init_params=stacked_ensemble[['id','flux_init','x_fit','y_fit','flux_fit']]
init_params['group_id'] = np.arange(len(init_params))+1
init_params['id'] = init_params['id'].astype(int)
init_params = Table.from_pandas(init_params)

#ensemble fluxes to use for std
ens_d=dict((el,[]) for el in eids)
nb=[]
aql=[]

#make df
cols=['filename','time', 'aql','neighbor'] + [str(e) for e in eids]
big_df = pd.DataFrame(0, index=np.arange(len(filelist)), columns=cols)
big_df['filename'] = filelist
big_df['time'] = pd.NaT 
big_df['aql'] = np.nan
big_df['neighbor'] = np.nan
for e in eids:
    big_df[str(e)] = np.nan 

showplot=True
nonexistent=[]
problems=[]
for ind, row in big_df.iterrows():
    print(f'working on {ind} of {len(filelist)}')
    file=row['filename']
    print('trying', file)
    try:
        imdata,hdr = fits.getdata(file,header=True)
        from scipy.ndimage import zoom
        #DELETE LATER KT
        imdata=zoom(imdata, 0.5, order=3)
    except:
        nonexistent.append(file)
        continue
    big_df.at[ind, 'time']=Time(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}")

    
    #background subtract data
    sigma_clip=SigmaClip(sigma=3.0)
    bkg_estimator=MedianBackground()
    fullbkg=Background2D(imdata, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkg_sub_full_data=imdata-fullbkg.background
    bkg_sub_full_data = np.nan_to_num(bkg_sub_full_data, nan=0.0)

    #plot sources on data just to double check
    '''
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(bkg_sub_full_data)
    #vmin2, vmax2 = interval.get_limits(stacked_trim)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    #norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
    #plt.imshow(stacked_trim, cmap='gray', origin='lower', norm=norm2, alpha=0.5)
    plt.scatter(init_params['x_fit'], init_params['y_fit'], marker='x')
    plt.scatter(neighborhood['x_fit'], neighborhood['y_fit'], marker='x')
    plt.scatter(sources['xcentroid'], sources['ycentroid'], marker='+', c='gold')

    for row in init_params:
        plt.annotate(
            str(row['id']), 
            (row['x_fit'], row['y_fit']),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
            color='red'
        )
    plt.show()
    '''
    #we need to get rid of the guys that are outside of this image if it's majorly shifted!! Some buffer around,
    #the edges are wonky....or we just trim everything to max shift. temp fix noted below

    #use the given positions of the good PSF stars to generate a new EPSF
    size=21
    nddata=NDData(data=bkg_sub_full_data)
    good_stars=extract_stars(nddata, sources, size=size)

    #temp fix:
    # Filter out stars with invalid or zero flux

    valid_stars = [star for star in good_stars 
                if np.isfinite(np.sum(star.data)) and np.sum(star.data) > 0]
    
    # filter out stars with the wrong shape
    valid_stars = [star for star in valid_stars if star.data.shape == (size, size)]


    try:
        epsf_input = EPSFStars(valid_stars)
        epsf_builder=EPSFBuilder(oversampling=2, maxiters=10)
        epsf, fitted_stars = epsf_builder(epsf_input)
    except:
        problems.append(file)
        continue
    

    #plot epsf just to see if it worked

    norm=simple_norm(epsf.data, 'log', percent=99.0)
    '''
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='gray')
    plt.show()
    '''

    #psf fitting, using init params
    psf_model = epsf
    fit_shape=(15,15)
    #grouper=SourceGrouper(min_separation=8)
    psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=5, xy_bounds=0.5)
    
    
    #make the xy pixel fit dist. 0.1, avg error in original fit so it doesn't move
    try:
        phot = psfphot(bkg_sub_full_data, init_params=init_params)
    except ValueError as e:
        raise e

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

    #just double checking there's no extra stars. from an earlier version when I psf fit the neighborhood too
    ens = phot[np.isin(phot['id'], eids)]
    
    #get scaled flux of neighbor for this particular exposure
    ave_ens_flux=np.nanmean(ens['flux_fit'])
    scaled_c_flux=stacked_flux_factor_c*ave_ens_flux
    scaled_b_flux=stacked_flux_factor_b*ave_ens_flux

    #using stacked x and y positions, subtract a psf of this neighbor star
    test_params=neighbors[['id','group_id', 'group_size', 'name']]
    test_params['x_0'], test_params['y_0'], test_params['flux']=neighbors['x_fit'], neighbors['y_fit'],np.array([scaled_b_flux, scaled_c_flux])
    test_params=Table.from_pandas(test_params)
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
    if showplot:
        plt.show()
    
    #normal image minus the neighbor
    final_data_unbkg=imdata-test
    
    #bkg subtracted img minus the neighbor
    final_data = bkg_sub_full_data - test
    
    #aperture photometry instead
    
    #using fitted positions from PSF photometry
    positions = np.transpose((phot['x_fit'], phot['y_fit']))
    
    #Manually add the aql position
    aql_position = np.array([[
        stacked_aql['x_fit'].iloc[0],
        stacked_aql['y_fit'].iloc[0]
    ]])
    positions = np.vstack([positions, aql_position])
    
    #do the ap phot
    apertures = CircularAperture(positions, r=r_ap)
    aper_phot = aperture_photometry(final_data, apertures)
    
    # map results by ID
    ids = list(phot['id'])
    ids.append('aql')
    fluxes = aper_phot['aperture_sum']
    
    flux_dict = dict(zip(ids, fluxes))
    
    
    #plot to check apertures
    '''
    plt.figure(figsize=(8, 8))
    plt.imshow(final_data, origin='lower', cmap='gray', norm=norm)
    
    # plot apertures
    apertures.plot(color='red', lw=1.5, alpha=0.8)
    
    # optionally label IDs
    for i, (x, y) in enumerate(positions):
        plt.annotate(
            str(ids[i]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            color='yellow',
            fontsize=8
        )
    plt.show()
    '''
    
    #new way of saving
    # ensemble stars
    for e in eids:
        if e in flux_dict:
            val = flux_dict[e]
            ens_d[e].append(val)
            big_df.at[ind, str(e)] = val
    
    #Aql X-1
    aql_val = flux_dict['aql']
    aql.append(aql_val)
    big_df.at[ind, 'aql'] = aql_val
    
    '''
    #old way of saving
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
    '''

    savefits=False
    #showplot=False
    if savefits:
        hdr['SUBTR']=True
        fits.writeto(f'/neta/xrb/{target}/{telescope}/opt/rccd/{band}_trimmed/sub_{file.split("/")[-1]}',final_data_unbkg, hdr, overwrite=True)
    if showplot:
        showyn=input('continue to show plots?')
        if 'y' in showyn:
            showplot=True
        else:
            showplot=False
    

big_df.to_csv(f'/home/kmc249/Downloads/phot_fluxes_{telescope}_{band}_apsize_{r_ap}.csv', index=False)

print('nonexistent: ', nonexistent)
print('problems: ', problems)