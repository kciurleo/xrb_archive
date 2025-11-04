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


#read in main df
fwhms=pd.read_csv('/home/kmc249/test_data/temp_aql_shifts.csv', low_memory=False)

bad_list=np.array(['130811.0067', '140529.0030','130504.0083','070524.0061','070809.0032','060710.0057', 
                   '070411.0052','070413.0070','030329.0212', '130504.0083','170730.0019', '171008.0030',
                   '130820.0064', '130820.0064', '150408.0159','051010.0021','050517.0149','050719.0101',
                   '040531.0042','120906.0021','090928.0041','070729.0050','030321.0139','030307.0212',
                   '160825.0042', '160904.0021', '170604.0070'])
bad_list='rccd'+bad_list+'.fits'

input_df=fwhms.loc[~fwhms['filename'].isin(bad_list)]

filelist=f'/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_'+np.array(input_df['filename'])

stacked_trim = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits')

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

#info about the neighbor
stacked_neighbor=Table.read('/home/kmc249/Downloads/neighbor_info.vot')
stacked_aql=Table.read('/home/kmc249/Downloads/aql_info.vot')
stacked_ensemble=Table.read('/home/kmc249/Downloads/ensemble_info.vot')
eids = list(stacked_ensemble['id'])

#put all the vots in the smaller x-y system
stacked_aql["x_init"]=stacked_aql["x_init"]-301
stacked_aql["y_init"]=stacked_aql["y_init"]-301
stacked_neighbor["x_init"]=stacked_neighbor["x_init"]-301
stacked_neighbor["y_init"]=stacked_neighbor["y_init"]-301
stacked_ensemble["x_init"]=stacked_ensemble["x_init"]-301
stacked_ensemble["y_init"]=stacked_ensemble["y_init"]-301

#flux factor (and error eventually)
stacked_flux_factor=stacked_neighbor['stacked_flux_factor'].value[0]

#init params to fit the neighbor and ensemble
init_params=stacked_ensemble['id','flux_init','x_fit','y_fit','flux_fit']
init_params.add_row(stacked_neighbor[0]['id','flux_init','x_fit','y_fit','flux_fit'])
init_params.add_row(stacked_aql[0]['id','flux_init','x_fit','y_fit','flux_fit'])
init_params['group_id'] = np.arange(len(init_params))

# now make Aql and neighbor share the same group_id
neighbor_id = stacked_neighbor['id'][0]
aql_id = stacked_aql['id'][0]
neighbor_idx = np.where(init_params['id'] == neighbor_id)[0][0]
aql_idx = np.where(init_params['id'] == aql_id)[0][0]
init_params['group_id'][aql_idx] = init_params['group_id'][neighbor_idx]

#ensemble fluxes to use for std
ens_d=dict((el,[]) for el in eids)
nb=[]
nb_modeled=[]
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

for ind, row in big_df.iterrows():
    file=row['filename']
    print('trying', file)
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

    #plot sources on data just to double check

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(bkg_sub_full_data)
    vmin2, vmax2 = interval.get_limits(stacked_trim)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(bkg_sub_full_data, cmap='gray', origin='lower', norm=norm)
    plt.imshow(stacked_trim, cmap='viridis', origin='lower', norm=norm2, alpha=0.5)
    plt.scatter(init_params['x_fit'], init_params['y_fit'], marker='x')

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
    '''
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='gray')
    plt.show()
    '''


    #####THIS PART SHOULDN'T EVEN BE NEEDED
    print(init_params)
    #psf fitting, using init params
    psf_model = epsf
    fit_shape=(7,7)
    #grouper=SourceGrouper(min_separation=8)
    psfphot=PSFPhotometry(psf_model, fit_shape, aperture_radius=8, xy_bounds=0.1)
    
    
    #make the xy pixel fit dist. 0.1, avg error in original fit so it doesn't move
    try:
        phot = psfphot(bkg_sub_full_data, init_params=init_params)
    except ValueError as e:
        print("init_params:", len(init_params))
        print("npixfit:", len(psfphot._ungroup(psfphot._group_results['npixfit'])))
        raise

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
    if showplot:
        plt.show()

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
        fits.writeto(f'/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/sub_{file.split("/")[-1]}',final_data, hdr, overwrite=True)
    if showplot:
        showyn=input('continue to show plots?')
        if 'y' in showyn:
            showplot=True
        else:
            showplot=False
    

big_df.to_csv('/home/kmc249/Downloads/psf_fluxes.csv', index=False)

print('nonexistent: ', nonexistent)