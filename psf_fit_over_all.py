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



filelist=glob.glob('/Users/katieciurleo/Downloads/aqlX1psfframes/*') #[list of files]
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

#info about the neighbor
stacked_neighbor=Table.read('/Users/katieciurleo/Downloads/yalestuff/neighbor_info.vot')
stacked_aql=Table.read('/Users/katieciurleo/Downloads/yalestuff/aql_info.vot')
stacked_ensemble=Table.read('/Users/katieciurleo/Downloads/yalestuff/ensemble_info.vot')
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
init_params=stacked_ensemble['id','group_id','flux_init','x_fit','y_fit','flux_fit']
init_params.add_row(stacked_neighbor[0]['id','group_id','flux_init','x_fit','y_fit','flux_fit'])
init_params.add_row(stacked_aql[0]['id','group_id','flux_init','x_fit','y_fit','flux_fit'])

print(init_params)
print(aksjhdjs)
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

for ind, row in big_df.iterrows():
    file=row['filename']
    fulldata,hdr = fits.getdata(file,header=True)
    big_df.at[ind, 'time']=Time(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}")

    #shift image to match the stacked image
    xshift, yshift = cross_image(stacked_full,fulldata,int(556), int(557), boxsize=400)
    
    imdata=shift_image(fulldata,xshift,yshift)[301:813,301:813]

    #background subtract data
    sigma_clip=SigmaClip(sigma=3.0)
    bkg_estimator=MedianBackground()
    fullbkg=Background2D(imdata, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkg_sub_full_data=imdata-fullbkg.background

    #plot sources on data just to double check
    '''
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
    '''

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


print(ens_d)
print(nb)
print(nb_modeled)
print(aql)

big_df.to_csv('/Users/katieciurleo/Downloads/yalestuff/psf_fluxes.csv', index=False)

# Convert keys to strings
ens_d_str = {str(k): v for k, v in ens_d.items()}

with open('/Users/katieciurleo/Downloads/yalestuff/photometry_results.json', 'w') as f:
    json.dump({
        #'ens_d': ens_d_str,
        'nb': [float(x) for x in nb],
        #'nb_modeled': [float(x) for x in nb_modeled],
        'aql': [float(x) for x in aql]
    }, f, indent=2)


#plotting

def f(x, a, c):
    return a*np.log10(x)+c

fig, axes = plt.subplots(figsize=(8, 8))
xdata, ydata=[],[]
for e in ens_d.keys():
    x=np.std(ens_d[e])
    y=-2.5*np.log10(np.nanmean(ens_d[e]))
    xdata.append(x)
    ydata.append(y)
    axes.scatter(x, y)
    axes.annotate(
        str(e),
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        color='red'
    )

axes.scatter(np.std(nb), -2.5*np.log10(np.nanmean(nb)))
axes.annotate(
    'neighbor',
    (np.std(nb), -2.5*np.log10(np.nanmean(nb))),
    textcoords="offset points",
    xytext=(5, 5),
    fontsize=8,
    color='red'
)
axes.scatter(np.std(nb_modeled), -2.5*np.log10(np.nanmean(nb_modeled)))
axes.annotate(
    'MODELED neighbor',
    (np.std(nb_modeled), -2.5*np.log10(np.nanmean(nb_modeled))),
    textcoords="offset points",
    xytext=(5, 5),
    fontsize=8,
    color='red'
)

axes.scatter(np.std(aql), -2.5*np.log10(np.nanmean(aql)))
axes.annotate(
    'aql',
    (np.std(aql), -2.5*np.log10(np.nanmean(aql))),
    textcoords="offset points",
    xytext=(5, 5),
    fontsize=8,
    color='red'
)

axes.set_xlabel('std')
axes.set_ylabel('mag')

popt, pcov = curve_fit(f, np.array(xdata), np.array(ydata))
x_arr=np.linspace(np.min(xdata), np.max(xdata),150)
axes.plot(x_arr, f(x_arr, *popt), 'g--')
plt.savefig('/Users/katieciurleo/Downloads/yalestuff/ensemble_variability.png', dpi=250)
plt.show()

