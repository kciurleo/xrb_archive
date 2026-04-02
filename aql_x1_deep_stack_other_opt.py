import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
import glob 
from smith_utils import *
from photutils.psf import fit_fwhm
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
import warnings
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
import os
from astropy.time import Time
import astroalign as aa
from scipy.ndimage import zoom
warnings.filterwarnings('ignore')


###USER DEFINED PARAMETERS
quiescence=pd.read_csv('/home/kmc249/Downloads/quiescence_mjd_ranges_v5.csv')

base='/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits'

fullbase='/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits'
savedir='/neta/xrb/AqlX-1/temp/'
#quantile window to choose for "good seeing"
q1, q2 = 0.25, 0.99

#functions
def register_with_flips(src, ref, **kwargs):
    """
    Try registering src → ref with astroalign.
    If it fails, try horizontal, vertical, and both flips.
    Returns aligned image and footprint, or (None, None) if all fail.
    """

    flip_funcs = [
        lambda x: x,                     # no flip
        np.fliplr,                       # horizontal
        np.flipud,                       # vertical
        lambda x: np.flipud(np.fliplr(x))# both
    ]

    for func in flip_funcs:
        try:
            test_img = func(src)
            aligned, footprint = aa.register(test_img, ref, **kwargs)
            return aligned, footprint
        except Exception as exc:
            print(exc)
            pass

    raise RuntimeError("astroalign failed for all flip orientations.")

#get base info
IM=fits.getdata(base)
#get hi res version
IM= zoom(IM, 2, order=3)
HDR=fits.getheader(base)
mean, median, std = sigma_clipped_stats(IM, sigma=3, maxiters=5)

daofinder = DAOStarFinder(threshold=5. * std, fwhm=6.)
sources = daofinder(IM)


positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=3 * 6.)

interval = ZScaleInterval()
vmin,vmax=interval.get_limits(IM)
norm=ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
'''
plt.figure(figsize=(10,10))
plt.imshow(IM, cmap='gray', origin='lower', norm=norm)
apertures.plot(color="red")
for i, (x, y) in enumerate(positions):
    plt.text(x+8, y+8, str(i),
             color='yellow',
             fontsize=10,
             ha='center', va='center')
plt.show()
'''
#picking these stars - the indices from above
names=['F', 'B', 'C']
#indices = np.array([356, 345, 326])
indices = np.array([198, 219, 210])

xcentroids = sources['xcentroid'][indices]
ycentroids = sources['ycentroid'][indices]

xypos = list(zip(xcentroids, ycentroids))
selected_apertures = CircularAperture(xypos, r=2*8.)

#checking these indices are correct
plt.figure(figsize=(10,10))
plt.imshow(IM, cmap='gray', origin='lower', norm=norm)

selected_apertures.plot(color='cyan', lw=2)
plt.show()

#%%

for band in ['I']:
    basedir=f"/neta/xrb/AqlX-1/1.3m/opt/rccd/{band}/"
    filelist=glob.glob(f'{basedir}*.fits')
    
    #align and trim, find fwhm of a handful of stars
    fwhms=pd.DataFrame(columns=['filename', 'maxshift', 'xshift', 'yshift', 'bad']+names)

    print(f'initial align and trim for {band}')
    for id, file in enumerate(filelist):
        print(f'working on {id} out of {len(filelist)}')
        
        fwhms.at[id, 'filename']=file  
        try:
            im,hdr = fits.getdata(file,header=True)
            exposure=hdr['EXPTIME']
            jd=hdr['JD']
            mjd = jd - 2400000.5  # convert JD to MJD
            fwhms.at[id, 'exp']=exposure
        except:
            print('bad file', file)
            continue
            
        #align and trim, save trimmed images
        img = np.asarray(fits.getdata(fullbase), dtype='<f8')
        img=zoom(img, 2, order=3)
        inp_img = np.asarray(fits.getdata(file), dtype='<f8')

        #manually flipping it
        try:
            img_aligned, footprint = register_with_flips(inp_img, img, detection_sigma=2.0, max_control_points=75)
        except:
            print('bad file :(')

        # now, apply the saved transform to master
        trimimg=img_aligned[2*301:2*813,2*301:2*813]
        
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(inp_img)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
        interval1 = ZScaleInterval()
        vmin2, vmax2 = interval1.get_limits(img)
        norm2 = ImageNormalize(vmin=vmin2, vmax=vmax2, stretch=SinhStretch())

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(inp_img, cmap='gray', interpolation='none', origin='lower', norm=norm)
        axes[0, 0].axis('off')
        axes[0, 0].set_title("Source Image")
        
        axes[0, 1].imshow(img, cmap='gray', interpolation='none', origin='lower', norm=norm2)
        axes[0, 1].axis('off')
        axes[0, 1].set_title("Master Image")

        axes[1, 0].imshow(img_aligned, cmap='gray', interpolation='none', origin='lower', norm=norm)
        axes[1, 0].axis('off')
        axes[1, 0].set_title("Result of aligning w/Example")

        axes[1, 1].imshow(trimimg, cmap='gray', interpolation='none', origin='lower', norm=norm)
        axes[1, 1].axis('off')
        axes[1, 1].set_title("Final Trim")
        
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.show()

        #save to trim:
        hdr['TRIM']=True
        filename=file.split('/')[-1]
        fits.writeto(f"/neta/xrb/AqlX-1/1.3m/opt/rccd/{band}_trimmed/trim_{filename}", trimimg, hdr, overwrite=True, output_verify='ignore')

        #check if in quiescence, skip those that aren't
        mask = (quiescence["q_start_mjd"] <= mjd) & (quiescence["q_end_mjd"] >= mjd)
        if not mask.any():
            print('not in quiescence')
            continue
        else:
            print('in quiescence!')
            
        #do some basic aperture photometry to get the count rate
        #basic bkg subtr
        sigma_clip=SigmaClip(sigma=3.0)
        bkg_estimator=MedianBackground()
        fullbkg=Background2D(im, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bkgsubim=im-fullbkg.background
        phot_table = aperture_photometry(bkgsubim, selected_apertures)
        phot_table['name']=names
        #print(phot_table)
        '''
        plt.figure(figsize=(10,10))
        plt.imshow(trimimg, cmap='gray', origin='lower', norm=norm)

        selected_apertures.plot(color='cyan', lw=2)
        plt.show()
        '''
        for num, name in enumerate(names):
            row = phot_table[phot_table['name'] == name]
            count=row['aperture_sum'][0]
            fwhms.at[id, name]=count/exposure
    
       
    fwhms.to_csv(f'{savedir}{band}_countrates.csv', index=False)
    
#%%

for band in ['I']:
    fwhms=pd.read_csv(f'{savedir}{band}_countrates.csv', low_memory=False)
    
    #pick some percentiles; was gonna do avg and std away, but we have a very skewed distribution here
    p20 = fwhms[names].quantile(q1)
    p40 = fwhms[names].quantile(q2)
    
    #mask out so we have just the ones w/fwhms in this range
    mask = ((fwhms[names] >= p20) & (fwhms[names] <= p40)).all(axis=1)
    winners = fwhms[mask]
    winners.reset_index(inplace=True)
    print(f'{len(winners)} files found for deep stack.')
    
    #realign just those of interest, stack, trim, save
    image_stack = np.empty([IM.shape[0],IM.shape[1],len(winners)])
    xshifts = {}
    yshifts = {}
    
    #make stack/realign
    print('final align and stack')
    for id, row in winners.iterrows():
        print(f'working on {id} out of {len(winners)}')
        dirpath, filename = os.path.split(row['filename'])
        new_dir = dirpath + "_trimmed"
        new_filename = "trim_" + filename
        
        new_path = os.path.join(new_dir, new_filename)
        im,hdr = fits.getdata(new_path,header=True)
        #scale the image to be the same countrate
        fmax=fwhms['F'].max()
        f=row['F']
        scalefactor=fmax/f
        try:
            image_stack[:,:,id] = im * scalefactor/row['exp']
            print('didit')
        except:
            print('skipping')

    
    #median image                    
    median_image = np.median(image_stack, axis=2)
    #median_image = np.mean(image_stack, axis=2)
    
    
    #add key word to nw HDR
    HDR['STACK']=True
    #save image and also save log of which fits files we threw into this image
    fits.writeto(f'{savedir}Aql_{band}_stack.fits',median_image,header=HDR, overwrite=True)
    winners.to_csv(f'{savedir}Aql_{band}_stack.csv', index=False)
    
    #Plotting just to tell
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(median_image)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,10))
    plt.imshow(median_image, cmap='gray', origin='lower', norm=norm)
    
    selected_apertures.plot(color='cyan', lw=2)
    plt.show()
    
    
#%%  
###special plot
# stack all images into a 3D array
all_images = []
for file in fwhms['filename']:
    try:
        im, hdr = fits.getdata(file, header=True)
        all_images.append(im)
    except:
        continue
all_images = np.stack(all_images, axis=0)  # shape: (n_images, ny, nx)

# define percentile ranges
percentile_ranges = [(25, 99), (50, 99), (75, 99)]

'''
# prepare figure
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(np.median(all_images, axis=0))
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())

for i, (p_low, p_high) in enumerate(percentile_ranges):
    # compute a simple metric for each image (mean of selected stars)
    star_indices = [32, 41, 51]  # indices of your selected stars
    star_flux = np.array([np.mean(im[[int(y) for y in star_indices],
                                     [int(x) for x in star_indices]]) for im in all_images])
    mask = (star_flux >= np.percentile(star_flux, p_low)) & (star_flux <= np.percentile(star_flux, p_high))
    stack = all_images[mask]

    # mean and median stacks
    mean_stack = np.mean(stack, axis=0)
    median_stack = np.median(stack, axis=0)

    # plot mean
    ax_mean = axes[i,0]
    im_mean = ax_mean.imshow(mean_stack, origin='lower', cmap='gray', norm=norm)
    ax_mean.set_title(f'Mean stack {p_low}-{p_high}%')
    ax_mean.text(5, 5, f'N = {stack.shape[0]}', color='yellow', fontsize=12, 
                 ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

    # plot median
    ax_med = axes[i,1]
    im_med = ax_med.imshow(median_stack, origin='lower', cmap='gray', norm=norm)
    ax_med.set_title(f'Median stack {p_low}-{p_high}%')
    ax_med.text(5, 5, f'N = {stack.shape[0]}', color='yellow', fontsize=12, 
                ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

plt.tight_layout()
plt.show()
'''