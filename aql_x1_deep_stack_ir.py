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
warnings.filterwarnings('ignore')

###USER DEFINED PARAMETERS
basedir="/neta/xrb/AqlX-1/1.3m/ir/proc/J"
all_dirs=glob.glob(f'{basedir}/night_*')
filelist=[i+'/final_stack_aligned_to_081031.fits' for i in all_dirs]

quiescence=pd.read_csv('/home/kmc249/Downloads/quiescence_mjd_ranges_v5.csv')

base='/neta/xrb/AqlX-1/1.3m/ir/proc/J/night_081031/final_stack_aligned_to_081031.fits'
savedir='/neta/xrb/AqlX-1/temp/'
#quantile window to choose for "good seeing"
q1, q2 = 0.75, 0.99


#get base info
IM=fits.getdata(base)
HDR=fits.getheader(base)
mean, median, std = sigma_clipped_stats(IM, sigma=3, maxiters=5)

daofinder = DAOStarFinder(threshold=5. * std, fwhm=6.)
sources = daofinder(IM)


positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=3 * 6.)

interval = ZScaleInterval()
vmin,vmax=interval.get_limits(IM)
norm=ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
plt.figure(figsize=(10,10))
plt.imshow(IM, cmap='gray', origin='lower', norm=norm)
apertures.plot(color="red")
for i, (x, y) in enumerate(positions):
    plt.text(x+8, y+8, str(i),
             color='yellow',
             fontsize=10,
             ha='center', va='center')
plt.show()

#picking these stars - the indices from above
names=['F', 'B', 'C']
indices = np.array([51, 41, 32])

xcentroids = sources['xcentroid'][indices]
ycentroids = sources['ycentroid'][indices]

xypos = list(zip(xcentroids, ycentroids))
selected_apertures = CircularAperture(xypos, r=3 * 6.)

#checking these indices are correct
plt.figure(figsize=(10,10))
plt.imshow(IM, cmap='gray', origin='lower', norm=norm)

selected_apertures.plot(color='cyan', lw=2)
plt.show()


phot_table = aperture_photometry(IM, selected_apertures)
phot_table['name']=names

#align and trim, find fwhm of a handful of stars
fwhms=pd.DataFrame(columns=['filename', 'maxshift', 'xshift', 'yshift', 'bad']+names)

print('initial align and trim')
for id, file in enumerate(filelist):
    direc=os.path.dirname(file)
    
    fwhms.at[id, 'filename']=file
    #check if in quiescence, skip those that aren't
    date_str = file.split('/')[-2].split('_')[-1]
    try:
        dt = pd.to_datetime(date_str, format="%y%m%d")
        mjd = Time(dt).mjd
    except:
        continue
    mask = (quiescence["q_start_mjd"] <= mjd) & (quiescence["q_end_mjd"] >= mjd)  
    if not mask.any():
        print('not in quiescence')
        continue
    else:
        print('in quiescence!')
    
    try:
        im,hdr = fits.getdata(file,header=True)
    except:
        print('no file', file)
        continue
    try:
        exphdr = fits.getheader(f'{direc}/aligned_frames/science_aligned_00.fits')
        exposure=exphdr['EXPTIME']
        fwhms.at[id, 'exp']=exposure
    except:
        print('no exposure hdr file', file)
        continue
    #do some basic aperture photometry to get the count rate
    #basic bkg subtr
    sigma_clip=SigmaClip(sigma=3.0)
    bkg_estimator=MedianBackground()
    fullbkg=Background2D(im, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkgsubim=im-fullbkg.background
    phot_table = aperture_photometry(bkgsubim, selected_apertures)
    phot_table['name']=names
    #print(phot_table)

    for num, name in enumerate(names):
        row = phot_table[phot_table['name'] == name]
        count=row['aperture_sum'][0]
        fwhms.at[id, name]=count/exposure

   
fwhms.to_csv(f'{savedir}ir_countrates.csv', index=False)

fwhms=pd.read_csv(f'{savedir}ir_countrates.csv', low_memory=False)
#fwhms=pd.read_csv(f'{savedir}temp_ir_countrates.csv')

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
    im,hdr = fits.getdata(row['filename'],header=True)
    #scale the image to be the same countrate
    fmax=fwhms['F'].max()
    f=row['F']
    scalefactor=fmax/f
    print(scalefactor)
    image_stack[:,:,id] = im * scalefactor/row['exp']
    print(im)
    print(im * scalefactor)

#median image                    
median_image = np.median(image_stack, axis=2)
#median_image = np.mean(image_stack, axis=2)


#add key word to nw HDR
HDR['STACK']=True
#save image and also save log of which fits files we threw into this image
fits.writeto(f'{savedir}Aql_J_stack_NEW_MEDIAN_SCALED.fits',median_image,header=HDR, overwrite=True)
winners.to_csv(f'{savedir}Aql_J_list_MEDIAN.csv', index=False)

plt.hist(fwhms['F'], bins=1000)
plt.xlim(0,10**6.1)

plt.show()

#Plotting just to tell
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(median_image)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
plt.figure(figsize=(10,10))
plt.imshow(median_image, cmap='gray', origin='lower', norm=norm)

selected_apertures.plot(color='cyan', lw=2)
plt.show()



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
#%%
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