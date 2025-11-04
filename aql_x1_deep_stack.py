import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
import glob 
from smith_utils import *
from photutils.psf import fit_fwhm
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
import warnings
import os
warnings.filterwarnings('ignore')

###USER DEFINED PARAMETERS
os.chdir('/scratch/temp_CD_data/AqlX-1/1.3m/rccd/')

#get these physical pixel coordinates from placing apertures and manually looking at the region file
names=['Aql X-1', 'Standard 2', 'Standard 3']
xcentroids= [556.90818,501.93133,543.40509,461.73843,575.07176]
ycentroids= [556.68056,553.30478,468.42824,551.76157,543.42824]
xypos=list(zip(xcentroids, ycentroids))

#exposure time and filter to use
exptime=float(600.0)
filt='R'

#if you want to trim to certain pixel size; else, trim will just do max x-y shift
trim_to=[]

#quantile window to choose for "good seeing"
q1, q2 = 0.05, 0.5

#working dir and where to save the file
savedir='/home/kmc249/test_data/'

#base image/hdr
base='/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits'

###END USER DEFINED PARAMETERS

#read in log
biglog=pd.read_csv('/home/kmc249/test_data/all_optical_log.csv',low_memory=False)
interest=biglog.loc[biglog['proper name'] == 'AqlX-1']
'''
#deal with multiple different file paths for right now
pathmap = {'opticaldirecs/xrb-archive_usb-data_AqlX-1_fitsimages_R/':'/OLD-NET-DRIVE-COLLECTION/xrb-archive/usb-data/AqlX-1/fitsimages/R/', 
    'opticaldirecs/SMARTS_xrb_AqlX-1/': '/netc/bailyn/SMARTS_xrb/AqlX-1/', 
    'opticaldirecs/xrb-archive_usb-data_AqlX-1_fitsimages_V/':'/OLD-NET-DRIVE-COLLECTION/xrb-archive/usb-data/AqlX-1/fitsimages/V/', 
    'opticaldirecs/xrb_ccd/': '/OLD-NET-DRIVE-COLLECTION/xrb/ccd/', 
    'opticaldirecs/xrb-archive_usb-data_AqlX-1_fitsimages_B/':'/OLD-NET-DRIVE-COLLECTION/xrb-archive/usb-data/AqlX-1/fitsimages/B/', 
    'opticaldirecs/xrb_ccd_reduced_2013/': '/OLD-NET-DRIVE-COLLECTION/xrb/ccd/reduced/2013/', 
    'opticaldirecs/xrb_ccd_reduced_2015/': '/OLD-NET-DRIVE-COLLECTION/xrb/ccd/reduced/2015/', 
    'optical_CD_header_logs/optical_CD_header_logs/': 'CD FAIL', 
    'opticaldirecs/NB_buxton/': '/OLD-NET-DRIVE-COLLECTION/xrb-archive/data/NB-buxton/AqlX1/optical/', 
    'opticaldirecs/xrb_ccd_reduced_2014/': '/OLD-NET-DRIVE-COLLECTION/xrb/ccd/reduced/2014/'}

interest['dir'] = interest['Location'].map(pathmap)

for id, row in interest.loc[interest['dir']=='/OLD-NET-DRIVE-COLLECTION/xrb-archive/data/NB-buxton/AqlX1/optical/'].iterrows():
    if '2011' in str(row['DATE-OBS']):
        interest.at[id, 'dir']='/OLD-NET-DRIVE-COLLECTION/xrb-archive/data/NB-buxton/AqlX1/optical/2011/'
    elif '2012' in str(row['DATE-OBS']):
        interest.at[id, 'dir']='/OLD-NET-DRIVE-COLLECTION/xrb-archive/data/NB-buxton/AqlX1/optical/2012/'
    elif '2013'in str(row['DATE-OBS']):
        interest.at[id, 'dir']='/OLD-NET-DRIVE-COLLECTION/xrb-archive/data/NB-buxton/AqlX1/optical/2013/'
    else:
        interest.at[id, 'dir']='FAIL'


#we're going to skip all the things that fail for right now, likely because they're actually on CDs oops
interest=interest.loc[~interest['dir'].isin(['FAIL', 'CD FAIL'])]
'''
print(f'list of acceptable fits files is is {len(interest)} long')

#only get the things in the exptime and filter requested
interest=interest.loc[(interest['CCDFLTID']==filt)]
interest.reset_index(inplace=True)

print(f'{len(interest)} files to look at in this filt/exp')

#get base info
IM=fits.getdata(base)
HDR=fits.getheader(base)
filelist=[]

#make file list
for id, row in interest.iterrows():
    #filelist.append(f'{row["dir"]}{row["filename"]}')
    filelist.append(f'{row["filename"]}')

#align and trim, find fwhm of a handful of stars
fwhms=pd.DataFrame(columns=['filename', 'maxshift', 'xshift', 'yshift', 'bad']+names)

print('initial align and trim')
for id, file in enumerate(filelist):
    fwhms.at[id, 'filename']=file
    try:
        im,hdr = fits.getdata(file,header=True)
    except:
        print('no file', file)
        print(interest.loc[interest['filename']==file])
        continue
    #do the fft crossimaging, centered on the star
    xshift, yshift = cross_image(IM, im, int(xcentroids[0]), int(ycentroids[0]), boxsize=400)
    newimg=shift_image(im,xshift,yshift)
    f = fit_fwhm(newimg, xypos=xypos, fit_shape=7)
    fwhms.at[id, 'maxshift']=np.max([xshift, yshift])
    fwhms.at[id, 'xshift']=xshift
    fwhms.at[id, 'yshift']=yshift
    for num, name in enumerate(names):
        fwhms.at[id, name]=f[num]
    trimimg=newimg[301:813,301:813]
    
    fig, ax = plt.subplots(figsize=(10,10))
    interval = ZScaleInterval()
    vmin,vmax=interval.get_limits(trimimg)
    norm=ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
    ax.imshow(trimimg, cmap='gray',origin='lower',norm=norm,alpha=0.5)
    plt.title(file)
    plt.show()
    print('just showed', file)

    fwhms.at[id, 'bad']=False
    hdr['TRIM']=True
    fits.writeto(f'/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_{file}', trimimg, hdr, overwrite=True)
    print('wrote trim_',file)
    
fwhms.to_csv('/home/kmc249/test_data/temp_aql_shifts.csv', index=False)
'''
plt.figure(figsize=(8,6))
plt.hist(fwhms['maxshift'], bins=40)
plt.show()
'''
plt.figure(figsize=(8,8))
plt.scatter(fwhms['xshift'], fwhms['yshift'])
plt.show()

'''
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
    #do the fft crossimaging, centered on the star
    xshifts[id], yshifts[id] = cross_image(IM, im, int(xcentroids[0]), int(ycentroids[0]), boxsize=400)
    image_stack[:,:,id] = shift_image(im,xshifts[id], yshifts[id])

#median image                    
median_image = np.median(image_stack, axis=2)

#trim logic
max_x_shift = int(np.max([xshifts[x] for x in xshifts.keys()]))
max_y_shift = int(np.max([yshifts[x] for x in yshifts.keys()]))

if len(trim_to)>0:
    final_image = median_image[trim_to[0]:trim_to[1],trim_to[2]:trim_to[3]]
elif (max_x_shift > 0) and (max_y_shift > 0):
    final_image = median_image[max_x_shift:-max_x_shift,max_y_shift:-max_y_shift]
elif max_x_shift > 0:
    final_image = median_image[max_x_shift:-max_x_shift,:]
elif max_x_shift > 0:
    final_image = median_image[:,max_y_shift:-max_y_shift]
else:
    final_image=median_image[:,:]

#add key word to nw HDR
HDR['STACK']=True
#save image and also save log of which fits files we threw into this image
fits.writeto(f'{savedir}{names[0]}_{filt}_{exptime}_stack.fits',final_image,header=HDR, overwrite=True)
winners.to_csv(f'{savedir}{names[0]}_{filt}_{exptime}_list.csv', index=False)

plt.hist(fwhms['Aql X-1'])
plt.show()

#Plotting just to tell
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(final_image)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())

plt.imshow(final_image, cmap='gray', origin='lower', norm=norm)
plt.colorbar()

plt.scatter(xcentroids,ycentroids, color='red', marker='x', s=3)
plt.show()
'''