#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 07:06:20 2025

@author: kmc249
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
import glob 
from photutils.psf import fit_fwhm
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
import warnings
import os
from astropy.time import Time
warnings.filterwarnings('ignore')

###USER DEFINED PARAMETERS
os.chdir('/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/')
savedir='/home/kmc249/Downloads/'

#get these physical pixel coordinates from placing apertures and manually looking at the region file
names=['Aql X-1', 'Standard 2', 'Standard 3']
xcentroids= np.array([556.90818,501.93133,543.40509,461.73843,575.07176])
ycentroids= np.array([556.68056,553.30478,468.42824,551.76157,543.42824])

xcentroids = xcentroids -70-420-1+200-12
ycentroids = ycentroids -200-300-1+200
xypos=list(zip(xcentroids, ycentroids))

#exposure time and filter to use
exptime=float(600.0)
filt='R'

allowed_mjd_spans = [(52747.0,53005.0),
    (53510.0, 53940.0),
    (53977.0, 54227.0),
    (54288.0, 54320.0),
    (54397.0, 54590.0),
    (54700.0, 55375.0),
    (56525.0, 56830.0),
    (56910.0, 57590.0),
    (57663.0, 57890.0),
]

def mjd_is_acceptable(mjd, spans):
    return any(lo <= mjd <= hi for lo, hi in spans)


#if you want to trim to certain pixel size; else, trim will just do max x-y shift
trim_to=[]

#quantile window to choose for "good seeing"
q1, q2 = 0.0, 0.45

filelist=glob.glob('trim*')


print(f'list of acceptable fits files is is {len(filelist)} long')

#align and trim, find fwhm of a handful of stars
fwhms=pd.DataFrame(columns=['filename', 'maxshift', 'xshift', 'yshift', 'bad']+names)

print('initial align and trim')
for id, file in enumerate(filelist):
    fwhms.at[id, 'filename']=file
    try:
        im,hdr = fits.getdata(file,header=True)
        mjd=Time(hdr['DATE-OBS']).mjd
        fwhms.at[id, 'exptime']=hdr['EXPTIME']
        fwhms.at[id, 'mjd']=mjd
        if not mjd_is_acceptable(mjd, allowed_mjd_spans): #hdr['EXPTIME']<800 and mjd>52747:#
            continue
    except:
        print('no file', file)
        continue
    #do the fft crossimaging, centered on the star
    f = fit_fwhm(im, xypos=xypos, fit_shape=7)
    for num, name in enumerate(names):
        fwhms.at[id, name]=f[num]
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    interval = ZScaleInterval()
    vmin,vmax=interval.get_limits(im)
    norm=ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
    ax.imshow(im, cmap='gray',origin='lower',norm=norm,alpha=0.5)
    plt.title(file)
    plt.show()
    print('just showed', file)
    '''
    fwhms.at[id, 'bad']=False

'''
plt.figure(figsize=(8,6))
plt.hist(fwhms['maxshift'], bins=40)
plt.show()

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
IM, HDR = fits.getdata(winners['filename'][0],header=True)
image_stack = np.full((IM.shape[0], IM.shape[1], len(winners)), np.nan)
xshifts = {}
yshifts = {}

#make stack/realign
print('final align and stack')
for id, row in winners.iterrows():
    im,hdr = fits.getdata(row['filename'],header=True)
    #do the fft crossimaging, centered on the star
    fig, ax = plt.subplots(figsize=(10,10))
    interval = ZScaleInterval()
    vmin,vmax=interval.get_limits(im)
    norm=ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
    ax.imshow(im, cmap='gray',origin='lower',norm=norm,alpha=0.5)
    plt.title(file)
    plt.show()
    print('just showed', row['filename'])
    good=input('was this good?')
    if 'y' in good:
        image_stack[:,:,id] = im
    else:
        image_stack[:, :, id] = np.nan
    

#median image                    
median_image = np.nanmedian(image_stack, axis=2)

#add key word to nw HDR
HDR['STACK']=True
#save image and also save log of which fits files we threw into this image
fits.writeto(f'{savedir}mostNEWEST_aql_{filt}_{exptime}_stack.fits',median_image,header=HDR, overwrite=True)
winners.to_csv(f'{savedir}mostNEWEST_aql_{filt}_{exptime}_list.csv', index=False)
print(winners)
print(winners['exptime'])
print(set(winners['exptime']))
print(fwhms.loc[fwhms['exptime']>600.])
print(image_stack[:,:,10])

#Plotting just to tell
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(median_image)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())

plt.imshow(median_image, cmap='gray', origin='lower', norm=norm)

plt.scatter(xcentroids,ycentroids, color='red', marker='x', s=3)
plt.show()
