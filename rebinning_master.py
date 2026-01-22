#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:44:36 2025

@author: kmc249
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smith_utils import *
from scipy.ndimage import zoom
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm

file='/home/kmc249/Downloads/list_for_master_img.txt'
dirloc='/scratch/temp_CD_data/AqlX-1/1.3m/rccd/'
trimdirloc='/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_'
filelist=[]

with open(file, 'r') as f:
    for line in f:
        filelist.append(line[5:-1])

#for file in filelist, we're gonna rerun the shifting but with an upsampling

upsample_factor = 2

#main file to align to, hires it up
IM_lores=fits.getdata(dirloc+filelist[0])
HDR=fits.getheader(dirloc+filelist[0])
IM = zoom(IM_lores, upsample_factor, order=3)
trim_IM=IM[upsample_factor*301:upsample_factor*813,upsample_factor*301:upsample_factor*813]

#realign just those of interest, stack, trim, save
image_stack = np.empty([IM.shape[0],IM.shape[1],len(filelist)])
xshifts = {}
yshifts = {}

for id, file in enumerate(filelist):
    print(f'working on file {id}')
    #load image and make it hires
    im_lores=fits.getdata(dirloc+file)
    im = zoom(im_lores, upsample_factor, order=3)
    
    #do the fft crossimaging, centered on the star

    xshifts[id], yshifts[id] = cross_image(IM, im, int(np.shape(IM)[0]/2-upsample_factor*17.116469062584883), int(np.shape(IM)[1]/2- upsample_factor*29.123442023369904), boxsize=400*upsample_factor)
    newimg=shift_image(im,xshifts[id], yshifts[id])
    #trimimg=newimg[upsample_factor*301:upsample_factor*813,upsample_factor*301:upsample_factor*813]
    image_stack[:,:,id] = newimg[:IM.shape[0],:IM.shape[1]]
    
    
    '''
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(im)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(trimimg, cmap='gray', origin='lower', norm=norm)
    plt.show()
    '''
median_image = np.median(image_stack, axis=2)
#%%
median_image = np.median(image_stack, axis=2)
stacked_full = fits.getdata('/home/kmc249/Downloads/AqlX-1_R_600.0_stack.fits')
zoomfull=zoom(stacked_full, upsample_factor, order=3)[:IM.shape[0], :IM.shape[1]]
print(np.shape(zoomfull))
xshift, yshift = cross_image(zoomfull, median_image,  int(np.shape(IM)[0]/2 - upsample_factor*17.116), int(np.shape(IM)[1]/2 - upsample_factor*29.123), boxsize=400*upsample_factor)
print(xshift, yshift)
final_image = shift_image(median_image,xshift, yshift)
final_image = final_image[upsample_factor*301:upsample_factor*813,upsample_factor*301:upsample_factor*813]
#sc = SigmaClip(sigma=3, maxiters=5)     # tune thresholds as needed
#median_image = np.median(sc(image_stack, axis=2), axis=2)

interval = ZScaleInterval()
vmin, vmax = interval.get_limits(median_image)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
plt.figure(figsize=(10,12))
plt.imshow(final_image, cmap='gray', origin='lower', norm=norm)
plt.show()

fits.writeto('/home/kmc249/Downloads/hires_master.fits',final_image,header=HDR, overwrite=True)

#go back to 512x512?
lores_median_image=zoom(final_image, 1/upsample_factor, order=3)
plt.figure(figsize=(10,12))
plt.imshow(lores_median_image, cmap='gray', origin='lower', norm=norm)
plt.show()
fits.writeto('/home/kmc249/Downloads/lores_master.fits',lores_median_image,header=HDR, overwrite=True)

stacked_trim = fits.getdata('/home/kmc249/Downloads/NEWEST_aql_R_600.0_stack.fits')

plt.figure(figsize=(10,12))
plt.imshow(stacked_trim, cmap='gray', origin='lower', norm=norm)
plt.show()
