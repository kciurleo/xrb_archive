#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:19:26 2025

@author: kmc249
"""
from astropy.io import fits
import numpy as np
import astroalign as aa
import glob
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch

#base image/hdr
#base='/home/kmc249/Downloads/AqlX-1_R_600.0_stack_NEWMASTER.fits'
base='/scratch/temp_CD_data/AqlX-1/1m/rccd/r0816_1798.014.fits'

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

#for file in ['/scratch/temp_CD_data/AqlX-1/1m/rccd/r0816_1798.014.fits']:
for file in glob.glob('/scratch/temp_CD_data/AqlX-1/1m/rccd/*'):
    if not file==base:
    #file='/scratch/temp_CD_data/AqlX-1/1m/rccd/rccd991108.0006.fits'
    #file='/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/sub_trim_rccd000405.0042.fits'
        
        img = np.asarray(fits.getdata(base), dtype='<f8')
        inp_img = np.asarray(fits.getdata(file), dtype='<f8')
  
        #manually flipping it
        tried=True
        test_img=np.rot90(np.flipud(inp_img), k=3)
        '''
        inp_img = np.ascontiguousarray(test_img[600:1524, 600:1524])
        img_aligned, footprint = aa.register(inp_img, img, detection_sigma=2.0, max_control_points=50)


        '''
        img_aligned, footprint = register_with_flips(inp_img, img, detection_sigma=2.0, max_control_points=75)
        #tried=False
        hand_aligned = test_img[800-50:1424-50, 800:1424]

        
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
        axes[0, 1].set_title("Target Image")
        if tried:
            axes[1, 0].imshow(img_aligned, cmap='gray', interpolation='none', origin='lower', norm=norm)
            axes[1, 0].axis('off')
            axes[1, 0].set_title("Source Image aligned with Target")
        else:
            axes[1, 0].imshow(test_img, cmap='gray', interpolation='none', origin='lower', norm=norm)
            axes[1, 0].axis('off')
            axes[1, 0].set_title("Rotated/flipped with Target")
        if tried:
            axes[1, 1].imshow(footprint, cmap='gray', interpolation='none', origin='lower')
            axes[1, 1].axis('off')
            axes[1, 1].set_title("Footprint of the transformation")
        else:
            axes[1, 1].imshow(hand_aligned, norm=norm, cmap='gray', interpolation='none', origin='lower')
            axes[1, 1].axis('off')
            axes[1, 1].set_title("Zoomed in (no longer 512x512)")
        
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.show()

