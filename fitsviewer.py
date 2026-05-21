#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:21:27 2025

@author: kmc249
"""

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
'''
plt.style.use('classic')
plt.figure(figsize=(10,7))
data=fits.getdata('/home/kmc249/Downloads/J1733+6834D.fits')
start=4700-938
wl=np.arange(start, start+len(data), 1)
plt.plot(wl, data, color='k')
plt.xlabel('Wavelength (A)', size=14)
plt.ylabel(r'$F_{\lambda}$ (erg/s/cm$^2$/A)', size=14)
plt.xlim()
plt.savefig('/home/kmc249/Downloads/dbsp.eps')
plt.show()
hdr=fits.getheader('/home/kmc249/Downloads/J1733+6834D.fits')
print(hdr)
print(np.shape(data))
'''
#file='/scratch/temp_CD_data/AqlX-1/1m/rccd/rccd991108.0006.fits'
import glob
#file='/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/sub_trim_rccd000405.0042.fits'
filev='/neta/xrb/A0620-00/1.3m/opt/rccd/V/rccd131026.0181.fits'
fileb='/neta/xrb/A0620-00/1.3m/opt/rccd/B/rccd051208.0015.fits'
filei='/neta/xrb/A0620-00/1.3m/opt/rccd/I/rccd031117.0087.fits'
fileh='/neta/xrb/A0620-00/1.3m/ir/raw/H/binir031029.0284.fits'
filehbackup='/neta/xrb/A0620-00/1.3m/ir/raw/H/binir051127.0248.fits'



for file in glob.glob('/neta/xrb/A0620-00/1.3m/ir/raw/H/*'):
    print(f'now showing{file}')
    data=fits.getdata(file)
    hdr=fits.getheader(file)
    print(hdr)
    try:
        filt=hdr['CCDFLTID']
    except:
        filt=hdr['FILTERID']

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    
    plt.imshow(data, cmap='gray', origin='lower', norm=norm)
    plt.title(filt)
    print(file)
    plt.show()

