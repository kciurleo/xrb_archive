#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:21:27 2025

@author: kmc249
"""

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
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