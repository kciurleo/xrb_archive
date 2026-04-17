#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:02:41 2026

@author: kmc249
"""

import matplotlib.pyplot as plt
import numpy as np

filters=['B','V','R', 'I', 'J', 'H', 'K']
wavelengths_nm = [440, 550, 640, 790, 1250, 1650, 2200]
our_ratios=[26, 17, 23, 31, 28, np.nan, np.nan]
our_errors=[17, 6, 6, 9, 3, 4, 6]
their_ratios=[np.nan, 11.8, 17, 22.5, np.nan, np.nan, 41]
their_errors=[np.nan, 1.1, 2.,1.5, np.nan, np.nan,9]

plt.figure(figsize=(8,6))
plt.xlabel('Filter')
plt.ylabel('Aql Flux (percent)')
plt.errorbar(filters, our_ratios, our_errors, fmt='.', capsize=2, alpha=0.75, label='From stacked PSF')
plt.errorbar(filters, their_ratios, their_errors, fmt='.', capsize=2, alpha=0.75, label='From lit.')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Aql Flux (percent)')

plt.errorbar(wavelengths_nm, our_ratios, our_errors,
             fmt='.', capsize=2, alpha=0.75, label='From stacked PSF')

plt.errorbar(wavelengths_nm, their_ratios, their_errors,
             fmt='.', capsize=2, alpha=0.75, label='From lit.')

plt.legend()
plt.tight_layout()
plt.xscale('log')
plt.show()