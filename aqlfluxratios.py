#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:02:41 2026

@author: kmc249
"""

import matplotlib.pyplot as plt
import numpy as np

filters=['B','V','R', 'I', 'J', 'H', 'K']
our_ratios=[23, 9, 20, 40, 47, 45, 54]
our_errors=[9, 2, 4, 15.5, 3, 4, 6]
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