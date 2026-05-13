#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:02:41 2026

@author: kmc249
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

filters=['B','V','R', 'I', 'J', 'H', 'K']
wavelengths_nm = np.array([440, 550, 640, 790, 1250, 1650, 2200])
#our_ratios=[26, 17, 21, 31, 28, np.nan, 52]
#our_errors=[17, 6, 3, 9, 3, 4, 4]
our_ratios=[np.nan, 17, 21, 31, 28, 35, 52]
our_errors=[np.nan, 6, 3, 9, 2, 2, 4]
their_ratios=np.array([np.nan, 11.8, 17, 22.5, np.nan, np.nan, 41])
their_errors=np.array([np.nan, 1.1, 2.,1.5, np.nan, np.nan,9])

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

#%%

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Linear model in log-space
def line(x, m, b):
    return m*x + b

# ---------- CLEAN DATA ----------
mask = (
    np.isfinite(wavelengths_nm) &
    np.isfinite(their_ratios) &
    np.isfinite(their_errors) &
    (their_errors > 0) &
    (wavelengths_nm > 0)
)

# log-space x values
x = np.log10(wavelengths_nm[mask])

y = their_ratios[mask]
err = their_errors[mask]

# ---------- FIT ----------
popt, pcov = curve_fit(
    line,
    x,
    y,
    sigma=err,
    absolute_sigma=True
)

m, b = popt
m_err, b_err = np.sqrt(np.diag(pcov))

print(f"Slope = {m:.3e} ± {m_err:.3e}")
print(f"Intercept = {b:.3e} ± {b_err:.3e}")

# ---------- PLOT ----------
xfit_nm = np.linspace(
    wavelengths_nm[mask].min(),
    wavelengths_nm[mask].max(),
    500
)

yfit = line(np.log10(xfit_nm), *popt)

plt.figure(figsize=(8,6))

plt.errorbar(
    wavelengths_nm,
    our_ratios,
    our_errors,
    fmt='.',
    capsize=2,
    alpha=0.75,
    label='From stacked PSF'
)
plt.errorbar(wavelengths_nm, their_ratios, their_errors,
             fmt='.', capsize=2, alpha=0.75, label='From lit.')


plt.plot(
    xfit_nm,
    yfit,
    label='Fit to lit.'
)

plt.xscale('log')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Aql Flux (percent)')

plt.legend()
plt.tight_layout()
plt.show()