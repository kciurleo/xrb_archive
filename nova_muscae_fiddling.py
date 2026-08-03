#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:56:59 2026

@author: kmc249
"""

#%%
from astropy.timeseries import LombScargle
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
import os
import pandas as pd


target='NovaMusca'

colors = {
    'B': 'blue',
    'V': 'green',
    'R': 'red',
    'I': 'chocolate'
}

outdir = f'/neta/xrb/{target}/product/first_pass_lightcurves'
savedir='/neta/xrb/PRODUCTS/'
#to iterate over
optical_bands=['B','V','R','I']
telescopes=['1.3m', '1m']
available_bands = []


def ellipsoidal_model(phi, C, A1, B1, A2, B2):

    return (
        C
        #+ A1*np.cos(2*np.pi*phi)
        + B1*np.sin(2*np.pi*phi)
        #+ A2*np.cos(4*np.pi*phi)
        + B2*np.sin(4*np.pi*phi)
    )
    
def linear_model(phi, m, c):
    return phi*0+c

for band in optical_bands:
    for tele in telescopes:
        fname = f'{outdir}/{target}_{tele}_{band}_first_pass_lc.csv'
        if os.path.exists(fname):
            available_bands.append(band)
            break

available_bands = list(dict.fromkeys(available_bands))  # preserve order


# Rough Nova Muscae period (days)
# Replace with your preferred value if you want fixed folding
nova_period = 10.38252/24  # ~10.38 hours

# Storage
band_results = {}

for band in available_bands:

    all_times = []
    all_mags = []
    all_errs = []

    # Combine telescopes for this band
    for tele in telescopes:

        fname = f'{outdir}/{target}_{tele}_{band}_first_pass_lc.csv'

        if not os.path.exists(fname):
            continue

        df = pd.read_csv(fname)

        if len(df) == 0:
            continue

        df['nice time'] = pd.to_datetime(
            df['time'],
            format='mixed',
            utc=True,
            errors='coerce'
        ).dt.tz_localize(None)

        valid = (
            df['target mag'].notna() &
            df['error'].notna() &
            df['nice time'].notna()
        )

        df = df.loc[valid]

        if len(df) == 0:
            continue

        # Convert to MJD
        mjd = Time(df['nice time']).mjd

        all_times.extend(mjd)
        all_mags.extend(df['target mag'].values)
        all_errs.extend(df['error'].values)


    if len(all_times) < 10:
        continue


    all_times = np.array(all_times)
    all_mags = np.array(all_mags)
    all_errs = np.array(all_errs)


    # Sort by time
    order = np.argsort(all_times)

    # Convert lists to arrays
    t = np.asarray(all_times)
    y = np.asarray(all_mags)
    dy = np.asarray(all_errs)
    
    # Sort by time
    order = np.argsort(t)
    
    t = t[order]
    y = y[order]
    dy = dy[order]
    
    # Sigma clip the magnitudes
    clipped = sigma_clip(
        y,
        sigma=5,
        maxiters=5,
        cenfunc='median',
        stdfunc='std'
    )
    
    mask = ~clipped.mask
    
    t = t[mask]
    y = y[mask]
    dy = dy[mask]
    
    print(f"Kept {mask.sum()} / {len(mask)} points after 3σ clipping.")
    
    ls = LombScargle(
        t,
        y,
        dy
    )

    # Search hours-to-days range
    '''
    frequency, power = ls.autopower(
        minimum_frequency=1/5,     # periods up to 5 days
        maximum_frequency=1/0.05,  # periods down to 1.2 hr
        samples_per_peak=20
    )
    '''
    frequency, power = ls.autopower(
        minimum_frequency=2.3115,   
        maximum_frequency=2.31175,  
        samples_per_peak=50
    )

    periods = 1/frequency

    best_period = periods[np.argmax(power)]

    print(
        f"{band}: LS period = {best_period:.6f} days "
        f"({best_period*24:.3f} hr)"
    )


    band_results[band] = {
        "time": t,
        "mag": y,
        "err": dy,
        "ls_period": best_period
    }


    # -------------------------
    # Plot LS periodogram
    # -------------------------

    plt.figure(figsize=(10,4))

    plt.plot(
        frequency,
        power,
        color=colors[band]
    )

    f_true = 1/nova_period
    
    plt.axvline(
        f_true,
        color='black',
        linestyle='--',
        label=f"Nova Muscae {f_true:.3f} d$^{{-1}}$"
    )
    '''
    for n in [-2,-1,1,2]:
        plt.axvline(
            abs(f_true+n),
            color='orange',
            linestyle=':',
            alpha=0.7
        )
    '''

    plt.xlabel("Frequency (1/day)")
    plt.ylabel("LS Power")
    plt.title(f"{target} {band}-band Lomb Scargle")
    plt.legend()

    plt.show()



    # -------------------------
    # Phase fold
    # -------------------------
    
    phase = ((t - t.min()) / nova_period) % 1
    
    sort = np.argsort(phase)
    
    plt.figure(figsize=(10,4))
    
    plt.errorbar(
        phase[sort],
        y[sort],
        yerr=dy[sort],
        fmt='.',
        ms=2,
        alpha=0.3,
        color=colors[band]
    )
    
    plt.gca().invert_yaxis()
    plt.xlabel("Orbital Phase")
    plt.ylabel(f"{band} magnitude")
    plt.title(f"{target} {band}-band")
    
    plt.xlim(0,1)
    plt.show()
    
    
    # ==========================================================
    # Median-bin the light curve
    # ==========================================================
    
    nbins = 16
    
    edges = np.linspace(0,1,nbins+1)
    centres = 0.5*(edges[:-1] + edges[1:])
    
    phase_bin = []
    mag_bin = []
    err_bin = []
    
    for lo, hi, cen in zip(edges[:-1], edges[1:], centres):
    
        inside = (phase >= lo) & (phase < hi)
    
        if inside.sum() < 3:
            continue
    
        phase_bin.append(cen)
    
        mag_bin.append(
            np.nanmedian(y[inside])
        )
    
        # median reported uncertainty
        err = np.nanmedian(dy[inside])
    
        # scatter inside the bin
        scatter = np.nanstd(y[inside])
    
        # use whichever is larger
        err_bin.append(
            max(err, scatter/np.sqrt(inside.sum()))
        )
    
    phase_bin = np.asarray(phase_bin)
    mag_bin = np.asarray(mag_bin)
    err_bin = np.asarray(err_bin)
    
    
    # ==========================================================
    #model
    p0 = [
        np.nanmedian(mag_bin),
        0,
        0,
        0.1,
        0
    ]
    
    pars, cov = curve_fit(
        ellipsoidal_model,
        phase_bin,
        mag_bin,
        sigma=err_bin,
        absolute_sigma=True,
        p0=p0
    )
    
    phi_fit = np.linspace(0,1,500)
    mag_fit = ellipsoidal_model(phi_fit,*pars)
    
    
    # ==========================================================
    # Plot everything
    # ==========================================================
    
    plt.figure(figsize=(10,5))
    
    # raw data
    plt.scatter(
        phase,
        y,
        s=6,
        color=colors[band],
        alpha=0.15
    )
    
    # binned points
    plt.errorbar(
        phase_bin,
        mag_bin,
        yerr=err_bin,
        fmt='ko',
        ms=6,
        capsize=2,
        label='Median bins'
    )
    
    # duplicate for second cycle
    plt.errorbar(
        phase_bin+1,
        mag_bin,
        yerr=err_bin,
        fmt='ko',
        ms=6,
        capsize=2
    )
    
    plt.plot(
        phi_fit,
        mag_fit,
        'r-',
        lw=3,
        label='Ellipsoidal fit'
    )
    
    plt.plot(
        phi_fit+1,
        mag_fit,
        'r-',
        lw=3
    )
    
    plt.xlim(0,2)
    
    plt.gca().invert_yaxis()
    
    plt.xlabel("Orbital Phase")
    plt.ylabel(f"{band} magnitude")
    
    plt.title(f"{target} {band}")
    
    plt.legend()
    
    plt.show()
    

    
    # ==========================================================
    # Ellipsoidal model
    # ==========================================================
    
    def ellipsoidal_model(phi, C, A1, B1, A2, B2):
    
        return (
            C
            + A1*np.cos(2*np.pi*phi)
            + B1*np.sin(2*np.pi*phi)
            + A2*np.cos(4*np.pi*phi)
            + B2*np.sin(4*np.pi*phi)
        )
    
    
    # ----------------------------------------------------------
    # Iterative passive-state fit
    # ----------------------------------------------------------
    
    p0 = [
        np.nanmedian(y),
        0,
        0,
        0.1,
        0
    ]
    
    keep = np.ones(len(y), dtype=bool)
    
    for i in range(15):
    
        pars, cov = curve_fit(
            ellipsoidal_model,
            phase[keep],
            y[keep],
            sigma=dy[keep],
            absolute_sigma=True,
            p0=p0,
            maxfev=20000
        )
        
        # Linear fit
        line_pars, line_cov = curve_fit(
            linear_model,
            phase_bin,
            mag_bin,
            sigma=err_bin,
            absolute_sigma=True
        )
        
        line_fit = linear_model(phi_fit, *line_pars)

    
        model = ellipsoidal_model(phase, *pars)
    
        resid = y - model
    
        # Estimate scatter from only the faint side
        sigma = np.std(resid[resid > 0])
    
        # Reject only bright excursions (active state)
        keep_new = resid > -2*sigma
    
        print(
            f"Iteration {i+1}: kept {keep_new.sum()} / {len(keep_new)}"
        )
    
        if np.all(keep_new == keep):
            break
    
        keep = keep_new
        p0 = pars
    
    
    phi_fit = np.linspace(0,1,500)
    mag_fit = ellipsoidal_model(phi_fit,*pars)
    
    # Linear fit to the passive-state points only
    line_pars, line_cov = curve_fit(
        linear_model,
        phase[keep],
        y[keep],
        sigma=dy[keep],
        absolute_sigma=True
    )
    
    line_fit = linear_model(phi_fit, *line_pars)

    print(np.max(mag_fit)-np.min(mag_fit))
    
    # Ellipsoidal model
    chi2_ell = np.sum(
        ((y[keep] - ellipsoidal_model(phase[keep], *pars))/dy[keep])**2
    )
    
    chi2_line = np.sum(
        ((y[keep] - linear_model(phase[keep], *line_pars))/dy[keep])**2
    )
    
    redchi_ell = chi2_ell/(keep.sum() - 5)
    redchi_line = chi2_line/(keep.sum() - 2)
    
    print(f"Ellipsoidal χ² = {chi2_ell:.2f}")
    print(f"Linear      χ² = {chi2_line:.2f}")
    
    
    print(f"Reduced χ² (ellipsoidal) = {redchi_ell:.2f}")
    print(f"Reduced χ² (linear)      = {redchi_line:.2f}")
    
    N = keep.sum()

    bic_line = chi2_line + 2*np.log(N)
    bic_ell  = chi2_ell  + 5*np.log(N)
    
    print(f"BIC(line) = {bic_line:.2f}")
    print(f"BIC(ell)  = {bic_ell:.2f}")
    print(f"ΔBIC = {bic_line - bic_ell:.2f}")
    
    plt.figure(figsize=(10,5))
    
    # rejected (active) points
    plt.scatter(
        phase[~keep],
        y[~keep],
        s=8,
        color='lightgray',
        alpha=0.5,
        label='"Active"'
    )
    
    # kept (passive) points
    plt.scatter(
        phase[keep],
        y[keep],
        s=8,
        color=colors[band],
        alpha=0.5,
        label='"Passive"'
    )
    
    # duplicate second orbit
    plt.scatter(
        phase[~keep]+1,
        y[~keep],
        s=8,
        color='lightgray',
        alpha=0.5
    )
    
    plt.scatter(
        phase[keep]+1,
        y[keep],
        s=8,
        color=colors[band],
        alpha=0.5
    )
    
    # median bins (for display only)
    plt.errorbar(
        phase_bin,
        mag_bin,
        yerr=0,#err_bin,
        fmt='ko',
        ms=6,
        capsize=2,
        label='Median bins'
    )
    
    plt.errorbar(
        phase_bin+1,
        mag_bin,
        yerr=err_bin,
        fmt='ko',
        ms=6,
        capsize=2
    )
    
    # fitted passive-state model
    plt.plot(
        phi_fit,
        mag_fit,
        'r-',
        lw=3,
        label='Ellipsoidal fit, red χ² = {redchi_ell:.2f}'
    )
    
    plt.plot(
        phi_fit+1,
        mag_fit,
        'r-',
        lw=3
    )
    
    plt.plot(
        phi_fit,
        line_fit,
        'b--',
        lw=2,
        label=f'Linear fit, red χ² = {redchi_line:.2f}'
    )
    
    plt.plot(
        phi_fit+1,
        line_fit,
        'b--',
        lw=2
    )

    
    plt.xlim(0,2)
    plt.gca().invert_yaxis()
    
    plt.xlabel("Orbital Phase")
    plt.ylabel(f"{band} magnitude")
    plt.title(f"{target} {band}")
    
    plt.legend()
    
    plt.show()
    
    


    
    
    
        
