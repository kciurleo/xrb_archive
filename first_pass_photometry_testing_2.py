#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:09:03 2026

@author: kmc249
"""


#This is another test version. instead of using the positions I got from the plate solved image, 
#I'm going to get the positions from the stacks.

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.psf import EPSFBuilder, EPSFStars, PSFPhotometry, IterativePSFPhotometry, CircularGaussianPRF, CircularGaussianPSF, SourceGrouper
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch, simple_norm
from astropy.table import QTable, Table
from scipy.ndimage import map_coordinates
from photutils.psf import extract_stars
from astropy.nddata import NDData
from smith_utils import *
from photutils.datasets import make_model_image
import glob
from scipy.optimize import curve_fit
import os
import pandas as pd

#aperture radius
r_ap = 5.0
fwhm=3.0
target='CenX-4'

outdir = f'/neta/xrb/{target}/product/first_pass_lightcurves'
os.makedirs(outdir, exist_ok=True)

#reference stars
reffile=glob.glob(f'/neta/xrb/{target}/product/*_ref_stars_kciurleo.csv')[0]
refstars=pd.read_csv(reffile)
eids = refstars.loc[refstars['type'] != 'target'].index.tolist()
cols=['filename','time', 'target'] + [str(e) for e in eids]

#label column
source_ids = []

for _, row in refstars.iterrows():
    if row['type'] == 'target':
        source_ids.append('target')
    else:
        #source_ids.append(str(row['objid']))
        source_ids.append(str(_))

#to iterate over
optical_bands=['B','V','R','I']
telescopes=['1.3m', '1m']


###DIFFERENT BIT

# ------------------------------------------------------------
# Get improved reference positions from stacked images
# ------------------------------------------------------------

savedir = '/neta/xrb/PRODUCTS/'
stacked_positions = {}

for band in optical_bands:

    stackfile = f'{savedir}/stacked_images_optical/{target}_{band}_stacked.fits'

    if not os.path.exists(stackfile):
        print(f'No stacked image found for {band}')
        continue

    print(f'Finding reference positions in stacked {band} image')

    # Read stacked image
    stack_data = fits.getdata(stackfile)

    # Background subtraction
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()

    stack_bkg = Background2D(
        stack_data,
        (20, 20),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator
    )

    stack_bkg_sub = stack_data - stack_bkg.background
    stack_bkg_sub = np.nan_to_num(stack_bkg_sub, nan=0.0)

    # Statistics for DAOStarFinder
    mean, median, std = sigma_clipped_stats(
        stack_bkg_sub,
        sigma=3,
        maxiters=5
    )

    # Run DAOStarFinder on the STACK
    daofinder = DAOStarFinder(
        threshold=2 * std,
        fwhm=fwhm
    )

    stack_sources = daofinder(stack_bkg_sub)

    if stack_sources is None or len(stack_sources) == 0:
        print(f'No sources found in stacked {band} image')
        continue

    print(
        f'Found {len(stack_sources)} sources in stacked {band} image'
    )

    # --------------------------------------------------------
    # Match stack detections to your existing reference stars
    # --------------------------------------------------------
    
    old_positions = refstars[['xpix', 'ypix']].to_numpy()

    stack_positions = np.transpose(
        (
            stack_sources['xcentroid'],
            stack_sources['ycentroid']
        )
    )

    max_match_distance = 5.0

    # Store positions in the SAME ORDER as source_ids
    new_positions = np.full(
        (len(source_ids), 2),
        np.nan
    )

    used_stack_indices = set()

    for ref_i, (ref_x, ref_y) in enumerate(old_positions):

        distances = np.sqrt(
            (stack_positions[:, 0] - ref_x)**2 +
            (stack_positions[:, 1] - ref_y)**2
        )

        # Find closest unused stack detection
        sorted_indices = np.argsort(distances)

        stack_i = None

        for candidate_i in sorted_indices:
            if candidate_i not in used_stack_indices:
                stack_i = candidate_i
                break

        if stack_i is None:
            continue

        min_distance = distances[stack_i]

        sid = source_ids[ref_i]

        if min_distance <= max_match_distance:

            new_positions[ref_i] = stack_positions[stack_i]

            used_stack_indices.add(stack_i)

            print(
                f'    {sid}: '
                f'old=({ref_x:.2f}, {ref_y:.2f}) '
                f'-> stack=({stack_positions[stack_i,0]:.2f}, '
                f'{stack_positions[stack_i,1]:.2f}), '
                f'offset={min_distance:.2f} pix'
            )

        else:

            print(
                f'    {sid}: NOT FOUND in stack '
                f'(nearest source = {min_distance:.2f} pix)'
            )

    # Save positions for this band
    stacked_positions[band] = new_positions

    # --------------------------------------------------------
    # Plot the stacked-image positions
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Plot original vs new stacked-image positions
    # --------------------------------------------------------

    norm = simple_norm(
        stack_data,
        'sqrt',
        percent=99
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(
        stack_data,
        origin='lower',
        cmap='gray',
        norm=norm
    )

    # --------------------------------------------------------
    # Plot ORIGINAL/reference positions in cyan
    # --------------------------------------------------------

    original_apertures = CircularAperture(
        old_positions,
        r=r_ap
    )

    original_apertures.plot(
        color='cyan',
        lw=1.2,
        alpha=0.8,
        linestyle='--'
    )

    # --------------------------------------------------------
    # Plot NEW stacked-image DAO positions in red
    # --------------------------------------------------------

    for i, (new_x, new_y) in enumerate(new_positions):

        sid = source_ids[i]

        old_x, old_y = old_positions[i]

        # ----------------------------------------------------
        # Star was successfully found in stack
        # ----------------------------------------------------

        if np.isfinite(new_x) and np.isfinite(new_y):

            # Red aperture at new position
            new_aperture = CircularAperture(
                [(new_x, new_y)],
                r=r_ap
            )

            new_aperture.plot(
                color='red',
                lw=1.5,
                alpha=0.9
            )

            # ------------------------------------------------
            # Draw line from old -> new position
            # ------------------------------------------------

            plt.plot(
                [old_x, new_x],
                [old_y, new_y],
                color='yellow',
                lw=1.0,
                alpha=0.8
            )

            # ------------------------------------------------
            # Label with source ID
            # ------------------------------------------------

            plt.annotate(
                str(sid),
                (new_x, new_y),
                textcoords='offset points',
                xytext=(5, 5),
                color='yellow',
                fontsize=8
            )

        # ----------------------------------------------------
        # Star was NOT found in stack
        # ----------------------------------------------------

        else:

            plt.annotate(
                str(sid) + ' (missed)',
                (old_x, old_y),
                textcoords='offset points',
                xytext=(5, 5),
                color='orange',
                fontsize=8
            )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            color='cyan',
            lw=1.2,
            linestyle='--',
            label='Original position'
        ),
        Line2D(
            [0], [0],
            color='red',
            lw=1.5,
            label='Stack DAO position'
        ),
        Line2D(
            [0], [0],
            color='yellow',
            lw=1.0,
            label='Position shift'
        ),
    ]

    plt.legend(
        handles=legend_elements,
        loc='upper right'
    )

    plt.title(
        f'{target} {band} — original vs stacked DAO positions'
    )

    plt.show()



for tele in telescopes:
    #Only do the things we have, and do them separately.
    print(f'Trying {tele}')
    if not os.path.exists(f'/neta/xrb/{target}/{tele}/'):
        print(f'Skipping {tele}')
        continue
    
    
    for band in optical_bands:
        trimdir=f'/neta/xrb/{target}/{tele}/opt/rccd/{band}_trimmed/'
        if not os.path.exists(trimdir):
            print(f'Skipping {tele} {band}')
            continue
        print(f'Working on {tele} {band}')

        #Get a file list
        filelist=sorted(glob.glob(f'{trimdir}*'))
        
        #initialize this df
        band_df = pd.DataFrame(0, index=np.arange(len(filelist)), columns=cols)
        band_df['filename'] = filelist
        band_df['time'] = pd.NaT 
        band_df['target'] = np.nan
        for e in eids:
            band_df[str(e)] = np.nan 
            
        for ind, row in band_df.iterrows():
            print(f'Working on {ind} out of {len(filelist)}')
            file=row['filename']
            #get data
            data=fits.getdata(file)
            hdr=fits.getheader(file)
            #hold onto time
            band_df.at[ind, 'time']=pd.to_datetime(f"{hdr['DATE-OBS']}T{hdr['TIME-OBS']}", errors='coerce')
            
            #background subtraction?
            #background subtract data
            sigma_clip=SigmaClip(sigma=3.0)
            bkg_estimator=MedianBackground()
            fullbkg=Background2D(data, (20,20), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            bkg_sub_full_data=data-fullbkg.background
            bkg_sub_full_data = np.nan_to_num(bkg_sub_full_data, nan=0.0)
            
            
            #Let DAOstarfinder get actual x y coords with initial conditions?
            '''
            mean, median, std = sigma_clipped_stats(data, sigma=3, maxiters=5)
            
            xycoords = refstars[['xpix', 'ypix']].to_numpy()
            
            daofinder = DAOStarFinder(
                threshold=2 * std,
                fwhm=fwhm,
                xycoords=xycoords
            )
            
            sources = daofinder(bkg_sub_full_data)
            
            sources['source_id'] = source_ids
            
            # aperture photometry
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            '''
            #or do we just use the things we have right now?
            positions = stacked_positions[band]
            
            # Only use stars that were actually found in the stack
            valid = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1])
            
            valid_positions = positions[valid]
            valid_source_ids = np.array(source_ids)[valid]
            
            apertures = CircularAperture(
                valid_positions,
                r=r_ap
            )
            
            aper_phot = aperture_photometry(
                bkg_sub_full_data,
                apertures
            )
            
            # Put fluxes into their consistent source IDs
            for sid, flux in zip(
                valid_source_ids,
                aper_phot['aperture_sum']
            ):
            
                band_df.at[ind, sid] = flux
            
            # plot to check apertures
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(data)
            
            norm = ImageNormalize(
                vmin=vmin,
                vmax=vmax,
                stretch=SinhStretch()
            )
            if ind == 0:
    
                plt.figure(figsize=(8, 8))
            
                norm = simple_norm(
                    data,
                    'sqrt',
                    percent=99
                )
            
                plt.imshow(
                    data,
                    origin='lower',
                    cmap='gray',
                    norm=norm
                )
            
                # Plot positions obtained from stacked image
                for i, (x, y) in enumerate(positions):
            
                    sid = source_ids[i]
            
                    if np.isfinite(x) and np.isfinite(y):
            
                        aperture = CircularAperture(
                            [(x, y)],
                            r=r_ap
                        )
            
                        aperture.plot(
                            color='red',
                            lw=1.5,
                            alpha=0.8
                        )
            
                        plt.annotate(
                            str(sid),
                            (x, y),
                            textcoords='offset points',
                            xytext=(5, 5),
                            color='yellow',
                            fontsize=8
                        )
            
                plt.title(
                    f'{tele} {band} — photometry using stacked-image positions'
                )
            
                plt.show()

        
        outfile = f'{outdir}/{target}_{tele}_{band}_first_pass_phot_with_stacked_inits.csv'
        band_df.to_csv(outfile, index=False)
        
        print(f'Saved {outfile}')