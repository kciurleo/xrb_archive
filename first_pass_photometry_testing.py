#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:09:03 2026

@author: kmc249
"""


#### This version is to test whether it really is the x-y coordinate issue that we're having. 
#### This version DAOphots each image to get the best star locations.

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
            
            mean, median, std = sigma_clipped_stats(data, sigma=3, maxiters=5)
            
            ###THIS IS THE DIFFERENT BIT
            
            # ------------------------------------------------------------
            # Reference star positions and IDs
            # ------------------------------------------------------------
            
            ref_positions = refstars[['xpix', 'ypix']].to_numpy()
            
            # Maximum distance (pixels) that DAO is allowed to move
            # from the original/reference position
            max_match_distance = 3.0
            
            # ------------------------------------------------------------
            # Run DAOStarFinder
            # ------------------------------------------------------------
            
            daofinder = DAOStarFinder(
                threshold=2 * std,
                fwhm=fwhm,
                xycoords=ref_positions
            )
            
            sources = daofinder(bkg_sub_full_data)
            
            # ------------------------------------------------------------
            # Initialize ALL stars as NaN
            # ------------------------------------------------------------
            
            for sid in source_ids:
                band_df.at[ind, sid] = np.nan
            
            # These are for plotting / diagnostics
            matched_positions = []
            matched_ids = []
            
            # ------------------------------------------------------------
            # Nothing found in this image
            # ------------------------------------------------------------
            
            if sources is None or len(sources) == 0:
            
                print("    DAOStarFinder found no sources")
            
            else:
            
                dao_positions = np.transpose(
                    (sources['xcentroid'], sources['ycentroid'])
                )
            
                # Keep track of which DAO detections have already
                # been assigned to a reference star
                used_dao_indices = set()
            
                # --------------------------------------------------------
                # Match each reference star to the nearest DAO detection
                # --------------------------------------------------------
            
                for ref_i, (ref_x, ref_y) in enumerate(ref_positions):
            
                    distances = np.sqrt(
                        (dao_positions[:, 0] - ref_x)**2 +
                        (dao_positions[:, 1] - ref_y)**2
                    )
            
                    # Sort DAO detections from closest to farthest
                    sorted_dao_indices = np.argsort(distances)
            
                    # Find the closest UNUSED DAO detection
                    dao_i = None
            
                    for candidate_i in sorted_dao_indices:
            
                        if candidate_i not in used_dao_indices:
                            dao_i = candidate_i
                            break
            
                    # If there aren't any unused DAO detections
                    if dao_i is None:
                        continue
            
                    min_distance = distances[dao_i]
            
                    sid = source_ids[ref_i]
            
                    # ----------------------------------------------------
                    # Accept the match only if it is close enough
                    # ----------------------------------------------------
            
                    if min_distance <= max_match_distance:
            
                        dao_x = dao_positions[dao_i, 0]
                        dao_y = dao_positions[dao_i, 1]
            
                        # Mark this DAO detection as used
                        used_dao_indices.add(dao_i)
            
                        # Save position and ID for plotting
                        matched_positions.append((dao_x, dao_y))
                        matched_ids.append(sid)
            
                        # ------------------------------------------------
                        # Aperture photometry at DAO position
                        # ------------------------------------------------
            
                        aperture = CircularAperture(
                            [(dao_x, dao_y)],
                            r=r_ap
                        )
            
                        aper_phot = aperture_photometry(
                            bkg_sub_full_data,
                            aperture
                        )
            
                        band_df.at[ind, sid] = aper_phot['aperture_sum'][0]
            
                        print(
                            f"    {sid}: "
                            f"reference=({ref_x:.2f}, {ref_y:.2f}), "
                            f"DAO=({dao_x:.2f}, {dao_y:.2f}), "
                            f"offset={min_distance:.2f} pix"
                        )
            
                    else:
            
                        # DAO did not find this star close enough
                        band_df.at[ind, sid] = np.nan
            
                        print(
                            f"    {sid}: NOT FOUND "
                            f"(nearest DAO source = {min_distance:.2f} pix)"
                        )
            
            ###END DIFFERENT BIT

            
            
            
            
            
            
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
                plt.imshow(
                    data,
                    origin='lower',
                    cmap='gray',
                    norm=norm
                )
            
                # --------------------------------------------------------
                # Original/reference positions
                # --------------------------------------------------------
            
                ref_apertures = CircularAperture(
                    ref_positions,
                    r=r_ap
                )
            
                ref_apertures.plot(
                    color='cyan',
                    lw=1.2,
                    alpha=0.7,
                    linestyle='--'
                )
            
                # --------------------------------------------------------
                # DAO-found positions
                # --------------------------------------------------------
            
                if len(matched_positions) > 0:
            
                    matched_positions = np.array(matched_positions)
            
                    dao_apertures = CircularAperture(
                        matched_positions,
                        r=r_ap
                    )
            
                    dao_apertures.plot(
                        color='red',
                        lw=1.5,
                        alpha=0.9
                    )
            
                    # Label DAO positions with stable reference IDs
                    for (x, y), sid in zip(
                        matched_positions,
                        matched_ids
                    ):
            
                        plt.annotate(
                            str(sid),
                            (x, y),
                            textcoords='offset points',
                            xytext=(5, 5),
                            color='yellow',
                            fontsize=8
                        )
            
                # --------------------------------------------------------
                # Plot formatting
                # --------------------------------------------------------
            
                plt.title(
                    f'{tele} {band} — reference vs DAO positions'
                )
            
                plt.show()

            '''
            #for src in sources: #put this back if you put back the DAOStarFinder thing
            for _, src in sources.iterrows():
                sid = src['source_id']
                flux = src['flux']
            
                band_df.at[ind, sid] = flux
            '''
        
        outfile = f'{outdir}/{target}_{tele}_{band}_first_pass_phot_with_DAO.csv'
        band_df.to_csv(outfile, index=False)
        
        print(f'Saved {outfile}')