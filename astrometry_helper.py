#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:04:12 2026

@author: kmc249
"""

import requests
import json
import time
import glob
from astrometry_net_client import Session
from astrometry_net_client import FileUpload
from astropy.io import fits
from astropy.wcs import WCS
import argparse
from astrometry_net_client import Settings

'''
#e.g.:
target='4U1543-47'
API_URL = "https://nova.astrometry.net/api/"
output_path = f'/neta/xrb/{target}/temp/{target}_wcs.fits'

txtfile = glob.glob(f'/neta/xrb/{target}/temp/{target}*ref.txt')[0]
data = {}
with open(txtfile) as f:
    for line in f:
        line = line.strip()
        if not line or ':' not in line:
            continue

        key, value = line.split(':', 1)

        key, value = line.split(':', 1)
        data[key.strip()] = value.strip()


most_files_band = data.get('band')
ref = data.get('reference_image')
x_pixel = float(data.get('x_pixel'))
y_pixel = float(data.get('y_pixel'))
'''

#where my api key is accessed
with open('/home/kmc249/Downloads/astrometry_api_key.txt', 'r') as f:
    API_KEY = f.read().strip()


def make_wcs_fits(ref, output_path, apikey=API_KEY):
    '''
    Parameters
    ----------
    ref : str
        path to fits file to wcs
    output_path : str
        output path to write new fits file to
    apikey : str, optional
        api key needed for astrometry.net, defaulting to my own

    Returns
    -------
    None - saves wcs fits file

    '''
    
    #api stuff
    s = Session(api_key=API_KEY)
    upl = FileUpload(ref, session=s)
    submission = upl.submit()
    submission.until_done()
    job = submission.jobs[0]
    job.until_done()
    if job.success():
        wcs = job.wcs_file()
    print(job.info())
    
    #write the file if successful
    if job.info()['status']=='success':
        hdul = fits.open(ref)
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(wcs)
        header.update(w.to_header(relax=True))
    
        fits.writeto(output_path, data, header, overwrite=True)
        
        print("WCS-added FITS written to:", output_path)
        
def make_centered_wcs_fits(ref, output_path, cra, cdec, apikey=API_KEY):
    '''
    Parameters
    ----------
    ref : str
        path to fits file to wcs
    output_path : str
        output path to write new fits file to
    cra: RA in degrees
    cdec: DEC in degrees
    apikey : str, optional
        api key needed for astrometry.net, defaulting to my own

    Returns
    -------
    None - saves wcs fits file

    '''
    
    #api stuff
    s = Session(api_key=API_KEY)
    upl = FileUpload(ref, session=s)
    settings = Settings()
    
    settings["center_ra"] = cra
    settings["center_dec"] = cdec
    settings["radius"] = 2.0
    
    settings["scale_units"] = "arcsecperpix"
    settings["scale_lower"] = 0.35
    settings["scale_upper"] = 0.40
    
    upl.settings = settings
    
    submission = upl.submit()
    submission.until_done()
    job = submission.jobs[0]
    job.until_done()
    if job.success():
        wcs = job.wcs_file()
    print(job.info())
    
    #write the file if successful
    if job.info()['status']=='success':
        hdul = fits.open(ref)
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(wcs)
        header.update(w.to_header(relax=True))
    
        fits.writeto(output_path, data, header, overwrite=True)
        
        print("WCS-added FITS written to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plate solve a FITS file using Astrometry.net and write WCS-added FITS."
    )

    parser.add_argument(
        "input_fits",
        help="Path to input FITS file"
    )

    parser.add_argument(
        "output_fits",
        help="Path to output WCS FITS file"
    )

    parser.add_argument(
        "--apikey",
        default=API_KEY,
        help="Astrometry.net API key"
    )

    args = parser.parse_args()

    make_wcs_fits(
        ref=args.input_fits,
        output_path=args.output_fits,
        apikey=args.apikey
    )
