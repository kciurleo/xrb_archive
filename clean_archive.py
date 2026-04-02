#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:05:54 2026

@author: kmc249
"""

from lookup_name import *
from astropy.io import fits
import os
import shutil
from datetime import datetime

#check that we don't have both ir and binir for the same thing, or rccd and ccd for the same thing
#check that the 1m data is in 1m spot and 1.3m data is in 1.3m spot
CUTOFF = datetime(2003, 1, 1)
DRY_RUN = False   # <<< SET TO False WHEN READY

for target in ['AqlX-1']: #xrb_list:
    homedir=f'/neta/xrb/{target}'
    
    all_files = []
    #walk the directory and get a list of files
    for root, dirs, files in os.walk(homedir):
        for f in files:
            if f.endswith('.fits') or f.endswith('.fits.gz'):
                all_files.append(os.path.join(root, f))
    
    #if we have both binir090802.1245.fits and ir090802.1245.fits, keep only the binir
    binir_files = {}
    ir_files = {}
    
    for path in all_files:
        fname = os.path.basename(path)
    
        # strip extension
        if fname.endswith('.fits.gz'):
            stem = fname[:-8]
        elif fname.endswith('.fits'):
            stem = fname[:-5]
        else:
            continue
    
        if stem.startswith('binir'):
            key = stem[len('binir'):]
            binir_files[key] = path
        elif stem.startswith('ir'):
            key = stem[len('ir'):]
            ir_files[key] = path
    
    # remove ir if binir exists
    removed_files = set()

    for key, ir_path in ir_files.items():
        if key in binir_files:
            print(f"Removing IR file: {ir_path}")
            removed_files.add(ir_path)
            if not DRY_RUN:
                os.remove(ir_path)
    
    # remove deleted files from all_files
    all_files = [f for f in all_files if f not in removed_files]
       
    #now, for each file, look at the date
    for file in all_files:
        try:
            hdr = fits.getheader(file)
            date_obs = hdr.get('DATE-OBS')
            if date_obs is None:
                continue

            obs_date = datetime.strptime(date_obs.strip(), "%Y-%m-%d")
        except:
            print(f"Skipping unreadable FITS: {file}")
            continue

        if obs_date < CUTOFF:
            correct_scope = '1m'
            wrong_scope = '1.3m'
        else:
            correct_scope = '1.3m'
            wrong_scope = '1m'

        if f'/{wrong_scope}/' not in file:
            continue  # already in correct place

        new_path = file.replace(f'/{wrong_scope}/', f'/{correct_scope}/')

        print(f"MOVE:\n  {file}\n  → {new_path}\n")

        if not DRY_RUN:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)



#somewhere here: should delete .gz files if the other one already exists