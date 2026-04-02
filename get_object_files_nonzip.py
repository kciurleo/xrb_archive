#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 10:17:32 2026

@author: kmc249
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import zipfile
from datetime import datetime
from lookup_name import *
import shutil

##This is the big file that will pull all the scrapelog and replog USB files for ALL of the xrbs. you can also move just one
#Because this is replog and scrapelog, I'm assuming all the files are the 1.3m (i.e. these are coming from 2009 or so onwards)


usb_roots = ["/USB1/archive", "/USB2/archive", "/USB3/archive", "/USB4/archive"]

#functions useful
def parse_span_folder(folder_name):
    """
    Converts a folder like '20090101thru0131' to a start and end date
    as datetime.date objects. Assumes folder_name is 'YYYYMMDDthruMMDD'.
    """
    try:
        start_str, end_str = folder_name.split("thru")
        # Start date is full YYYYMMDD
        start_date = datetime.strptime(start_str, "%Y%m%d").date()
        # End date might be missing the year, assume same year as start
        if len(end_str) == 4:  # MMDD
            end_date = datetime.strptime(str(start_date.year) + end_str, "%Y%m%d").date()
        else:  # full YYYYMMDD
            end_date = datetime.strptime(end_str, "%Y%m%d").date()
        return start_date, end_date
    except:
        # Not a date range folder
        return None, None

def build_folder_map(usb_roots):
    """
    Recursively walk all archive folders and build a map of date ranges to folder paths.
    """
    folder_map = {}
    for root in usb_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for dirname in dirnames:
                if "thru" in dirname:
                    start, end = parse_span_folder(dirname)
                    if start and end:
                        folder_map[(start, end)] = os.path.join(dirpath, dirname)
    return folder_map

def find_file(file_date_str, filename, folder_map):
    """
    Given a file date like '20090105' and filename like 'binir090105.0296',
    find its full path using the folder_map.
    """

    file_date = datetime.strptime(file_date_str, "%Y%m%d").date()

    if not filename.endswith(".fits"):
        filename += ".fits"

    for (start, end), span_path in sorted(folder_map.items()):
        if start <= file_date <= end:

            daily_folder = os.path.join(span_path, file_date_str)
            if not os.path.isdir(daily_folder):
                return None

            # walk ALL subdirectories (ir/, ccd/, etc)
            for root, dirs, files in os.walk(daily_folder):
                if filename in files:
                    return os.path.join(root, filename)

            # only fail AFTER walking everything
            return None

    return None
###testing to get object files for a given target

#Build the folder map once
folder_map = build_folder_map(usb_roots)


def main(target, band):
    csv1_path = f'/home/kmc249/usbdrive_logs/usbdrivereplog_{target}.csv'
    csv2_path = f'/home/kmc249/usbdrive_logs/usbdrivescrapelog_{target}.csv'
    
    # Read CSVs only if they exist and are not empty
    if os.path.exists(csv1_path) and os.path.getsize(csv1_path) > 0:
        csv1 = pd.read_csv(csv1_path)
    else:
        print(f"Replog CSV missing or empty: {csv1_path}. Skipping replog processing.")
        csv1 = pd.DataFrame()  # empty, safe for filtering
    
    if os.path.exists(csv2_path) and os.path.getsize(csv2_path) > 0:
        csv2 = pd.read_csv(csv2_path)
    else:
        print(f"Scrapelog CSV missing or empty: {csv2_path}. Skipping scrapelog processing.")
        csv2 = pd.DataFrame()  # empty, safe for filtering
    
    #filter if csv exists
    if not csv2.empty:
        if band=='ir':
            interest = csv2.loc[(csv2['filename'].str.startswith(('binir'))) | (csv2['filename'].str.startswith(('ir')))]
        elif band=='opt':
            interest = csv2.loc[(csv2['filename'].str.startswith(('rccd'))) | (csv2['filename'].str.startswith(('ccd')))]
    else:
        interest = pd.DataFrame()
    
    if not csv1.empty:
        if band=='ir':
            interest2 = csv1.loc[(csv1['Filename'].str.startswith(('binir'))) | (csv1['Filename'].str.startswith(('ir')))]
        elif band=='opt':
            interest2 = csv1.loc[(csv1['Filename'].str.startswith(('rccd'))) | (csv1['Filename'].str.startswith(('ccd')))]
    else:
        interest2 = pd.DataFrame()
    
    #we need to deal with the replog separately, because we don't have the full file path. filename is the file name only
    
    if band=='ir':
        flt='IRFLTID'
        kind='raw'
    else:
        flt='CCDFLTID'
    
    #move files in the scrapelog
    errors = []
    skipped = []
    nontarget = []
    if not csv2.empty:
        for path in set(interest["full path"]):
            #print('writing ',path)
            basefilename = os.path.basename(path)
            if band=='opt':
                if basefilename.startswith('ccd'):
                    kind='ccd'
                elif  basefilename.startswith('rccd'):
                    kind='rccd'
            try:
                hdr=fits.getheader(path)
                name=hdr['OBJECT']
                filt=hdr[flt]
                
                if get_proper_name(name)!=target:
                    print(f'not {target}. skipping')
                    nontarget.append(path)
                    continue
            except:
                print('error reading hdr')
                errors.append(path)
                continue
            dest=f'/neta/xrb/{target}/1.3m/{band}/{kind}/{filt}/'
            destfile = os.path.join(dest, basefilename)
            
            #make destination if it doesn't have it already
            os.makedirs(dest, exist_ok=True)
            
            if os.path.exists(destfile):
                print(f"already exists, skipping: {destfile}")
                #skipped.append(path)
            elif os.path.isdir(dest):
                shutil.copy2(path, destfile)
                #print(f'moved {path} -> {destfile}')
            else:
                print(f"error with destination: {dest}")
                errors.append(path)
                
        print('finished scrapelog. doing replog.')
        
    if not csv1.empty:
        #Use it on all the guys:
        for id, row in interest2.iterrows():
            file_date=row['replog'].split('/')[-1][6:-3]
            #specifically look for the bin ir or the rccd:
            if row['Filename'].startswith('ir'):
                filename='bin'+row['Filename']+'.fits'
            elif row['Filename'].startswith('ccd'):
                filename='r'+row['Filename']+'.fits'
            else:
                filename=row['Filename']+'.fits'
        
            path = find_file(file_date, filename, folder_map)
            if path:
                basefilename = os.path.basename(path)
                if band=='opt':
                    if basefilename.startswith('ccd'):
                        kind='ccd'
                    elif  basefilename.startswith('rccd'):
                        kind='rccd'
                try:
                    hdr=fits.getheader(path)
                    name=hdr['OBJECT']
                    filt=hdr[flt]
                    
                    if get_proper_name(name)!=target:
                        print(f'not {target}. skipping')
                        nontarget.append(path)
                        continue
                except:
                    print('error reading hdr')
                    errors.append(path)
                    continue
                dest=f'/neta/xrb/{target}/1.3m/{band}/{kind}/{filt}/'
                destfile = os.path.join(dest, basefilename)
                
                #make destination if it doesn't have it already
                os.makedirs(dest, exist_ok=True)
                
                if os.path.exists(destfile):
                    print(f"already exists, skipping: {destfile}")
                    #skipped.append(path)
                elif os.path.isdir(dest):
                    shutil.copy2(path, destfile)
                    #print(f'moved {path} -> {destfile}')
                else:
                    print(f"error with destination: {dest}")
                    errors.append(path)
            else:
                print(f"File not found: {filename}")
                errors.append(filename)
        
    print(skipped)
    print(errors)
    print(f"Files skipped: {len(skipped)}")
    print(f"Files errored: {len(errors)}")
    
    output_file = f"/neta/xrb/META/copy_errors_{band}_{target}.txt"
    
    with open(output_file, "w") as f:
        f.write("=== Skipped Files ===\n")
        for s in skipped:
            f.write(f"{s}\n")
        f.write(f"\nFiles skipped: {len(skipped)}\n\n")
        
        f.write("=== Nontarget Files ===\n")
        for s in nontarget:
            f.write(f"{s}\n")
        f.write(f"\nFiles ignored: {len(nontarget)}\n\n")
        
        f.write("=== Errored Files ===\n")
        for e in errors:
            f.write(f"{e}\n")
        f.write(f"\nFiles errored: {len(errors)}\n")
        
    print(f"Error log written to {output_file}")


if __name__ == "__main__":

    target_input = input("Enter target (press Enter to run all): ").strip()
    band_input = input("Enter band 'ir' or 'opt' (press Enter to run all): ").strip().lower()

    targets_to_run = [target_input] if target_input else xrb_list
    bands_to_run = [band_input] if band_input in ['ir', 'opt'] else ['ir', 'opt']

    for target in targets_to_run:
        for band in bands_to_run:
            print(f"\n=== Running for {target} {band} ===")
            main(target, band)