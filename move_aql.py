#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:07:45 2026

@author: kmc249
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from collections import Counter
import glob
import os
import re
import shutil
from lookup_name import *
from datetime import datetime
'''
infile='/home/kmc249/test_data/xrb_archive/internal_plots/AqlX-1/incomplete_table_AqlX-1.csv'
maindir='/OLD-NET-DRIVE/xrb-archive/data/aqlx1_yalo_IR/'
dstdir='/neta/xrb/IRCALIBS/'
subdirs=['2001/','2002/']

df=pd.read_csv(infile, low_memory=False)

ir=df.loc[df["filename"].str.contains("ir", na=False)]
print(len(ir))

print(Counter(ir['Location']))
'''
'''
calib_re = re.compile(r'^ir\d+\.[^.]*?(on|off|dark|dome|flat)[^.]*?\.', re.IGNORECASE)
number_re = re.compile(r'^ir\d+\.\d+\.')

caliblist=[]
numberlist=[]
junk=[]
for i in subdirs:
    fullpath=maindir+i
    #just get ir fits files in case there's other stuff in here
    irlist=glob.glob(f'{fullpath}ir*fits')
    for file in irlist:
        f = os.path.basename(file)
        if calib_re.search(f):
            caliblist.append(f)
        elif number_re.search(f):
            numberlist.append(file)
        else:
            junk.append(f)

byhand=['ir010621nickdark.0001.fits', 'ir010621nickdark.0002.fits', 'ir010621nickdark.0003.fits', 'ir010621nickdark.0004.fits', 'ir010621nickdark.0005.fits', 'ir010621nickdark.0006.fits', 'ir010621nickdark.0008.fits', 'ir010621nickdark.0009.fits', 'ir010621nickdark.0010.fits', 'ir010621nickdark10.0001.fits', 'ir010621nickdark10.0002.fits', 'ir010621nickdark10.0003.fits', 'ir010621nickdark10.0004.fits', 'ir010621nickdark10.0005.fits', 'ir010621nickdark10.0006.fits', 'ir010621nickdark10.0007.fits', 'ir010621nickdark10.0008.fits', 'ir010621nickdark10.0009.fits', 'ir010621nickdark10.0010.fits', 'ir010621nickdark10.0011.fits', 'ir010621nickdark10.0013.fits', 'ir010621nickjoff.0001.fits', 'ir010621nickjoff.0002.fits', 'ir010621nickjon.0001.fits', 'ir010621nickjon.0002.fits', 'ir010621nickdark.0007.fits', 'ir010621nickdark10.0012.fits']
caliblist=caliblist+byhand

year_re = re.compile(r'^ir(\d{2})')

for f in caliblist:
    m = year_re.match(f)
    if not m:
        print("Could not parse year:", f)
        continue

    yy = m.group(1)

    if yy == "01":
        year_dir = "2001"
    elif yy == "02":
        year_dir = "2002"
    else:
        print("Unknown year:", f)
        continue

    src = os.path.join(maindir, year_dir, f)
    dst = os.path.join(dstdir, year_dir, f)

    shutil.copy2(src, dst)

#same thing but check that it's aql    
for file in numberlist:
    f = os.path.basename(file)
    #read hdr
    name=fits.getheader(file)['OBJECT']
    filt=fits.getheader(file)['IRFLTID']
    #they're all J for now. work smarter not harder
    
    if get_proper_name(name)!='AqlX-1':
        continue
    

    dst = os.path.join('/neta/xrb/AqlX-1/1m/ir/raw/J/', f)
    shutil.copy2(file, dst)
'''
'''
#some code to help with replog from it which shall not be named:
usb_roots = ["/USB1/archive", "/USB2/archive", "/USB3/archive", "/USB4/archive"]

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
    print(file_date)
    # Step 1: Find the span folder
    span_folder_path = None
    for (start, end), path in folder_map.items():
        if start <= file_date <= end:
            span_folder_path = path
            break
    
    if not span_folder_path:
        return None  # No folder found for this date
    
    # Step 2: Go into the daily folder named exactly like the date
    daily_folder = os.path.join(span_folder_path, file_date_str)
    if not os.path.isdir(daily_folder):
        return None  # Daily folder doesn't exist
    
    # Step 3: Check for the file in that folder
    file_path = os.path.join(daily_folder, filename)
    if os.path.isfile(file_path):
        return file_path
    else:
        return None

#Build the folder map once
folder_map = build_folder_map(usb_roots)
print(folder_map)

errors=[]
replog=ir.loc[ir['Physical loc']=='Data replog']
print(replog)
for id, row in replog.head(10).iterrows():
    yy=row['filename'][2:4]
    mm=row['filename'][4:6]
    dd=row['filename'][6:8]
    if dd[-1]=='.':
        yy=row['filename'][2:3]
        mm=row['filename'][3:5]
        dd=row['filename'][5:7]
    
    #need to be smart!! if there's whitespace, chop this off
    basefilename = row['filename'].split()[0] + '.fits'
    print(basefilename)
    
    #figuring out the destination:
    try:
        if int(yy)>2 and int(yy)<98:
            tel='1.3m'
        elif yy in ['00', '01', '02']:
            tel='1m'
        else:
            errors.append(row['filename'])
            continue
    except:
        errors.append(row['filename'])
        continue
    if row['IRFLTID'] in ['H','J','K','Y']:
        filt=row['IRFLTID']
    else:
        errors.append(row['filename'])
        continue
    dest=f'/neta/xrb/AqlX-1/{tel}/ir/raw/{filt}/{basefilename}'
    
    #finding a source
    if int(yy)<98:
        filedate='20'+yy+mm+dd
    elif int(yy)>=98:
        filedate='19'+yy+mm+dd
    #print(filedate)
    
    found_path = find_file(filedate, 'bin'+basefilename, folder_map)
    if found_path:
        print(f"File found: {found_path}")
    else:
        print("File not found.")
        continue
    
    shutil.copy2(found_path, dest)
'''
#startdir='/OLD-NET-DRIVE/xrb-archive/usb-data/AqlX-1/fitsimages/R/'
startdir='/OLD-NET-DRIVE/xrb-archive/data/AqlX1-2003/ir/'
filelist=glob.glob(f'{startdir}*')

errors=[]
nontarget=[]
skipped=[]
for file in filelist:
    filename = os.path.basename(file)
    if filename.endswith('.gz'):
        basefilename = filename.removesuffix('.gz')
    else:
        basefilename = os.path.basename(file)
    try:
        hdr=fits.getheader(file)
        name=hdr['OBJECT']
        filt=hdr['IRFLTID']
        
        if get_proper_name(name)!='AqlX-1':
            print(f'not aql. skipping')
            nontarget.append(file)
            continue
    except:
        print('error reading hdr')
        errors.append(file)
        continue
    dest=f'/neta/xrb/AqlX-1/1.3m/ir/raw/{filt}/'
    destfile = os.path.join(dest, basefilename)
    
    #make destination if it doesn't have it already
    os.makedirs(dest, exist_ok=True)
    
    if os.path.exists(destfile):
        print(f"already exists, skipping: {destfile}")
        #skipped.append(path)
    elif os.path.isdir(dest):
        shutil.copy2(file, destfile)
        print(f'moved {file} -> {destfile}')
    else:
        print(f"error with destination: {dest}")
        errors.append(file)
    

print(len(errors))
print(errors)
