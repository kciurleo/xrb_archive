#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:02:09 2026

@author: kmc249
"""
import os
import re
from astropy.io import fits
import shutil
from lookup_name import *
### this is actually just moving everything from like late 2012 which somehow didn't get picked up
#july 2012 onward
#starts=['/USB2/archive/20120601thru0630','/USB2/archive/20120901thru0930', '/USB2/archive/20121201thru1231', '/USB2/archive/20120701thru0731', '/USB2/archive/20121001thru1031', '/USB2/archive/20120801thru0831', '/USB2/archive/20121101thru1130']
#starts=['/OLD-NET-DRIVE/xrb-archive/data/AqlX1-2003/ir','/OLD-NET-DRIVE/xrb-archive/tapes/AQLX1-IR-20030219-20030414', '/OLD-NET-DRIVE/xrb-archive/tapes/AQLX1-IR-20030418-20030526', '/OLD-NET-DRIVE/xrb-archive/tapes/AQLX1-IR-20030527-20030730','/OLD-NET-DRIVE/xrb-archive/tapes/AQLX1-IR-20030801-20031114',]
starts=['/OLD-NET-DRIVE/xrb-archive/data', '/OLD-NET-DRIVE/xrb-archive/tapes']
pattern = re.compile(r'^(ir|binir|rccd)\d{6,8}\.\d{4}\.fits$')

all_files = []

for start in starts:
    for root, dirs, files in os.walk(start):
        for name in files:
            if pattern.match(name):
                all_files.append(os.path.join(root, name))

print(f"Found {len(all_files)} files")

errors=[]
moved=[]
for file in all_files:
    basefilename = os.path.basename(file)
    try:
        hdr=fits.getheader(file)
        target=get_proper_name(hdr['OBJECT'])
    except:
        continue
    
    if target!='AqlX-1':#not is_xrb(target):
        continue
    print(f'found an xrb!! {target}')
    band = 'ir' if basefilename.startswith(('binir', 'ir')) else 'opt'
    if band=='ir':
        flt='IRFLTID'
        kind='raw'
    else:
        flt='CCDFLTID'
        if basefilename.startswith('ccd'):
            kind='ccd'
        elif  basefilename.startswith('rccd'):
            kind='rccd'
    filt=hdr[flt]
    
    dest=f'/neta/xrb/{target}/1.3m/{band}/{kind}/{filt}/'
    destfile = os.path.join(dest, basefilename)
    
    #make destination if it doesn't have it already
    os.makedirs(dest, exist_ok=True)
    
    if os.path.exists(destfile):
        print(f"already exists, skipping: {destfile}")
        #skipped.append(path)
    elif os.path.isdir(dest):
        shutil.copy2(file, destfile)
        print(f'moved {file} -> {destfile}')
        moved.append(file)
    else:
        print(f"error with destination: {dest}")
        errors.append(file)
        
print(errors)
print(len(errors))
print(moved)