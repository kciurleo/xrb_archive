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
from datetime import datetime

starts=['/scratch/reading_CDs/irGX339.20030802thru1102/']
#starts=['/scratch/temp_CD_data/other_nonxrb/']
remove=True #delete things if they exist already?
move=True # as opposed to just copying (which you'd put false)

pattern = re.compile(
    r'^(binir|rccd|ir|ccd)\d{6,8}\.\d{4}\.fits(\.gz)?$'
    r'|^r\d{4}_\d{4}\.\d{3}\.fits(\.gz)?$'
)

#calib stuff
dest_root = '/neta/xrb/IRCALIBS'

pattern2 = re.compile(
    r'(binir|ir).*?(dark|dome|on|off|flat).*\.fits(\.gz)?$',
    re.IGNORECASE
)

year_pattern = re.compile(r'(\d{2})\d{4}')

CUTOFF = datetime(2003, 1, 1)
errors=[]
moved=[]
unreadable_hdrs=[]
unrecog=[]
calibs=[]
for start in starts:
    print('working on', start)
    for root, dirs, files in os.walk(start):
        for name in files:
            hdr_failure=False
            skip_copy = False
            #calibs first
            if pattern2.search(name) and year_pattern.search(name):
                file=os.path.join(root, name)
                basefilename = os.path.basename(file)
                match=year_pattern.search(basefilename)
                if basefilename.endswith('.fits.gz'):
                    compare_name = basefilename[:-3]   # remove '.gz'
                else:
                    compare_name = basefilename
                yy = int(match.group(1))
    
                if yy >= 90:
                    year = f"19{yy}"
                else:
                    year = f"20{yy:02d}"
                dest_dir = os.path.join(dest_root, year)

                dest_path = os.path.join(dest_dir, compare_name)

                #don't overwrite existing files
                if os.path.exists(dest_path):
                    print(f"already exists, skipping and deleting: {dest_path}")
                    if remove:
                        os.remove(file)
                    continue

                try:
                    if move:
                        shutil.move(file, dest_path)
                    else:
                        shutil.copy2(file, dest_path)
                    calibs.append(file)
                    print(f"moved: {file} → {dest_path}")
                except Exception as e:
                    print(f"ERROR moving {file}: {e}")
                    
            #all the rest of them
            if pattern.match(name):
                file=os.path.join(root, name)
                basefilename = os.path.basename(file)
                if basefilename.endswith('.fits.gz'):
                    compare_name = basefilename[:-3]   # remove '.gz'
                else:
                    compare_name = basefilename
                try:
                    hdr=fits.getheader(file)
                except:
                    unreadable_hdrs.append(file)
                    continue
                try:
                    target=get_proper_name(hdr['OBJECT'])
                except:
                    unrecog.append(file)
                    continue
                
                for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
                    try:
                        date_obs = hdr.get('DATE-OBS')
                        obs_date = datetime.strptime(date_obs.strip(), fmt)
                        break
                    except:
                        hdr_failure=True
                    
                if not is_xrb(target):
                    continue
                #print(f'found an xrb!! {target}')
                
                #move bad hdrs to location
                if hdr_failure:
                    dest=f'/neta/xrb/{target}/bad_hdrs/'
                    destfile = os.path.join(dest, compare_name)
                else:
                    if basefilename.startswith(('binir', 'ir')):
                        band='ir'
                        flt='IRFLTID'
                        if basefilename.startswith('binir'):
                            kind='raw'
                        elif basefilename.startswith('ir'):
                            kind='unbinned'
                    elif basefilename.startswith(('ccd', 'rccd')):
                        band='opt'
                        flt='CCDFLTID'
                        if basefilename.startswith('ccd'):
                            kind='ccd'
                        elif  basefilename.startswith('rccd'):
                            kind='rccd'
                    #dealing with early 1m stuff
                    else:
                        try:
                            filt = hdr['CCDFLTID'].strip().lower()
                            realfilt=hdr['CCDFLTID']
                            flt='CCDFLTID'
                        except:
                            filt = hdr['FILTERID'].strip().lower()
                            realfilt=hdr['FILTERID']
                            flt='FILTERID'
                        if filt in ['b','v','r','i','r wide', 'wide r']:
                            band='opt'
                            kind='rccd'
                        elif filt in ['y','h','k','j']:
                            band='ir'
                            kind='raw'
                        else:
                            errors.append(file)
                    filt=hdr[flt]
                    if obs_date < CUTOFF:
                        scope = '1m'
                    else:
                        scope = '1.3m'
                    
                    dest=f'/neta/xrb/{target}/{scope}/{band}/{kind}/{filt}/'
                    destfile = os.path.join(dest, compare_name)
                
                
                    if kind in ['ccd', 'unbinned'] and os.path.isdir(dest):
                    #see if the better version already exists
                        if kind == 'ccd':
                            alt_kind = 'rccd'
                            prefix='r'
                        elif kind == 'unbinned':
                            alt_kind = 'raw'
                            prefix='bin'
                            
                        altdest=f'/neta/xrb/{target}/{scope}/{band}/{alt_kind}/{filt}/'
                        altdestfile = os.path.join(altdest, f'{prefix}{compare_name}')
                        if os.path.exists(altdestfile):
                            skip_copy=True
                    
                #make destination if it doesn't have it already
                os.makedirs(dest, exist_ok=True)
                
                if os.path.exists(destfile) or skip_copy:
                    print(f"already exists, skipping and deleting: {destfile}")
                    if remove:
                        os.remove(file)
                    #skipped.append(path)
                    pass
                elif os.path.isdir(dest):
                    if move:
                        shutil.move(file, destfile)
                    else:
                        shutil.copy2(file, destfile)
                    print(f'moved {file} -> {destfile}')
                    moved.append(file)
                else:
                    print(f"error with destination: {dest}")
                    errors.append(file)
        
print('errors:',len(errors))
print('moved:',len(moved))
print('unreadable headers:', len(unreadable_hdrs))
print('unrecognized:', len(unrecog))
print(f'calibs moved: {len(calibs)}')

filesave='/neta/xrb/META/scratch_move_errors.txt'

with open(filesave, 'w') as f:
    f.write(f"unreadable_hdrs = {unreadable_hdrs!r}\n\n")
    f.write(f"errors = {errors!r}\n")