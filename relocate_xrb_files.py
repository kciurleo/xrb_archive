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
### this is actually just moving everything from like late 2012 which somehow didn't get picked up
#july 2012 onward
#starts=['/USB2/archive/20120601thru0630','/USB2/archive/20120901thru0930', '/USB2/archive/20121201thru1231', '/USB2/archive/20120701thru0731', '/USB2/archive/20121001thru1031', '/USB2/archive/20120801thru0831', '/USB2/archive/20121101thru1130']
#starts=['/OLD-NET-DRIVE/xrb-archive', '/OLD-NET-DRIVE/xrb', '/OLD-NET-DRIVE/glast',  '/OLD-NET-DRIVE/net_bailyn', '/OLD-NET-DRIVE/smarts','/XRB']
#starts=['/scratch']
starts=['/scratch/TAPE4/']
pattern = re.compile(
    r'^(binir|rccd|ir|ccd)\d{6,8}\.\d{4}\.fits$'
    r'|^r\d{4}_\d{4}\.\d{3}\.fits$'
)
CUTOFF = datetime(2003, 1, 1)
errors=[]
moved=[]
unreadable_hdrs=[]
unrecog=[]

for start in starts:
    print('working on', start)
    for root, dirs, files in os.walk(start):
        for name in files:
            hdr_failure=False
            skip_copy = False
            if pattern.match(name):
                file=os.path.join(root, name)
                basefilename = os.path.basename(file)
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
                    destfile = os.path.join(dest, basefilename)
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
                    destfile = os.path.join(dest, basefilename)
                
                
                    if kind in ['ccd', 'unbinned'] and os.path.isdir(dest):
                    #see if the better version already exists
                        if kind == 'ccd':
                            alt_kind = 'rccd'
                            prefix='r'
                        elif kind == 'unbinned':
                            alt_kind = 'raw'
                            prefix='bin'
                            
                        altdest=f'/neta/xrb/{target}/{scope}/{band}/{alt_kind}/{filt}/'
                        altdestfile = os.path.join(altdest, f'{prefix}{basefilename}')
                        if os.path.exists(altdestfile):
                            skip_copy=True
                    
                #make destination if it doesn't have it already
                os.makedirs(dest, exist_ok=True)
                
                if os.path.exists(destfile) or skip_copy:
                    print(f"already exists, skipping: {destfile}")
                    #skipped.append(path)
                    pass
                elif os.path.isdir(dest):
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

filesave='/neta/xrb/META/scratch_move_errors.txt'

with open(filesave, 'w') as f:
    f.write(f"unreadable_hdrs = {unreadable_hdrs!r}\n\n")
    f.write(f"errors = {errors!r}\n")