#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:33:49 2026

@author: kmc249
"""
from astropy.io import fits
import re
import os
from lookup_name import *
from datetime import datetime
namespace = {}

with open('/neta/xrb/META/scratch_move_errors.txt') as f:
    exec(f.read(), namespace)

bad_hdrs = namespace['bad_hdrs']
errors = namespace['errors']
for file in bad_hdrs:
    if 'r0904_0598' in file:
        print(file)
'''
pattern = re.compile(
    r'^(binir|rccd|ir|ccd)\d{6,8}\.\d{4}\.fits$'
    r'|^r\d{4}_\d{4}\.\d{3}\.fits$'
)
CUTOFF = datetime(2003, 1, 1)

moved=[]
unreadable_hdrs=[]
unrecog=[]


for file in errors:
    
    hdr_failure=False
    skip_copy = False

    basefilename = os.path.basename(file)
    try:
        hdr=fits.getheader(file)
        #print(hdr)
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
                print('errored here!!!')
                continue
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
        if filt !='WIDE R':
            print('file is', file)
            print(f"error with destination: {dest}")

'''
