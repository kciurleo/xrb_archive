#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 09:23:08 2025

@author: kmc249
"""
import glob
import os
from astropy.io import fits
from lookup_name import *
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)

paths=['/USB4/archive', '/USB2/archive/20130101thru0130', '/USB2/archive/20130201thru0228', '/USB2/archive/20130301thru0331', '/USB2/archive/20130401thru0430']

paths2= ['20131201thru1231',
 '20170601ThruEnd',
 '20161001ThruEnd',
 '20170501ThruEnd',
 '20160101ThruEnd',
 '20130901thru0929',
 '20150601ThruEnd',
 '20150201ThruEnd',
 '20140301ThruEnd',
 '20140901ThruEnd',
 '20170101ThruEnd',
 '20170701ThruEnd',
 '20150701ThruEnd',
 '20130401thru0430',
 '20170901ThruEnd',
 '20141201ThruEnd',
 '20150501ThruEnd',
 '20170801ThruEnd',
 '20161101ThruEnd',
 '20141001ThruEnd',
 '20150401ThruEnd',
 '20130501thru0531',
 '20130601thru0630',
 '20150301ThruEnd',
 '20130701thru0731',
 '20140701ThruEnd',
 '20150901ThruEnd',
 '20140801ThruEnd',
 '20140401ThruEnd',
 '20131101thru1130',
 '20130801thru0831',
 '20140101ThruEnd',
 '20140501ThruEnd',
 '20131001thru1031',
 '20170201ThruEnd',
 '20170301ThruEnd',
 '20160201ThruEnd',
 '20140201ThruEnd',
 '20160701ThruEnd',
 '20160401ThruEnd',
 '20160901ThruEnd',
 '20150801ThruEnd',
 '20160501ThruEnd',
 '20141101ThruEnd',
 '20140601ThruEnd',
 '20150101ThruEnd',
 '20151101ThruEnd',
 '20151201ThruEnd',
 '20160301ThruEnd',
 '20151001ThruEnd',
 '20161201ThruEnd',
 '20170401ThruEnd',
 '20160601ThruEnd',
 '20160801ThruEnd']

for p in paths2:
    newp=f'/USB3/archive/{p}'
    paths.append(newp)

#make a dictionary to hold all the rows
df_dict = {xrb: [] for xrb in xrb_list}

keywords=['OBJECT','RA','DEC','DATE-OBS','TIME-OBS','JD','EXPTIME','SECZ','CCDFLTID','IRFLTID','TILT1','TILT2','TILT3','OWNER']



new_names=[]
objs=[]
bad_hdr=[]
for i, repo in enumerate(paths):
    print(f'working on repo {i}/{len(paths)}')
    for dirpath, dirnames, filenames in os.walk(repo):
        print(f'current dir: {dirpath}')
        for f in filenames:
            if (f.endswith('.fits') or f.endswith('.fits.gz')) and not f.startswith('ccd'):
                full_path = os.path.join(dirpath, f)
                try:
                    hdr=fits.getheader(full_path)
                    obj=hdr['OBJECT']
                except:
                    bad_hdr.append(full_path)
                try:
                    name=get_proper_name(obj)
                    if is_xrb_quick(name):
                        objs.append(name)
                        #add all the things if this is an xrb
                        row=[f, full_path]
                        for k in keywords:
                            try:
                                row.append(hdr[k])
                            except:
                                row.append(np.nan)
                        df_dict[name].append(row)
                    else:
                        continue
                except:
                    new_names.append(obj)

for key in df_dict:
    df=pd.DataFrame(df_dict[key], columns=['filename', 'full path'] + keywords)
    df.to_csv(f'/home/kmc249/usbdrive_logs/usbdrivescrapelog_{key}.csv', index=False) 

new_name=list(set(new_names))
new_name.sort()

print('new names: ', new_name)
print('objs found:', set(objs))
print('bad_hdr:', bad_hdr)