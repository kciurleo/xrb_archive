#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:13:13 2025

@author: kmc249
"""

import glob
import os
import pandas as pd
import numpy as np
from lookup_name import *

#make a dictionary to hold all the rows
df_dict = {xrb: [] for xrb in xrb_list}

#scrape logs first
filelist=glob.glob('/home/kmc249/usbdrive_logs/*scrape*')
badnames=[]
for file in filelist:
    name=file.split('/')[-1][18:-4]

    csv = pd.read_csv(file, on_bad_lines='skip')
    counts = csv['OWNER'].value_counts().to_dict()
    df_dict[name]=counts

print(df_dict)
    
    
filelist=glob.glob('/home/kmc249/usbdrive_logs/*rep*')
for file in filelist:
    name=file.split('/')[-1][15:-4]
    
    csv=pd.read_csv(file, on_bad_lines='skip')
    counts = csv['ProjectID'].value_counts().to_dict()
    for owner, count in counts.items():
        df_dict[name][owner] = df_dict[name].get(owner, 0) + count
print(df_dict)
print(f'bad names: {badnames}')