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
import matplotlib.pyplot as plt

#NOTE: the resulting plots will likely not include a lot of the CDs, which should be mostly yale.
#this will skew slightly non-yale then.

#make a dictionary to hold all the rows
df_dict = {xrb: [] for xrb in xrb_list}

#scrape logs first
filelist=glob.glob('/home/kmc249/usbdrive_logs/*scrape*')
for file in filelist:
    name=file.split('/')[-1][18:-4]

    csv = pd.read_csv(file, on_bad_lines='skip')
    counts = csv['OWNER'].value_counts().to_dict()
    df_dict[name]=counts
    
    
filelist=glob.glob('/home/kmc249/usbdrive_logs/*rep*')
for file in filelist:
    name=file.split('/')[-1][15:-4]
    
    csv=pd.read_csv(file, on_bad_lines='skip')
    counts = csv['ProjectID'].value_counts().to_dict()
    for owner, count in counts.items():
        df_dict[name][owner] = df_dict[name].get(owner, 0) + count

for name in xrb_list:
    try:
        csv=pd.read_csv(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name}/incomplete_table_{name}.csv', low_memory=False)
        counts = csv['OWNER'].value_counts().to_dict()
        for owner, count in counts.items():
            df_dict[name][owner] = df_dict[name].get(owner, 0) + count
        D=df_dict[name]
        plt.figure(figsize=(6,10))
        plt.barh(range(len(D)), list(D.values()), align='center')
        plt.yticks(range(len(D)), list(D.keys()))
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name}/owners_{name}.png', dpi=200)
        plt.show()
    except:
        print('nothing for ', name)