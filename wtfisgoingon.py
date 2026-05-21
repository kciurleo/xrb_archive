#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:16:53 2026

@author: kmc249
"""


import numpy as np
import pandas as pd
from astropy.time import Time
import glob


#read in all the dfs
filelist=glob.glob('/home/kmc249/Downloads/take3/*')
print(filelist)

data = {}

def read_corrected_txt(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=[
            "MJD",
            "mag_corr",
            "mag_corr_err",
            "flag",
            "mag",
            "mag_err",
            "error_flag"
        ]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

for file in filelist:
    if 'SMARTS' in file:
        tele='SMARTS'
    else:
        tele='LCO'
    filename=file.split('/')[-1]
    band=filename.split('_')[0]
    data[f'{band} {tele}']={'df':read_corrected_txt(file)}

#Outburst list
full = pd.read_csv("/home/kmc249/Downloads/full_outbursts.csv")
mini = pd.read_csv("/home/kmc249/Downloads/mini_outbursts.csv")

#Mask out quiescence
intervals = list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))

def get_quiescent(df, intervals):
    mask = np.ones(len(df), dtype=bool)
    
    for start, end in intervals:
        mask &= ~((df["MJD"] >= start) & (df["MJD"] <= end))
    
    return df[mask].copy()

for key in data:
    df = data[key]['df']
    data[key]['quiescence'] = get_quiescent(df, intervals)
    data[key]['global med q mag'] = data[key]['quiescence']['mag_corr'].median()
    data[key]['global mean q mag'] = data[key]['quiescence']['mag_corr'].mean()
    print(key, data[key]['global mean q mag'])




