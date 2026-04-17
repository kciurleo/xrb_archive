#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:13:46 2026

@author: kmc249
"""
import pandas as pd
import pandas as pd
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt
import os
from prefixspan import PrefixSpan
import ast
import datetime
import matplotlib.dates as mdates
from collections import Counter

#my logfile
logfile=pd.read_csv('/neta/xrb/A0620-00/product/log_files_A0620-00.csv')
logfile['datetime']=pd.to_datetime(logfile['datetime'], format='mixed')
#get rid fo bad dates for now
logfile = logfile.dropna(subset=['datetime'])

#victor's 1m data
v = pd.read_csv(
    '/home/kmc249/Downloads/1m_VJDs.dat',
    sep=r'\s+',
    skiprows=1,
    names=['JD', 'date', 'time']
)
v['datetime'] = pd.to_datetime(v['date'] + ' ' + v['time'])

i=pd.read_csv(
    '/home/kmc249/Downloads/1m_IJDs.dat',
    sep=r'\s+',
    skiprows=1,
    names=['JD', 'date', 'time']
)
i['datetime'] = pd.to_datetime(i['date'] + ' ' + i['time'])
h=pd.read_csv(
    '/home/kmc249/Downloads/1m_HJDs.dat',
    sep=r'\s+',
    skiprows=1,
    names=['JD', 'date', 'time']
)
h['datetime'] = pd.to_datetime(h['date'] + ' ' + h['time'])


#### V band

#check to see if I have all this stuff
log_v = logfile.loc[logfile['filter'] == 'V'].copy()
log_v = log_v.sort_values('datetime').reset_index(drop=True)

v = v.sort_values('datetime').reset_index(drop=True)

v_1m = pd.merge_asof(
    v,
    log_v.reset_index().rename(columns={'index': 'log_idx'}),
    on='datetime',
    direction='nearest',
    tolerance=pd.Timedelta('900s')
)
print(v_1m[['datetime','filename', 'filter']])


#number failed
n_failed = v_1m['filename'].isna().sum()
print('failed:',n_failed)


# count how often each logfile row is used
counts = v_1m['filename'].value_counts()
print(counts.head())


#### I band

#check to see if I have all this stuff
log_i = logfile.loc[logfile['filter'] == 'I'].copy()
log_i = log_i.sort_values('datetime').reset_index(drop=True)

i = i.sort_values('datetime').reset_index(drop=True)

i_1m = pd.merge_asof(
    i,
    log_i.reset_index().rename(columns={'index': 'log_idx'}),
    on='datetime',
    direction='nearest',
    tolerance=pd.Timedelta('900s')
)
print(i_1m[['datetime','filename', 'filter']])


#number failed
n_failed = i_1m['filename'].isna().sum()
print('failed:',n_failed)


# count how often each logfile row is used
counts = i_1m['filename'].value_counts()
print(counts.head())


#### H band

#check to see if I have all this stuff
log_h = logfile.loc[logfile['filter'] == 'H'].copy()
log_h = log_h.sort_values('datetime').reset_index(drop=True)

h = h.sort_values('datetime').reset_index(drop=True)

h_1m = pd.merge_asof(
    h,
    log_h.reset_index().rename(columns={'index': 'log_idx'}),
    on='datetime',
    direction='nearest',
    tolerance=pd.Timedelta('50s')
)
print(h_1m[['datetime','filename', 'filter']])


#number failed
n_failed = h_1m['filename'].isna().sum()
print('failed:',n_failed)


# count how often each logfile row is used
counts = h_1m['filename'].value_counts()
print(counts.head())



#### combining
all_1m = pd.concat([v_1m, i_1m, h_1m], ignore_index=True)
all_1m = all_1m.sort_values('datetime').reset_index(drop=True)
print(all_1m[['datetime','filename', 'filter']])


#plotting for sanity

#list of patterns, should correspond
filt_pats=[['B', 'V', 'I', 'V', 'I'], ['B', 'V', 'I'],['B', 'V', 'I'], ['B', 'V', 'I'], ['V', 'I'], ['V', 'I'], ['I'], ['B'], ['V']]
exp_pats=[[360.0, 360.0, 360.0, 360.0, 360.0],[360.0, 360.0, 360.0], [240.0, 240.0, 240.0], [300.0, 240.0, 240.0], [360.0, 360.0], [660.0, 660.0], [240.0], [360.0], [360.0]]
colors=['sienna','g','violet','k','yellow', 'blue', 'cyan', 'darkorange','indigo']


filter_map2 = {'B': 0, 'V': 1, 'R': 2, 'I': 3, 'H': 4}
all_1m['filter num'] = all_1m['filter'].map(filter_map2)

#set up years array
years=np.arange(1999,2003,1)
#do some sorting

all_1m['in FULL pattern'] = 'No'
for pnum in range(len(filt_pats)):
    n=len(filt_pats[pnum])
    for i in range(len(all_1m) - n + 1):
        window = all_1m.loc[i:i+n-1, 'filter'].tolist()
        window2 = all_1m.loc[i:i+n-1, 'EXPTIME'].tolist()
        if window == filt_pats[pnum] and window2 == exp_pats[pnum] and set(all_1m.loc[i:i+n-1, 'in FULL pattern'])==set(['No']):
            #only do this if it's not actually already in another pattern
            all_1m.loc[i:i+n-1, 'in FULL pattern'] = pnum

#plot things
f, a = plt.subplots(2, 2, figsize=(11, 5),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=all_1m.loc[all_1m['datetime'].dt.year==year]

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['in FULL pattern']==pnum]
        axes[id].scatter(inpat['datetime'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    outpat=yrtable.loc[yrtable['in FULL pattern']=='No']
    axes[id].scatter(outpat['datetime'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3, 4])
    axes[id].set_yticklabels(['B', 'V', 'R', 'I', 'H']) 
    
    if year!=2000 and year!=2002:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

#save as a file
plt.suptitle('A0620-00')
plt.legend(ncol=2)

'''
for pnum in range(len(filt_pats)):
    df=all_1m.loc[all_1m['in FULL pattern']==pnum]
    print(df[['filename','DATE-OBS','TIME-OBS','JD','EXPTIME','CCDFLTID','in FULL pattern']])
'''
#plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/real_patterns_A0620-00.png', dpi=300)
plt.show()

#%%
#make array of filters; group roughly by night to use prefix span and identify nightly patterns
filters=[]
exptimes=[]
optonly=all_1m.loc[all_1m['filter']!='H']
dategrp=optonly.groupby(['DATE-OBS'])
for n, dg in dategrp:
    filters.append(list(dg['filter']))
    exptimes.append(list(dg['EXPTIME']))
ps = PrefixSpan(filters)
res=ps.topk(k=5, closed=True,filter=lambda patt, freq: len(patt) >= 3)
if len(res)>0:
    print(res)
else:
    print(ps.topk(k=5))