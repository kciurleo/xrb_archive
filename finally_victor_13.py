#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:13:46 2026

@author: kmc249
"""
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
#logfile=pd.read_csv('/neta/xrb/A0620-00/product/log_files_A0620-00.csv')
logfile=pd.read_csv('/Users/katieciurleo/Downloads/a0620/log_files_A0620-00.csv')
logfile['datetime']=pd.to_datetime(logfile['datetime'], format='mixed')
#get rid fo bad dates for now
logfile = logfile.dropna(subset=['datetime'])

basedir='/Users/katieciurleo'

#victor's 1m data
v = pd.read_csv(
    f'{basedir}/Downloads/1.3m_VJDs.dat',
    sep=r'\s+',
    skiprows=1,
    names=['JD', 'date', 'time']
)
v['datetime'] = pd.to_datetime(v['date'] + ' ' + v['time'])

i=pd.read_csv(
    f'{basedir}/Downloads/1.3m_IJDs.dat',
    sep=r'\s+',
    skiprows=1,
    names=['JD', 'date', 'time']
)
i['datetime'] = pd.to_datetime(i['date'] + ' ' + i['time'])

b=pd.read_csv(
    f'{basedir}/Downloads/1.3m_BJDs.dat',
    sep=r'\s+',
    skiprows=1,
    names=['JD', 'date', 'time']
)
b['datetime'] = pd.to_datetime(b['date'] + ' ' + b['time'])


h=pd.read_csv(
    f'{basedir}/Downloads/1.3m_HJDs.dat',
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
#all the 1m stuff has the same exptime. fill the blanks because we know thisis all v
v_1m["filter"] = v_1m["filter"].fillna("V")
v_1m["EXPTIME"] = v_1m["EXPTIME"].fillna(240.0)


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
i_1m["filter"] = i_1m["filter"].fillna("I")
i_1m["EXPTIME"] = i_1m["EXPTIME"].fillna(240.0)


#number failed
n_failed = i_1m['filename'].isna().sum()
print('failed:',n_failed)


# count how often each logfile row is used
counts = i_1m['filename'].value_counts()
print(counts.head())


##### B band

#check to see if I have all this stuff
log_b = logfile.loc[logfile['filter'] == 'B'].copy()
log_b = log_b.sort_values('datetime').reset_index(drop=True)

b = b.sort_values('datetime').reset_index(drop=True)

b_1m = pd.merge_asof(
    b,
    log_b.reset_index().rename(columns={'index': 'log_idx'}),
    on='datetime',
    direction='nearest',
    tolerance=pd.Timedelta('900s')
)
print(b_1m[['datetime','filename', 'filter']])
b_1m["filter"] = b_1m["filter"].fillna("B")
b_1m["EXPTIME"] = b_1m["EXPTIME"].fillna(240.0)


#number failed
n_failed = b_1m['filename'].isna().sum()
print('failed:',n_failed)


# count how often each logfile row is used
counts = b_1m['filename'].value_counts()
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
h_1m["filter"] = h_1m["filter"].fillna("H")
h_1m["EXPTIME"] = h_1m["EXPTIME"].fillna(90.02)


#number failed
n_failed = h_1m['filename'].isna().sum()
print('failed:',n_failed)


# count how often each logfile row is used
counts = h_1m['filename'].value_counts()
print(counts.head())



#### combining
all_1m = pd.concat([v_1m, i_1m, h_1m, b_1m], ignore_index=True)
all_1m = all_1m.sort_values('datetime').reset_index(drop=True)
print(all_1m[['datetime','filename', 'filter']])


#plotting for sanity

#list of patterns, should correspond
filt_pats=[['B', 'V', 'I', 'V', 'I'], ['B', 'V', 'I'],['B', 'V', 'I'], ['B', 'V', 'I'], ['V', 'I'], ['V', 'I'], ['I'], ['B'], ['V']]
exp_pats=[[360.0, 360.0, 360.0, 360.0, 360.0],[360.0, 360.0, 360.0], [240.0, 240.0, 240.0], [300.0, 240.0, 240.0], [360.0, 360.0], [660.0, 660.0], [240.0], [360.0], [360.0]]

#filt_pats=[['K','K','K','K','K', 'K', 'K', 'K','J','J','J','J','J','J', 'H', 'H','H','H','H','H'],['J','J', 'J', 'J', 'J','H', 'H', 'H', 'H','H','K','K', 'K', 'K', 'K','K'], ['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H'],['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H']]
#exp_pats=[[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04],[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04],[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04], [90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03,90.03, 90.03]]
colors=['blue','g','violet','k','yellow', 'blue', 'cyan', 'darkorange','indigo']


filter_map2 = {'B': 0, 'V': 1, 'R': 2, 'I': 3, 'H': 4}
all_1m['filter num'] = all_1m['filter'].map(filter_map2)

#set up years array
years=np.arange(2003,2020,1)
#do some sorting

all_1m['in FULL pattern'] = np.nan
for pnum in range(len(filt_pats)):
    n=len(filt_pats[pnum])
    for i in range(len(all_1m) - n + 1):
        window = all_1m.loc[i:i+n-1, 'filter'].tolist()
        window2 = all_1m.loc[i:i+n-1, 'EXPTIME'].tolist()
        if window == filt_pats[pnum] and window2 == exp_pats[pnum] and all_1m.loc[i:i+n-1, 'in FULL pattern'].isna().all():
            #only do this if it's not actually already in another pattern
            all_1m.loc[i:i+n-1, 'in FULL pattern'] = pnum

#plot things
f, a = plt.subplots(9, 2, figsize=(18, 10),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=all_1m.loc[all_1m['datetime'].dt.year==year]

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['in FULL pattern']==pnum]
        axes[id].scatter(inpat['datetime'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    outpat = yrtable.loc[yrtable['in FULL pattern'].isna()]
    axes[id].scatter(outpat['datetime'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3, 4])
    axes[id].set_yticklabels(['B', 'V', 'R', 'I', 'H']) 
    
    if year!=2010 and year!=2019:
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
all_1m[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern']].to_csv(f'{basedir}/Downloads/all_1.3m.csv', index=False)
#%%
#make array of filters; group roughly by night to use prefix span and identify nightly patterns
filters=[]
exptimes=[]
optonly=all_1m.loc[all_1m['filter']!='H']
ironly=all_1m.loc[all_1m['filter']=='H']
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
# %%
optonly[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern']].to_csv(f'{basedir}/Downloads/optonly.csv', index=False)
ironly[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern']].to_csv(f'{basedir}/Downloads/ironly.csv', index=False)


#%%

#just opt


#list of patterns, should correspond
filt_pats=[['V', 'I'], [ 'I', 'V']]
exp_pats=[[660.0, 660.0], [660.0, 660.0]]
colors=['blue','g','violet','k','yellow', 'blue', 'cyan', 'darkorange','indigo']

#set up years array
years=np.arange(1999,2003,1)

#do some sorting
optonly = optonly.sort_values('datetime').reset_index(drop=True)

max_gap = 800  # seconds (~10 min, tweak this)

seq_ids = [0]

for i in range(1, len(optonly)):
    dt = (optonly.loc[i, 'datetime'] - optonly.loc[i-1, 'datetime']).total_seconds()
    
    if dt > max_gap:
        seq_ids.append(seq_ids[-1] + 1)  # new sequence
    else:
        seq_ids.append(seq_ids[-1])

optonly['seq_id'] = seq_ids

optonly['in FULL pattern'] = np.nan

for seq_id, group in optonly.groupby('seq_id'):
    idx = group.index.tolist()
    
    for pnum in range(len(filt_pats)):
        n = len(filt_pats[pnum])
        
        for k in range(len(idx) - n + 1):
            inds = idx[k:k+n]
            
            window = optonly.loc[inds, 'filter'].tolist()
            window2 = optonly.loc[inds, 'EXPTIME'].tolist()
            
            if (window == filt_pats[pnum] and
                window2 == exp_pats[pnum] and
                optonly.loc[inds, 'in FULL pattern'].isna().all()):
                
                optonly.loc[inds, 'in FULL pattern'] = pnum

#plot things
f, a = plt.subplots(9, 2, figsize=(18, 5),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=optonly.loc[optonly['datetime'].dt.year==year]

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['in FULL pattern']==pnum]
        axes[id].scatter(inpat['datetime'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    outpat = yrtable.loc[yrtable['in FULL pattern'].isna()]
    axes[id].scatter(outpat['datetime'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3, 4])
    axes[id].set_yticklabels(['B', 'V', 'R', 'I', 'H']) 
    
    if year!=2010 and year!=2019:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

#save as a file
plt.suptitle('A0620-00')
plt.legend(ncol=2)
plt.show()

print(optonly[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern', 'seq_id']])

#%%
#grouping optical to ir
opt_groups = optonly.groupby("seq_id").agg(
    start=("datetime", "min"),
    end=("datetime", "max")
).reset_index()

pad = pd.Timedelta("13min")

opt_groups["start"] -= pad
opt_groups["end"]   += pad

def assign_seq(row, groups):
    match = groups[(groups.start <= row.datetime) & (row.datetime <= groups.end)]
    if len(match):
        return match.seq_id.values[0]
    return np.nan

ironly["seq_id"] = ironly.apply(assign_seq, axis=1, groups=opt_groups)

print(ironly[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern', 'seq_id']])


optonly[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern', 'seq_id']].to_csv(f'{basedir}/Downloads/optonly.csv', index=False)
ironly[['filename', 'datetime','filter', 'EXPTIME', 'in FULL pattern', 'seq_id']].to_csv(f'{basedir}/Downloads/ironly.csv', index=False)

#%%

offsets = []

for seq_id, grp in optonly.groupby("seq_id"):
    grp = grp.sort_values("datetime")

    t0 = grp["datetime"].iloc[0]
    dt = (grp["datetime"] - t0).dt.total_seconds()

    offsets.append(pd.DataFrame({
        "seq_id": seq_id,
        "offset_s": dt
    }))
    
offset_df = pd.concat(offsets, ignore_index=True)

for seq_id, grp in offset_df.groupby("seq_id"):
    plt.plot(grp["offset_s"], [seq_id]*len(grp), '.')
    
#%%
# stupid stuff
vicguys=all_1m.loc[all_1m['datetime'].dt.year>=2017]
vicopt=vicguys.loc[vicguys['filter']!='H']
vicopt = vicopt.reset_index(drop=True)
filt_pats=[['B','V', 'I'],['V','I', 'B'], [ 'I', 'V'], [ 'V', 'I']]
exp_pats=[[240.0,240.0,240.0],[240.0,240.0,240.0], [240.0,240.0], [240.0,240.0]]

years=np.arange(2017,2020,1)
vicopt['in FULL pattern'] = np.nan
for pnum in range(len(filt_pats)):
    n=len(filt_pats[pnum])
    for i in range(len(vicopt) - n + 1):
        window = vicopt.loc[i:i+n-1, 'filter'].tolist()
        window2 = vicopt.loc[i:i+n-1, 'EXPTIME'].tolist()
        if window == filt_pats[pnum] and window2 == exp_pats[pnum] and vicopt.loc[i:i+n-1, 'in FULL pattern'].isna().all():
            #only do this if it's not actually already in another pattern
            vicopt.loc[i:i+n-1, 'in FULL pattern'] = pnum

#plot things
f, a = plt.subplots(3, 1, figsize=(9, 5),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=vicopt.loc[vicopt['datetime'].dt.year==year]

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['in FULL pattern']==pnum]
        axes[id].scatter(inpat['datetime'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    outpat = yrtable.loc[yrtable['in FULL pattern'].isna()]
    axes[id].scatter(outpat['datetime'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3, 4])
    axes[id].set_yticklabels(['B', 'V', 'R', 'I', 'H']) 
    
    if year!=2019:
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
#different timing based thing for patterns
max_gap = 800  # or whatever makes sense

vicopt = vicopt.sort_values('datetime').reset_index(drop=True)

dt = vicopt['datetime'].diff().dt.total_seconds()

vicopt['seq_id'] = (dt > max_gap).cumsum()

# IMPORTANT: match longer patterns first
order = np.argsort([-len(p) for p in filt_pats])

vicopt['in FULL pattern'] = np.nan

for seq_id, group in vicopt.groupby('seq_id'):
    idx = group.index.to_list()
    
    for pidx in order:
        pat = filt_pats[pidx]
        exp = exp_pats[pidx]
        n = len(pat)
        
        for i in range(len(idx) - n + 1):
            inds = idx[i:i+n]
            
            if not vicopt.loc[inds, 'in FULL pattern'].isna().all():
                continue
            
            window_f = vicopt.loc[inds, 'filter'].tolist()
            window_e = vicopt.loc[inds, 'EXPTIME'].tolist()
            
            if window_f == pat and window_e == exp:
                vicopt.loc[inds, 'in FULL pattern'] = pidx
                
#plot things
f, a = plt.subplots(3, 1, figsize=(9, 5),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=vicopt.loc[vicopt['datetime'].dt.year==year]

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['in FULL pattern']==pnum]
        axes[id].scatter(inpat['datetime'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    outpat = yrtable.loc[yrtable['in FULL pattern'].isna()]
    axes[id].scatter(outpat['datetime'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3, 4])
    axes[id].set_yticklabels(['B', 'V', 'R', 'I', 'H']) 
    
    if year!=2019:
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

vicopt['offset_s'] = vicopt.groupby('seq_id')['datetime'].transform(
    lambda x: (x - x.iloc[0]).dt.total_seconds()
)

for ptype in [0.0,1.0,2.0,3.0]:
    subset = vicopt[(vicopt['in FULL pattern'] == ptype) & (vicopt['offset_s'] > 0)]
    
    plt.hist(subset['offset_s'], bins=40)
    plt.title(f"{ptype} timing")
    plt.xlabel("Offset (s)")
    plt.ylabel("Count")
    plt.show()