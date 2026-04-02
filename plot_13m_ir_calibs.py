#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 10:14:19 2026

@author: kmc249
"""
import glob
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import os
import numpy as np
import matplotlib.dates as mdates
import datetime
from matplotlib.lines import Line2D
rows = []
yrs = np.arange(2003, 2020, 1)

for yr in yrs:
    print('working on ', yr)
    filelist = glob.glob(f'/neta/xrb/IRCALIBS/{yr}/*')
    for file in filelist:
        try:
            hdr = fits.getheader(file)
        except:
            print('bad file: ',file)
        
        basefile = os.path.basename(file)
        
        #filetype
        f_lower = basefile.lower()
        if 'dark' in f_lower:
            ftype = 'dark'
        elif 'on' in f_lower:
            ftype = 'on'
        elif 'off' in f_lower:
            ftype = 'off'
        elif 'flat' in f_lower:
            ftype = 'flat'
        else:
            ftype = 'error'
        
        #time
        date_obs = hdr['DATE-OBS']
        time_obs = hdr['TIME-OBS']
        if date_obs and time_obs:
            datetime_obs = pd.to_datetime(f"{date_obs} {time_obs}", errors='coerce')
        elif date_obs:
            datetime_obs = pd.to_datetime(f"{date_obs}", errors='coerce')
        else:
            datetime_obs=pd.NaT                           
        
        #filter
        try:
            filter=hdr['IRFLTID']
        except:
            filter=hdr['FILTERID']
        
        rows.append({
            'filename': basefile,
            'year': yr,
            'type': ftype,
            'filter': filter,
            'datetime': datetime_obs
        })

df = pd.DataFrame(rows)

filter_map1 = {
    'Y': 'gold',
    'J': 'darkorange',
    'H': 'magenta',
    'K': 'purple'
}


ty_map2 = {
    'dark': 3,
    'on': 2,
    'off': 1,
    'flat': 0
}

df['ty num'] = df['type'].map(ty_map2)

f, a = f, a = plt.subplots(9, 2, figsize=(10, 8), sharey=True)
axes = np.ravel(a, order='F')
     
for id, year in enumerate(yrs):

    yrtable = df.loc[df['year'] == year]
    
    for filt, sub in yrtable.groupby('filter'):
        if filt in filter_map1:
            
            axes[id].scatter(
                sub['datetime'],
                sub['ty num'],
                c=filter_map1.get(filt, 'k'),
                s=4,
            )

    axes[id].set_ylabel(year)
    axes[id].set_ylim(-0.5, 3.5)
    axes[id].set_xlim(datetime.datetime(year, 1, 1), datetime.datetime(year, 12, 31))
    axes[id].set_yticks(range(4))
    axes[id].set_yticklabels(['flat','off','on','dark']) 

    if year not in [2008, 2019]:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
legend_elements = [
    Line2D([0], [0],
           marker='o',
           color='w',
           label=filt,
           markerfacecolor=color,
           markersize=6)
    for filt, color in filter_map1.items()
]
f.legend(
    handles=legend_elements,
    title="Filter",
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98)
) 
plt.suptitle('1m IR Calibs')
plt.tight_layout()

plt.savefig('/neta/xrb/META/ir_calib_calendar.png')
plt.show()

for current_filter in filter_map1.keys():

    df_filt = df[df['filter'] == current_filter]

    f, a = plt.subplots(9, 2, figsize=(10, 8), sharey=True)
    axes = np.ravel(a, order='F')

    for id, year in enumerate(yrs):

        yrtable = df_filt.loc[df_filt['year'] == year]

        axes[id].scatter(
            yrtable['datetime'],
            yrtable['ty num'],
            c=filter_map1[current_filter],
            s=4,
        )

        axes[id].set_ylabel(year)
        axes[id].set_ylim(-0.5, 3.5)
        axes[id].set_xlim(datetime.datetime(year, 1, 1),
                          datetime.datetime(year, 12, 31))
        axes[id].set_yticks(range(4))
        axes[id].set_yticklabels(['flat','off','on','dark'])

        if year not in [2008, 2019]:
            axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            axes[id].xaxis.set_major_locator(mdates.MonthLocator())
            axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.suptitle(f'1m IR Calibs – {current_filter} Filter')
    plt.tight_layout()

    plt.savefig(f'/neta/xrb/META/ir_calib_calendar_{current_filter}.png')
    plt.show()