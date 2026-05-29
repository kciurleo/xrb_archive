#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:42:17 2026

@author: kmc249
"""

import pandas as pd
from astropy.visualization import ZScaleInterval, ImageNormalize, SinhStretch
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from astropy.time import Time

#df=pd.read_csv('/neta/xrb/AqlX-1/product/AqlX-1_R_corrected_lc_with_outliers.csv', low_memory=False)
#df=pd.read_csv('/neta/xrb/AqlX-1/product/AqlX-1_R_corrected_lc.csv', low_memory=False)
#df=pd.read_csv('/neta/xrb/AqlX-1/product/combo_subtracted_shifted/AqlX-1_R_corrected_lc_4_27.csv', low_memory=False)
df=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv', low_memory=False)


#cycle through. look at images. if bad, hold onto. 
bad_times=['1998-07-18 08:12:35.300',
'1998-08-08 06:37:16.900',
'1998-08-16 05:53:37.100',
'1998-08-19 05:47:05.500',
'1998-08-25 05:55:44.600',
'1998-09-06 04:42:29.400',
'1998-09-27 00:47:33.400',
'1998-09-29 00:55:40.200',
'1998-10-10 02:20:38.300',
'1998-10-13 02:04:53.200',
'1998-10-23 01:05:36.800',
'1998-11-18 00:09:31.600',
'1998-11-19 00:05:36.500',
'1998-11-21 00:05:23.600',
'1999-03-18 09:53:20.700',
'1999-03-21 10:00:58.100',
'1999-05-12 08:09:04.800',
'1999-05-16 06:38:11.500',
'1999-06-04 03:51:07.500',
'1999-06-04 04:18:50.300',
'1999-07-04 05:40:49.000',
'1999-07-04 06:50:28.100',
'1999-07-04 07:22:40.400',
'1999-07-04 07:52:35.700',
'1999-07-04 08:22:26.600',
'1999-07-04 08:46:35.700',
'1999-07-29 00:30:18.600',
'1999-08-06 06:32:26.400']

bad_times_df=df.loc[df['nice time'].isin(bad_times)]

bad_guys=list(bad_times_df['filename'])
print(bad_guys)

#%%
skipto='2003-04-22 08:50:52.000'
skipover=True
for id, row in df.iterrows():
    if skipover:
        if row['nice time'] == skipto:
            skipover = False
        else:
            continue
    print(row['nice time'])
    filename=row['filename']
    try:
        im=fits.getdata(filename)
    except:
        print('NO FILE')
        continue
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(im)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(im, cmap='gray', origin='lower', norm=norm)
    plt.show()
    isbad=input('If bad type b')
    if 'b' in isbad:
        bad_guys.append(filename)
    if 'q' in isbad:
        break
    
print(bad_guys)

#%%

#bad_guys=['/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1029_3098.003.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000629.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000819.0024.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010624.0087.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020329.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020426.0048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020505.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020511.0037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050721.0079.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd090524.0062.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100330.0175.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110608.0137.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141026.0030.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd170818.0067.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001001.0018.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001015.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0717_1898.067.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0807_0898.046.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0815_1698.039.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0818_1998.033.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0824_2598.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0905_0698.029.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0926_2798.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0928_2998.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1009_1098.010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1012_1398.006.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1022_2398.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1117_1898.002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1118_1998.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1120_2198.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0317_1899.041.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0320_2199.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0511_1299.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0515_1699.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.053.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.057.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.061.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.066.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.069.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990728.0013.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990805.0017.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000406.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000423.0052.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0035.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000522.0032.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000819.0024.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001002.0016.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001122.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001123.0002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001124.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001125.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001127.0003.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001127.0005.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010428.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010603.0068.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010708.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010803.0021.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010902.0010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010912.0011.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010925.0009.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011112.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011115.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011116.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011120.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011121.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd030307.0213.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd030414.0175.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd030418.0180.fits']

bad_guys=['/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0717_1898.067.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1009_1098.010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1120_2198.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0515_1699.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.057.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.061.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.066.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.069.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990805.0017.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100423.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141031.0025.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd180527.0142.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd180628.0094.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990727.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020511.0037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050620.0131.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd080327.0142.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd080426.0131.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100330.0175.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110318.0139.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110608.0137.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd150730.0062.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0807_0898.046.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0815_1698.039.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0818_1998.033.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0824_2598.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0905_0698.029.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0926_2798.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0928_2998.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1012_1398.006.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1022_2398.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1029_3098.003.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1117_1898.002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1118_1998.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0317_1899.041.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0320_2199.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0511_1299.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.053.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990728.0013.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000406.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0035.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000522.0032.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000629.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000819.0024.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001002.0016.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001015.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001122.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001123.0002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001124.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001125.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010428.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010624.0087.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010708.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010902.0010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010912.0011.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011112.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011120.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011121.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020329.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020426.0048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020505.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd030307.0213.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050721.0079.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd090524.0062.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141026.0030.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd170818.0067.fits']
df['nice time'] = pd.to_datetime(df['nice time'], errors='coerce')
bad_df=df.loc[df['filename'].isin(bad_guys)]

df = df.sort_values('nice time')

# difference with previous and next points
diff_prev = (df['Rmag_corr'] - df['Rmag_corr'].shift(1)).abs()
diff_next = (df['Rmag_corr'] - df['Rmag_corr'].shift(-1)).abs()

# flag points that differ by >2 from BOTH neighbors
blue_mask = (diff_prev > 2) & (diff_next > 2)
outliers = df[blue_mask]
outliers = outliers[~outliers['filename'].isin(bad_df['filename'])]

ymin = min(df['Rmag_corr'].min(), df['Rmag'].min())
ymax = max(df['Rmag_corr'].max(), df['Rmag'].max())

fig, axes = plt.subplots(
    7, 1, figsize=(16, 24),
    gridspec_kw={'height_ratios': [1,1,1,1,1,1,1]}
)

# get start and end times
t_start = df['nice time'].min()
t_end   = df['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=8)  # 4 chunks = 5 edges

# now split df into 4 chunks by time range
time_chunks = [df[(df['nice time'] >= edges[i]) & (df['nice time'] < edges[i+1])]
               for i in range(7)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = df[(df['nice time'] >= edges[-2]) & (df['nice time'] <= edges[-1])]

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax)
    mask2 = (bad_df['nice time'] >= tmin) & (bad_df['nice time'] <= tmax)
        
    mask3 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax) & blue_mask
    
    # --- MAIN LIGHT CURVE ---
    '''
    ax_main.errorbar(df.loc[mask1, 'nice time'],
                    df.loc[mask1,  'Rmag_corr'], yerr=np.abs(df.loc[mask1,  'e_Rmag_corr']), 
                    fmt='.', color='black', markersize=3, label='Corrected')
    '''
    ax_main.errorbar(df.loc[mask1, 'nice time'],
                    df.loc[mask1,  'Rmag_corr'], yerr=df.loc[mask1,  'e_Rmag_corr'],
                    fmt='.', color='black', markersize=3, label='Corrected Rmag')
    ax_main.errorbar(df.loc[mask3, 'nice time'],
                     df.loc[mask3, 'Rmag_corr'],
                     yerr=df.loc[mask3, 'e_Rmag_corr'],
                     fmt='.', color='cyan', markersize=4, label='Outliers')
    ax_main.errorbar(bad_df.loc[mask2, 'nice time'],
                    bad_df.loc[mask2,  'Rmag_corr'], yerr=bad_df.loc[mask2,  'e_Rmag_corr'],
                    fmt='.', color='red', markersize=3, label='Excluded')
    
    ax_main.set_xlim(tmin, tmax)
    #ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    ax_main.set_ylim(ymax, ymin) 
    
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')

plt.tight_layout()
#plt.savefig('/home/kmc249/Downloads/df.png', dpi=300)
plt.show()


#%%
real_bads=[]
#just the outliers
bad_df = bad_df.sort_values('nice time')
outliers = outliers.sort_values('nice time')
for id, row in outliers.iterrows():

    print(row['nice time'])
    filename=row['filename']
    try:
        im=fits.getdata(filename)
    except:
        print('NO FILE')
        continue
    
    fig, axes = plt.subplots(
        7, 1, figsize=(16, 24),
        gridspec_kw={'height_ratios': [1,1,1,1,1,1,1]}
    )

    # get start and end times
    t_start = df['nice time'].min()
    t_end   = df['nice time'].max()

    # create 4 evenly spaced edges
    edges = pd.date_range(start=t_start, end=t_end, periods=8)  # 4 chunks = 5 edges

    # now split df into 4 chunks by time range
    time_chunks = [df[(df['nice time'] >= edges[i]) & (df['nice time'] < edges[i+1])]
                   for i in range(7)]

    # note: last chunk includes the last timestamp exactly
    time_chunks[-1] = df[(df['nice time'] >= edges[-2]) & (df['nice time'] <= edges[-1])]

    for i, chunk in enumerate(time_chunks):
        ax_main = axes[i]
            
        tmin = chunk['nice time'].min()
        tmax = chunk['nice time'].max()
        
        # masks
        mask1 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax)
        mask2 = (bad_df['nice time'] >= tmin) & (bad_df['nice time'] <= tmax)
            
        mask3 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax) & blue_mask
        
        # --- MAIN LIGHT CURVE ---
        '''
        ax_main.errorbar(df.loc[mask1, 'nice time'],
                        df.loc[mask1,  'Rmag_corr'], yerr=np.abs(df.loc[mask1,  'e_Rmag_corr']), 
                        fmt='.', color='black', markersize=3, label='Corrected')
        '''
        ax_main.errorbar(df.loc[mask1, 'nice time'],
                        df.loc[mask1,  'Rmag_corr'], yerr=df.loc[mask1,  'e_Rmag'],
                        fmt='.', color='black', markersize=3, label='Uncorrected')
        ax_main.errorbar(df.loc[mask3, 'nice time'],
                         df.loc[mask3, 'Rmag_corr'],
                         yerr=df.loc[mask3, 'e_Rmag'],
                         fmt='.', color='blue', markersize=4, label='Outliers')
        ax_main.errorbar(bad_df.loc[mask2, 'nice time'],
                        bad_df.loc[mask2,  'Rmag_corr'], yerr=bad_df.loc[mask2,  'e_Rmag'],
                        fmt='.', color='red', markersize=3, label='Uncorrected')
        ax_main.errorbar(row['nice time'],
                        row['Rmag_corr'], yerr=row['e_Rmag'],
                        fmt='.', color='lime', markersize=10, label='This guy')
        
        ax_main.set_xlim(tmin, tmax)
        #ax_main.set_ylim(20, 15.3)
        #ax_main.invert_yaxis()
        ax_main.set_ylim(ymax, ymin) 
        
        
        # formatting
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # legend once
    axes[0].legend(loc='upper right')

    plt.tight_layout()
    #plt.savefig('/home/kmc249/Downloads/df.png', dpi=300)
    plt.show()
    
    
    
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(im)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch())
    plt.figure(figsize=(10,12))
    plt.imshow(im, cmap='gray', origin='lower', norm=norm)
    plt.show()
    isbad=input('If bad type b')
    if 'b' in isbad:
        real_bads.append(filename)
    if 'q' in isbad:
        break
print(real_bads)

#%%
print(len(df))

bad_guys=['/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010705.0027.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0717_1898.067.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1009_1098.010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1120_2198.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0515_1699.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.057.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.061.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.066.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.069.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990805.0017.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100423.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141031.0025.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd180527.0142.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd180628.0094.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990727.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020511.0037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050620.0131.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd080327.0142.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd080426.0131.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd100330.0175.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110318.0139.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd110608.0137.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd150730.0062.fits','/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0807_0898.046.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0815_1698.039.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0818_1998.033.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0824_2598.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0905_0698.029.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0926_2798.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r0928_2998.004.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1012_1398.006.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1022_2398.008.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_r1029_3098.003.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1117_1898.002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r1118_1998.001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0317_1899.041.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0320_2199.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0511_1299.037.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0603_0499.048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.045.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_wideR/trim_r0703_0499.053.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd990728.0013.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000406.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000517.0035.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000522.0032.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000629.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd000819.0024.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001002.0016.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001015.0007.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001122.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001123.0002.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001124.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd001125.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010428.0084.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010624.0087.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010708.0018.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010902.0010.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd010912.0011.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011112.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011120.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd011121.0001.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020329.0031.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020426.0048.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1_ccd_R/trim_rccd020505.0042.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd030307.0213.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd050721.0079.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd090524.0062.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd141026.0030.fits', '/scratch/temp_CD_data/AqlX-1/trimmed_1.3_ccd_R/trim_rccd170818.0067.fits']
df['nice time'] = pd.to_datetime(df['nice time'], errors='coerce')
bad_df=df.loc[df['filename'].isin(bad_guys)]
print(len(bad_df))
final_df=df[~df['filename'].isin(bad_df['filename'])]
print(len(final_df))
df=final_df
df=df.loc[df['Rmag_corr']<=24]
#df.to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv')
#bad_df.to_csv('/neta/xrb/AqlX-1/temp/AqlX-1_R_badimgs.csv')

fig, axes = plt.subplots(
    7, 1, figsize=(16, 24),
    gridspec_kw={'height_ratios': [1,1,1,1,1,1,1]}
)

# get start and end times
t_start = df['nice time'].min()
t_end   = df['nice time'].max()

# create 4 evenly spaced edges
edges = pd.date_range(start=t_start, end=t_end, periods=8)  # 4 chunks = 5 edges

# now split df into 4 chunks by time range
time_chunks = [df[(df['nice time'] >= edges[i]) & (df['nice time'] < edges[i+1])]
               for i in range(7)]

# note: last chunk includes the last timestamp exactly
time_chunks[-1] = df[(df['nice time'] >= edges[-2]) & (df['nice time'] <= edges[-1])]

new_mask=(df['nice time'].dt.year==2001) & (df['nice time'].dt.month==7)

for i, chunk in enumerate(time_chunks):
    ax_main = axes[i]
        
    tmin = chunk['nice time'].min()
    tmax = chunk['nice time'].max()
    
    # masks
    mask1 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax)
    mask2 = (bad_df['nice time'] >= tmin) & (bad_df['nice time'] <= tmax)
        
    mask3 = (df['nice time'] >= tmin) & (df['nice time'] <= tmax) & new_mask
    
    # --- MAIN LIGHT CURVE ---
    '''
    ax_main.errorbar(df.loc[mask1, 'nice time'],
                    df.loc[mask1,  'Rmag_corr'], yerr=np.abs(df.loc[mask1,  'e_Rmag_corr']), 
                    fmt='.', color='black', markersize=3, label='Corrected')
    '''
    ax_main.errorbar(df.loc[mask1, 'nice time'],
                    df.loc[mask1,  'Rmag_corr'], yerr=df.loc[mask1,  'e_Rmag_corr'],
                    fmt='.', color='black', markersize=3, label='Corrected Rmag')
    ax_main.errorbar(df.loc[mask3, 'nice time'],
                     df.loc[mask3, 'Rmag_corr'],
                     yerr=df.loc[mask3, 'e_Rmag_corr'],
                     fmt='.', color='cyan', markersize=4, label='Outliers')
    
    ax_main.set_xlim(tmin, tmax)
    #ax_main.set_ylim(20, 15.3)
    #ax_main.invert_yaxis()
    ax_main.set_ylim(ymax, ymin) 
    
    
    # formatting
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# legend once
axes[0].legend(loc='upper right')

plt.tight_layout()
#plt.savefig('/home/kmc249/Downloads/df.png', dpi=300)
plt.show()
