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
df=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_I_corrected_lc_4_27.csv', low_memory=False)
df=pd.read_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/unshifted/AqlX-1_V_corrected_lc_4_27.csv', low_memory=False)


#cycle through. look at images. if bad, hold onto. 
bad_guys=[]


#For I:
bad_guys_I=['/neta/xrb/AqlX-1/1m/opt/rccd/I_trimmed/trim_r0818_1998.035.fits', '/neta/xrb/AqlX-1/1m/opt/rccd/I_trimmed/trim_r0821_2298.044.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090325.0102.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090413.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090421.0107.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090426.0090.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090509.0080.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090519.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090521.0099.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090605.0055.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090608.0082.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd090831.0105.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160816.0053.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160821.0029.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160821.0032.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160821.0035.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160901.0074.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160901.0077.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160912.0037.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160914.0005.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd160928.0028.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd161003.0025.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd161010.0020.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170604.0145.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170608.0068.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170613.0144.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170614.0058.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170624.0046.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170712.0059.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170804.0062.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170822.0092.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170823.0112.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170902.0078.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170903.0062.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170918.0069.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd170928.0039.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171006.0032.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171011.0016.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171015.0017.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171020.0015.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171022.0014.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd171024.0014.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180330.0157.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180331.0166.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180418.0194.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180427.0165.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180428.0139.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180428.0142.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180501.0064.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180506.0165.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180508.0123.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180514.0175.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180527.0143.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180625.0071.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180626.0068.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180627.0067.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180628.0095.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180701.0095.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180705.0081.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180725.0098.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180725.0101.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180730.0060.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180803.0072.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180811.0110.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd180823.0118.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd181015.0026.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190421.0136.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190507.0169.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190515.0166.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190518.0115.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190518.0116.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190520.0135.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190521.0116.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190603.0195.fits', '/neta/xrb/AqlX-1/1.3m/opt/rccd/I_trimmed/trim_rccd190622.0162.fits']



skipto='2003-04-22 08:50:52.000'
skipover=False
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
df=df.loc[df['Rmag_corr']<=21]
df.to_csv('/neta/xrb/AqlX-1/product/just_subtracted_shifted/AqlX-1_R_corrected_lc.csv')
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
                    df.loc[mask1,  'Rmag_corr'], yerr=df.loc[mask1,  'e_Rmag'],
                    fmt='.', color='black', markersize=3, label='Corrected Rmag')
    ax_main.errorbar(df.loc[mask3, 'nice time'],
                     df.loc[mask3, 'Rmag_corr'],
                     yerr=df.loc[mask3, 'e_Rmag'],
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
