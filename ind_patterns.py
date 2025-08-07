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

infile='/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/objlog_A0620-00.csv'

#list of patterns, should correspond
filt_pats=[['B', 'V', 'I', 'V', 'I'], ['B', 'V', 'I'],['B', 'V', 'I'], ['B', 'V', 'I'], ['V', 'I'], ['V', 'I'],['I','V'], ['I'], ['B'], ['V']]
exp_pats=[[360.0, 360.0, 360.0, 360.0, 360.0],[360.0, 360.0, 360.0], [240.0, 240.0, 240.0], [300.0, 240.0, 240.0], [360.0, 360.0], [660.0, 660.0],[240.0,240.0], [240.0], [360.0], [360.0]]
colors=['sienna','g','violet','k','yellow', 'gray', 'blue','cyan', 'darkorange','indigo']

table=pd.read_csv(infile)

table['DATE-OBS']=pd.to_datetime(table['DATE-OBS'], errors='coerce')

for id, row in table.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            table.at[id, 'datetimeobs']=Time(row['JD'], format='jd')
            table.at[id, 'nice time']=table.at[id, 'datetimeobs'].datetime
        except:
            table.at[id, 'datetimeobs']=np.nan
            table.at[id, 'nice time']=np.nan

table['nice time'] = pd.to_datetime(table['nice time'], errors='coerce')
#set up years array
years=np.arange(1998,2020,1)
#do some sorting
table.sort_values(by=['nice time'], ascending=True, inplace=True)
table.reset_index(drop=True, inplace=True)

table['in FULL pattern'] = 'No'
for pnum in range(len(filt_pats)):
    n=len(filt_pats[pnum])
    for i in range(len(table) - n + 1):
        window = table.loc[i:i+n-1, 'CCDFLTID'].tolist()
        window2 = table.loc[i:i+n-1, 'EXPTIME'].tolist()
        if window == filt_pats[pnum] and window2 == exp_pats[pnum] and set(table.loc[i:i+n-1, 'in FULL pattern'])==set(['No']):
            #only do this if it's not actually already in another pattern
            table.loc[i:i+n-1, 'in FULL pattern'] = pnum

#plot things
f, a = plt.subplots(11, 2, figsize=(22, 16),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=table.loc[table['DATE-OBS'].dt.year==year]

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['in FULL pattern']==pnum]
        axes[id].scatter(inpat['nice time'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    outpat=yrtable.loc[yrtable['in FULL pattern']=='No']
    axes[id].scatter(outpat['nice time'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3])
    axes[id].set_yticklabels(['B', 'V', 'R', 'I']) 
    
    if year!=2008 and year!=2019:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

#save as a file
plt.suptitle('A0620-00')
plt.legend(ncol=2)

if not os.path.exists(f'/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/'):
    os.makedirs(f'/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/')

plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/real_patterns_A0620-00.png', dpi=300)
plt.show()