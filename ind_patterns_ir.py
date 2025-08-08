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

infile='/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/ir_objlog_A0620-00.csv'

#list of patterns, should correspond
filt_pats=[['K','K','K','K','K', 'K', 'K', 'K','J','J','J','J','J','J', 'H', 'H','H','H','H','H'],['J','J', 'J', 'J', 'J','H', 'H', 'H', 'H','H','K','K', 'K', 'K', 'K','K'], ['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H'],['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H']]
exp_pats=[[30.04, 30.04, 30.04, 30.04,30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04,30.04, 30.04,30.04, 30.04, 30.04, 30.04,30.04, 30.04],[30.04, 30.04, 30.04, 30.04,30.04, 30.04, 30.04, 30.04,30.04, 30.04,30.04, 30.04, 30.04, 30.04,30.04, 30.04],[30.04, 30.04, 30.04, 30.04,30.04, 30.04, 30.04, 30.04,30.04, 30.04, 30.04, 30.04,30.04, 30.04], [90.03, 90.03, 90.03, 90.03,90.03, 90.03, 90.03, 90.03,90.03, 90.03, 90.03, 90.03,90.03, 90.03]]
exp_pats = [[int(x) for x in sub] for sub in exp_pats]
colors=['blue', 'violet', 'g','gold']
#colors=['sienna','g','violet','k','yellow', 'blue', 'cyan', 'darkorange','indigo']

table=pd.read_csv(infile)
table=table.loc[table['OBJECT'].isin(['A0620','A0620-00'])]
table = table[table['filename'].str.startswith(('ir', 'binir'))]

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
        window = table.loc[i:i+n-1, 'IRFLTID'].tolist()
        window2raw = table.loc[i:i+n-1, 'EXPTIME'].tolist()
        window2 = [int(w) if not np.isnan(w) else 0 for w in window2raw]
        if window == filt_pats[pnum] and window2 == exp_pats[pnum] and set(table.loc[i:i+n-1, 'in FULL pattern'])==set(['No']):
            #only do this if it's not actually already in another pattern
            table.loc[i:i+n-1, 'in FULL pattern'] = pnum

#plot things
f, a = plt.subplots(11, 2, figsize=(11, 8),layout='constrained')
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
    axes[id].set_yticklabels(['Y', 'J', 'H', 'K']) 
    
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

for pnum in range(len(filt_pats)):
    df=table.loc[table['in FULL pattern']==pnum]
    print(df[['filename','DATE-OBS','TIME-OBS','JD','EXPTIME','IRFLTID','in FULL pattern']].head(30))

plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/A0620-00/real_ir_patterns_A0620-00.png', dpi=300)
plt.show()
table.to_csv('/home/kmc249/test_data/xrb_archive/A0620_test_ir.csv')