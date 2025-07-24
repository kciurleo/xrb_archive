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

infile='/home/kmc249/test_data/all_optical_log.csv'

table=pd.read_csv(infile, low_memory=False)

#immediately drop anything that's not an xrb.
table=table.loc[table['xrb']==True]

#deal with some date stuff and some filter stuff
filter_map1 = {'B': 'blue', 'V': 'g', 'R': 'red', 'I': 'sienna'}
filter_map2 = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
filter_map3 = {'B': '.', 'V': 'x', 'R': '_', 'I': '|'}
table['filter num'] = table['CCDFLTID'].map(filter_map2)
table['DATE-OBS']=pd.to_datetime(table['DATE-OBS'], errors='coerce')

for id, row in table.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            table.at[id, 'datetimeobs']=Time(row['JD'], format='jd')
            table.at[id, 'nice time']=table.at[id, 'datetimeobs'].datetime
        except:
            table.at[id, 'datetimeobs']=np.nan
            table.at[id, 'nice time']=np.nan

grp=table.groupby(['proper name'])

#set up years array
years=np.arange(1998,2020,1)

for name, g in grp:
    #do some sorting
    g.sort_values(by=['nice time'], ascending=True, inplace=True)
    g.reset_index(drop=True, inplace=True)
    
    txt=f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/info_{name[0]}.txt'

    #read txt file and assign variables accordingly
    f=open(txt)
    lines=f.readlines()
    filt_pat=[]
    exp_pat=[]
    for i in lines[5].split('\t'):
        filt_pat.append(ast.literal_eval(i))
    for i in lines[7].split('\t'):
        exp_pat.append(ast.literal_eval(i))
    m=len(filt_pat[1])
    n=len(filt_pat[0])

    pattern1=filt_pat[0]
    pattern2=filt_pat[1]
    pattern3=filt_pat[0]
    pattern4=filt_pat[1]

    g['in filt pattern'] = 'No'
    for i in range(len(g) - n + 1):
        window = g.loc[i:i+n-1, 'CCDFLTID'].tolist()
        if window == pattern1:
            g.loc[i:i+n-1, 'in filt pattern'] = '1'
    if m>0:
        for i in range(len(g) - m + 1):
            window = g.loc[i:i+m-1, 'CCDFLTID'].tolist()
            if window == pattern2:
                #only do this if it's not actually already in pattern 1
                if set(g.loc[i:i+m-1, 'in filt pattern'])==set(['No']): 
                    g.loc[i:i+m-1, 'in filt pattern'] = '2'


    pat1str=''
    for f in pattern1:
        pat1str+=f'{f} '
    pat2str=''
    for f in pattern2:
        pat2str+=f'{f} '

    #plot things
    f, a = plt.subplots(11, 2, figsize=(22, 16),layout='constrained')
    axes=np.ravel(a, order='F')
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=g.loc[g['DATE-OBS'].dt.year==year]
    
        inpat=yrtable.loc[yrtable['in filt pattern']=='1']
        inpat2=yrtable.loc[yrtable['in filt pattern']=='2']
        outpat=yrtable.loc[yrtable['in filt pattern']=='No']

        axes[id].scatter(inpat['nice time'], inpat['filter num'], c='g', s=2, label=f'Pattern: {pat1str}')
        if m>0:
            axes[id].scatter(inpat2['nice time'], inpat2['filter num'], c='gold', s=2, label=f'Pattern: {pat2str}')
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
    plt.suptitle(name[0])
    plt.legend()

    if not os.path.exists(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/'):
        os.makedirs(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/')

    plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/filters_{name[0]}_big.png', dpi=300)
    plt.show()

    #plot things
    f, a = plt.subplots(11, 2, figsize=(22,16), sharey=True, layout='constrained')
    axes=np.ravel(a, order='F')
    ylimlo=g['EXPTIME'].min()-5
    ylimhi=g['EXPTIME'].max()+5
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=g.loc[g['DATE-OBS'].dt.year==year]

        for filt in ['B', 'V', 'R', 'I']:
            sub = yrtable.loc[yrtable['CCDFLTID'] == filt]
            if not sub.empty:
                axes[id].scatter(sub['nice time'], sub['EXPTIME'], c=filter_map1[filt], marker=filter_map3[filt], s=2)

        axes[id].set_ylabel(year)
        xlimlo = datetime.datetime(year, 1, 1)
        xlimhi = datetime.datetime(year, 12, 31)
        axes[id].set_xlim(xlimlo, xlimhi)
        axes[id].set_ylim(ylimlo, ylimhi)
        
        if year!=2008 and year!=2019:
            axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            axes[id].xaxis.set_major_locator(mdates.MonthLocator())
            axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    #save as a file
    plt.suptitle(name[0])

    plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/exptimes_{name[0]}_big.png', dpi=300)
    plt.show()