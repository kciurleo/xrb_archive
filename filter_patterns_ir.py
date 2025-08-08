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

infile='/home/kmc249/test_data/all_ir_log.csv'

table=pd.read_csv(infile, low_memory=False)

#immediately drop anything that's not an xrb.
table=table.loc[table['xrb']==True]

#deal with some date stuff and some filter stuff
filter_map1 = {'Y': 'blue', 'J': 'g', 'H': 'red', 'K': 'sienna'}
filter_map2 = {'Y': 0, 'J': 1, 'H': 2, 'K': 3}
filter_map3 = {'Y': '.', 'J': 'x', 'H': '_', 'K': '|'}
table['filter num'] = table['IRFLTID'].map(filter_map2)
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

skipnames=['4U 1254-690',
'4U 1538-52',
'4U 1543-47',
'4U 1543-624',
'4U 1608-52',
'4U 1630-47',
'4U 1636-536',
'4U 1658-298',
'4U 1702-429',
'4U 1735-444',
'4U 1755-338',
'4U 1822-371',
'4U 1907+097',
'A0620-00',
'Aql X-1',
'Cen X-3',
'Cen X-4',
'Cir X-1',
'GRO J0433+32',
'GRO J1655-40',
'GRS 1009-45',
'GRS 1716-249',
'GRS 1739-278',
'GRS 1915+105',
'GS 1354-645',
'GX 13+1',
'GX 1+4',
'GX 17+2',
'GX 301-2',
'GX 339-4',
'GX 349+2',
'GX 354-0',
'GX 5-1',
'GX 9+9']

dolist=['A0620-00']

for name, g in grp:
    print(name[0])
    if name[0] not in dolist:
        continue
    #do some sorting
    g.sort_values(by=['nice time'], ascending=True, inplace=True)
    g.reset_index(drop=True, inplace=True)
    
    #make array of filters; group roughly by night to use prefix span and identify nightly patterns
    filters=[]
    exptimes=[]
    dategrp=g.groupby(['DATE-OBS'])
    for n, dg in dategrp:
        filters.append(list(dg['IRFLTID']))
        exptimes.append(list(dg['EXPTIME']))
    ps = PrefixSpan(filters)
    res=ps.topk(k=5, closed=True,filter=lambda patt, freq: len(patt) >= 3)
    if len(res)<0:
        print(res)
    else:
        print(ps.topk(k=15))

    #while loop until you're happy with the results
    happy=False
    while not happy:
        #get the user to input 1 or 2 patterns
        print("Note: pattern 1 should typically be chosen to be the longer one \n(or at least not containing pattern 2)")
        pattern1 = ast.literal_eval(input("Input a list of the first pattern: "))
        pattern2_input = input("Input a list of a secondary pattern, leave blank else: ")
        pattern2 = ast.literal_eval(pattern2_input) if pattern2_input.strip() else []

        #sliding windows to check if the patterns are matched
        g['in filt pattern'] = 'No'
        n = len(pattern1)
        for i in range(len(g) - n + 1):
            window = g.loc[i:i+n-1, 'IRFLTID'].tolist()
            if window == pattern1:
                g.loc[i:i+n-1, 'in filt pattern'] = '1'
        m = len(pattern2)
        if m>0:
            for i in range(len(g) - m + 1):
                window = g.loc[i:i+m-1, 'IRFLTID'].tolist()
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
        f, a = plt.subplots(11, 2, figsize=(10,8),layout='constrained')
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
            axes[id].set_yticklabels(['Y', 'J', 'H', 'K']) 
            
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

        plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/ir_filters_{name[0]}.png', dpi=300)
        plt.show()

        #find out if you're happy; rerun if not
        get_happy=input('Happy with this?')
        if 'y' in get_happy or 'Y' in get_happy or 'true' in get_happy or 'True' in get_happy:
            happy=True
    
    #we're going to use the filter patterns to determine EXPTIME patterns, since they should be linked
    exptime_windows1 = []
    for i in range(len(g) - n + 1):
        if g.loc[i:i+n-1, 'IRFLTID'].tolist() == pattern1:
            exptime_windows1.append(tuple(g.loc[i:i+n-1, 'EXPTIME'].tolist()))

    #most common exp time pattern
    pattern3 = list(Counter(exptime_windows1).most_common(1)[0][0])
    print(f"Exposure time pattern for {pattern1}: {pattern3}")

    #if there's a second pattern
    exptime_windows2 = []
    if pattern2:
        for i in range(len(g) - m + 1):
            if g.loc[i:i+m-1, 'IRFLTID'].tolist() == pattern2:
                exptime_windows2.append(tuple(g.loc[i:i+m-1, 'EXPTIME'].tolist()))
        pattern4 = list(Counter(exptime_windows2).most_common(1)[0][0])
        print(f"Exposure time pattern for {pattern2}: {pattern4}")
    else:
        pattern4 = []

    #plot things
    f, a = plt.subplots(11, 2, figsize=(10,8), sharey=True, layout='constrained')
    axes=np.ravel(a, order='F')
    ylimlo=g['EXPTIME'].min()-5
    ylimhi=g['EXPTIME'].max()+5
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=g.loc[g['DATE-OBS'].dt.year==year]

        for filt in ['Y', 'J', 'H', 'K']:
            sub = yrtable.loc[yrtable['IRFLTID'] == filt]
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

    plt.savefig(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/ir_exptimes_{name[0]}.png', dpi=300)
    plt.show()

    #then save this pretty detailed log and also maybe a txt file with info like obj name and nicknames,
    #ra and dec, patterns, any percentages
    g.to_csv(f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/ir_objlog_{name[0]}.csv', index=False)
    

    txtfile=f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/ir_info_{name[0]}.txt'
    with open(txtfile, 'w') as r:
        r.write(f'Name:\n{name[0]}\n')
        r.write(f'Nicknames:\n{list(set(g["OBJECT"]))}\n')
        r.write(f'Filter pattern(s):\n{pattern1}\t{pattern2}\n')
        r.write(f'Exptime pattern(s):\n{pattern3}\t{pattern4}')
