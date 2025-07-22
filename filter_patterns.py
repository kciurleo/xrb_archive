import pandas as pd
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt
import os
from prefixspan import PrefixSpan
import ast
import datetime
import matplotlib.dates as mdates

infile='/home/kmc249/test_data/all_optical_log.csv'

table=pd.read_csv(infile, low_memory=False)

#immediately drop anything that's not an xrb.
table=table.loc[table['xrb']==True]

#deal with some date stuff and some filter stuff
filter_map1 = {'B': 'blue', 'V': 'g', 'R': 'red', 'I': 'sienna'}
filter_map2 = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
table['filter num'] = table['CCDFLTID'].map(filter_map2)
table['filt color'] = table['CCDFLTID'].map(filter_map1)
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
    
    #make array of filters; group roughly by night to use prefix span and identify nightly patterns
    filters=[]
    exptimes=[]
    dategrp=g.groupby(['DATE-OBS'])
    for n, dg in dategrp:
        filters.append(list(dg['CCDFLTID']))
        exptimes.append(list(dg['EXPTIME']))
    ps = PrefixSpan(filters)
    res=ps.topk(k=5, closed=True,filter=lambda patt, freq: len(patt) >= 3)
    if len(res)>0:
        print(res)
    else:
        print(ps.topk(k=5))

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
            window = g.loc[i:i+n-1, 'CCDFLTID'].tolist()
            if window == pattern1:
                g.loc[i:i+n-1, 'in filt pattern'] = '1'
        m = len(pattern2)
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
        f, a = plt.subplots(11, 2, figsize=(20,16),layout='constrained')
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

        if not os.path.exists(f'/home/kmc249/test_data/internal_plots/{name[0]}/'):
            os.makedirs(f'/home/kmc249/test_data/internal_plots/{name[0]}/')

        plt.savefig(f'/home/kmc249/test_data/internal_plots/{name[0]}/filters_{name[0]}.png', dpi=250)
        plt.show()

        #find out if you're happy; rerun if not
        get_happy=input('Happy with this?')
        if 'y' in get_happy or 'Y' in get_happy or 'true' in get_happy or 'True' in get_happy:
            happy=True
    
    #now do the same thing but for exp times
    ps2 = PrefixSpan(exptimes)
    res2=ps2.topk(k=5, closed=True,filter=lambda patt, freq: len(patt) >= 3)
    if len(res2)>0:
        print(res2)
    else:
        print(ps2.topk(k=5))

    #while loop until you're happy with the results
    happy2=False
    while not happy2:
        #get the user to input 1 or 2 patterns
        print("Note: in this case, pattern should probably be the shortest thing, \nunless BVRI are different or something")
        print("Pattern of ")
        pattern3 = ast.literal_eval(input("Input a list of the first pattern: "))
        pattern4_input = input("Input a list of a secondary pattern, leave blank else: ")
        pattern4 = ast.literal_eval(pattern4_input) if pattern4_input.strip() else []

        #sliding windows to check if the patterns are matched
        g['in ex pattern'] = 'No'
        n = len(pattern3)
        for i in range(len(g) - n + 1):
            window = g.loc[i:i+n-1, 'EXPTIME'].tolist()
            if window == pattern3:
                g.loc[i:i+n-1, 'in ex pattern'] = '1'
        m = len(pattern4)
        if m>0:
            for i in range(len(g) - m + 1):
                window = g.loc[i:i+m-1, 'EXPTIME'].tolist()
                if window == pattern4:
                    #only do this if it's not actually already in pattern 1
                    if set(g.loc[i:i+m-1, 'in ex pattern'])==set(['No']): 
                        g.loc[i:i+m-1, 'in ex pattern'] = '2'

        #plot things
        f, a = plt.subplots(11, 2, figsize=(20,16), sharey=True, layout='constrained')
        axes=np.ravel(a, order='F')
        ylimlo=g['EXPTIME'].min()-5
        ylimhi=g['EXPTIME'].max()+5
        
        #for each year, find the associate obs
        for id, year in enumerate(years):
            yrtable=g.loc[g['DATE-OBS'].dt.year==year]
        
            inpat=yrtable.loc[yrtable['in ex pattern']=='1']
            inpat2=yrtable.loc[yrtable['in ex pattern']=='2']
            outpat=yrtable.loc[yrtable['in ex pattern']=='No']

            axes[id].scatter(inpat['nice time'], inpat['EXPTIME'], c=inpat['filt color'], s=2)
            if m>0:
                axes[id].scatter(inpat2['nice time'], inpat2['EXPTIME'], c=inpat2['filt color'], s=2)
            axes[id].scatter(outpat['nice time'], outpat['EXPTIME'], c=outpat['filt color'], s=2, marker='x')

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
        
        plt.legend()

        plt.savefig(f'/home/kmc249/test_data/internal_plots/{name[0]}/exptimes_{name[0]}.png', dpi=250)
        plt.show()

        #find out if you're happy; rerun if not
        get_happy2=input('Happy with this?')
        if 'y' in get_happy2 or 'Y' in get_happy2 or 'true' in get_happy2 or 'True' in get_happy2:
            happy2=True

    #then save this pretty detailed log and also maybe a txt file with info like obj name and nicknames,
    #ra and dec, patterns, any percentages
    g.to_csv(f'/home/kmc249/test_data/internal_plots/{name[0]}/objlog_{name[0]}.csv', index=False)
    

    txtfile=f'/home/kmc249/test_data/internal_plots/{name[0]}/info_{name[0]}.txt'
    with open(txtfile, 'w') as r:
        r.write(f'Name:\n{name[0]}\n')
        r.write(f'Nicknames:\n{list(set(g["OBJECT"]))}\n')
        r.write(f'Filter pattern(s):\n{pattern1}\t{pattern2}\n')
        r.write(f'Exptime pattern(s):\n{pattern3}\t{pattern4}')
