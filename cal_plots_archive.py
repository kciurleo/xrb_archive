import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from astropy.time import Time
import re
from lookup_name import *

#functions
def clean_table(table):
    for id, row in table.iterrows():
        if pd.isna(row['datetime']):
            try:
                tempyr = row['filename'].split('.')[0][-6:-4]
                yr = f'20{tempyr}' if int(tempyr) < 20 else f'19{tempyr}'
                mo = row['filename'].split('.')[0][-4:-2]
                day = row['filename'].split('.')[0][-2:]
                table.at[id, 'DATE-OBS'] = f'{yr}-{mo}-{day}'
            except:
                continue
    table['DATE-OBS'] = pd.to_datetime(table['DATE-OBS'], errors='coerce')
    return table

def normalize_filename(fn):
    fn = re.sub(r'^(r|bin)', '', fn)
    fn = re.sub(r'\.gz$', '', fn)
    return fn

#deal with some date stuff and some filter stuff
# optical + IR filters
filter_map1 = {
    'B': 'blue',
    'V': 'g',
    'R': 'red',
    'WIDE R': 'black',
    'I': 'sienna',
    'Y': 'gold',
    'J': 'darkorange',
    'H': 'magenta',
    'K': 'purple'
}

filter_map2 = {
    'B': 7,
    'V': 6,
    'R': 5,
    'R wide': 5,
    'I': 4,
    'Y': 3,
    'J': 2,
    'H': 1,
    'K': 0
}


years = np.arange(1998, 2020, 1)

for target in xrb_list:
    print(f'doing {target}')
    repo=f'/neta/xrb/{target}/'
    
    df=pd.read_csv(f'{repo}/product/log_files_{target}.csv', low_memory=False)
    
    g = clean_table(df)

    f, a = plt.subplots(11, 2, figsize=(10, 8), sharey=True)
    axes = np.ravel(a, order='F')
         
    for id, year in enumerate(years):
    
        yrtable = g.loc[g['DATE-OBS'].dt.year == year]
        
        for jd, row in yrtable.iterrows():
            start = row['DATE-OBS']
            end = start + pd.Timedelta(days=1)
    
            # hdr issues:
            if pd.isna(row['TIME-OBS']):
                c = 'red'
                z = 1.5
            else:
                c='black'
                z=1.45
    
            # choose vertical position depending on source
            if row['band'] == 'ir':
                y_pos = 0.75
            else:
                y_pos = 0.25
    
            axes[id].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)
    
            # handle span crossing year
            if end.year == year + 1:
                axes[id+1].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)
    
        axes[id].set_ylabel(year)
        axes[id].set_ylim(0, 1)
        axes[id].set_xlim(datetime.datetime(year, 1, 1), datetime.datetime(year, 12, 31))
        axes[id].set_yticks([]) 
        axes[id].set_yticklabels([])
    
        if year not in [2008, 2019]:
            axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            axes[id].xaxis.set_major_locator(mdates.MonthLocator())
            axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            if year==2019:
                axes[id].annotate('IR\n\nOpt', xy=(0.93, 0.92), xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top')
    
    plt.suptitle(target)
    plt.tight_layout()
    
    plt.savefig(f'{repo}/product/combined_calendar_{target}.png')
    #plt.show()
    plt.close(f)

    
    #now for filters plot:
    g['filter num'] = g['filter'].map(filter_map2)
    g['DATE-OBS']=pd.to_datetime(g['DATE-OBS'], errors='coerce')
    
    g['nice time'] = pd.NaT

    for id, row in g.iterrows():
        if pd.notna(row['DATE-OBS']) and pd.notna(row['TIME-OBS']):
            try:
                g.at[id, 'nice time'] = pd.to_datetime(
                    f"{row['DATE-OBS'].date()} {row['TIME-OBS']}",
                    errors='coerce'
                )
            except:
                g.at[id, 'nice time'] = pd.NaT

                
    g.sort_values(by=['nice time'], ascending=True, inplace=True)
    g.reset_index(drop=True, inplace=True)

    #plot things
    f, a = plt.subplots(11, 2, figsize=(20, 16),layout='constrained')
    axes=np.ravel(a, order='F')
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=g.loc[g['DATE-OBS'].dt.year==year]
        

        for filt, sub in yrtable.groupby('filter'):
            if filt in filter_map2:
                
                axes[id].scatter(
                    sub['nice time'],
                    sub['filter num'],
                    c=filter_map1.get(filt, 'k'),
                    s=4,
                )

        axes[id].set_ylabel(year)
        xlimlo = datetime.datetime(year, 1, 1)
        xlimhi = datetime.datetime(year, 12, 31)
        axes[id].set_xlim(xlimlo, xlimhi)
        axes[id].set_yticks(range(8))
        axes[id].set_yticklabels(['K','H','J','Y','I','R','V','B'], fontsize=8) 
        
        if year!=2008 and year!=2019:
            axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            axes[id].xaxis.set_major_locator(mdates.MonthLocator())
            axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    #save as a file
    plt.suptitle(target)


    plt.savefig(f'{repo}/product/filter_patterns_{target}.png', dpi=300)
    #plt.show()
    plt.close(f)

    #plot things
    f, a = plt.subplots(11, 2, figsize=(20,16), sharey=True, layout='constrained')
    axes=np.ravel(a, order='F')
    ir=g.loc[g['filter'].isin(['K','H','J','Y'])]
    if not ir.empty:
        ylimlo = ir['EXPTIME'].min() - 5
        ylimhi = ir['EXPTIME'].max() + 5
    else:
        ylimlo, ylimhi = 0, 1   # fallback safe limits
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=ir.loc[ir['DATE-OBS'].dt.year==year]

        for filt in ['K','H','J','Y']:
            sub = yrtable.loc[yrtable['filter'] == filt]
            if not sub.empty:
                axes[id].scatter(sub['nice time'], sub['EXPTIME'], c=filter_map1[filt],marker='|', s=6)

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
    plt.suptitle(target)

    plt.savefig(f'{repo}/product/exposure_patterns_ir_{target}.png', dpi=300)
    #plt.show()
    plt.close(f)
    
    f, a = plt.subplots(11, 2, figsize=(20,16), sharey=True, layout='constrained')
    axes=np.ravel(a, order='F')
    opt=g.loc[g['filter'].isin(['I','R','V','B','WIDE R'])]
    if not opt.empty:
        ylimlo = opt['EXPTIME'].min() - 5
        ylimhi = opt['EXPTIME'].max() + 5
    else:
        ylimlo, ylimhi = 0, 1   # fallback safe limits
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=opt.loc[opt['DATE-OBS'].dt.year==year]

        for filt in ['I','R','V','B','WIDE R']:
            sub = yrtable.loc[yrtable['filter'] == filt]
            if not sub.empty:
                axes[id].scatter(sub['nice time'], sub['EXPTIME'], c=filter_map1[filt],marker='|', s=6)

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
    plt.suptitle(target)

    plt.savefig(f'{repo}/product/exposure_patterns_optical_{target}.png', dpi=300)
    #plt.show()
    plt.close(f)
    
    

