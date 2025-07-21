import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from astropy.time import Time
import os

infile='/home/kmc249/test_data/all_optical_log.csv'

table=pd.read_csv(infile, low_memory=False)

#immediately drop anything that's not an xrb.
table=table.loc[table['xrb']==True]

#deal with some date stuff
for id, row in table.iterrows():
    if pd.isna(row['DATE-OBS']):
        try:
            tempyr=row['filename'].split('.')[0][-6:-4]
            if int(tempyr)<20:
                yr=f'20{tempyr}'
            else:
                yr=f'19{tempyr}'
            mo = row['filename'].split('.')[0][-4:-2]
            day =row['filename'].split('.')[0][-2:]
            table.at[id, 'DATE-OBS']=f'{yr}-{mo}-{day}'
        except:
            continue
table['DATE-OBS']=pd.to_datetime(table['DATE-OBS'], errors='coerce')


#table['datetimeobs']=Time(f"{table['DATE-OBS']}T{table['TIME-OBS']}")
grp=table.groupby(['proper name'])

#set up years array
years=np.arange(1998,2020,1)

#for each object: make a plot
for name, g in grp:

    f, a = plt.subplots(11, 2, figsize=(10,8))
    axes=np.ravel(a, order='F')
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=g.loc[g['DATE-OBS'].dt.year==year]
        for jd, row in yrtable.iterrows():
            start = row['DATE-OBS']
            end=start+pd.Timedelta(days=1)
            
            #color by location:
            if pd.isna(row['TIME-OBS']):
                #failed to read on CD/DVD
                c='red'
                z=5
            elif row['Physical loc']=='CD':
                #on a CD/DVD
                c='gold'
                z=2
            elif row['Physical loc']=='Disk':
                #on disk already
                c='lightgreen'
                z=3

            axes[id].barh(y=0, width=(end - start).days, left=start, height=0.4, color=c, zorder=z)
            #account for span crossing year
            if end.year==year+1:
                axes[id+1].barh(y=0, width=(end - start).days, left=start, height=0.4, color=c, zorder=z)


        axes[id].set_ylabel(year)
        xlimlo = datetime.datetime(year, 1, 1)
        xlimhi = datetime.datetime(year, 12, 31)
        axes[id].set_xlim(xlimlo, xlimhi)
        axes[id].set_yticklabels([]) 
        axes[id].set_yticks([])
        
        if year!=2008 and year!=2019:
            axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            axes[id].xaxis.set_major_locator(mdates.MonthLocator())
            axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    #save as a file
    plt.suptitle(name[0])
    plt.tight_layout()
    if not os.path.exists(f'/home/kmc249/test_data/internal_plots/{name[0]}/'):
        os.makedirs(f'/home/kmc249/test_data/internal_plots/{name[0]}/')

    plt.savefig(f'/home/kmc249/test_data/internal_plots/{name[0]}/calendar_{name[0]}.png')
    plt.show()