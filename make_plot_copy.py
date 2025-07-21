import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime

infile='/home/kmc249/test_data/massive_log_temp.csv'
repo='optical_CD_header_logs'

#reload jic
table=pd.read_csv(infile, low_memory=False)
table['DATE-OBS']=pd.to_datetime(table['DATE-OBS'], errors='coerce')
years=np.arange(1998,2020,1)
#total plot
f, a = plt.subplots(11, 2, figsize=(10,8))
axes=np.ravel(a, order='F')

#for each year, find the associate obs
for id, year in enumerate(years):
    yrtable=table.loc[table['DATE-OBS'].dt.year==year]
    for jd, row in yrtable.iterrows():
        start = row['DATE-OBS']
        end = row['DATE-OBS']
        #account for single day spans
        if end==start:
            end=end+pd.Timedelta(days=1)
        axes[id].barh(y=0, width=(end - start).days, left=start, height=0.4, color='lightgreen')
        #account for span crossing year
        if end.year==year+1:
            axes[id+1].barh(y=0, width=(end - start).days, left=start, height=0.4, color='lightgreen')


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
plt.suptitle('Any')
plt.tight_layout()
plt.savefig(f'/home/kmc249/test_data/{repo}/any.png')
plt.show()
'''
#reload jic
table=pd.read_csv(infile)
table['DATE-OBS']=pd.to_datetime(table['DATE-OBS'], errors='coerce')
#sort
grp=table.groupby(['OBJECT'])

#set up years array
years=np.arange(1998,2020,1)
print(table)
#for each object, make a plot
for name, g in grp:

    f, a = plt.subplots(11, 2, figsize=(10,8))
    axes=np.ravel(a, order='F')
    
    #for each year, find the associate obs
    for id, year in enumerate(years):
        yrtable=g.loc[g['DATE-OBS'].dt.year==year]
        for jd, row in yrtable.iterrows():
            start = row['DATE-OBS']
            end = row['DATE-OBS']
            #account for single day spans
            if end==start:
                end=end+pd.Timedelta(days=1)
            axes[id].barh(y=0, width=(end - start).days, left=start, height=0.4, color='lightgreen')
            #account for span crossing year
            if end.year==year+1:
                axes[id+1].barh(y=0, width=(end - start).days, left=start, height=0.4, color='lightgreen')


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


    
    # Create two subplots and unpack the output array immediately
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(f'/home/kmc249/test_data/{repo}/{name}.png')
    plt.show()

'''