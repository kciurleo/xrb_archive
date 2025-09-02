import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from astropy.time import Time
import os
import re
from lookup_name import *

# Load both datasets
optical = pd.read_csv('/home/kmc249/test_data/all_optical_log.csv', low_memory=False)
optical['source'] = 'optical'

ir = pd.read_csv('/home/kmc249/test_data/all_ir_log.csv', low_memory=False)
ir['source'] = 'IR'

tapes=pd.read_csv('/home/kmc249/test_data/inventory_tapes_08_19_25.csv')
tapes['Start Date'] = pd.to_datetime(tapes['Start Date'], errors='coerce')
tapes['End Date'] = pd.to_datetime(tapes['End Date'], errors='coerce')

#load by night tape things
night_tape_df=pd.read_csv("/home/kmc249/test_data/early_by_night.csv")
night_tape_df['Date']=pd.to_datetime(night_tape_df['Date'], errors='coerce')
night_tape_df=night_tape_df.set_index('Date')

#functions
def clean_table(table):
    table = table.loc[table['xrb'] == True]
    for id, row in table.iterrows():
        if pd.isna(row['DATE-OBS']):
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



optical = clean_table(optical)
ir = clean_table(ir)

# Combine datasets
table = pd.concat([optical, ir], ignore_index=True)

years = np.arange(1998, 2020, 1)

for name in xrb_list:
    g=table.loc[table['proper name']==name]
    ###addition: the new usb drives
    try:
        on_usb=pd.read_csv(f'/home/kmc249/usbdrive_logs/usbdrivereplog_{name}.csv', low_memory=False)
        on_usb['DATE-OBS'] = pd.to_datetime(on_usb['UTDate'], errors='coerce')
        
        g_filenames_normalized = g['filename'].dropna().apply(normalize_filename)
        on_usb['normalized'] = on_usb['Filename'].dropna().apply(normalize_filename)
        skip=False
    except:
        skip=True
    try:
        tapedates = list(night_tape_df.index[night_tape_df[name]])
    except:
        tapedates = []
        
    #temp tapes thing
    gtape=tapes.loc[tapes['Obj ID']==name]
    
    f, a = plt.subplots(11, 2, figsize=(10, 8), sharey=True)
    axes = np.ravel(a, order='F')
    
    to_upload=[]
     
    for id, year in enumerate(years):
        if len(tapedates)>0:
            filtered = [ts for ts in tapedates if ts.year == year]
            for item in filtered:
                start=item
                end = start + pd.Timedelta(days=1)
                c = 'saddlebrown'
                z = 1.45
                y_vals=[0.25, 0.75]
                for y_pos in y_vals:
                    axes[id].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)
                
        yrtable = g.loc[g['DATE-OBS'].dt.year == year]
        if len(gtape)>0:
            yrtable3 = gtape.loc[gtape['Start Date'].dt.year == year]
            for jd, row in yrtable3.iterrows():
                start=row['Start Date']
                end=row['End Date']
                if end==start:
                    end = start + pd.Timedelta(days=1)
                c = 'cyan'
                z = 1.4
               
                y_vals=[]
                # choose vertical position depending on source
                if 'Optical' in row['Optical/IR']:
                    y_vals.append(0.25)
                elif 'IR' in row['Optical/IR']:
                    y_vals.append(0.75)
                elif 'Both' in row['Optical/IR']:
                    y_vals.append(0.75)
                    y_vals.append(0.25)
                
                for y_pos in y_vals:
                    axes[id].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)
    
                    # handle span crossing year
                    if end.year == year + 1:
                        axes[id+1].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)

        if not skip:
            yrtable2 = on_usb.loc[on_usb['DATE-OBS'].dt.year == year]
            for jd, row in yrtable2.iterrows():
                start = row['DATE-OBS']
                end = start + pd.Timedelta(days=1)
    
                c = 'indigo'
                z=1.65
    
                # choose vertical position depending on source
                if 'ccd' in row['Filename']:
                    y_pos = 0.25
                else:
                    y_pos = 0.75
    
                axes[id].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)
    
                # handle span crossing year
                if end.year == year + 1:
                    axes[id+1].barh(y=y_pos, width=(end - start).days, left=start, height=0.2, color=c, zorder=z)

        for jd, row in yrtable.iterrows():
            start = row['DATE-OBS']
            end = start + pd.Timedelta(days=1)

            # color by location:
            if pd.isna(row['TIME-OBS']):
                c = 'red'
                z = 1.5
            elif row['Location'] == 'miniarchive':
                c='black'
                z=1.95
            elif row['Physical loc'] == 'CD':
                c = 'gold'
                z = 1.6
                to_upload.append(row.to_dict())
            elif row['Physical loc'] == 'Disk':
                c = 'lightgreen'
                z = 1.8
            elif row['Physical loc'] == 'Data scrape':
                c = 'violet'
                z = 1.7

            # choose vertical position depending on source
            if row['source'] == 'optical':
                y_pos = 0.25
            else:
                y_pos = 0.75

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

    plt.suptitle(name)
    plt.tight_layout()
    outdir = f'/home/kmc249/test_data/xrb_archive/internal_plots/{name}/'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/combined_calendar_{name}.png')
    #plt.show()
    plt.close(f)

    #make a list of the CDs I need to upload per obj
    upload_df = pd.DataFrame(to_upload)
    inv=pd.read_csv('/home/kmc249/test_data/inventory_bydisk_08_11_25.csv', low_memory=False)
    if not upload_df.empty:
        upload_df = upload_df.merge(inv, how='left', left_on='log name', right_on='Logname')
        for id, row in upload_df.iterrows():
            upload_df.at[id, 'spot'] = f"{row['Location_y']}:{row['Page']}:{row['Slot']}"
        
        st=set(upload_df['spot'])
        print(f"{name}: {st}")
    else:
        print(f'nothing for {name}')
    #print(f'did {name}')
