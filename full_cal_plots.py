import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from astropy.time import Time
import os

# Load both datasets
optical = pd.read_csv('/home/kmc249/test_data/all_optical_log.csv', low_memory=False)
optical['source'] = 'optical'

ir = pd.read_csv('/home/kmc249/test_data/all_ir_log.csv', low_memory=False)
ir['source'] = 'IR'

# Apply your same cleaning to both datasets (put into a function for convenience)
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

optical = clean_table(optical)
ir = clean_table(ir)

# Combine datasets
table = pd.concat([optical, ir], ignore_index=True)

grp = table.groupby(['proper name'])
years = np.arange(1998, 2020, 1)

for name, g in grp:
    f, a = plt.subplots(11, 2, figsize=(10, 8), sharey=True)
    axes = np.ravel(a, order='F')

    for id, year in enumerate(years):
        yrtable = g.loc[g['DATE-OBS'].dt.year == year]

        for jd, row in yrtable.iterrows():
            start = row['DATE-OBS']
            end = start + pd.Timedelta(days=1)

            # color by location:
            if pd.isna(row['TIME-OBS']):
                c = 'red'
                z = 5
            elif row['Physical loc'] == 'CD':
                c = 'gold'
                z = 2
            elif row['Physical loc'] == 'Disk':
                c = 'lightgreen'
                z = 3

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

    plt.suptitle(name[0])
    plt.tight_layout()
    outdir = f'/home/kmc249/test_data/xrb_archive/internal_plots/{name[0]}/'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/combined_calendar_{name[0]}.png')
    #plt.show()
    plt.close(f)
    print(f'did {name[0]}')
