import pandas as pd
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt
import os

infile='/home/kmc249/test_data/all_optical_log.csv'

table=pd.read_csv(infile, low_memory=False)

#immediately drop anything that's not an xrb.
table=table.loc[table['xrb']==True]

#deal with some date stuff and some filter stuff
filter_map = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
table['filter_num'] = table['CCDFLTID'].map(filter_map)

for id, row in table.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            table.at[id, 'datetimeobs']=Time(f"{row['DATE-OBS']}T{row['TIME-OBS']}")
            table.at[id, 'nice time']=table.at[id, 'datetimeobs'].datetime
        except:
            table.at[id, 'datetimeobs']=np.nan
            table.at[id, 'nice time']=np.nan

grp=table.groupby(['proper name'])

for name, g in grp:
    #do some sorting
    g.sort_values(by=['nice time'], ascending=True, inplace=True)
    '''

    plt.figure(figsize=(16,4))
    plt.scatter(g['nice time'], g['filter_num'])
    plt.yticks([0, 1, 2, 3], ['B', 'V', 'R', 'I'])
    plt.xlabel('Datetime')
    plt.ylabel('Filter')
    plt.title(f'{name[0]}')
    plt.tight_layout()
    if not os.path.exists(f'/home/kmc249/test_data/internal_plots/{name[0]}/'):
        os.makedirs(f'/home/kmc249/test_data/internal_plots/{name[0]}/')

    #plt.savefig(f'/home/kmc249/test_data/internal_plots/{name[0]}/filters_{name[0]}.png')
    plt.show()
    '''
    print(name[0])
    print(list(g['CCDFLTID']))