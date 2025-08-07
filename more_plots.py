import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
import datetime
import matplotlib.dates as mdates

infile='/home/kmc249/test_data/massive_log_temp.csv'
infile='/home/kmc249/test_data/all_ir_log.csv'

table=pd.read_csv(infile, low_memory=False)
table=table.loc[table['proper name']=='A0620-00']
table['DATE-OBS']=pd.to_datetime(table['DATE-OBS'], errors='coerce')

for id, row in table.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            table.at[id, 'datetimeobs']=Time(row['JD'], format='jd')
            table.at[id, 'nice time']=table.at[id, 'datetimeobs'].datetime
        except:
            table.at[id, 'datetimeobs']=np.nan
            table.at[id, 'nice time']=np.nan
table.sort_values(by=['nice time'], ascending=True, inplace=True)
table['TILT1'] = pd.to_numeric(table['TILT1'], errors='coerce')
table['TILT2'] = pd.to_numeric(table['TILT2'], errors='coerce')
table['TILT3'] = pd.to_numeric(table['TILT3'], errors='coerce')
print(table)
years=np.arange(1998,2020,1)
f, a = plt.subplots(11, 2, figsize=(10,8),layout='constrained')
axes=np.ravel(a, order='F')
for id, year in enumerate(years):
    yrtable=table.loc[table['DATE-OBS'].dt.year==year]

    axes[id].scatter(yrtable['nice time'], yrtable['TILT1'], alpha=0.2, label='TILT1', s=2)
    axes[id].scatter(yrtable['nice time'], yrtable['TILT2'], alpha=0.2, label='TILT2', s=2)
    axes[id].scatter(yrtable['nice time'], yrtable['TILT3'], alpha=0.2, label='TILT3', s=2)

    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    
    if year!=2008 and year!=2019:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

#save as a file
plt.suptitle('A0620-00')
plt.legend()
#plt.savefig('/home/kmc249/test_data/tiltstime.png')
plt.show()

'''
#filters
#table=table.loc[table['DATE-OBS']< pd.Timestamp('2004-05-10')]
filter_map = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
table['filter_num'] = table['CCDFLTID'].map(filter_map)
plt.figure(figsize=(12,8))
plt.scatter(table['DATE-OBS'], table['filter_num'])
plt.yticks([0, 1, 2, 3], ['B', 'V', 'R', 'I'])
plt.xlabel('DATE-OBS')
plt.ylabel('Filter')
plt.title('V4641 (e.g.)')
#plt.savefig('/home/kmc249/test_data/V4641_filterstime.png')
plt.show()

#exposure times
#table=table.loc[table['DATE-OBS']< pd.Timestamp('2004-05-10')]
plt.figure(figsize=(12,8))
plt.scatter(table['DATE-OBS'], table['EXPTIME'])
plt.xlabel('DATE-OBS')
plt.ylabel('Exp. time')
plt.title('V4641 (e.g.)')
#plt.savefig('/home/kmc249/test_data/V4641_exptime.png')
plt.show()

'''