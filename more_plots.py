import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

infile='/home/kmc249/test_data/massive_log_temp.csv'

table=pd.read_csv(infile, low_memory=False)
table['DATE-OBS'] = pd.to_datetime(table['DATE-OBS'], errors='coerce')
table['TILT1'] = pd.to_numeric(table['TILT1'], errors='coerce')
table['TILT2'] = pd.to_numeric(table['TILT2'], errors='coerce')
table['TILT3'] = pd.to_numeric(table['TILT3'], errors='coerce')
table=table.loc[table['OBJECT']=='V4641']

plt.figure(figsize=(12,8))
plt.scatter(table['DATE-OBS'], table['TILT1'], alpha=0.2, label='TILT1')
plt.scatter(table['DATE-OBS'], table['TILT2'], alpha=0.2, label='TILT2')
plt.scatter(table['DATE-OBS'], table['TILT3'], alpha=0.2, label='TILT3')
plt.yscale('log')
plt.xlabel('DATE-OBS')
plt.legend()
plt.title('all')
#plt.savefig('/home/kmc249/test_data/tiltstime.png')
plt.show()


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

