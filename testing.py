import pandas as pd
import numpy as np
import os
import glob
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt
'''
inputcsv=pd.read_csv('/home/kmc249/test_data/xrb_ccd_reduced_2014/LOG_2014.20140102thru20141231.csv')

have=inputcsv['filename']
os.chdir('/OLD-NET-DRIVE-COLLECTION/xrb/ccd/photometry')
other=glob.glob('*ccd*.fits')
print(f'have: {len(set(have))}')
print(f'other: {len(set(other))}')
print(f'need: {len(set(other)-set(have))}')
print(len(set(other) & set(have)))
print(have[0])
print(other[0])
'''
table=pd.read_csv('/home/kmc249/test_data/all_optical_log.csv', low_memory=False)
bad=0
good=0
badoffsets=[]
alloffsets=[]
for id, row in table.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            table.at[id, 'datetimeobs']=Time(f"{row['DATE-OBS']}T{row['TIME-OBS']}")
            table.at[id, 'jdobs']=Time(row['JD'], format='jd')
            diff=table.at[id, 'datetimeobs']-table.at[id, 'jdobs']
            if diff.to_value(u.s)>1 or diff.to_value(u.s)<-1:
                bad+=1
                badoffsets.append(diff.to_value(u.s))
                alloffsets.append(diff.to_value(u.s))
            else:
                alloffsets.append(diff.to_value(u.s))
                good+=0
        except:
            table.at[id, 'datetimeobs']=np.nan
            table.at[id, 'jdobs']=np.nan

print('ave bad offset:', np.mean(np.abs(np.array(badoffsets))))
print('median bad offset:', np.median(np.abs(np.array(badoffsets))))
print('std of bad offset:', np.std(np.abs(np.array(badoffsets))))
print('ave offset:', np.mean(np.abs(np.array(alloffsets))))
print('median offset:', np.median(np.abs(np.array(alloffsets))))
print('std of offset:', np.std(np.abs(np.array(alloffsets))))
plt.figure(figsize=(8,6))
plt.hist(np.array(badoffsets), bins=100)
plt.xscale('log')
plt.title('bad offsets')
plt.show()
plt.figure(figsize=(8,6))
plt.hist(np.array(alloffsets), bins=100)
plt.xscale('log')
plt.title('all offsets')
plt.show()

#this comment is to test if I can edit and use git on this linux machine too