import pandas as pd
import numpy as np
import os
import glob

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