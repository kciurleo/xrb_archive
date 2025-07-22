#finds the oldest and youngest fits file dates, assuming all files are the same obj

import numpy as np
import sys
import os
from astropy.io import fits
import pandas as pd
import warnings

#ignore astropy warnings please
warnings.filterwarnings('ignore', category=UserWarning, append=True)

#take current repository
repo=os.getcwd()

#get all the files specifically rccd right now 
files = [f for f in os.listdir(repo) if f.endswith('fits')]

files = []
for dirpath, dirnames, filenames in os.walk(repo):
    for f in filenames:
        if f.endswith('.fits') and f.startswith('rccd'):
            full_path = os.path.join(dirpath, f)
            files.append(full_path)

#initialize data frame
keywords=['OBJECT','RA','DEC','DATE-OBS','TIME-OBS','JD','EXPTIME','SECZ','CCDFLTID','IRFLTID','TILT1','TILT2','TILT3']
df=pd.DataFrame(columns=['filename'] + keywords)
print('Reading headers...')

#read from the header into df
for id, file in enumerate(files):
    df.at[id, 'filename']=file

    #if reading the whole file is an issue and takes longer than 15 seconds
    try:
        #hdr=fits.getheader(repo+'/'+file)
        hdr=fits.getheader(file)
    except:
        print(f'Could not read header for {file}')
        for keyword in keywords:
                df.at[id, keyword]=np.nan
        continue
    
    #if a specific keyword doesn't exist
    for keyword in keywords:
        try:
            df.at[id, keyword]=hdr[keyword]
        except:
            print(f'Failed to find {keyword} for {file}')
            df.at[id, keyword]=np.nan

#specifically fix the nan to naT issue
df['DATE-OBS'] = pd.to_datetime(df['DATE-OBS'], errors='coerce')

#stop if more than one obj in set
multiple=''
if len(set(df['OBJECT']))>1:
    print(f"Multiple objects in set: {set(df['OBJECT'])}")
    multiple=input("Are there truly multiple objs? (y/n)")
    if 'y' not in multiple and 'Y' not in multiple:
        name = input("Enter obj's real name if not: ")
        print(f'Obj: {name}')
        #for now, just print all dates and oldest and youngest dates
        print(f"Oldest: {df['DATE-OBS'].min()}")
        print(f"Youngest date: {df['DATE-OBS'].max()}")
else:
    name = list(set(df['OBJECT']))[0]
    print(f'Obj: {name}')
    #for now, just print all dates and oldest and youngest dates
    print(f"Oldest: {df['DATE-OBS'].min()}")
    print(f"Youngest date: {df['DATE-OBS'].max()}")

df=df.sort_values(by='DATE-OBS')
for id, row in df.iterrows():
    df.at[id,'filename']=df.at[id,'filename'].split('/')[-1]
print(df)

#naming log files in a smart way to always include the name of the obj
if 'y' in multiple or 'Y' in multiple:
    names= " ".join(list(set(df['OBJECT'])))
    string=f"LOG_MULTI_{names}.{str(df['DATE-OBS'].min()).split(' ')[0].replace('-', '')}thru{str(df['DATE-OBS'].max()).split(' ')[0].replace('-', '')}"
else:
    string=f"LOG_{repo.split('/')[-1]}.{str(df['DATE-OBS'].min()).split(' ')[0].replace('-', '')}thru{str(df['DATE-OBS'].max()).split(' ')[0].replace('-', '')}"

#save log
df.to_csv(f'/home/kmc249/test_data/{string}.csv', index=False)
