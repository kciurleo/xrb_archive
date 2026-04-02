#finds the oldest and youngest fits file dates, assuming all files are the same obj

import numpy as np
import sys
import os
from astropy.io import fits
import pandas as pd
import warnings
from lookup_name import *

#ignore astropy warnings please
warnings.filterwarnings('ignore', category=UserWarning, append=True)

#take current repository
#repo=os.getcwd()
for target in xrb_list:
    repo=f'/neta/xrb/{target}/'
    
    print(target)
    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        #if 'temp_CD_data' in dirpath:
            #continue
        for f in filenames:
            if f.endswith('.fits') or f.endswith('.fits.gz') :# and (f.startswith('rccd') or f.startswith('ccd')):
                full_path = os.path.join(dirpath, f)
                # Only include if '1m' or '1.3m' is in the full path, so we only get the raw/base files or whatever
                if 'proc' in full_path:
                    continue
                if '1m' in full_path or '1.3m' in full_path:
                    files.append(full_path)
    
    print("Number of files:", len(files))
    #initialize data frame
    keywords=['OBJECT','RA','DEC','DATE-OBS','TIME-OBS','EXPTIME','SECZ','CCDFLTID','IRFLTID','FILTERID','TELESCOP','OWNER']
    df=pd.DataFrame(columns=['filename', 'band', 'filter'] + keywords)
    print('Reading headers...')
    headerissues=[]
    #read from the header into df
    for id, file in enumerate(files):
        
        ###THE LOGIC BELOW FOR BAND AND FILTER IS DUMB
        basefilename = os.path.basename(file)
        df.at[id, 'filename'] = basefilename
        try:
            print(f'Reading {file}')
            hdr=fits.getheader(file)
        except:
            print(f'Could not read header for {file}')
            for keyword in keywords:
                    df.at[id, keyword]=np.nan
                    headerissues.append(file)
            continue
        
        #if a specific keyword doesn't exist
        for keyword in keywords:
            try:
                df.at[id, keyword]=hdr[keyword]
            except:
                print(f'Failed to find {keyword} for {file}')
                df.at[id, keyword]=np.nan
        if basefilename.startswith(('ir', 'binir')):
            band='ir'
            realfilt=hdr['IRFLTID']
        elif basefilename.startswith(('rccd', 'ccd')):
            band='opt'
            realfilt=hdr['CCDFLTID']
        elif basefilename.startswith('r'):
            try:
                filt = hdr['CCDFLTID'].strip().lower()
                realfilt=hdr['CCDFLTID']
            except:
                filt = hdr['FILTERID'].strip().lower()
                realfilt=hdr['FILTERID']
            if filt in ['b','v','r','i','r wide', 'wide r']:
                band='opt'
            elif filt in ['y','h','k','j']:
                band='ir'
            else:
                band=np.nan
            
        df.at[id, 'band']=band
        df.at[id, 'filter']=realfilt
        
    ####THIS LOGIC ABOVE IS DUMB
    #specifically fix the nan to naT issue
    df['datetime'] = pd.to_datetime(df['DATE-OBS'] + ' ' + df['TIME-OBS'], format='mixed', errors='coerce')
    df = df.sort_values(by='datetime')
    
    for id, row in df.iterrows():
        df.at[id,'filename']=df.at[id,'filename'].split('/')[-1]
    print(df)
    #save log
    df.to_csv(f'{repo}/product/log_files_{target}.csv', index=False)
    print('header issues: ',headerissues)
