import numpy as np
import os
import glob
import pandas as pd
from lookup_name import *

os.chdir('/home/kmc249/test_data/')
logs=glob.glob('**/LOG*csv', recursive=True)

#scrapelogs and replogs:
scrapelogs=glob.glob('/home/kmc249/usbdrive_logs/*scrape*')
replogs=glob.glob('/home/kmc249/usbdrive_logs/*replog*')

#for replog header
rename_map = {
    'Filename': 'filename',
    'Object': 'OBJECT',
    'RA': 'RA',
    'Dec': 'DEC',
    'UTDate': 'DATE-OBS',
    'UTC': 'TIME-OBS',
    'JD': 'JD',
    'ExpTime': 'EXPTIME',
    'SecZ': 'SECZ',
}

current_header = ['ProjectID','ImType','Object','RA','Dec','ExpTime','Filter','SecZ','LST','UTDate','UTC','JD','Filename','Logged_UT','replog','proper name']
goal_header = ['filename','OBJECT','RA','DEC','DATE-OBS','TIME-OBS','JD','EXPTIME','SECZ','CCDFLTID','IRFLTID','TILT1','TILT2','TILT3','log name','Location','prop name help','Physical loc','OWNER','proper name','xrb']


#make a list of all the logs
dflist=[]
for log in logs:
    df=pd.read_csv(log, low_memory=False)
    df['log name']=f"{log.split('/')[-2]}/{log.split('/')[-1]}"
    if 'other' in log:
        df['Location']='miniarchive'
    else:
        df['Location']=log.split('LOG_')[0]
    df['prop name help']=log.split('LOG_')[1].split('.')[0]
    #if df['Location'][0]=='optical_CD_header_logs/':
    if 'CD_header_logs' in df['Location'][0]:
        df['Physical loc']='CD'
    else:
        df['Physical loc']='Disk'
    dflist.append(df)
    
for log in scrapelogs:
    try:
        df=pd.read_csv(log, low_memory=False, on_bad_lines='skip')
    except:
        raise Exception(f'{log} messed up')
    df['log name']=log.split('/')[-1]
    df['Location']=df['full path']
    df['prop name help']=''
    df['Physical loc']='Data scrape'
    dflist.append(df)
    
for log in replogs:
    df=pd.read_csv(log, low_memory=False)
    df['CCDFLTID'] = df['Filter']
    df['IRFLTID'] = df['Filter']
    df = df.rename(columns=rename_map)
    df = df[[col for col in df.columns if col in goal_header]]


    df['log name']=log.split('/')[-1]
    try:
        df['Location']=df['replog']
    except:
        df['Location']=''
    df['prop name help']=''
    df['Physical loc']='Data replog'

biglog = pd.concat(dflist, ignore_index=True)

for band in ['ir', 'optical']:
    #filter to ignore some stuff, check it's just rccd
    if band=='optical':
        biglog = biglog[biglog['filename'].str.contains('rccd', case=False, na=False)]
    else:
        biglog = biglog[biglog['filename'].str.contains('ir', case=False, na=False)]
    pattern = r'focus|BIAS| for |JUNK|dome|dark|shift|faint|bright|dither|flat|test|sky|summed'
    biglog = biglog[~biglog['OBJECT'].str.contains(pattern, case=False, na=False)]
    
    
    #make .gz also count as duplicates
    biglog['filename_base'] = biglog['filename'].str.replace(r'\.gz$', '', regex=True)
    # drop duplicates based on 'filename', 'OBJECT', and 'DATE-OBS', keeping disk as default
    biglog = biglog.sort_values('Physical loc', ascending=False)
    biglog = biglog.drop_duplicates(subset=['filename_base', 'OBJECT', 'DATE-OBS', 'TIME-OBS'])
    biglog = biglog.drop(columns='filename_base')
    biglog.reset_index(drop=True, inplace=True)
    for id, row in biglog.iterrows():
        try:
            biglog.at[id, 'proper name']=get_proper_name(row['OBJECT'])
            biglog.at[id, 'xrb']=is_xrb_quick(biglog.at[id, 'proper name'])
        except:
            #Get rid of the errors that were disk guys, but keep CD guys that errored
            if row['Physical loc']=='Disk':    
                biglog.at[id, 'xrb']=False
                biglog.at[id, 'proper name']=np.nan
            else:
                #if it's on a CD, try looking up the name of the CD object
                biglog.at[id, 'xrb']=True
                try:
                    biglog.at[id, 'proper name']=get_proper_name(row['prop name help'])
                    biglog.at[id, 'xrb']=is_xrb_quick(biglog.at[id, 'proper name'])
                except:
                    biglog.at[id, 'xrb']=False
                    biglog.at[id, 'proper name']=np.nan
                    
    
    
    biglog.to_csv(f'all_{band}_log.csv', index=False)