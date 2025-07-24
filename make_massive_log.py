import numpy as np
import os
import glob
import pandas as pd
from lookup_name import *

os.chdir('/home/kmc249/test_data/')
logs=glob.glob('**/LOG*csv', recursive=True)

#make a list of all the logs
dflist=[]
for log in logs:
    df=pd.read_csv(log)
    df['Location']=log.split('LOG_')[0]
    df['prop name help']=log.split('LOG_')[1].split('.')[0]
    if df['Location'][0]=='optical_CD_header_logs/':
        df['Physical loc']='CD'
    else:
        df['Physical loc']='Disk'
    dflist.append(df)

biglog = pd.concat(dflist, ignore_index=True)

#filter to ignore some stuff, check it's just rccd
biglog = biglog[biglog['filename'].str.contains('rccd', case=False, na=False)]
biglog = biglog[~biglog['OBJECT'].str.contains('flat', case=False, na=False)]
biglog = biglog[~biglog['OBJECT'].str.contains('focus', case=False, na=False)]
biglog = biglog[~biglog['OBJECT'].str.contains('BIAS', case=False, na=False)]
biglog = biglog[~biglog['OBJECT'].str.contains(' for ', case=False, na=False)]
biglog = biglog[~biglog['OBJECT'].str.contains(' JUNK ', case=False, na=False)]


# drop duplicates based on 'filename', 'OBJECT', and 'DATE-OBS', keeping disk as default
biglog = biglog.sort_values('Physical loc', ascending=False)
biglog = biglog.drop_duplicates(subset=['filename', 'OBJECT', 'DATE-OBS', 'TIME-OBS'])
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
                


biglog.to_csv('all_optical_log.csv', index=False)