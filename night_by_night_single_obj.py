import glob
import os
from astropy.io import fits
from lookup_name import *
import pandas as pd
import numpy as np

#this should be the proper name of the object you want to search for
interest=''

#replog version
paths=['/USB2/archive/REP-LOGS-2012/','/USB2/archive/2011.lastpart/REP-LOGS/', '/USB1/archive/REP-LOGS/2009/', '/USB1/archive/REP-LOGS/2010/', '/USB1/archive/REP-LOGS/2011firstpart/']

loglist=[]
for path in paths:
    loglist1=glob.glob(f'{path}/*')
    loglist+=loglist1
    

#fixed width location for most of 2012 at least
colspecs = [
    (0, 15),    # ProjectID
    (15, 22),   # ImType
    (22, 39),   # Object
    (39, 50),   # RA
    (50, 60),   # Dec
    (60, 68),   # ExpTime
    (68, 76),   # Filter
    (76, 81),   # SecZ
    (81, 90),   # LST
    (90, 101),  # UTDate
    (101, 112), # UTC
    (112, 125), # JD
    (125, 148), # Filename
    (148, None) # [Logged@UT]
]

colnames = [
    "ProjectID", "ImType", "Object", "RA", "Dec", "ExpTime", "Filter",
    "SecZ", "LST", "UTDate", "UTC", "JD", "Filename", "Logged_UT"
]

#another version of the colspecs
colspecs2 = [
    (0, 15),    # ProjectID
    (15, 22),   # ImType
    (22, 39),   # Object
    (39,47),   # ExpTime
    (47, 55),   # Filter
    (55, 64),   # LST
    (64, 75),   # UTDate
    (75, 86),   # UTC
    (86, 99),   # JD
    (99, 122),  # Filename
    (122, None) # [Logged@UT]
]

colnames2 = [
    "ProjectID", "ImType", "Object",  "ExpTime", "Filter",
    "LST", "UTDate", "UTC", "JD", "Filename", "Logged_UT"
]

obj_rows=[]
for log in loglist:
    #figure out where to skip to
    with open(log) as f:
        lines = f.readlines()
    header_line=0
    footer_line=0
    for i, line in enumerate(lines):
        if line.strip().startswith("Project"):
            header_line = i
        #in case they add comments after the table
        if line.startswith("--------") and header_line!=0 and i>header_line:
            footer_line=i
        
    #if we didn't find that log, then there was no observing that night, so skip
    if header_line==0:
        no_obs_list.append(log)
        continue
    
    df = pd.read_fwf(log, skiprows=header_line+1, colspecs=colspecs,skipfooter=footer_line, names=colnames)

    #use the second format if necessary
    if df['Logged_UT'].isna().any():
        df=pd.read_fwf(log, skiprows=header_line+1, colspecs=colspecs2, skipfooter=footer_line, names=colnames2)
    
    df['replog']=log
    
    
    #find all the objects
    for id, row in df.iterrows():
        try:
            df.at[id, 'proper name']=get_proper_name(row['Object'])
        except:
            df.at[id, 'proper name']='FAIL'
    
    for id, row in df.iterrows():

        if row['proper name'] == interest:
            obj_rows.append(row)
        else:
            continue

### find the xrbs and what files exist for each of them. 
#we're going to make a massive log to be used with the nice calendar thing
    
replogdf=pd.DataFrame(xrb_rows)

replogdf.to_csv(f'/home/kmc249/usbdrive_logs/single_replog_{interest}.csv', index=False)


#scrapelog version
paths=['/USB4/archive', '/USB2/archive/20130101thru0130', '/USB2/archive/20130201thru0228', '/USB2/archive/20130301thru0331', '/USB2/archive/20130401thru0430']

paths2= ['20131201thru1231',
 '20170601ThruEnd',
 '20161001ThruEnd',
 '20170501ThruEnd',
 '20160101ThruEnd',
 '20130901thru0929',
 '20150601ThruEnd',
 '20150201ThruEnd',
 '20140301ThruEnd',
 '20140901ThruEnd',
 '20170101ThruEnd',
 '20170701ThruEnd',
 '20150701ThruEnd',
 '20130401thru0430',
 '20170901ThruEnd',
 '20141201ThruEnd',
 '20150501ThruEnd',
 '20170801ThruEnd',
 '20161101ThruEnd',
 '20141001ThruEnd',
 '20150401ThruEnd',
 '20130501thru0531',
 '20130601thru0630',
 '20150301ThruEnd',
 '20130701thru0731',
 '20140701ThruEnd',
 '20150901ThruEnd',
 '20140801ThruEnd',
 '20140401ThruEnd',
 '20131101thru1130',
 '20130801thru0831',
 '20140101ThruEnd',
 '20140501ThruEnd',
 '20131001thru1031',
 '20170201ThruEnd',
 '20170301ThruEnd',
 '20160201ThruEnd',
 '20140201ThruEnd',
 '20160701ThruEnd',
 '20160401ThruEnd',
 '20160901ThruEnd',
 '20150801ThruEnd',
 '20160501ThruEnd',
 '20141101ThruEnd',
 '20140601ThruEnd',
 '20150101ThruEnd',
 '20151101ThruEnd',
 '20151201ThruEnd',
 '20160301ThruEnd',
 '20151001ThruEnd',
 '20161201ThruEnd',
 '20170401ThruEnd',
 '20160601ThruEnd',
 '20160801ThruEnd']

for p in paths2:
    newp=f'/USB3/archive/{p}'
    paths.append(newp)


keywords=['OBJECT','RA','DEC','DATE-OBS','TIME-OBS','JD','EXPTIME','SECZ','CCDFLTID','IRFLTID','TILT1','TILT2','TILT3','OWNER']


rows=[]
new_names=[]
objs=[]
for repo in paths:
    for dirpath, dirnames, filenames in os.walk(repo):
        for f in filenames:
            if (f.endswith('.fits') or f.endswith('.fits.gz')) and not f.startswith('ccd'):
                full_path = os.path.join(dirpath, f)
                hdr=fits.getheader(full_path)
                obj=hdr['OBJECT']
                try:
                    name=get_proper_name(obj)
                    if name==interest:
                        objs.append(name)
                        #add all the things if this is an xrb
                        row=[f, full_path]
                        for k in keywords:
                            try:
                                row.append(hdr[k])
                            except:
                                row.append(np.nan)
                        rows.append(row)
                    else:
                        continue
                except:
                    continue


df=pd.DataFrame(rows, columns=['filename', 'full path'] + keywords)
df.to_csv(f'/home/kmc249/usbdrive_logs/single_scrapelog_{interest}.csv', index=False) 

print('objs found in 20013-2019:', set(objs))