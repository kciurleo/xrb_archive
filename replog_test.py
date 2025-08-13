import pandas as pd
import numpy as np
import glob
from lookup_name import *

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

no_obs_list=[]
error_list=[]
all_names=[]
xrb_rows=[]
for log in loglist:
    print(log)
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
    #new names to add to my name lookup
    try:
        names=set(df.loc[df['proper name']=='FAIL']['Object'])
        all_names.append(names)
    except:
        print('what the heck')
        print(df)
        error_list.append(log)
    
    for id, row in df.iterrows():
        try:
            if is_xrb_quick(row['proper name']):
                xrb_rows.append(row)
        except:
            print('faulty nickname')

unique_names = set().union(*all_names)
#print(unique_names)
newnames=list(unique_names)
newnames.sort()
#print(newnames)
#print(no_obs_list)
#print(error_list)


### find the xrbs and what files exist for each of them. 
#we're going to make a massive log to be used with the nice calendar thing
    
xrbdf=pd.DataFrame(xrb_rows)
print(xrbdf.head())
print(len(xrbdf))

grp = xrbdf.groupby(['proper name'])

for name, g in grp:
    g.to_csv(f'/home/kmc249/usbdrive_logs/usbdrivereplog_{name[0]}.csv', index=False)