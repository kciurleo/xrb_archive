import pandas as pd
import numpy as np
import glob
from lookup_name import *

path='/scratch/REPLOGS'

loglist=glob.glob(f'{path}/**/[0-9]*')
    

#fixed width location for most of 2012 at least
colspecs = [
    (0, 9),    # ProjectID
    (9, 16),   # ImType
    (16, 33),   # Object
    (33, 44),   # RA
    (44, 54),   # Dec
    (54, 62),   # ExpTime
    (62, 70),   # Filter
    (70, 75),   # SecZ
    (75, 84),   # LST
    (84, 95),  # UTDate
    (95, 105), # UTC
    (105, 118), # JD
    (118, 137), # Filename
    (137, None) # [Logged@UT]
]

colnames = [
    "ProjectID", "ImType", "Object", "RA", "Dec", "ExpTime", "Filter",
    "SecZ", "LST", "UTDate", "UTC", "JD", "Filename", "Logged_UT"
]

#another version of the colspecs
colspecs2 = [
    (0, 9),    # ProjectID
    (9, 16),   # ImType
    (16, 42),   # Object
    (42, 53),   # RA
    (53, 63),   # Dec\
    (63, 71),   # Equinox
    (71, 79),   # ExpTime
    (79, 88),   # Filter
    (88, 93),   # SecZ
    (93, 104),   # LST
    (104, 115),  # UTDate
    (115, 126), # UTC
    (126, 136), # JD
    (136, None) # Filename
]

colnames2 = [
    "ProjectID", "ImType", "Object",  "RA", "Dec", "Equinox", "ExpTime", "Filter",
    "SecZ", "LST", "UTDate", "UTC", "JD", "Filename"
]

colspecs3 = [
    (0, 9),    # ProjectID
    (9, 16),   # ImType
    (16, 33),   # Object
    (33, 44),   # RA
    (44, 54),   # Dec
    (54, 62),   # ExpTime
    (62, 70),   # Filter
    (70, 75),   # SecZ
    (75, 84),   # LST
    (84, 95),  # UTDate
    (95, 106), # UTC
    (106, 116), # JD
    (116, None), # Filename
]

colnames3 = [
        "ProjectID", "ImType", "Object", "RA", "Dec", "ExpTime", "Filter",
        "SecZ", "LST", "UTDate", "UTC", "JD", "Filename"
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
    
    
    header_line = None
    footer_line = 0
    templn=''
    for i, line in enumerate(lines):
        if line.strip().startswith("Project") or line.strip().startswith("Owner"):
            header_line = i
        if header_line and line.strip() == "" and footer_line==0:
            footer_line = len(lines)-i
 

    #if we didn't find that log, then there was no observing that night, so skip
    if header_line is None:
        no_obs_list.append(log)
        print(f'NO LOG: {log}')
        continue
    print('first try')
    df = pd.read_fwf(log, skiprows=header_line+1, colspecs=colspecs2,skipfooter=footer_line, names=colnames2)
    needed_second=False
    #use the second format if necessary
    sety = [str(i) for i in set(df['Equinox'])]
    if '2000.0' not in sety:
        print('second try')
        needed_second=True
        print(df)
        df=pd.read_fwf(log, skiprows=header_line+1, colspecs=colspecs, skipfooter=footer_line, names=colnames)
    
    if df['Filename'].isna().any():
        print('third try')
        df=pd.read_fwf(log, skiprows=header_line+1, colspecs=colspecs3, skipfooter=footer_line, names=colnames3)
    
    
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
    if needed_second:
        print(df)

unique_names = set().union(*all_names)
#print(unique_names)
newnames=list(unique_names)
newnames.sort()
print(newnames)
#print(no_obs_list)
print(error_list)

'''
### find the xrbs and what files exist for each of them. 
#we're going to make a massive log to be used with the nice calendar thing
    
xrbdf=pd.DataFrame(xrb_rows)
print(xrbdf.head())
print(len(xrbdf))

grp = xrbdf.groupby(['proper name'])

for name, g in grp:
    g.to_csv(f'/home/kmc249/usbdrive_logs/tapereplog_{name[0]}.csv', index=False)
'''