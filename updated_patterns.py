import pandas as pd
import numpy as np
from astropy.time import Time
from datetime import timedelta
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

#infiles, assuming at least datetime, filename, filter
infile='/home/kmc249/test_data/all_ir_log.csv'
df1=pd.read_csv(infile, low_memory=False)
df1=df1.loc[df1['proper name']=='A0620-00']
infile2='/home/kmc249/test_data/all_optical_log.csv'
df2=pd.read_csv(infile2, low_memory=False)
df2=df2.loc[df2['proper name']=='A0620-00']
df = pd.concat([df1, df2], ignore_index=True)


df['DATE-OBS']=pd.to_datetime(df['DATE-OBS'], errors='coerce')


#filter_map2 = {'Y': 0, 'J': 1, 'H': 2, 'K': 3}
#df['filter num'] = df['IRFLTID'].map(filter_map2)

for id, row in df.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            df.at[id, 'datetimeobs']=Time(row['JD'], format='jd')
            df.at[id, 'nice time']=df.at[id, 'datetimeobs'].datetime
        except:
            df.at[id, 'datetimeobs']=np.nan
            df.at[id, 'nice time']=np.nan

df['nice datetime'] = pd.to_datetime(df['nice time'], errors='coerce')



#make actual filter col
def get_actual_filter(row):
    fn = str(row['filename'])
    if fn.startswith('r'):
        return row['CCDFLTID']
    elif fn.startswith('binir'):
        return row['IRFLTID']
    else:
        return np.nan

#cleaning the df and sort by time
df['actual filter'] = df.apply(get_actual_filter, axis=1)
df.dropna(subset=['actual filter'], inplace=True)
df.dropna(subset=['nice datetime'], inplace=True)
df.drop_duplicates(subset=['filename'], inplace=True)

df = df.sort_values('nice datetime').reset_index(drop=True)

filter_map2 = {'B': 3, 'V': 2, 'R': 1, 'I': 0, 'K': -1, 'H': -2, 'J': -3, 'Y': -4}
df['filter num'] = df['actual filter'].map(filter_map2)


#matching pattenrns
def match_pattern(df, start_index, filt_pattern, exp_pattern,
                  readout):
    """
    df: dataframe sorted by time
    start_index: index into df to try matching the pattern
    filt_pattern: list, e.g. ['V','I','H','H','H']
    exp_pattern: corresponding list, e.g. [90.,90.,60.,60.,60.]
    readout: acceptableextra time between exposures (sec)
    
    Returns: (True/False, list_of_indices)
    """
    k = len(filt_pattern)
    if start_index + k > len(df):
        return (False, [])

    idxs = list(range(start_index, start_index + k))
    window = df.iloc[idxs]

    #check filters
    if window['actual filter'].tolist() != filt_pattern:
        return (False, [])

    #check exposures
    if not np.allclose(window['EXPTIME'].astype(float),
                   exp_pattern, rtol=0, atol=2):
        return (False, [])

    #check timing
    #initial/final times
    t0 = window['nice datetime'].iloc[0]
    t_last = window['nice datetime'].iloc[-1]
    
    #expected total time for the pattern + last time
    total_expected = float(np.sum(exp_pattern) + readout * (len(exp_pattern)))
    t_last_expected = t0 + timedelta(seconds=total_expected)

    #last exposure should be within readout seconds of expectation
    if (t_last - t_last_expected).total_seconds() > readout:
        return (False, [])


    return (True, idxs)


#finding patterns

def find_all_pattern_matches(df, filt_patterns, exp_patterns, readout):
    """
    filt_patterns: list of filter sequences
    exp_patterns:  list of exposure sequences (same length)
    """
    
    df['pattern id'] = -1  
    
    for pat_id, (fpat, epat) in enumerate(zip(filt_patterns, exp_patterns)):
        used = set()  #indices already used by earlier matches

        for i in range(len(df)):
            #skip if already used
            if i in used:
                continue

            ok, idxs = match_pattern(df, i, fpat, epat, readout=readout)
            if ok:
                #assign the pattern id, keep track of the ones we've already assigned a pattern to
                df.loc[idxs, 'pattern id'] = pat_id
                used.update(idxs)

    return df

#defined patterns

filt_pats1=[['B', 'V', 'I', 'V', 'I'], ['B', 'V', 'I'],['B', 'V', 'I'], ['B', 'V', 'I'], ['V', 'I'], ['V', 'I'], ['I'], ['B'], ['V']]
exp_pats1=[[360.0, 360.0, 360.0, 360.0, 360.0],[360.0, 360.0, 360.0], [240.0, 240.0, 240.0], [300.0, 240.0, 240.0], [360.0, 360.0], [660.0, 660.0], [240.0], [360.0], [360.0]]
colors1=['sienna','g','violet','k','yellow', 'blue', 'cyan', 'darkorange','indigo']


filt_pats2=[['K','K','K','K','K', 'K', 'K', 'K','J','J','J','J','J','J', 'H', 'H','H','H','H','H'],['J','J', 'J', 'J', 'J','H', 'H', 'H', 'H','H','K','K', 'K', 'K', 'K','K'], ['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H'],['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H'], ['H', 'H', 'H']]
exp_pats2=[[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04],[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04],[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04], [90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03,90.03, 90.03], [30.04, 30.04, 30.04]]
exp_pats2 = [[int(x) for x in sub] for sub in exp_pats2]
colors2=['blue', 'violet', 'g','gold']

# Combine patterns
filt_pats = filt_pats1 + filt_pats2
exp_pats = exp_pats1 + exp_pats2
colors = colors1 + colors2


df = find_all_pattern_matches(df, filt_pats, exp_pats, readout=80)
print(df[['filename', 'nice datetime','actual filter', 'pattern id']])
print(len(df.loc[df['pattern id']==-1]))
df.to_csv('/home/kmc249/Downloads/updated_ao620_patterns.csv')

filt_pats1=[['B', 'V', 'I', 'V', 'I'], ['B', 'V', 'I'],['B', 'V', 'I'], ['B', 'V', 'I'], ['V', 'I'], ['V', 'I'], ['I'], ['B'], ['V']]
exp_pats1=[[360.0, 360.0, 360.0, 360.0, 360.0],[360.0, 360.0, 360.0], [240.0, 240.0, 240.0], [300.0, 240.0, 240.0], [360.0, 360.0], [660.0, 660.0], [240.0], [360.0], [360.0]]
colors1=['sienna','g','violet','k','yellow', 'blue', 'cyan', 'darkorange','indigo']


filt_pats2=[['K','K','K','K','K', 'K', 'K', 'K','J','J','J','J','J','J', 'H', 'H','H','H','H','H'],['J','J', 'J', 'J', 'J','H', 'H', 'H', 'H','H','K','K', 'K', 'K', 'K','K'], ['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H'],['H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H', 'H', 'H','H', 'H'], ['H', 'H', 'H']]
exp_pats2=[[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04],[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04],[30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04, 30.04], [90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03, 90.03,90.03, 90.03], [30.04, 30.04, 30.04]]
exp_pats2 = [[int(x) for x in sub] for sub in exp_pats2]
colors2=['blue', 'violet', 'g','gold', 'gray']

# Combine patterns
filt_pats = filt_pats1 + filt_pats2
exp_pats = exp_pats1 + exp_pats2
colors = colors1 + colors2
df=pd.read_csv('/home/kmc249/Downloads/updated_ao620_patterns.csv', low_memory=False)

df['DATE-OBS']=pd.to_datetime(df['DATE-OBS'], errors='coerce')
for id, row in df.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            df.at[id, 'datetimeobs']=Time(row['JD'], format='jd')
            df.at[id, 'nice time']=df.at[id, 'datetimeobs'].datetime
        except:
            df.at[id, 'datetimeobs']=np.nan
            df.at[id, 'nice time']=np.nan

df['nice datetime'] = pd.to_datetime(df['nice time'], errors='coerce')
#set up years array
years=np.arange(1998,2020,1)
#plot things
f, a = plt.subplots(11, 2, figsize=(11, 16),layout='constrained')
axes=np.ravel(a, order='F')

#for each year, find the associate obs

for id, year in enumerate(years):
    yrtable=df.loc[df['DATE-OBS'].dt.year==year]
    outpat=yrtable.loc[yrtable['pattern id']==-1]
    axes[id].scatter(outpat['nice time'], outpat['filter num'], c='red', s=2, label= 'Out of pattern')

    for pnum in range(len(filt_pats)):
        inpat=yrtable.loc[yrtable['pattern id']==pnum]
        axes[id].scatter(inpat['nice time'], inpat['filter num'], c=colors[pnum], s=2, label=f'Pattern {pnum}')
    
    axes[id].set_ylabel(year)
    xlimlo = datetime.datetime(year, 1, 1)
    xlimhi = datetime.datetime(year, 12, 31)
    axes[id].set_xlim(xlimlo, xlimhi)
    axes[id].set_yticks([0, 1, 2, 3])
    axes[id].set_yticklabels(['Y', 'J', 'H', 'K', 'I','V','R','B']) 
    
    if year!=2008 and year!=2019:
        axes[id].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        axes[id].xaxis.set_major_locator(mdates.MonthLocator())
        axes[id].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

#save as a file
plt.suptitle('A0620-00')
plt.legend(ncol=2)

plt.savefig(f'/home/kmc249/Downloads/updated_patterns_A0620-00.png', dpi=300)
plt.show()
