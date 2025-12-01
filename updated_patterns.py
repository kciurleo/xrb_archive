import pandas as pd
import numpy as np
from astropy.time import Time
from datetime import timedelta
import re

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

infile='/home/kmc249/test_data/all_optical_log.csv'
df=pd.read_csv(infile, low_memory=False)
df=df.loc[df['proper name']=='A0620-00']

df['DATE-OBS']=pd.to_datetime(df['DATE-OBS'], errors='coerce')
filter_map2 = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
df['filter num'] = df['CCDFLTID'].map(filter_map2)

for id, row in df.iterrows():
    if not pd.isna(row['TIME-OBS']):
        try:
            df.at[id, 'datetimeobs']=Time(row['JD'], format='jd')
            df.at[id, 'nice time']=df.at[id, 'datetimeobs'].datetime
        except:
            df.at[id, 'datetimeobs']=np.nan
            df.at[id, 'nice time']=np.nan

df['nice time'] = pd.to_datetime(df['nice time'], errors='coerce')
#set up years array
years=np.arange(1998,2020,1)
# Sort by time
df = df.sort_values('nice datetime').reset_index(drop=True)

# ------------------------------------------------------------
# 2. CREATE `actual_filter` COLUMN
# ------------------------------------------------------------

# Optical images start with 'rccd...', IR images start with 'binir...'
def get_actual_filter(row):
    fn = str(row['filename'])
    if fn.startswith('rccd'):
        return row['ccdfltid']
    elif fn.startswith('binir'):
        return row['irfltid']
    else:
        return np.nan

df['actual_filter'] = df.apply(get_actual_filter, axis=1)

# ------------------------------------------------------------
# 3. FUNCTION TO CHECK IF A SEQUENCE MATCHES A PATTERN
# ------------------------------------------------------------

def match_pattern(df, start_index, filt_pattern, exp_pattern,
                  max_slop=20):
    """
    df: dataframe sorted by time
    start_index: index into df to try matching the pattern
    filt_pattern: e.g. ['R','B','H','H','H']
    exp_pattern:  e.g. [90.,90.,60.,60.,60.]
    max_slop: maximum allowed extra time between exposures (sec)
    
    Returns: (True/False, list_of_indices)
    """
    k = len(filt_pattern)
    if start_index + k > len(df):
        return (False, [])

    idxs = list(range(start_index, start_index + k))
    window = df.iloc[idxs]

    # 1. FILTER MATCH
    if window['actual_filter'].tolist() != filt_pattern:
        return (False, [])

    # 2. EXPOSURE TIME MATCH
    if not np.allclose(window['exptime'].values.astype(float),
                       np.array(exp_pattern).astype(float)):
        return (False, [])

    # 3. TIMING CHECK
    # Compute expected times
    t0 = window['nice datetime'].iloc[0]

    expected_times = [t0]
    for ex in exp_pattern[:-1]:
        # each next exposure should begin after exp + some small readout
        expected_times.append(expected_times[-1] + timedelta(seconds=ex + max_slop))

    # Actual start times:
    actual_times = window['nice datetime'].tolist()

    # Allow deviations up to max_slop as long as total spacing is right-ish
    for t_act, t_exp in zip(actual_times, expected_times):
        if abs((t_act - t_exp).total_seconds()) > max_slop:
            return (False, [])

    return (True, idxs)


# ------------------------------------------------------------
# 4. FIND ALL PATTERN MATCHES (ENSURING NO OVERLAP)
# ------------------------------------------------------------

def find_all_pattern_matches(df, filt_patterns, exp_patterns, max_slop=20):
    """
    filt_patterns: list of filter sequences
    exp_patterns:  list of exposure sequences (same length)
    """
    
    df['pattern_id'] = -1  # -1 = no pattern match
    
    for pat_id, (fpat, epat) in enumerate(zip(filt_patterns, exp_patterns)):
        used = set()  # indices already used by earlier matches

        for i in range(len(df)):
            # skip if already used
            if i in used:
                continue

            ok, idxs = match_pattern(df, i, fpat, epat, max_slop=max_slop)
            if ok:
                # assign the pattern ID
                df.loc[idxs, 'pattern_id'] = pat_id
                used.update(idxs)  # avoid overlap

    return df

# ------------------------------------------------------------
# 5. RUN PATTERN MATCHER
# ------------------------------------------------------------

filt_pats = [
    ['R','B','H','H','H'],
]
exp_pats = [
    [90.,90.,60.,60.,60.]
]

df = find_all_pattern_matches(df, filt_pats, exp_pats, max_slop=25)

# Now df['pattern_id'] contains:
#   -1  → not in any pattern
#   0   → pattern 0
#   1   → pattern 1
# etc.

