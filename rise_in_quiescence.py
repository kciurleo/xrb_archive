#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:07:50 2026

@author: kmc249
"""
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from astropy.time import Time
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#some functions
def read_corrected_txt(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=[
            "MJD",
            "mag_corr",
            "mag_corr_err",
            "flag",
            "mag",
            "mag_err",
            "alt_mag_corr"
        ]
    )
    
    df["nice time"] = pd.to_datetime(Time(df["MJD"], format="mjd").to_datetime())
    return df

def get_quiescent(df, intervals):
    mask = np.ones(len(df), dtype=bool)
    
    for start, end in intervals:
        mask &= ~((df["MJD"] >= start) & (df["MJD"] <= end))
    
    return df[mask].copy()

#read in 1.3 and 1 m stuff
Rtable = pd.read_csv('/neta/xrb/AqlX-1/product/combo_subtracted_shifted/AqlX-1_R_corrected_lc_4_27.csv', low_memory=False) #change this girl
Jtable=pd.read_csv('/neta/xrb/AqlX-1/product/combo_subtracted_shifted/AqlX-1_J_corrected_lc_4_27.csv', low_memory=False)

#rename columns to be consistent
Rtable = Rtable.rename(columns={
    'Rmag_corr': 'mag_corr',
    'e_Rmag_corr': 'mag_corr_err',
    'Rmag': 'mag',
    'e_Rmag': 'mag_err',
    'Rmag Divided Version': 'alt_mag_corr'
})
Jtable = Jtable.rename(columns={
    'Jmag_corr': 'mag_corr',
    'e_Jmag_corr': 'mag_corr_err',
    'Jmag': 'mag',
    'e_Jmag': 'mag_err',
    'Jmag Divided Version': 'alt_mag_corr'
})

#LCO quiescent data
R_LCO_banzai_corr = read_corrected_txt('/home/kmc249/Downloads/R_usable_banzai_corrected.txt')  
R_LCO_orac_corr = read_corrected_txt('/home/kmc249/Downloads/R_usable_orac_corrected.txt')
R_LCO_corr=pd.concat([R_LCO_banzai_corr, R_LCO_orac_corr], ignore_index=True)
Rp_LCO_corr = read_corrected_txt('/home/kmc249/Downloads/rp_usable_banzai_corrected.txt') 

#ip band corr
ip_LCO_banzai_corr = read_corrected_txt('/home/kmc249/Downloads/ip_usable_banzai_corrected.txt')  
ip_LCO_orac_corr = read_corrected_txt('/home/kmc249/Downloads/ip_usable_orac_corrected.txt')
ip_LCO_corr=pd.concat([ip_LCO_banzai_corr, ip_LCO_orac_corr], ignore_index=True)

#v band corr
v_LCO_banzai_corr = read_corrected_txt('/home/kmc249/Downloads/V_usable_banzai_corrected.txt')  
v_LCO_orac_corr = read_corrected_txt('/home/kmc249/Downloads/V_usable_orac_corrected.txt')
v_LCO_corr=pd.concat([v_LCO_banzai_corr, v_LCO_orac_corr], ignore_index=True)


#only get stuff in quiescence
#Outburst list
full = pd.read_csv("/home/kmc249/Downloads/full_outbursts.csv")
mini = pd.read_csv("/home/kmc249/Downloads/mini_outbursts.csv")

#Mask out quiescence
count_mini=False
full_intervals=list(zip(full["Start MJD"], full["End MJD"])) + \
            list(zip(mini["Start MJD"], mini["End MJD"]))
if count_mini:
    intervals = full_intervals
else:
    intervals = list(zip(full["Start MJD"], full["End MJD"]))

#find quiescence values for all tables which have MJD already
quiescent_tables={}
tables = {
    "R Smarts": Rtable,
    "J Smarts": Jtable,
    "R LCO": R_LCO_corr,
    "rp LCO": Rp_LCO_corr,
    "ip LCO": ip_LCO_corr,
    "V LCO": v_LCO_corr,
    
}

for name, table in tables.items():
    quiescent_tables[name] = get_quiescent(table, full_intervals)



#%%

#Fiddling with quiescence

# Sort intervals just in case
intervals = sorted(intervals, key=lambda x: x[0])

# Build quiescent intervals (gaps between outbursts)
quiescent_intervals = []

for i in range(len(intervals) - 1):
    end_current = intervals[i][1]
    start_next = intervals[i+1][0]
    
    if start_next > end_current:
        quiescent_intervals.append((end_current, start_next))
 
#%%
#Iterating and plotting

for key, oldtable in quiescent_tables.items():
    segments_J = []
    
    for start, end in quiescent_intervals:
        seg_J = oldtable[(oldtable["MJD"] >= start) & (oldtable["MJD"] <= end)].copy()
        
        if len(seg_J) > 0:
            segments_J.append(seg_J)
            
    #Durations
    durations_J = []
    
    for seg in segments_J:
        t0 = seg["MJD"].min()
        t1 = seg["MJD"].max()
        durations_J.append(t1 - t0)
    
    durations_J = np.array(durations_J)          
        
    
    norm = mcolors.Normalize(vmin=durations_J.min(), vmax=durations_J.max())
    cmap = cm.viridis  # you can try 'plasma', 'coolwarm', etc.

    ##Aligned by start
    
    all_x = []
    all_y = []
    
    for seg in segments_J:
        t0 = seg["MJD"].min()
        shifted_time = seg["MJD"] - t0
        
        all_x.append(shifted_time.values)
        all_y.append(seg["alt_mag_corr"].values)
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    
    
    slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
    
    fig, ax = plt.subplots()
    
    # scatter (your existing loop)
    for seg, dur in zip(segments_J, durations_J):
        t0 = seg["MJD"].min()
        shifted_time = seg["MJD"] - t0
        color = cmap(norm(dur))
        
        ax.scatter(shifted_time, seg["alt_mag_corr"], color=color, alpha=0.5)
    
    # best-fit line
    x_fit = np.linspace(all_x.min(), all_x.max(), 500)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, color='black', linewidth=2, label=f'y={slope:.4f}x+{intercept:2f}')
    ax.set_title("Quiescence aligned by start")
    ax.set_xlabel("Time since quiescence start (days)")
    ax.set_ylabel(f"{key} (corrected)")
    ax.invert_yaxis()
    ax.legend()
    
    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Quiescence duration (days)")
    
    plt.show()
    
    ##Aligned by end
    
    all_x = []
    all_y = []
    
    for seg in segments_J:
        t1 = seg["MJD"].max()
        shifted_time = seg["MJD"] - t1   # <-- key change
        
        all_x.append(shifted_time.values)
        all_y.append(seg["alt_mag_corr"].values)
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
    
    fig, ax = plt.subplots()
    
    # scatter
    for seg, dur in zip(segments_J, durations_J):
        t1 = seg["MJD"].max()
        shifted_time = seg["MJD"] - t1
        color = cmap(norm(dur))
        
        ax.scatter(shifted_time, seg["alt_mag_corr"], color=color, alpha=0.5)
    
    # best-fit line
    x_fit = np.linspace(all_x.min(), all_x.max(), 500)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, color='black', linewidth=2,
            label=f'y={slope:.4f}x+{intercept:.4f}')
    
    ax.set_title("Quiescence aligned by end")
    ax.set_xlabel("Time until next outburst (days)")
    ax.set_ylabel(f"{key} (corrected)")
    ax.invert_yaxis()
    ax.legend()
    
    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Quiescence duration (days)")
    
    plt.show()
    
    
    ##Scaled
    
    
    all_x = []
    all_y = []
    
    for seg in segments_J:
        t0 = seg["MJD"].min()
        t1 = seg["MJD"].max()
        
        # avoid divide-by-zero for very short segments
        if (t1 - t0) <= 0:
            continue
        
        scaled_time = (seg["MJD"] - t0) / (t1 - t0)
        
        all_x.append(scaled_time.values)
        all_y.append(seg["alt_mag_corr"].values)
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
    
    fig, ax = plt.subplots()
    
    # scatter
    for seg, dur in zip(segments_J, durations_J):
        t0 = seg["MJD"].min()
        t1 = seg["MJD"].max()
        
        if (t1 - t0) <= 0:
            continue
        
        scaled_time = (seg["MJD"] - t0) / (t1 - t0)
        color = cmap(norm(dur))
        
        ax.scatter(scaled_time, seg["alt_mag_corr"], color=color, alpha=0.5)
    
    # best-fit line
    x_fit = np.linspace(0, 1, 500)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, color='black', linewidth=2,
            label=f'y={slope:.4f}x+{intercept:.4f}')
    
    ax.set_title("Scaled quiescence (start=0, end=1)")
    ax.set_xlabel("Normalized quiescence phase")
    ax.set_ylabel(f"{key} (corrected)")
    ax.invert_yaxis()
    ax.legend()
    
    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Quiescence duration (days)")
    
    plt.show()



    #same thing but 'Jmag Divided Version'
    
    ##Aligned by start
    
    all_x = []
    all_y = []
    
    for seg in segments_J:
        t0 = seg["MJD"].min()
        shifted_time = seg["MJD"] - t0
        
        all_x.append(shifted_time.values)
        all_y.append(seg['alt_mag_corr'].values)
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    
    
    slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
    
    fig, ax = plt.subplots()
    
    # scatter (your existing loop)
    for seg, dur in zip(segments_J, durations_J):
        t0 = seg["MJD"].min()
        shifted_time = seg["MJD"] - t0
        color = cmap(norm(dur))
        
        ax.scatter(shifted_time, seg['alt_mag_corr'], color=color, alpha=0.5)
    
    # best-fit line
    x_fit = np.linspace(all_x.min(), all_x.max(), 500)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, color='black', linewidth=2, label=f'y={slope:.4f}x+{intercept:2f}')
    ax.set_title("Quiescence aligned by start")
    ax.set_xlabel("Time since quiescence start (days)")
    ax.set_ylabel(f"{key} (divided only)")
    ax.invert_yaxis()
    ax.legend()
    
    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Quiescence duration (days)")
    
    plt.show()
    
    ##Aligned by end
    
    all_x = []
    all_y = []
    
    for seg in segments_J:
        t1 = seg["MJD"].max()
        shifted_time = seg["MJD"] - t1   # <-- key change
        
        all_x.append(shifted_time.values)
        all_y.append(seg['alt_mag_corr'].values)
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
    
    fig, ax = plt.subplots()
    
    # scatter
    for seg, dur in zip(segments_J, durations_J):
        t1 = seg["MJD"].max()
        shifted_time = seg["MJD"] - t1
        color = cmap(norm(dur))
        
        ax.scatter(shifted_time, seg['alt_mag_corr'], color=color, alpha=0.5)
    
    # best-fit line
    x_fit = np.linspace(all_x.min(), all_x.max(), 500)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, color='black', linewidth=2,
            label=f'y={slope:.4f}x+{intercept:.4f}')
    
    ax.set_title("Quiescence aligned by end")
    ax.set_xlabel("Time until next outburst (days)")
    ax.set_ylabel(f"{key} (divided only)")
    ax.invert_yaxis()
    ax.legend()
    
    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Quiescence duration (days)")
    
    plt.show()
    
    
    ##Scaled
    
    
    all_x = []
    all_y = []
    
    for seg in segments_J:
        t0 = seg["MJD"].min()
        t1 = seg["MJD"].max()
        
        # avoid divide-by-zero for very short segments
        if (t1 - t0) <= 0:
            continue
        
        scaled_time = (seg["MJD"] - t0) / (t1 - t0)
        
        all_x.append(scaled_time.values)
        all_y.append(seg['alt_mag_corr'].values)
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    slope, intercept, r_value, p_value, std_err = linregress(all_x, all_y)
    
    fig, ax = plt.subplots()
    
    # scatter
    for seg, dur in zip(segments_J, durations_J):
        t0 = seg["MJD"].min()
        t1 = seg["MJD"].max()
        
        if (t1 - t0) <= 0:
            continue
        
        scaled_time = (seg["MJD"] - t0) / (t1 - t0)
        color = cmap(norm(dur))
        
        ax.scatter(scaled_time, seg['alt_mag_corr'], color=color, alpha=0.5)
    
    # best-fit line
    x_fit = np.linspace(0, 1, 500)
    y_fit = slope * x_fit + intercept
    
    ax.plot(x_fit, y_fit, color='black', linewidth=2,
            label=f'y={slope:.4f}x+{intercept:.4f}')
    
    ax.set_title("Scaled quiescence (start=0, end=1)")
    ax.set_xlabel("Normalized quiescence phase")
    ax.set_ylabel(f"{key} (divided only)")
    ax.invert_yaxis()
    ax.legend()
    
    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Quiescence duration (days)")
    
    plt.show()


#%%
#put this all in one

def align_start(seg):
    t0 = seg["MJD"].min()
    return seg["MJD"] - t0

def align_end(seg):
    t1 = seg["MJD"].max()
    return seg["MJD"] - t1

def align_scaled(seg):
    t0 = seg["MJD"].min()
    t1 = seg["MJD"].max()
    if t1 - t0 <= 0:
        return None
    return (seg["MJD"] - t0) / (t1 - t0)

methods = {
    "Start": align_start,
    "End": align_end,
    "Scaled": align_scaled
}

keys = list(quiescent_tables.keys())

fig, axes = plt.subplots(
    len(keys),
    len(methods),
    figsize=(4 * len(methods), 3 * len(keys)),
    sharey='row'
)

for row_idx, (name, table) in enumerate(quiescent_tables.items()):

    # build segments
    segments = []
    for start, end in quiescent_intervals:
        seg = table[(table["MJD"] >= start) & (table["MJD"] <= end)].copy()
        if len(seg) > 0:
            segments.append(seg)

    durations = np.array([
        seg["MJD"].max() - seg["MJD"].min()
        for seg in segments
    ])

    norm = mcolors.Normalize(vmin=durations.min(), vmax=durations.max())
    cmap = cm.viridis

    for col_idx, (method_name, func) in enumerate(methods.items()):

        ax = axes[row_idx, col_idx]

        all_x, all_y = [], []

        for seg, dur in zip(segments, durations):

            x = func(seg)
            if x is None:
                continue

            y = seg["alt_mag_corr"].values

            all_x.append(x.values)
            all_y.append(y)

            ax.scatter(
                x,
                y,
                color=cmap(norm(dur)),
                alpha=0.5,
                s=10
            )

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        slope, intercept, *_ = linregress(all_x, all_y)

        x_fit = np.linspace(all_x.min(), all_x.max(), 200)
        if slope>=0:
            c='crimson'
        else:
            c='black'
        ax.plot(x_fit, slope * x_fit + intercept, color=c, label=f'y={slope:.4f}x+{intercept:.4f}')

        ax.invert_yaxis()

        # labels only on edges
        if row_idx == 0:
            ax.set_title(method_name)

        if col_idx == 0:
            ax.set_ylabel(name)
        ax.legend()
            
axes[len(keys)-1,2].set_xlabel("Normalized quiescence phase")
axes[len(keys)-1,1].set_xlabel("Time until next outburst (days)")
axes[len(keys)-1,0].set_xlabel("Time since quiescence start (days)")
plt.tight_layout()        
plt.show()

#%%
#sigma clipped version
def table_sigma_clip(x, y, mean, std, sigma=5):
    y = np.asarray(y)
    mask = np.abs(y - mean) < sigma * std
    return x[mask], y[mask]

table_stats = {}

for name, table in quiescent_tables.items():
    vals = table["mag_corr"].values
    vals = vals[~np.isnan(vals)]

    table_stats[name] = {
        "mean": np.nanmean(vals),
        "std": np.nanstd(vals)
    }

fig, axes = plt.subplots(
    len(keys),
    len(methods),
    figsize=(4 * len(methods), 3 * len(keys)),
    sharey='row'
)

for row_idx, (name, table) in enumerate(quiescent_tables.items()):

    # build segments
    segments = []
    for start, end in quiescent_intervals:
        seg = table[(table["MJD"] >= start) & (table["MJD"] <= end)].copy()
        if len(seg) > 0:
            segments.append(seg)

    durations = np.array([
        seg["MJD"].max() - seg["MJD"].min()
        for seg in segments
    ])

    norm = mcolors.Normalize(vmin=durations.min(), vmax=durations.max())
    cmap = cm.viridis

    for col_idx, (method_name, func) in enumerate(methods.items()):

        ax = axes[row_idx, col_idx]

        all_x, all_y = [], []

        for seg, dur in zip(segments, durations):

            
            x = func(seg)
            if x is None:
                continue

            y = seg["alt_mag_corr"].values

            # sigma clip per segment
            stats = table_stats[name]

            xv = x.values
            yv = y
            
            xv, yv = table_sigma_clip(
                xv,
                yv,
                stats["mean"],
                stats["std"],
                sigma=5
            )

            all_x.append(xv)
            all_y.append(yv)

            ax.scatter(
                xv,
                yv,
                color=cmap(norm(dur)),
                alpha=0.5,
                s=10
            )
            

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        slope, intercept, *_ = linregress(all_x, all_y)

        x_fit = np.linspace(all_x.min(), all_x.max(), 200)
        if slope>=0:
            c='crimson'
        else:
            c='black'
        ax.plot(x_fit, slope * x_fit + intercept, color=c, label=f'y={slope:.4f}x+{intercept:.4f}')

        ax.invert_yaxis()

        # labels only on edges
        if row_idx == 0:
            ax.set_title(method_name)

        if col_idx == 0:
            ax.set_ylabel(name)
        ax.legend()
            
axes[len(keys)-1,2].set_xlabel("Normalized quiescence phase")
axes[len(keys)-1,1].set_xlabel("Time until next outburst (days)")
axes[len(keys)-1,0].set_xlabel("Time since quiescence start (days)")
plt.tight_layout()        
plt.show()

#%%

#buffer on quiescence
buffer = 5  # days


full_buffered_intervals = [
    (start - buffer, end + buffer)
    for start, end in full_intervals
]
if count_mini:
    buffered_intervals=full_buffered_intervals
else:
    buffered_intervals = [
        (start - buffer, end + buffer)
        for start, end in intervals
    ]

buffered_quiescent_tables={}
for name, table in tables.items():
    buffered_quiescent_tables[name] = get_quiescent(table, full_buffered_intervals)

# Build quiescent intervals (gaps between outbursts)
buffered_quiescent_intervals = []

for i in range(len(buffered_intervals) - 1):
    end_current = buffered_intervals[i][1]
    start_next = buffered_intervals[i+1][0]
    
    if start_next > end_current:
        buffered_quiescent_intervals.append((end_current, start_next))


fig, axes = plt.subplots(
    len(keys),
    len(methods),
    figsize=(4 * len(methods), 3 * len(keys)),
    sharey='row'
)

for row_idx, (name, table) in enumerate(buffered_quiescent_tables.items()):

    # build segments
    segments = []
    for start, end in buffered_quiescent_intervals:
        seg = table[(table["MJD"] >= start) & (table["MJD"] <= end)].copy()
        if len(seg) > 0:
            segments.append(seg)

    durations = np.array([
        seg["MJD"].max() - seg["MJD"].min()
        for seg in segments
    ])

    norm = mcolors.Normalize(vmin=durations.min(), vmax=durations.max())
    cmap = cm.viridis

    for col_idx, (method_name, func) in enumerate(methods.items()):

        ax = axes[row_idx, col_idx]

        all_x, all_y = [], []

        for seg, dur in zip(segments, durations):

            x = func(seg)
            if x is None:
                continue

            y = seg["alt_mag_corr"].values

            all_x.append(x.values)
            all_y.append(y)

            ax.scatter(
                x,
                y,
                color=cmap(norm(dur)),
                alpha=0.5,
                s=10
            )

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        slope, intercept, *_ = linregress(all_x, all_y)

        x_fit = np.linspace(all_x.min(), all_x.max(), 200)
        if slope>=0:
            c='crimson'
        else:
            c='black'
        ax.plot(x_fit, slope * x_fit + intercept, color=c, label=f'y={slope:.4f}x+{intercept:.4f}')

        ax.invert_yaxis()

        # labels only on edges
        if row_idx == 0:
            ax.set_title(method_name)

        if col_idx == 0:
            ax.set_ylabel(name)
        ax.legend()
            
axes[len(keys)-1,2].set_xlabel("Normalized quiescence phase")
axes[len(keys)-1,1].set_xlabel("Time until next outburst (days)")
axes[len(keys)-1,0].set_xlabel("Time since quiescence start (days)")
plt.tight_layout()        
plt.show()

#%%

#what happens if I collapse these
def build_raw(table, intervals):
    return get_quiescent(table, intervals)

def build_buffered(table, buffered_intervals):
    return get_quiescent(table, buffered_intervals)

def sigma_clip_global(df, mean, std, sigma=5):
    m = np.abs(df["alt_mag_corr"] - mean) < sigma * std
    return df[m].copy()

raw_tables = quiescent_tables
buffered_tables = buffered_quiescent_tables

table_stats = {}
for name, table in quiescent_tables.items():
    vals = table["alt_mag_corr"].dropna().values
    table_stats[name] = (np.mean(vals), np.std(vals))
    
def align_start(seg):
    t0 = seg["MJD"].min()
    return seg["MJD"] - t0

def align_end(seg):
    t1 = seg["MJD"].max()
    return seg["MJD"] - t1

def align_scaled(seg):
    t0 = seg["MJD"].min()
    t1 = seg["MJD"].max()
    if t1 - t0 <= 0:
        return None
    return (seg["MJD"] - t0) / (t1 - t0)

methods = {
    "Start": align_start,
    "End": align_end,
    "Scaled": align_scaled
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)

row_names = ["Raw", "Sigma clipped", "Buffered"]

for row_idx, mode in enumerate(row_names):

    for col_idx, (method_name, func) in enumerate(methods.items()):

        ax = axes[row_idx, col_idx]

        all_x, all_y = [], []

        # choose dataset
        if mode == "Raw":
            data_dict = quiescent_tables
            intervals_use = quiescent_intervals

        elif mode == "Buffered":
            data_dict = buffered_quiescent_tables
            intervals_use = buffered_quiescent_intervals

        elif mode == "Sigma clipped":
            data_dict = {}
            for name, table in quiescent_tables.items():
                mean, std = table_stats[name]
                data_dict[name] = sigma_clip_global(table, mean, std)

            intervals_use = quiescent_intervals

        # flatten ALL bands together
        for name, table in data_dict.items():
            meanmag=table["alt_mag_corr"].mean()

            segments = []
            for start, end in intervals_use:
                seg = table[(table["MJD"] >= start) & (table["MJD"] <= end)].copy()
                if len(seg) > 0:
                    segments.append(seg)

            for seg in segments:

                x = func(seg)
                if x is None:
                    continue

                y = seg["alt_mag_corr"].values
                
                #subtract the mean value from y to put it on the same scale
                y-=meanmag

                all_x.append(x.values)
                all_y.append(y)

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        # fit
        slope, intercept, *_ = linregress(all_x, all_y)

        # intercept-normalised plotting (collapse offsets)
        y_shifted = all_y - intercept

        ax.scatter(all_x, y_shifted, s=5, alpha=0.3)

        x_fit = np.linspace(all_x.min(), all_x.max(), 300)
        if slope>=0:
            c='crimson'
        else:
            c='black'
        ax.plot(
            x_fit,
            slope * x_fit,
            color=c,
            lw=2,
            label=f"slope={slope:.4f}"
        )

        ax.invert_yaxis()

        if row_idx == 0:
            ax.set_title(method_name)

        if col_idx == 0:
            ax.set_ylabel(row_names[row_idx])

        ax.legend()

axes[-1, 0].set_xlabel("Start aligned")
axes[-1, 1].set_xlabel("End aligned")
axes[-1, 2].set_xlabel("Scaled phase")

plt.tight_layout()
plt.show()


#%%

all_mjd = np.concatenate([t["MJD"].values for t in tables.values()])
all_mag = np.concatenate([t["mag_corr"].values for t in tables.values()])
ymin, ymax = np.nanmin(all_mag), np.nanmax(all_mag)
t_min, t_max = np.min(all_mjd), np.max(all_mjd)

edges = np.linspace(t_min, t_max, 5)  # 4 chunks → 5 edges

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

colors = {
    "R Smarts": "crimson",
    "J Smarts": "saddlebrown",
    "R LCO": "orange",
    "rp LCO": "violet",
    "ip LCO": "chocolate",
    "V LCO": "green",
}

for i in range(4):
    ax = axes[i]

    t_start, t_end = edges[i], edges[i+1]

    # --- plot light curves (clipped to time window) ---
    for name, table in tables.items():
        mask = (table["MJD"] >= t_start) & (table["MJD"] < t_end)

        ax.scatter(
            table.loc[mask, "MJD"],
            table.loc[mask, "mag_corr"],
            s=2,
            alpha=0.2,
            color=colors.get(name, "gray"),
            label=name
        )
        
    for name, table in quiescent_tables.items():
        mask = (table["MJD"] >= t_start) & (table["MJD"] < t_end)

        ax.scatter(
            table.loc[mask, "MJD"],
            table.loc[mask, "mag_corr"],
            s=2,
            alpha=1,
            color=colors.get(name, "gray"),
            label=name
        )

    # --- overlays (same logic but clipped) ---
    for start, end in intervals:
        if end < t_start or start > t_end:
            continue
        ax.axvspan(start, end, color="red", alpha=0.15)

    for start, end in quiescent_intervals:
        if end < t_start or start > t_end:
            continue
        ax.axvspan(start, end, color="gray", alpha=0.12)

    for start, end in buffered_intervals:
        if end < t_start or start > t_end:
            continue
        ax.axvspan(start, end, color="black", alpha=0.08)

    ax.set_xlim(t_start, t_end)


    ax.set_title(f"MJD {t_start:.0f} → {t_end:.0f}")

    if i == 0:
        ax.legend(fontsize=8)
for ax in axes:
    ax.set_ylim(ymax, ymin)  # reversed on purpose
axes[-1].set_xlabel("MJD")

plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

row_names = ["Raw", "Sigma clipped", "Buffered"]

for row_idx, mode in enumerate(row_names):

    for col_idx, (method_name, func) in enumerate(methods.items()):

        ax = axes[row_idx, col_idx]
        slopes = []

        # choose dataset
        if mode == "Raw":
            data_dict = quiescent_tables
            intervals_use = quiescent_intervals

        elif mode == "Buffered":
            data_dict = buffered_quiescent_tables
            intervals_use = buffered_quiescent_intervals

        elif mode == "Sigma clipped":
            data_dict = {}
            for name, table in quiescent_tables.items():
                mean, std = table_stats[name]
                data_dict[name] = sigma_clip_global(table, mean, std)

            intervals_use = quiescent_intervals

        # --- loop over ALL tables ---
        for name, table in data_dict.items():

            for start, end in intervals_use:

                seg = table[(table["MJD"] >= start) & (table["MJD"] <= end)]

                if len(seg) < 5:   # avoid garbage fits
                    continue

                x = func(seg)
                if x is None:
                    continue

                y = seg["alt_mag_corr"].values

                slope, intercept, *_ = linregress(x.values, y)

                slopes.append(slope)

        slopes = np.array(slopes)

        # --- plot histogram ---
        ax.hist(slopes, bins=30, alpha=0.7)
        
        # --- median slope ---
        if len(slopes) > 0:
            med = np.median(slopes)
            if med>=0:
                c='crimson'
            else:
                c='black'
            ax.axvline(med, color=c, linewidth=2,
                       label=f"median={med:.4f}")

        # vertical line at 0
        ax.axvline(0, color='black', linestyle='--')

        if row_idx == 0:
            ax.set_title(method_name)

        if col_idx == 0:
            ax.set_ylabel(row_names[row_idx])

        ax.set_xlabel("Slope")
        ax.legend()

plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(
    len(keys),
    len(methods),
    figsize=(4 * len(methods), 3 * len(keys)),
)

for row_idx, (name, table) in enumerate(quiescent_tables.items()):

    # build segments
    segments = []
    for start, end in quiescent_intervals:
        seg = table[(table["MJD"] >= start) & (table["MJD"] <= end)].copy()
        if len(seg) > 2:  # need at least 3 points for a meaningful fit
            segments.append(seg)

    for col_idx, (method_name, func) in enumerate(methods.items()):

        ax = axes[row_idx, col_idx]

        slopes = []

        for seg in segments:

            x = func(seg)
            if x is None:
                continue

            y = seg["alt_mag_corr"].values

            xv = x.values
            yv = y

            # skip tiny segments
            if len(xv) < 3:
                continue

            slope, intercept, *_ = linregress(xv, yv)
            slopes.append(slope)

        slopes = np.array(slopes)

        # --- histogram ---
        ax.hist(slopes, bins=20, alpha=0.7, color="steelblue")

        # --- vertical line at 0 (important reference) ---
        ax.axvline(0, color='black', linestyle='--', linewidth=1)

        # --- median slope ---
        if len(slopes) > 0:
            med = np.median(slopes)
            if med>=0:
                c='crimson'
            else:
                c='black'
            ax.axvline(med, color=c, linewidth=2,
                       label=f"median={med:.4f}")

        # labels
        if row_idx == 0:
            ax.set_title(method_name)

        if col_idx == 0:
            ax.set_ylabel(name)

        ax.legend(fontsize=8)

axes[-1, 0].set_xlabel("Slope (mag/day)")
axes[-1, 1].set_xlabel("Slope (mag/day)")
axes[-1, 2].set_xlabel("Slope (mag/day)")

plt.tight_layout()
plt.show()